//! Supervised pre-training of the policy from a teacher's moves.
//!
//! Reinforcement learning has to discover a good move ordering from a scalar
//! reward spread over ~35 decisions. A teacher hands over the answer for every
//! single position, densely and exactly, which removes credit assignment,
//! reward shaping, advantage variance and the critic from the problem at once.

use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement as _;
use burn::tensor::{Int, Tensor, TensorData};

use super::{PPOMoveSelector, ACTION_SIZE, STATE_SIZE};

/// Positions labelled with the move a teacher chose.
#[derive(Default, Debug)]
pub struct Dataset {
    pub states: Vec<f32>,
    pub masks: Vec<f32>,
    pub targets: Vec<i32>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Write to `path` as raw little-endian arrays, so a sweep does not have
    /// to regenerate labels (depth-2 collection costs ~10 minutes).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.targets.len() as u64).to_le_bytes())?;
        for v in &self.states {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.masks {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.targets {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let mut at = 8;
        let mut take_f32 = |count: usize, at: &mut usize| -> Vec<f32> {
            let out = bytes[*at..*at + count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *at += count * 4;
            out
        };
        let states = take_f32(n * STATE_SIZE, &mut at);
        let masks = take_f32(n * ACTION_SIZE, &mut at);
        let targets = bytes[at..at + n * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { states, masks, targets })
    }

    pub fn push(&mut self, state: &[f32], mask: &[f32], target: usize) {
        self.states.extend_from_slice(state);
        self.masks.extend_from_slice(mask);
        self.targets.push(target as i32);
    }
}



/// Enough to replay the games a label set came from.
///
/// Stored instead of the gamestates themselves. The encoding is derived data:
/// change `gs_to_array_for` and every stored state vector is stale, but the
/// move and value labels are properties of the position and stay valid. Given
/// the seed and the moves actually played, every position can be reconstructed
/// exactly and re-encoded without re-running a single search -- which is the
/// difference between a few minutes and several hours.
///
/// Deliberately not serialising `Gamestate`: its fields are private, and its
/// layout is being changed concurrently for unrelated performance work.
#[derive(Default, Debug)]
pub struct Replay {
    /// Seed per game.
    pub seeds: Vec<u64>,
    /// Number of plies (and so of labelled positions) per game.
    pub plies: Vec<u32>,
    /// Move indices actually played, flattened, in game then ply order.
    pub played: Vec<i32>,
}

impl Replay {
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.seeds.len() as u64).to_le_bytes())?;
        f.write_all(&(self.played.len() as u64).to_le_bytes())?;
        for v in &self.seeds {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.plies {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.played {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let b = std::fs::read(path)?;
        let games = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
        let moves = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        let mut at = 16;
        let seeds = b[at..at + games * 8]
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        at += games * 8;
        let plies = b[at..at + games * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        at += games * 4;
        let played = b[at..at + moves * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { seeds, plies, played })
    }

    /// Load every replay file in `dir` whose name starts with `prefix`,
    /// concatenated in filename order so it lines up with the label shards.
    pub fn load_dir(dir: &std::path::Path, prefix: &str) -> std::io::Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
            })
            .collect();
        paths.sort();
        let mut out = Self::default();
        for p in paths {
            let r = Self::load(&p)?;
            out.seeds.extend(r.seeds);
            out.plies.extend(r.plies);
            out.played.extend(r.played);
        }
        Ok(out)
    }
}

/// Positions labelled by several search depths at once.
///
/// A deeper search subsumes the work of the shallower ones, so labelling a
/// position at depths 1..=D costs little more than labelling it at D alone --
/// and it yields a curriculum of teachers plus, from each search's root value,
/// a supervised target for the critic. Training the value head on a real
/// evaluation is worth having: left to reinforcement learning it has only ever
/// explained a fraction of the return variance.
#[derive(Default, Debug)]
pub struct MultiDataset {
    pub states: Vec<f32>,
    pub masks: Vec<f32>,
    /// Which depths were run, e.g. `[1, 2, 3]`.
    pub depths: Vec<u8>,
    /// `targets[d][i]`: the move `depths[d]` chose at position `i`.
    pub targets: Vec<Vec<i32>>,
    /// `values[d][i]`: the root value `depths[d]` returned at position `i`.
    pub values: Vec<Vec<f32>>,
}

impl MultiDataset {
    pub fn len(&self) -> usize {
        self.targets.first().map_or(0, |t| t.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write one shard.
    ///
    /// Sharded rather than one file: a multi-hour generation run that only
    /// writes at the end loses everything to a restart, which has already
    /// happened once here.
    pub fn save_shard(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.len() as u64).to_le_bytes())?;
        f.write_all(&[self.depths.len() as u8])?;
        f.write_all(&self.depths)?;
        for v in &self.states {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.masks {
            f.write_all(&v.to_le_bytes())?;
        }
        for d in 0..self.depths.len() {
            for v in &self.targets[d] {
                f.write_all(&v.to_le_bytes())?;
            }
            for v in &self.values[d] {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        f.flush()
    }

    fn load_shard(path: &std::path::Path) -> std::io::Result<Self> {
        let b = std::fs::read(path)?;
        let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
        let nd = b[8] as usize;
        let depths = b[9..9 + nd].to_vec();
        let mut at = 9 + nd;
        let mut f32s = |count: usize, at: &mut usize| -> Vec<f32> {
            let out = b[*at..*at + count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *at += count * 4;
            out
        };
        let states = f32s(n * STATE_SIZE, &mut at);
        let masks = f32s(n * ACTION_SIZE, &mut at);
        let (mut targets, mut values) = (Vec::new(), Vec::new());
        for _ in 0..nd {
            let t = b[at..at + n * 4]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            at += n * 4;
            targets.push(t);
            values.push(f32s(n, &mut at));
        }
        Ok(Self { states, masks, depths, targets, values })
    }

    /// Load every shard in `dir` whose name starts with `prefix`.
    pub fn load_dir(dir: &std::path::Path, prefix: &str) -> std::io::Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
            })
            .collect();
        paths.sort();
        let mut out = Self::default();
        for p in paths {
            let s = Self::load_shard(&p)?;
            if out.depths.is_empty() {
                out.depths = s.depths.clone();
                out.targets = vec![Vec::new(); out.depths.len()];
                out.values = vec![Vec::new(); out.depths.len()];
            }
            out.states.extend(s.states);
            out.masks.extend(s.masks);
            for d in 0..out.depths.len() {
                out.targets[d].extend(&s.targets[d]);
                out.values[d].extend(&s.values[d]);
            }
        }
        Ok(out)
    }

    /// The single-depth view the policy trainer expects.
    pub fn for_depth(&self, depth: u8) -> Option<Dataset> {
        let d = self.depths.iter().position(|&x| x == depth)?;
        Some(Dataset {
            states: self.states.clone(),
            masks: self.masks.clone(),
            targets: self.targets[d].clone(),
        })
    }
}


/// Initialise the value head from a search's root evaluations.
///
/// Caveat on the target: reinforcement learning trains the critic to predict
/// the *return*, the discounted sum of future per-move rewards, whereas a
/// search returns the differential score as it stands now. Those are not the
/// same quantity, so this is an initialisation rather than the real target --
/// a head that already encodes "who is ahead here" is a far better starting
/// point than noise, and fine-tuning adapts it. Values are scaled by the same
/// /10 the reward uses, so the magnitudes are at least comparable.
pub fn pretrain_value<B: AutodiffBackend>(
    mut ppo: PPOMoveSelector<B>,
    data: &Dataset,
    values: &[f32],
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    device: &B::Device,
) -> PPOMoveSelector<B> {
    let mut optimiser = AdamConfig::new().init();
    let n = data.len() * 9 / 10;
    let val = data.len() - n;

    for epoch in 0..epochs {
        let (mut total, mut batches) = (0.0f32, 0usize);
        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let b = end - start;
            let states = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.states[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
                    [b, STATE_SIZE],
                ),
                device,
            );
            let targets = Tensor::<B, 2>::from_data(
                TensorData::new(
                    values[start..end].iter().map(|v| v / 10.0).collect::<Vec<f32>>(),
                    [b, 1],
                ),
                device,
            );
            let preds = ppo.value.value(states);
            // Squared error by multiplication, not powi_scalar: burn's WGSL
            // backend emits a duplicate powf_primitive definition for that and
            // the shader fails to compile.
            let diff = preds - targets;
            let loss = (diff.clone() * diff).mean();
            total += loss.clone().into_scalar().to_f32();
            batches += 1;
            let grads = loss.backward();
            let params = GradientsParams::from_grads(grads, &ppo.value);
            ppo.value = optimiser.step(learning_rate, ppo.value, params);
        }
        // Explained variance on held-out positions: the same measure the
        // training loop reports, so the two are directly comparable.
        let mut se = 0.0f64;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for start in (n..n + val).step_by(batch_size) {
            let end = (start + batch_size).min(n + val);
            let b = end - start;
            let states = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.states[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
                    [b, STATE_SIZE],
                ),
                device,
            );
            let preds: Vec<f32> = ppo
                .value
                .value(states)
                .to_data()
                .to_vec()
                .unwrap();
            for (k, p) in preds.iter().enumerate() {
                let t = values[start + k] / 10.0;
                se += ((t - p) as f64).powi(2);
                sum += t as f64;
                sq += (t as f64).powi(2);
            }
        }
        let m = sum / val as f64;
        let var = sq / val as f64 - m * m;
        println!(
            "value epoch {epoch}: mse {:.4}, val explained variance {:+.3}",
            total / batches.max(1) as f32,
            1.0 - (se / val as f64) / var.max(1e-9)
        );
    }
    ppo
}

/// Train the policy to reproduce the teacher's choices by cross-entropy.
///
/// Only the policy is touched; the critic is left to reinforcement learning,
/// which is the part that needs it.
pub fn behaviour_clone<B: AutodiffBackend>(
    mut ppo: PPOMoveSelector<B>,
    data: &Dataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    device: &B::Device,
) -> PPOMoveSelector<B> {
    let mut optimiser = AdamConfig::new().init();
    // Hold out the tail as validation: training agreement alone cannot tell
    // "too small to fit" from "memorising".
    let n = data.len() * 9 / 10;
    let val = data.len() - n;

    for epoch in 0..epochs {
        let (mut total, mut batches, mut correct) = (0.0f32, 0usize, 0usize);
        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let b = end - start;

            let states = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.states[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
                    [b, STATE_SIZE],
                ),
                device,
            );
            let masks = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.masks[start * ACTION_SIZE..end * ACTION_SIZE].to_vec(),
                    [b, ACTION_SIZE],
                ),
                device,
            );
            let targets = Tensor::<B, 2, Int>::from_data(
                TensorData::new(data.targets[start..end].to_vec(), [b, 1]),
                device,
            );

            let log_probs = log_softmax(ppo.policy.action(states) + masks, 1);
            // Negative log likelihood of the teacher's move.
            let picked = log_probs.clone().gather(1, targets.clone());
            let loss = -picked.mean();

            total += loss.clone().into_scalar().to_f32();
            batches += 1;

            let chosen = log_probs.argmax(1);
            correct += chosen
                .equal(targets)
                .int()
                .sum()
                .into_scalar()
                .to_usize();

            let grads = loss.backward();
            let params = GradientsParams::from_grads(grads, &ppo.policy);
            ppo.policy = optimiser.step(learning_rate, ppo.policy, params);
        }
        // Validation agreement, no gradient.
        let mut val_correct = 0usize;
        for start in (n..n + val).step_by(batch_size) {
            let end = (start + batch_size).min(n + val);
            let b = end - start;
            let states = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.states[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
                    [b, STATE_SIZE],
                ),
                device,
            );
            let masks = Tensor::<B, 2>::from_data(
                TensorData::new(
                    data.masks[start * ACTION_SIZE..end * ACTION_SIZE].to_vec(),
                    [b, ACTION_SIZE],
                ),
                device,
            );
            let targets = Tensor::<B, 2, Int>::from_data(
                TensorData::new(data.targets[start..end].to_vec(), [b, 1]),
                device,
            );
            let lp = log_softmax(ppo.policy.action(states) + masks, 1);
            val_correct += lp
                .argmax(1)
                .equal(targets)
                .int()
                .sum()
                .into_scalar()
                .to_usize();
        }
        println!(
            "bc epoch {epoch}: loss {:.4}, train {:.1}%, val {:.1}%",
            total / batches.max(1) as f32,
            100.0 * correct as f32 / n as f32,
            100.0 * val_correct as f32 / val.max(1) as f32
        );
    }
    ppo
}
