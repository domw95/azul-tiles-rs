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

use crate::gamestate::{Gamestate, Move};
use crate::players::nn::{gs_to_array_ordered, FactoryOrder};

use super::{PPOMoveSelector, ACTION_SIZE, STATE_SIZE};

/// Points per unit of network output.
///
/// The critic is trained on scores divided by this, because reinforcement
/// learning's rewards are scaled the same way and the two have to be
/// commensurate. Anything reading a value back out -- a search evaluator, say
/// -- has to multiply by it again, so the number lives here rather than being
/// written `10.0` at each end and drifting apart silently.
pub const VALUE_SCALE: f32 = 10.0;

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

    /// Append a pre-encoded position.
    ///
    /// `mask` and `target` must be in *canonical* action space, the space the
    /// policy is indexed by. Prefer [`Self::push_position`], which cannot get
    /// that wrong.
    pub fn push(&mut self, state: &[f32], mask: &[f32], target: usize) {
        self.states.extend_from_slice(state);
        self.masks.extend_from_slice(mask);
        self.targets.push(target as i32);
    }

    /// Encode a position and label it with the move the teacher chose.
    ///
    /// The encoder sorts the factory displays into a canonical order, so a
    /// move's action index is not `Move::to_index` -- it is that index with
    /// the source remapped through [`FactoryOrder`]. Labelling a dataset with
    /// the raw index would train the policy to name a display by its original
    /// slot while being shown the sorted one, and nothing downstream would
    /// complain; it would simply never learn. Encoding here, once, removes the
    /// chance of the two disagreeing.
    ///
    /// `moves` are the legal moves in `gs`, and `chosen` must be one of them.
    pub fn push_position(&mut self, gs: &Gamestate<2, 6>, moves: &[Move], chosen: &Move) {
        let order = FactoryOrder::canonical(gs);
        let state = gs_to_array_ordered(gs, gs.current_player() as usize, &order);

        let mut mask = [-1e8f32; ACTION_SIZE];
        for m in moves {
            mask[order.canonical_index(m)] = 0.0;
        }

        let target = order.canonical_index(chosen);
        debug_assert_eq!(mask[target], 0.0, "the teacher's move was not legal");
        self.push(state.as_slice(), &mask, target);
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

/// A borrowed view of labelled positions, for training without copying.
#[derive(Clone, Copy, Debug)]
pub struct DataView<'a> {
    pub states: &'a [f32],
    pub masks: &'a [f32],
    pub targets: &'a [i32],
}

impl DataView<'_> {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl Dataset {
    pub fn view(&self) -> DataView<'_> {
        DataView {
            states: &self.states,
            masks: &self.masks,
            targets: &self.targets,
        }
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

    pub fn load_shard(path: &std::path::Path) -> std::io::Result<Self> {
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

    /// Every shard in `dir` named `prefix*`, in the order the loaders read
    /// them.
    fn shard_paths(
        dir: &std::path::Path,
        prefix: &str,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
            })
            .collect();
        paths.sort();
        Ok(paths)
    }

    /// Load every shard in `dir` whose name starts with `prefix`.
    pub fn load_dir(dir: &std::path::Path, prefix: &str) -> std::io::Result<Self> {
        let paths = Self::shard_paths(dir, prefix)?;
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

    /// Load only what fitting a critic needs: the encoded states, and one
    /// depth's root values.
    ///
    /// [`Self::load_dir`] also brings in the action masks and the move
    /// targets, which a value fit never looks at. On the 2.1M position set
    /// that is 1.5 GB of masks held for nothing, and the peak matters more
    /// than the steady state: this box has no swap and has OOM-killed work
    /// that merely spiked on the way in.
    ///
    /// So the shard headers are read first and the buffers allocated once at
    /// their final size. Growing them by `extend` transiently needs about
    /// three times the final size at the last reallocation, which is the
    /// difference between fitting and not.
    pub fn load_dir_values(
        dir: &std::path::Path,
        prefix: &str,
        depth: u8,
    ) -> std::io::Result<(Vec<f32>, Vec<f32>)> {
        use std::io::{Read, Seek};

        let paths = Self::shard_paths(dir, prefix)?;
        let mut counts = Vec::with_capacity(paths.len());
        let mut total = 0usize;
        for path in &paths {
            let mut head = [0u8; 9];
            std::fs::File::open(path)?.read_exact(&mut head)?;
            let n = u64::from_le_bytes(head[0..8].try_into().unwrap()) as usize;
            counts.push(n);
            total += n;
        }

        let mut states = Vec::with_capacity(total * STATE_SIZE);
        let mut values = Vec::with_capacity(total);
        let mut buf = Vec::new();
        for (path, n) in paths.iter().zip(counts) {
            let mut f = std::fs::File::open(path)?;
            let mut head = [0u8; 9];
            f.read_exact(&mut head)?;
            let nd = head[8] as usize;
            let mut depths = vec![0u8; nd];
            f.read_exact(&mut depths)?;
            let Some(d) = depths.iter().position(|&x| x == depth) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}: no depth {depth} in {depths:?}", path.display()),
                ));
            };

            buf.resize(n * STATE_SIZE * 4, 0);
            f.read_exact(&mut buf)?;
            states.extend(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );

            // Past the masks, then past every earlier depth's targets and
            // values, then past this depth's targets.
            let skip = (n * ACTION_SIZE * 4) + (d * n * 8) + (n * 4);
            f.seek(std::io::SeekFrom::Current(skip as i64))?;
            buf.resize(n * 4, 0);
            f.read_exact(&mut buf)?;
            values.extend(
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );
        }
        Ok((states, values))
    }

    /// Borrow the labels for one depth.
    ///
    /// A view, not a copy: at 2.1M positions the states and masks are several
    /// gigabytes, and cloning them to select a depth doubled peak memory --
    /// enough, alongside a training run, to lose the training run.
    pub fn view_depth(&self, depth: u8) -> Option<DataView<'_>> {
        let d = self.depths.iter().position(|&x| x == depth)?;
        Some(DataView {
            states: &self.states,
            masks: &self.masks,
            targets: &self.targets[d],
        })
    }
}


/// Fit the value head to a search's root evaluations.
///
/// The root value a search returns is in the *root player's* frame, and the
/// state is encoded from that same seat, so the two agree and the head learns
/// "how far ahead is the side to move". [`crate::players::nn_eval::NnEvaluator`]
/// is what reads it back out, and it is that evaluator which has to undo the
/// seat convention, because `minimaxer::Evaluate` wants the first player's
/// frame instead.
///
/// Caveat on using this for reinforcement learning specifically: RL trains the
/// critic to predict the *return*, the discounted sum of future per-move
/// rewards, whereas a search returns the differential score as it stands now.
/// Those are not the same quantity, so as a critic initialisation this is a
/// starting point rather than the target -- and it measured as a null in that
/// role (+0.94 on held-out search values, +0.01 transfer to RL returns). As a
/// search evaluator, which is what issue #3 is about, the quantity it predicts
/// is exactly the right one.
///
/// Stopping is [`CloneStop`], shared with [`behaviour_clone`]; the metric it
/// compares is validation explained variance rather than move agreement, both
/// being "higher is better" on roughly the same scale. Returning the last
/// epoch rather than the best was the flaw cloning had already lost, and it
/// matters more here: mean squared error on a heavy-tailed target overfits
/// visibly within a handful of epochs.
///
/// `on_best` is called with the model each time the held-out score improves,
/// which is where a caller writes a checkpoint. An epoch over 2.1M positions
/// takes minutes on a contended box, so a run that only saves when it returns
/// can be hours of work with nothing on disk -- and the reason the run ends is
/// as often a kill as a stopping rule. The same lesson the PPO loop learned
/// from losing 1845 episodes.
pub fn pretrain_value<B: AutodiffBackend>(
    mut ppo: PPOMoveSelector<B>,
    states_all: &[f32],
    values: &[f32],
    stop: CloneStop,
    batch_size: usize,
    learning_rate: f64,
    device: &B::Device,
    mut on_best: impl FnMut(&PPOMoveSelector<B>, usize, f32),
) -> (PPOMoveSelector<B>, CloneSummary) {
    let mut optimiser = AdamConfig::new().init();
    let mut best_value = ppo.value.clone();
    let mut best_val = f32::NEG_INFINITY;
    let mut best_epoch = 0usize;
    let mut epochs_run = 0usize;
    let mut stopped_early = false;
    // The split is positional, not shuffled, and deliberately so: the labels
    // arrive in game order, so a random split would put positions from the
    // same game on both sides and report a leak as generalisation.
    let n = values.len() * 9 / 10;
    let val = values.len() - n;

    for epoch in 0..stop.max_epochs {
        epochs_run = epoch + 1;
        let (mut total, mut batches) = (0.0f32, 0usize);
        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let b = end - start;
            let states = Tensor::<B, 2>::from_data(
                TensorData::new(
                    states_all[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
                    [b, STATE_SIZE],
                ),
                device,
            );
            let targets = Tensor::<B, 2>::from_data(
                TensorData::new(
                    values[start..end]
                        .iter()
                        .map(|v| v / VALUE_SCALE)
                        .collect::<Vec<f32>>(),
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
                    states_all[start * STATE_SIZE..end * STATE_SIZE].to_vec(),
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
                let t = values[start + k] / VALUE_SCALE;
                se += ((t - p) as f64).powi(2);
                sum += t as f64;
                sq += (t as f64).powi(2);
            }
        }
        let m = sum / val as f64;
        let var = sq / val as f64 - m * m;
        let ev = (1.0 - (se / val as f64) / var.max(1e-9)) as f32;
        let improved = ev > best_val + stop.min_delta;
        if improved {
            best_val = ev;
            best_epoch = epoch;
            best_value = ppo.value.clone();
            // `ppo` is the best model at this instant, so hand it over before
            // the next epoch moves it on.
            on_best(&ppo, epoch, ev);
        }
        println!(
            "value epoch {epoch}: mse {:.4}, val explained variance {ev:+.3}{}",
            total / batches.max(1) as f32,
            if improved { " *" } else { "" }
        );
        if epoch >= best_epoch + stop.patience {
            println!(
                "value stopped: no gain for {} epochs; best {best_val:+.3} at epoch {best_epoch}",
                stop.patience
            );
            stopped_early = true;
            break;
        }
    }

    if !stopped_early {
        println!(
            "value hit the epoch cap at {epochs_run}; best {best_val:+.3} at epoch {best_epoch} -- raise max_epochs if that is near the end"
        );
    }
    // Return the best, not the last.
    ppo.value = best_value;
    (ppo, CloneSummary { epochs_run, best_epoch, best_val, stopped_early })
}

/// When to stop cloning.
///
/// This used to be a bare epoch count chosen by feel, returning whatever the
/// last epoch produced. Both are flaws: a 16-epoch run was still gaining
/// validation agreement when it stopped, and once a run does start
/// overfitting, returning the last epoch silently keeps the worse model.
#[derive(Debug, Clone, Copy)]
pub struct CloneStop {
    /// Hard cap.
    pub max_epochs: usize,
    /// Stop after this many epochs with no improvement in validation agreement.
    pub patience: usize,
    /// Improvement smaller than this does not count as progress.
    pub min_delta: f32,
}

impl Default for CloneStop {
    fn default() -> Self {
        Self { max_epochs: 200, patience: 5, min_delta: 0.0005 }
    }
}

/// What a cloning run produced.
#[derive(Debug, Clone, Copy)]
pub struct CloneSummary {
    pub epochs_run: usize,
    pub best_epoch: usize,
    pub best_val: f32,
    /// Training agreement in the same epoch as `best_val`. The gap between
    /// the two is what says whether a net is too small or is memorising.
    pub best_train: f32,
    pub stopped_early: bool,
}

/// Train the policy to reproduce the teacher's choices by cross-entropy.
///
/// Only the policy is touched; the critic is left to reinforcement learning,
/// which is the part that needs it.
pub fn behaviour_clone<B: AutodiffBackend>(
    mut ppo: PPOMoveSelector<B>,
    data: DataView<'_>,
    stop: CloneStop,
    batch_size: usize,
    learning_rate: f64,
    device: &B::Device,
) -> (PPOMoveSelector<B>, CloneSummary) {
    let mut optimiser = AdamConfig::new().init();
    // Snapshot of the best policy by validation agreement, so a run that
    // starts overfitting still returns its best model rather than its last.
    let mut best_policy = ppo.policy.clone();
    let mut best_val = f32::NEG_INFINITY;
    let mut best_train = 0.0f32;
    let mut best_epoch = 0usize;
    let mut epochs_run = 0usize;
    let mut stopped_early = false;
    // Hold out the tail as validation: training agreement alone cannot tell
    // "too small to fit" from "memorising".
    let n = data.len() * 9 / 10;
    let val = data.len() - n;

    for epoch in 0..stop.max_epochs {
        epochs_run = epoch + 1;
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
        let val_acc = val_correct as f32 / val.max(1) as f32;
        let train_acc = correct as f32 / n.max(1) as f32;
        let improved = val_acc > best_val + stop.min_delta;
        if improved {
            best_val = val_acc;
            best_train = train_acc;
            best_epoch = epoch;
            best_policy = ppo.policy.clone();
        }
        println!(
            "bc epoch {epoch}: loss {:.4}, train {:.1}%, val {:.1}%{}",
            total / batches.max(1) as f32,
            100.0 * train_acc,
            100.0 * val_acc,
            if improved { " *" } else { "" }
        );
        if epoch >= best_epoch + stop.patience {
            println!(
                "bc stopped: no validation gain for {} epochs; best {:.1}% at epoch {best_epoch}",
                stop.patience,
                100.0 * best_val
            );
            stopped_early = true;
            break;
        }
    }

    if !stopped_early {
        println!(
            "bc hit the epoch cap at {epochs_run}; best {:.1}% at epoch {best_epoch} -- raise max_epochs if that is near the end",
            100.0 * best_val
        );
    }
    // Return the best, not the last.
    ppo.policy = best_policy;
    (ppo, CloneSummary { epochs_run, best_epoch, best_val, best_train, stopped_early })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dataset row has to line up with what the policy is indexed by: the
    /// label must be a legal slot in its own mask, and both must be in the
    /// canonical action space the state was encoded in.
    #[test]
    fn a_pushed_position_labels_a_legal_canonical_action() {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(0, 0);
        let mut data = Dataset::default();
        let mut rows = 0;

        for ply in 0..30 {
            let moves = gs.get_moves();
            if moves.is_empty() {
                break;
            }
            let chosen = moves[ply % moves.len()];
            data.push_position(&gs, &moves, &chosen);
            rows += 1;

            let target = data.targets[rows - 1] as usize;
            let mask = &data.masks[(rows - 1) * ACTION_SIZE..rows * ACTION_SIZE];
            assert_eq!(mask[target], 0.0, "ply {ply}: label is masked out");
            assert_eq!(
                mask.iter().filter(|&&m| m == 0.0).count(),
                moves.len(),
                "ply {ply}: mask opens a different number of slots than there \
                 are legal moves, so two moves collided"
            );

            // The state is the acting player's view, at the declared width.
            assert_eq!(data.states.len(), rows * STATE_SIZE);

            gs.play_move(chosen);
        }

        assert!(rows > 5, "only {rows} rows, test proves little");
    }
}
