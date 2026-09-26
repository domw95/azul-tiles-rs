//! Training over a label set that stays on disk.
//!
//! [`MultiDataset::load_dir`] decodes every shard into one set of `Vec<f32>`
//! and hands back a borrow of it, so the whole dataset lives in the process
//! for as long as training does. That is fine at the scale it was written for
//! and stops being fine quickly: a position costs 2,012 bytes of states, masks
//! and labels, so 4.1M positions is 8.3 GiB before the model, the optimiser or
//! anything else on the box. Generation currently adds about 1.2M positions a
//! day.
//!
//! This module reads the same shards a few at a time instead. Resident cost is
//! the buffer, not the dataset: 64 shards of 500 positions is about 64 MiB, so
//! the ceiling moves from "how much RAM is there" to "how much disk is there",
//! which is a far larger number and one that compresses 5x.
//!
//! **None of this touches the GPU.** [`super::pretrain::behaviour_clone`]
//! already builds one tensor per batch and drops it, so device memory is a
//! function of batch size and model size alone and never saw the dataset. A
//! 256-row batch is 513 KiB of inputs against a 264k-parameter model; the
//! dataset could be 40M positions or 40 thousand and the GPU would not know.
//! Streaming changes host RAM only.

use std::path::{Path, PathBuf};

use rand::seq::SliceRandom;
use rand::Rng;

use super::pretrain::{DataView, MultiDataset};

/// One shuffled chunk of positions, owned and ready to slice into batches.
pub struct Chunk {
    pub states: Vec<f32>,
    pub masks: Vec<f32>,
    pub targets: Vec<i32>,
    pub values: Vec<f32>,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Borrow rows `start..end` in the shape the trainer wants.
    ///
    /// The rows are already shuffled, so a contiguous range is a random
    /// sample and the batch needs no gather.
    pub fn view(&self, start: usize, end: usize) -> DataView<'_> {
        DataView {
            states: &self.states[start * super::STATE_SIZE..end * super::STATE_SIZE],
            masks: &self.masks[start * super::ACTION_SIZE..end * super::ACTION_SIZE],
            targets: &self.targets[start..end],
        }
    }
}

/// A label set addressed by its shard files rather than its contents.
#[derive(Clone, Debug)]
pub struct ShardSet {
    shards: Vec<PathBuf>,
    depth: u8,
    len: usize,
}

impl ShardSet {
    /// Index `dir` without reading the bulk of it.
    ///
    /// Only each shard's header is read, which is enough for the position
    /// count and the depths present. Counting by opening 28,000 files is still
    /// slow enough to be worth doing once, so the total is cached here rather
    /// than recomputed per epoch.
    pub fn open(dir: &Path, prefix: &str, depth: u8) -> std::io::Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
            })
            .collect();
        // Sorted so a train/validation split by position in this list is the
        // same split on every run and every machine.
        shards.sort();
        let mut len = 0usize;
        for p in &shards {
            let mut f = std::fs::File::open(p)?;
            let mut head = [0u8; 8];
            use std::io::Read;
            f.read_exact(&mut head)?;
            len += u64::from_le_bytes(head) as usize;
        }
        Ok(Self { shards, depth, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Split off the tail as validation, **by shard rather than by position**.
    ///
    /// A position-level split would put plies from one game on both sides:
    /// consecutive positions in a game differ by a single move, so the model
    /// would be validated on near-duplicates of what it trained on and every
    /// agreement figure would read high. Whole shards never straddle the line,
    /// and with one worker per shard the held-out games come from seeds the
    /// training side never saw.
    pub fn split(&self, val_fraction: f64) -> (Self, Self) {
        let n_val = ((self.shards.len() as f64 * val_fraction).round() as usize)
            .clamp(1, self.shards.len().saturating_sub(1));
        let cut = self.shards.len() - n_val;
        let mk = |paths: Vec<PathBuf>| {
            let len = paths.iter().map(|p| shard_len(p).unwrap_or(0)).sum();
            Self { shards: paths, depth: self.depth, len }
        };
        (mk(self.shards[..cut].to_vec()), mk(self.shards[cut..].to_vec()))
    }

    /// Chunks for one pass, shard order reshuffled and rows shuffled within
    /// each chunk.
    ///
    /// Shuffling matters more here than it does for an in-memory set, because
    /// on disk the rows arrive in game order: consecutive positions are one
    /// move apart, so an unshuffled batch is a handful of near-identical
    /// states and the gradient it produces is close to a single sample's.
    /// `buffer_shards` sets how far a row can travel -- larger is a better
    /// shuffle and more resident memory.
    pub fn epoch<'a, R: Rng>(
        &'a self,
        buffer_shards: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = std::io::Result<Chunk>> + 'a {
        let mut order: Vec<usize> = (0..self.shards.len()).collect();
        order.shuffle(rng);
        let groups: Vec<Vec<usize>> =
            order.chunks(buffer_shards.max(1)).map(|c| c.to_vec()).collect();
        groups.into_iter().map(move |group| {
            let mut chunk = self.load_group(&group)?;
            shuffle_rows(&mut chunk, rng);
            Ok(chunk)
        })
    }

    /// Chunks in a fixed order with no shuffling, for validation.
    pub fn sequential(
        &self,
        buffer_shards: usize,
    ) -> impl Iterator<Item = std::io::Result<Chunk>> + '_ {
        let groups: Vec<Vec<usize>> = (0..self.shards.len())
            .collect::<Vec<_>>()
            .chunks(buffer_shards.max(1))
            .map(|c| c.to_vec())
            .collect();
        groups.into_iter().map(move |group| self.load_group(&group))
    }

    fn load_group(&self, group: &[usize]) -> std::io::Result<Chunk> {
        let mut chunk =
            Chunk { states: Vec::new(), masks: Vec::new(), targets: Vec::new(), values: Vec::new() };
        for &i in group {
            let m = MultiDataset::load_shard(&self.shards[i])?;
            let d = m.depths.iter().position(|&x| x == self.depth).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{}: depth {} not among {:?}",
                        self.shards[i].display(),
                        self.depth,
                        m.depths
                    ),
                )
            })?;
            chunk.states.extend(m.states);
            chunk.masks.extend(m.masks);
            chunk.targets.extend(&m.targets[d]);
            chunk.values.extend(&m.values[d]);
        }
        Ok(chunk)
    }
}

fn shard_len(path: &Path) -> std::io::Result<usize> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut head = [0u8; 8];
    f.read_exact(&mut head)?;
    Ok(u64::from_le_bytes(head) as usize)
}

/// Permute rows in place, so a contiguous batch is a random sample.
///
/// Done by building the permuted arrays rather than swapping in place: a row
/// is 2,004 bytes spread over two arrays, and a Fisher-Yates over that many
/// bytes costs more than the copy.
fn shuffle_rows<R: Rng>(chunk: &mut Chunk, rng: &mut R) {
    let n = chunk.len();
    if n < 2 {
        return;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    let (ss, as_) = (super::STATE_SIZE, super::ACTION_SIZE);
    let mut states = Vec::with_capacity(n * ss);
    let mut masks = Vec::with_capacity(n * as_);
    let mut targets = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for &i in &idx {
        states.extend_from_slice(&chunk.states[i * ss..(i + 1) * ss]);
        masks.extend_from_slice(&chunk.masks[i * as_..(i + 1) * as_]);
        targets.push(chunk.targets[i]);
        values.push(chunk.values[i]);
    }
    chunk.states = states;
    chunk.masks = masks;
    chunk.targets = targets;
    chunk.values = values;
}

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement as _;
use burn::tensor::{Int, Tensor, TensorData};

use super::pretrain::{CloneStop, CloneSummary};
use super::{PPOMoveSelector, ACTION_SIZE, STATE_SIZE};

/// Move one chunk's rows onto the device, batch by batch.
fn batch_tensors<B: AutodiffBackend>(
    chunk: &Chunk,
    start: usize,
    end: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2, Int>) {
    let b = end - start;
    let v = chunk.view(start, end);
    (
        Tensor::from_data(TensorData::new(v.states.to_vec(), [b, STATE_SIZE]), device),
        Tensor::from_data(TensorData::new(v.masks.to_vec(), [b, ACTION_SIZE]), device),
        Tensor::from_data(TensorData::new(v.targets.to_vec(), [b, 1]), device),
    )
}

/// Fraction of `set` where the policy's best legal move is the teacher's.
///
/// Runs on `B::InnerBackend` via [`AutodiffModule::valid`], and that is not a
/// tidiness point -- it is the difference between 73 MiB and 5.4 GiB.
///
/// Scoring builds the same graph training does, but there is no `backward()`
/// here to consume it, so on the autodiff backend every forward pass is
/// retained until the pass ends. The cost is then proportional to the
/// validation set, which is proportional to the dataset -- so it grows with
/// exactly the quantity streaming exists to stop mattering, and it grows in a
/// phase that looks idle from the outside. Measured on 4.17M positions:
/// training held 49-73 MiB throughout and peak RSS still came out at 5,475
/// MiB, all of it accrued during validation.
///
/// The inner backend has no graph to retain, so this now costs one batch.
pub fn agreement<B: AutodiffBackend>(
    ppo: &PPOMoveSelector<B>,
    set: &ShardSet,
    batch_size: usize,
    buffer_shards: usize,
    device: &B::Device,
) -> std::io::Result<f32> {
    type Inner<B> = <B as AutodiffBackend>::InnerBackend;
    let policy = ppo.policy.valid();
    let (mut seen, mut correct) = (0usize, 0usize);
    for chunk in set.sequential(buffer_shards) {
        let chunk = chunk?;
        for start in (0..chunk.len()).step_by(batch_size) {
            let end = (start + batch_size).min(chunk.len());
            let b = end - start;
            let v = chunk.view(start, end);
            let states = Tensor::<Inner<B>, 2>::from_data(
                TensorData::new(v.states.to_vec(), [b, STATE_SIZE]),
                device,
            );
            let masks = Tensor::<Inner<B>, 2>::from_data(
                TensorData::new(v.masks.to_vec(), [b, ACTION_SIZE]),
                device,
            );
            let targets = Tensor::<Inner<B>, 2, Int>::from_data(
                TensorData::new(v.targets.to_vec(), [b, 1]),
                device,
            );
            let scores = log_softmax(policy.action(states) + masks, 1);
            correct += scores.argmax(1).equal(targets).int().sum().into_scalar().to_usize();
            seen += b;
        }
    }
    Ok(correct as f32 / seen.max(1) as f32)
}

/// [`super::pretrain::behaviour_clone`], but reading shards from disk.
///
/// Same objective, same early stopping, same best-model selection. Two things
/// differ, and both are consequences of the data no longer fitting in memory:
///
/// - Rows are shuffled, within a window of `buffer_shards` shards. The
///   in-memory version walks the dataset in order, which was survivable when
///   `load_dir` had already concatenated every worker's output; reading shards
///   one at a time, an unshuffled batch would be consecutive plies of a single
///   game -- states one move apart, carrying roughly one sample's worth of
///   gradient between them.
/// - Validation is a separate [`ShardSet`] split off by shard rather than the
///   tail 10% of an array, so no game has plies on both sides of the split.
///
/// Returns the best policy by validation agreement, not the last.
#[allow(clippy::too_many_arguments)]
pub fn behaviour_clone_streaming<B: AutodiffBackend>(
    mut ppo: PPOMoveSelector<B>,
    train: &ShardSet,
    val: &ShardSet,
    stop: CloneStop,
    batch_size: usize,
    buffer_shards: usize,
    learning_rate: f64,
    device: &B::Device,
) -> std::io::Result<(PPOMoveSelector<B>, CloneSummary)> {
    let mut optimiser = AdamConfig::new().init();
    let mut rng = rand::thread_rng();
    let mut best_policy = ppo.policy.clone();
    let mut best_val = f32::NEG_INFINITY;
    let mut best_epoch = 0usize;
    let mut epochs_run = 0usize;
    let mut stopped_early = false;

    for epoch in 0..stop.max_epochs {
        epochs_run = epoch + 1;
        let (mut loss_sum, mut batches) = (0.0f32, 0usize);
        for chunk in train.epoch(buffer_shards, &mut rng) {
            let chunk = chunk?;
            for start in (0..chunk.len()).step_by(batch_size) {
                let end = (start + batch_size).min(chunk.len());
                let (states, masks, targets) =
                    batch_tensors::<B>(&chunk, start, end, device);
                let log_probs = log_softmax(ppo.policy.action(states) + masks, 1);
                // Negative log likelihood of the teacher's move.
                let loss = -log_probs.gather(1, targets).mean();
                loss_sum += loss.clone().into_scalar().to_f32();
                batches += 1;
                let grads = loss.backward();
                let params = GradientsParams::from_grads(grads, &ppo.policy);
                ppo.policy = optimiser.step(learning_rate, ppo.policy, params);
            }
        }

        let v = agreement(&ppo, val, batch_size, buffer_shards, device)?;
        log::info!(
            "epoch {epochs_run}: loss {:.4}, val agreement {:.4}",
            loss_sum / batches.max(1) as f32,
            v
        );
        if v > best_val + stop.min_delta {
            best_val = v;
            best_epoch = epochs_run;
            best_policy = ppo.policy.clone();
        } else if epochs_run - best_epoch >= stop.patience {
            stopped_early = true;
            break;
        }
    }

    ppo.policy = best_policy;
    Ok((ppo, CloneSummary { epochs_run, best_epoch, best_val, stopped_early }))
}
