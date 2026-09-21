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

    pub fn push(&mut self, state: &[f32], mask: &[f32], target: usize) {
        self.states.extend_from_slice(state);
        self.masks.extend_from_slice(mask);
        self.targets.push(target as i32);
    }
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
    let n = data.len();

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
        println!(
            "bc epoch {epoch}: loss {:.4}, top-1 agreement {:.1}%",
            total / batches.max(1) as f32,
            100.0 * correct as f32 / n as f32
        );
    }
    ppo
}
