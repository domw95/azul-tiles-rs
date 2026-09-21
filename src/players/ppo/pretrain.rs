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
