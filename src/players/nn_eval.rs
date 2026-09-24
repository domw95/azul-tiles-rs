//! The critic's value head, wrapped as a search evaluator.
//!
//! Substituting a network for [`ScoreEvaluator`](super::minimax::ScoreEvaluator)
//! is trivial -- `Minimaxer` is generic over its evaluator -- and that is
//! precisely why it is not the interesting part. The interesting part is cost.
//! Alpha-beta visits thousands of leaves per move and calls `evaluate` at every
//! one; `ScoreEvaluator` is a handful of arithmetic ops, while a burn forward
//! at batch 1 allocates tensors, dispatches kernels and rebuilds a 321-float
//! input vector from scratch. Expect three to four orders of magnitude.
//!
//! Run it with `MATMUL_NUM_THREADS=1` set. Left alone, burn's ndarray backend
//! fans a 321x320 gemv out across sixteen threads and then waits for them, and
//! on a loaded box that costs 2.7 ms a call against 311 us single-threaded --
//! system time running to twice user time. Threading a matrix-vector product
//! this small cannot pay for itself under any load; it is pure loss here.
//!
//! So this type exists to be *measured*, not to be fast. Raced at equal time
//! against the arithmetic evaluator it answers the question that decides
//! whether the NNUE-style path in issue #3 is worth building: is a learned
//! evaluation losing because it is bad, or because it is slow? Equal depth
//! cannot tell those apart, which is how naive attempts at this fail
//! invisibly.

use burn::tensor::backend::Backend;
use burn::tensor::cast::ToElement as _;
use burn::tensor::{Tensor, TensorData};

use crate::gamestate::Gamestate;
use crate::players::nn::gs_to_array_for;
use crate::players::ppo::pretrain::VALUE_SCALE;
use crate::players::ppo::{CheckpointError, PPOMoveSelector, STATE_SIZE};

/// Evaluate positions with a trained value head.
///
/// The model sits behind a mutex because burn stores each parameter in a
/// `OnceCell` for lazy initialisation, which makes a module `Send` but not
/// `Sync`, and `minimaxer::Evaluate` requires both. Uncontended that costs
/// tens of nanoseconds against a forward pass costing tens of microseconds,
/// so it is invisible here -- but it would serialise a `parallel` search
/// completely, and any hand-rolled inference replacing this should drop it.
/// Cloning shares the weights rather than copying them, which is what the
/// per-thread clone in the search wants anyway.
#[derive(Debug, Clone)]
pub struct NnEvaluator<B: Backend> {
    net: std::sync::Arc<std::sync::Mutex<PPOMoveSelector<B>>>,
    device: B::Device,
}

impl<B: Backend> NnEvaluator<B> {
    pub fn new(net: PPOMoveSelector<B>, device: &B::Device) -> Self {
        Self {
            net: std::sync::Arc::new(std::sync::Mutex::new(net)),
            device: device.clone(),
        }
    }

    /// Load a checkpoint written by [`PPOMoveSelector::save`].
    ///
    /// The policy comes along unused; it shares the checkpoint with the critic
    /// and costs only the load.
    pub fn from_checkpoint(
        dir: &std::path::Path,
        tag: impl std::fmt::Display,
        device: &B::Device,
    ) -> Result<Self, CheckpointError> {
        Ok(Self::new(PPOMoveSelector::from_checkpoint(dir, tag, device)?, device))
    }
}

impl<B: Backend> minimaxer::Evaluate<Gamestate<2, 6>> for NnEvaluator<B> {
    /// The score lead, in points, from *player zero's* point of view.
    ///
    /// Two frames meet here and neither is negotiable. The encoding is written
    /// from the acting player's seat, and the value head was fitted against
    /// search root values, which negamax also reports in the root player's
    /// frame -- so the network's output is "how far ahead is the side to
    /// move". `minimaxer::Evaluate` is documented as the first player's frame,
    /// because negamax applies the aim multiplier itself afterwards. Hence the
    /// negation at seat one.
    ///
    /// Getting this backwards is silent: the search would simply play for its
    /// opponent, still legally, still at a plausible node rate, and only the
    /// win rate would say so. `seat_one_is_the_negation_of_seat_zero` pins it.
    fn evaluate(&mut self, g: &Gamestate<2, 6>) -> f32 {
        let seat = g.current_player() as usize;
        let state = gs_to_array_for(g, seat);
        let input = Tensor::<B, 1>::from_data(
            TensorData::new(state.as_slice().to_vec(), [STATE_SIZE]),
            &self.device,
        );
        let from_seat = self
            .net
            .lock()
            .expect("value head")
            .value(input)
            .into_scalar()
            .to_f32()
            * VALUE_SCALE;
        if seat == 0 {
            from_seat
        } else {
            -from_seat
        }
    }
}

/// Spend the expensive evaluation only where the round has ended.
///
/// `is_terminal` is `is_round_over` here, so a search bottoms out at either
/// the depth limit or the end of a round, and both kinds of leaf call
/// `evaluate`. Round ends are the minority, so paying a lot for them and
/// almost nothing elsewhere costs the average of the two weighted by how often
/// each occurs -- measured by `leaf_mix` at 0.1% of evaluations under a fixed
/// depth and 5.2% under a clock, the latter because iterative deepening runs
/// far past the nominal depth in late-round positions, where the tree is
/// truncated by the round ending and so is cheap to exhaust.
///
/// Round end is also the natural place for a learned value: it is the one
/// position whose labels can be made *exact*, since a search with no depth cap
/// terminates there, and it is where a score differential is least informative
/// about what the position is actually worth.
///
/// **A head fitted by `fit_value` on `gen_labels` output has never seen a
/// round-over position.** That generator records a position only when it has
/// legal moves, so every training position had a non-empty factory. Using one
/// here is therefore out of distribution, and a poor result says as much about
/// the labels as about the idea. The exhaustive round-end labels are what this
/// wants.
#[derive(Debug, Clone)]
pub struct RoundEndOnly<S, D> {
    pub shallow: S,
    pub deep: D,
}

impl<S, D> minimaxer::Evaluate<Gamestate<2, 6>> for RoundEndOnly<S, D>
where
    S: minimaxer::Evaluate<Gamestate<2, 6>>,
    D: minimaxer::Evaluate<Gamestate<2, 6>>,
{
    fn evaluate(&mut self, g: &Gamestate<2, 6>) -> f32 {
        if g.is_round_over() {
            self.deep.evaluate(g)
        } else {
            self.shallow.evaluate(g)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::ppo::PPOConfig;
    use burn::backend::NdArray;
    use minimaxer::Evaluate as _;

    /// The same position, reached with either player to move, must evaluate to
    /// the same number with the sign flipped.
    ///
    /// This is the seat convention and nothing else. A random head is enough
    /// to test it -- indeed better than a trained one, whose outputs might be
    /// near enough symmetric to hide a sign error.
    #[test]
    fn seat_one_is_the_negation_of_seat_zero() {
        let device = Default::default();
        let net = PPOMoveSelector::<NdArray>::new(PPOConfig::default(), &device);
        let mut eval = NnEvaluator::new(net, &device);

        // Two gamestates identical but for whose turn it is. Playing one move
        // from a fresh position passes the turn, so evaluate before and after
        // and compare against the encoding directly.
        let gs = Gamestate::<2, 6>::new_2_player_with_seed(7, 0);
        assert_eq!(gs.current_player(), 0);
        let zero = eval.evaluate(&gs);

        let swapped = Gamestate::<2, 6>::new_2_player_with_seed(7, 1);
        assert_eq!(swapped.current_player(), 1);
        // Same seed, so the same deal; both boards are empty, so the encoding
        // is identical and only the seat convention can differ.
        let one = eval.evaluate(&swapped);
        assert!(
            (zero + one).abs() < 1e-4,
            "seat 0 gave {zero}, seat 1 gave {one}; they should be negatives"
        );
        assert!(zero.abs() > 0.0, "a constant-zero head would pass vacuously");
    }
}
