//! Players, and the trait the game runner drives them through.

pub mod minimax;

#[cfg(feature = "full")]
pub mod nn;
#[cfg(feature = "full")]
pub mod nn_eval;
#[cfg(feature = "full")]
pub mod ppo;

// Everything but the minimax search pulls in the training and GUI stack. The
// re-export keeps the public paths (`players::RandomPlayer` and friends)
// exactly where they were.
#[cfg(feature = "full")]
mod agents;
#[cfg(feature = "full")]
pub use agents::*;
