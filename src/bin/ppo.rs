#![recursion_limit = "256"]
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::train::PPOTrainer;
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::{Device, Tensor};
use minimaxer::negamax::SearchOptions;

// Wgpu needs a Vulkan ICD present; without one this binary cannot start at
// all, so the CPU backend is the default and the GPU is opt-in:
//     cargo run --release --bin ppo --features gpu
#[cfg(feature = "gpu")]
type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type Backend = burn::backend::Autodiff<burn::backend::NdArray>;

/// Search depth of the minimax opponent the agent trains against.
///
/// Depth 1 is barely a search at all, so a policy that beats it has not yet
/// beaten "minimax" in any meaningful sense. Raise this once the agent is
/// reliably winning.
const OPPONENT_DEPTH: u8 = 1;

fn main() {
    let device = Device::<Backend>::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );
    // Create policy and value networks. The shapes are saved with each
    // checkpoint, so nothing downstream has to restate them.
    let ppo = PPOMoveSelector::<Backend>::new(PPOConfig::default(), &device);

    println!("PPO Move Selector: {:?}", ppo);

    // Create a basic opponent
    let opponent = Box::new(Minimaxer::new(
        SearchOptions {
            max_depth: Some(OPPONENT_DEPTH),
            ..Default::default()
        },
        format!("Depth{OPPONENT_DEPTH}"),
        ScoreEvaluator,
    ));

    let (_, summary) = PPOTrainer::new(ppo, opponent, &device).train();
    println!("{summary:?}");
}
