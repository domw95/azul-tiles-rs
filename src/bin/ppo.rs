use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::train::PPOTrainer;
use azul_tiles_rs::players::ppo::{PPOMoveSelector, PolicyConfig, ValueConfig};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::{Device, Tensor};
use minimaxer::negamax::SearchOptions;

use burn::backend::{Autodiff, NdArray, Wgpu};

type Backend = Autodiff<Wgpu>; //Wgpu; //NdArray;

fn main() {
    let device = Device::<Backend>::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::OpenGl>(
        &device,
        Default::default(),
    );
    // Create policy and value networks
    let policy_config = PolicyConfig {
        input_size: 150,
        hidden_size: 320,
    };
    let value_config = ValueConfig {
        input_size: 150,
        hidden_size: 320,
    };
    let mut ppo = PPOMoveSelector::<Backend>::new(policy_config, value_config, &device);

    println!("PPO Move Selector: {:?}", ppo);

    // Create a basic opponent
    let opponent = Box::new(Minimaxer::new(
        SearchOptions {
            max_depth: Some(1),
            ..Default::default()
        },
        "Depth1",
        ScoreEvaluator,
    ));

    let mut trainer = PPOTrainer::new(ppo, opponent, &device);

    trainer.train();
}
