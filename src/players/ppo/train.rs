use std::ops::AddAssign;

use burn::module::Module;
use burn::nn::loss::HuberLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{self, DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement as _;
use burn::{prelude::Backend, tensor::Tensor};
use log::trace;
use nalgebra::{DVector, SVector};

use crate::gamestate::{Gamestate, State};
use crate::players::{ppo::PPOMoveSelector, Player};
/// Train a PPO agent against another player
///
/// Runs a matchup, collecting state and rewards
/// then trains the player based on outcome
pub struct PPOTrainer<B: Backend> {
    ppo: PPOMoveSelector<B>,
    opponent: Box<dyn Player<2, 6>>,
    device: B::Device,
}

impl<B: AutodiffBackend> PPOTrainer<B> {
    pub fn new(
        ppo: PPOMoveSelector<B>,
        opponent: Box<dyn Player<2, 6>>,
        device: &B::Device,
    ) -> Self {
        Self {
            ppo,
            opponent,
            device: device.clone(),
        }
    }

    pub fn train(mut self) {
        // create optimiser for policy and critic
        let mut policy_optimiser = AdamConfig::new().init();
        // let mut critic_optimiser = AdamConfig::new().init();
        let mut critic_optimiser = AdamConfig::new().init();

        let mut ppo = self.ppo;
        let mut opponent = self.opponent;
        let device = self.device;

        let gamma = 0.99;
        let epsilon = 0.1;
        let episodes = 1000;
        let epochs = 5;
        let batch_size = 128;
        let games_per_episode = 40;
        let learning_rate = 0.001;

        // Create dir to store progress
        let dir = std::path::Path::new("ppo");
        std::fs::create_dir_all(dir).unwrap();
        let mut recorder: record::NamedMpkFileRecorder<FullPrecisionSettings> =
            DefaultFileRecorder::default();

        for episode in 0..episodes {
            println!("Episode: {}", episode);
            let mut data = Data::default();
            let results = play_games(&mut ppo, &mut opponent, games_per_episode);
            // Convert each result into a batch and append to batch
            for result in results {
                let returns = returns(&device, &result.rewards, gamma);
                let advantages = advantages(&device, &returns, &result.values);
                data += Data {
                    states: result.states,
                    returns,
                    advantages,
                    action_logs: result.action_logs,
                    actions: result.actions,
                    action_masks: result.action_masks,
                };
            }
            println!(
                " Collected {} states from {} games",
                data.states.len(),
                games_per_episode
            );
            // Detach the tensors from the computation graph
            data.detach();

            for epoch in 0..epochs {
                let mut batch = 0;
                // Iterate over batches of batch_size
                while batch * batch_size < data.states.len() {
                    let start = batch * batch_size;
                    let end = ((batch + 1) * batch_size).min(data.states.len());
                    let states = &data.states[start..end];
                    let returns = &data.returns[start..end];
                    let advantages = &data.advantages[start..end];
                    let action_logs = &data.action_logs[start..end];
                    let actions = &data.actions[start..end];
                    let action_masks = &data.action_masks[start..end];

                    // calculate softmax of masked actions of current policy and predicted value
                    let (value_preds, action_log_new): (Vec<Tensor<B, 1>>, Vec<Tensor<B, 1>>) =
                        states
                            .iter()
                            .zip(action_masks)
                            .map(|(s, m)| {
                                (
                                    ppo.value(s.clone()),
                                    softmax(ppo.action(s.clone()) + m.clone(), 0),
                                )
                            })
                            .unzip();
                    // calculate the surrogate loss
                    let surrogate_loss = surrogate_loss(
                        &device,
                        action_logs,
                        &action_log_new,
                        advantages,
                        epsilon,
                        actions,
                    );
                    // println!("Surrogate loss: {:?}", surrogate_loss);
                    // Get losses
                    let (policy_loss, critic_loss) =
                        calculate_losses(&device, surrogate_loss, returns.to_vec(), value_preds);
                    // println!("Policy loss: {}", policy_loss);
                    // println!("Critic loss: {}", critic_loss);
                    let policy_grad = policy_loss.backward();
                    let gradient_params = GradientsParams::from_grads(policy_grad, &ppo.policy);
                    // println!("Gradient params: {:?}", gradient_params);
                    let policy = policy_optimiser.step(learning_rate, ppo.policy, gradient_params);
                    let critic_grad = critic_loss.backward();
                    let critic_gradient_params =
                        GradientsParams::from_grads(critic_grad, &ppo.value);
                    let critic =
                        critic_optimiser.step(learning_rate, ppo.value, critic_gradient_params);

                    // Reconstruct PPO
                    ppo = PPOMoveSelector {
                        device: device.clone(),
                        policy,
                        value: critic,
                    };
                    batch += 1;
                }
            }
            // Save model checkpoints
            ppo.policy
                .clone()
                .save_file(dir.join(format!("checkpoint_{episode}.pt")), &recorder)
                .unwrap();
        }
    }
}

#[derive(Debug, Default)]
struct Data<B: Backend> {
    states: Vec<Tensor<B, 1>>,
    returns: Vec<Tensor<B, 1>>,
    advantages: Vec<Tensor<B, 1>>,
    action_logs: Vec<Tensor<B, 1>>,
    actions: Vec<usize>,
    action_masks: Vec<Tensor<B, 1>>,
}

impl<B: Backend> AddAssign for Data<B> {
    fn add_assign(&mut self, other: Self) {
        self.states.extend(other.states);
        self.returns.extend(other.returns);
        self.advantages.extend(other.advantages);
        self.action_logs.extend(other.action_logs);
        self.actions.extend(other.actions);
        self.action_masks.extend(other.action_masks);
    }
}

impl<B: Backend> Data<B> {
    fn detach(&mut self) {
        self.action_logs = self.action_logs.drain(..).map(|l| l.detach()).collect();
    }
}

fn returns<B: Backend>(device: &B::Device, rewards: &[f32], gamma: f32) -> Vec<Tensor<B, 1>> {
    // Calculate the discounted rewards for each state

    let mut returns = DVector::<f32>::zeros(rewards.len());
    let mut cumulative = 0.0;
    for (reward, discounted) in rewards.iter().zip(returns.iter_mut()).rev() {
        cumulative = *reward + gamma * cumulative;
        *discounted = cumulative;
    }

    // Remove mean and divide by std
    let mean = returns.mean();
    let std = returns.variance().sqrt() + 1e-8;
    let returns = returns.map(|r| (r - mean) / std);

    returns
        .iter()
        .map(|&r| Tensor::from_data([r].as_slice(), device))
        .collect()
}

fn advantages<B: Backend>(
    device: &B::Device,
    returns: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
) -> Vec<Tensor<B, 1>> {
    let advantages = values
        .iter()
        .zip(returns.iter())
        .map(|(v, r)| r.clone() - v.clone().detach())
        .collect::<Vec<_>>();
    // Normalise by mean and std
    let mean: f32 = advantages
        .iter()
        .map(|a| a.clone().into_scalar().to_f32())
        .sum::<f32>()
        / advantages.len() as f32;
    let var = advantages
        .iter()
        .map(|x| (x.clone().into_scalar().to_f32() - mean).powi(2))
        .sum::<f32>()
        / advantages.len() as f32;
    let std = var.sqrt() + 1e-8;

    advantages
        .iter()
        .map(|a| {
            Tensor::from_data(
                [(a.clone().into_scalar().to_f32() - mean) / std].as_slice(),
                device,
            )
        })
        .collect()
}

fn surrogate_loss<B: Backend>(
    device: &B::Device,
    action_log_old: &[Tensor<B, 1>],
    action_log_new: &[Tensor<B, 1>],
    advantages: &[Tensor<B, 1>],
    epsilon: f32,
    actions: &[usize],
) -> Vec<Tensor<B, 1>> {
    action_log_new
        .into_iter()
        .zip(action_log_old.into_iter())
        .zip(advantages.iter())
        .zip(actions)
        .map(|(((new, old), adv), &action)| {
            // Policy ratio r
            let ratio = (new.clone() - old.clone()).exp();
            let s1 = ratio.clone() * adv.clone();
            let s2 = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * adv.clone();
            s1.min_pair(s2)
                .select(0, Tensor::from_data([action].as_slice(), device))
        })
        .collect::<Vec<_>>()
}

fn calculate_losses<B: Backend>(
    device: &B::Device,
    surrogate_loss: Vec<Tensor<B, 1>>,
    returns: Vec<Tensor<B, 1>>,
    value_preds: Vec<Tensor<B, 1>>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    // Policy loss is sum of surrogate loss
    let policy_loss = -(surrogate_loss
        .into_iter()
        .fold(Tensor::zeros([1], device), |acc, x| acc + x));
    // Convert returns and values to tensors
    let returns: Tensor<B, 2> = Tensor::stack(returns, 1);
    let value_preds = Tensor::stack(value_preds, 1);
    // calculate huber loss instead of smooth l1 loss
    let huber = HuberLoss {
        delta: 1.0,
        lin_bias: 0.0,
    };
    let critic_loss = huber.forward(returns, value_preds, burn::nn::loss::Reduction::Sum);
    (policy_loss, critic_loss)
}

// Play the same game with each person starting first once
// fn play_double_game

/// Play a number of games, stacking all the results
fn play_games<B: Backend>(
    ppo: &mut PPOMoveSelector<B>,
    opponent: &mut Box<dyn Player<2, 6>>,
    num_games: usize,
) -> Vec<GameResult<B>> {
    let mut results = Vec::with_capacity(num_games);
    let mut scores = Vec::new();
    for seed in 0..num_games {
        let result = play_game(ppo, opponent, Some(seed as u64));
        scores.push(result.score);
        results.push(result);
    }
    // Print the sum of ppo score
    let sum: u32 = scores.iter().map(|s| s[0] as u32).sum();
    let wins = scores.iter().filter(|s| s[0] > s[1]).count();
    println!("Sum of scores: {sum}, Wins: {wins}");
    results
}

/// Play a game and collect the results
fn play_game<B: Backend>(
    ppo: &mut PPOMoveSelector<B>,
    opponent: &mut Box<dyn Player<2, 6>>,
    seed: Option<u64>,
) -> GameResult<B> {
    let mut result = GameResult::default();
    // Create a random game
    let mut gs = if let Some(seed) = seed {
        Gamestate::new_2_player_with_seed(seed, 0)
    } else {
        Gamestate::new_2_player()
    };

    // Play the game
    loop {
        // Get the moves that can be played
        let moves = gs.get_moves();
        let state = match gs.current_player() {
            0 => {
                // PPO
                let pick = ppo.pick_move_train(&gs, moves);
                // Save the pick for training
                result.states.push(pick.state);
                result.action_logs.push(pick.action_probs);
                result.values.push(pick.value);
                result.action_masks.push(pick.action_mask);
                result.actions.push(pick.action);
                let prev_score = gs.boards()[0].predicted_score as f32;
                let state = gs.play_move(pick.picked_move);
                let score = gs.boards()[0].predicted_score as f32;
                let delta = (score - prev_score) / 10.0;
                if score == 0.0 {
                    result.rewards.push(delta.min(-1.0));
                } else {
                    result.rewards.push(delta);
                }

                state
            }
            1 => {
                // Opponent
                gs.play_move(opponent.pick_move(&gs, moves))
            }
            _ => unreachable!(),
        };
        if state == State::RoundEnd {
            trace!("Round ended");
            if gs.end_round() == State::GameEnd {
                trace!("Game ended");
                break;
            }
        }
    }
    result.score = gs.scores();
    result
}

#[derive(Debug, Default)]
struct GameResult<B: Backend> {
    /// Each state that was passed to the PPO agent
    states: Vec<Tensor<B, 1>>,
    /// The softmax action vectors from policy agent
    action_logs: Vec<Tensor<B, 1>>,
    /// The masks for the actions
    action_masks: Vec<Tensor<B, 1>>,
    /// The action taken
    actions: Vec<usize>,
    /// The value estimates from the critic
    values: Vec<Tensor<B, 1>>,
    /// Each reward that was received from the environment
    rewards: Vec<f32>,
    /// The scores
    score: [u8; 2],
}
