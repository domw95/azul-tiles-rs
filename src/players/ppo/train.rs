use std::path::PathBuf;

use burn::nn::loss::{HuberLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, TensorData};
use burn::{prelude::Backend, tensor::Tensor};
use log::trace;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::gamestate::{Gamestate, State};
use crate::players::ppo::{PPOMoveSelector, ACTION_SIZE, STATE_SIZE};
use crate::players::Player;

/// Seed offset for evaluation games.
///
/// Evaluation must not reuse the seeds training plays on, or the stopping
/// metric measures memorisation of a fixed set of deals rather than skill.
const EVAL_SEED_BASE: u64 = 1_000_000;

/// How training decides it is done.
///
/// Episode count is a poor stopping rule: it says nothing about whether the
/// agent is still improving, and a fixed budget either stops early or burns
/// hours after the curve has flattened.
#[derive(Debug, Clone)]
pub struct StopCondition {
    /// Stop once the greedy agent wins at least this fraction of eval games.
    pub target_win_rate: f32,
    /// Stop if the *smoothed* eval margin has not improved for this many
    /// episodes.
    pub patience: usize,
    /// Never stop on patience before this many episodes.
    ///
    /// Early episodes are the noisiest, and an unlucky early high water mark
    /// would otherwise end a run that is still climbing.
    pub min_episodes: usize,
    /// Hard cap, so a run that never converges still terminates.
    pub max_episodes: usize,
    /// EMA weight applied to the eval margin, in (0, 1]. Lower is smoother.
    ///
    /// Patience runs off the smoothed value, not the raw one: a single good
    /// episode is luck, a rising average is progress.
    pub smoothing: f32,
}

impl Default for StopCondition {
    fn default() -> Self {
        Self {
            target_win_rate: 0.55,
            // Patience of 60 over min_episodes 100 measured too impatient: a
            // run stopped at 156 keeping an episode-90 checkpoint (7.5%
            // held-out) where a flat 200-episode budget reached 13.5%. The
            // eval signal is noisy enough that a long flat stretch is not
            // evidence of a real plateau.
            patience: 400,
            min_episodes: 600,
            max_episodes: 4000,
            smoothing: 0.2,
        }
    }
}

/// Why training stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the target win rate.
    TargetReached,
    /// No improvement for `patience` episodes.
    Plateaued,
    /// Hit `max_episodes`.
    BudgetExhausted,
}

/// How a per-move reward is derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardMode {
    /// The agent's own predicted-score delta, read straight after its move.
    ///
    /// Simple, but blind to the opponent: it cannot express "I held them down".
    Immediate,
    /// Change in the score *differential*, settled once the opponent has
    /// replied.
    ///
    /// Correct as an MDP -- the opponent is part of the environment, so a
    /// move's consequences are only visible after the reply -- but it folds
    /// the opponent's noise into every reward.
    DifferentialDelayed,
    /// Own predicted-score delta, minus the change in floor penalty.
    ///
    /// Fixes the flaw in [`RewardMode::Immediate`]: `predicted_score` is
    /// clamped at zero, so while a board is underwater every move scores a
    /// delta of zero and the old code substituted a flat -1.0. Measured over
    /// 5301 moves that fired on **77%** of them, making most of the signal a
    /// constant unrelated to move quality. Tracking the floor penalty directly
    /// restores a graded signal below zero.
    Shaped,
    /// Nothing per move; the final score margin fed back through the whole
    /// game by [`returns`].
    ///
    /// Unbiased against the real objective, but one number per ~35 decisions.
    /// Pair with `gamma = 1.0`, or the opening is discounted for no reason.
    TerminalOnly,
    /// Nothing per move; the margin gained over each round credited to the
    /// last move of that round, plus the final margin.
    ///
    /// Five signals per game instead of one, and round ends are where
    /// wall-tiling and floor penalties actually resolve.
    RoundMargin,
}

/// Which seat(s) the agent trains in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatMode {
    /// Always seat 0, always moving first.
    First,
    /// Alternate seats by game, so the agent can also play second.
    Alternate,
}

/// Rollout settings shared by training and evaluation.
#[derive(Debug, Clone, Copy)]
struct Rollout {
    selection: Selection,
    seats: SeatMode,
    reward_mode: RewardMode,
    terminal_reward: f32,
}

/// Training hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub games_per_episode: usize,
    /// Greedy games played each episode to measure progress.
    pub eval_games: usize,
    /// Discount applied to future rewards.
    pub gamma: f32,
    /// PPO clip range.
    pub epsilon: f32,
    pub learning_rate: f64,
    /// Weight of the entropy bonus in the policy loss.
    pub entropy_coeff: f32,
    /// Reward added at the end of a game for winning, subtracted for losing.
    ///
    /// Without this the agent is never told the outcome: shaped per-move
    /// rewards alone optimise score accumulation, not winning.
    pub terminal_reward: f32,
    /// How per-move rewards are derived.
    pub reward_mode: RewardMode,
    /// Which seat(s) the agent trains in.
    pub seats: SeatMode,
    /// Huber delta for the critic loss.
    ///
    /// Huber is linear beyond `delta`, so its gradient magnitude is capped
    /// there however wrong the prediction is. If returns are larger than
    /// `delta` the critic spends the whole run in that regime and fits slowly.
    pub huber_delta: f32,
    /// Critic learning rate as a multiple of `learning_rate`.
    pub critic_lr_multiplier: f64,
    /// Shuffle minibatches each epoch.
    ///
    /// Without this the batches are contiguous slices in game order, so every
    /// one is a run of consecutive states from the same game and every epoch
    /// sees the identical partition.
    pub shuffle_minibatches: bool,
    /// Standardise the critic's targets across each episode.
    ///
    /// Advantages are then computed in the same standardised space, so the
    /// baseline stays consistent with what the critic predicts.
    pub normalise_value_targets: bool,
    /// Directory checkpoints are written to.
    pub dir: PathBuf,
    pub stop: StopCondition,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 5,
            batch_size: 128,
            games_per_episode: 40,
            eval_games: 40,
            gamma: 0.99,
            epsilon: 0.1,
            learning_rate: 0.001,
            entropy_coeff: 0.01,
            terminal_reward: 1.0,
            // Measured, not assumed. A 2x2 over these two at a fixed
            // 200-episode budget gave, as held-out win rate / mean score:
            //   Immediate           + First     17% / 37.1
            //   Immediate           + Alternate  7% / 29.4
            //   DifferentialDelayed + First      0% /  3.9
            //   DifferentialDelayed + Alternate  2% /  6.5
            // The delayed differential reward is theoretically the better
            // formulation -- the opponent is part of the environment -- but it
            // folds the opponent's reply into every reward, and in practice the
            // critic cannot fit the result.
            reward_mode: RewardMode::Immediate,
            // Seat 0 only. With the perspective fix the encoding is relative to
            // whoever is acting, so a seat-0 model transfers to seat 1 on its
            // own (15% there), and splitting the budget across seats measured
            // worse rather than better.
            seats: SeatMode::First,
            // Measured over 150 episodes, critic explained variance:
            //   d=1  lr=1x        +0.158   (was the default)
            //   d=10 lr=1x        +0.250
            //   d=1  lr=5x        -0.046
            //   d=10 lr=5x        +0.073
            //   d=1  lr=1x, norm  -0.207
            // Only the wider delta helps. A faster critic destabilises it, and
            // standardising the targets per episode makes the critic chase a
            // scale that moves every episode -- worse than predicting the mean.
            huber_delta: 10.0,
            critic_lr_multiplier: 1.0,
            normalise_value_targets: false,
            shuffle_minibatches: true,
            dir: PathBuf::from("ppo"),
            stop: StopCondition::default(),
        }
    }
}

/// Result of playing a set of evaluation games with the greedy policy.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EvalResult {
    pub win_rate: f32,
    /// Mean (agent score - opponent score). Negative means losing.
    pub margin: f32,
    pub mean_score: f32,
}

/// Outcome of a training run.
#[derive(Debug, Clone)]
pub struct TrainSummary {
    pub episodes_run: usize,
    pub best: EvalResult,
    pub best_episode: usize,
    pub reason: StopReason,
    /// Critic explained variance on the final episode. See
    /// [`explained_variance`].
    pub explained_variance: f32,
}

/// Train a PPO agent against another player
///
/// Runs a matchup, collecting state and rewards
/// then trains the player based on outcome
pub struct PPOTrainer<B: Backend> {
    ppo: PPOMoveSelector<B>,
    opponent: Box<dyn Player<2, 6>>,
    device: B::Device,
    options: TrainOptions,
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
            options: TrainOptions::default(),
        }
    }

    /// Override the default hyperparameters.
    pub fn with_options(mut self, options: TrainOptions) -> Self {
        self.options = options;
        self
    }

    /// Run training, returning the trained agent and a summary.
    ///
    /// The best agent by evaluation margin is written to `checkpoint_best_*`,
    /// so the saved model is the best one seen rather than whichever happened
    /// to be current when the run stopped.
    pub fn train(self) -> (PPOMoveSelector<B>, TrainSummary) {
        let mut policy_optimiser = AdamConfig::new().init();
        let mut critic_optimiser = AdamConfig::new().init();

        let mut ppo = self.ppo;
        let opponent = self.opponent;
        let device = self.device;
        let options = self.options;
        let config = ppo.config().clone();

        std::fs::create_dir_all(&options.dir).unwrap();

        // Two notions of "best", deliberately: the raw result decides which
        // weights are worth keeping, the smoothed one decides when to stop.
        let mut best = EvalResult {
            margin: f32::NEG_INFINITY,
            ..Default::default()
        };
        let mut best_episode = 0;
        let mut smoothed: Option<f32> = None;
        let mut best_smoothed = f32::NEG_INFINITY;
        let mut best_smoothed_episode = 0;
        let mut episodes_run = 0;
        let mut last_ev = f32::NAN;
        let mut reason = StopReason::BudgetExhausted;

        for episode in 0..options.stop.max_episodes {
            episodes_run = episode + 1;

            // Rollout runs on the inner backend: it is pure inference, so
            // building an autodiff graph for it would be wasted work, and the
            // plain results are cheap to move between threads.
            let inference = ppo.valid();
            // Fresh deals each episode. Reusing seeds 0..games_per_episode
            // every time meant the agent only ever saw that many distinct
            // games, and memorised them instead of learning to play.
            let results = play_games(
                &inference,
                opponent.as_ref(),
                options.games_per_episode,
                (episode * options.games_per_episode) as u64,
                Rollout {
                    selection: Selection::Sampled,
                    seats: options.seats,
                    reward_mode: options.reward_mode,
                    terminal_reward: options.terminal_reward,
                },
            );

            let mut data = Batch::default();
            for result in &results {
                let returns = returns(&result.rewards, options.gamma);
                data.extend(result, returns);
            }
            if options.normalise_value_targets {
                data.normalise_returns();
            }
            data.compute_advantages();
            // Before normalising: advantages are still returns - values, so
            // this measures how much of the return variance the critic
            // actually accounts for.
            let ev = explained_variance(&data.returns, &data.advantages);
            data.normalise_advantages();

            let mut order: Vec<usize> = (0..data.len()).collect();
            for _ in 0..options.epochs {
                if options.shuffle_minibatches {
                    order.shuffle(&mut rand::thread_rng());
                }
                for chunk in order.chunks(options.batch_size) {
                    let (policy_loss, critic_loss) =
                        data.losses(&ppo, chunk, &device, &options);

                    let policy_grad = policy_loss.backward();
                    let gradient_params = GradientsParams::from_grads(policy_grad, &ppo.policy);
                    let policy =
                        policy_optimiser.step(options.learning_rate, ppo.policy, gradient_params);

                    let critic_grad = critic_loss.backward();
                    let critic_gradient_params =
                        GradientsParams::from_grads(critic_grad, &ppo.value);
                    let critic = critic_optimiser.step(
                        options.learning_rate * options.critic_lr_multiplier,
                        ppo.value,
                        critic_gradient_params,
                    );

                    ppo = PPOMoveSelector {
                        device: device.clone(),
                        config: config.clone(),
                        policy,
                        value: critic,
                    };
                }
            }

            // Measure the greedy policy on held-out deals: that, not the
            // sampled training games, is what "can it play" means.
            let eval = evaluate(
                &ppo.valid(),
                opponent.as_ref(),
                options.eval_games,
                EVAL_SEED_BASE,
                options.seats,
            );
            let smooth = match smoothed {
                None => eval.margin,
                Some(prev) => {
                    options.stop.smoothing * eval.margin + (1.0 - options.stop.smoothing) * prev
                }
            };
            smoothed = Some(smooth);

            println!(
                "episode {episode}: {} states | ev {ev:+.2} | eval win {:.0}% margin {:+.1} (avg {:+.1}) score {:.1}",
                data.len(),
                100.0 * eval.win_rate,
                eval.margin,
                smooth,
                eval.mean_score
            );
            last_ev = ev;

            if eval.margin > best.margin {
                best = eval;
                best_episode = episode;
                ppo.save(&options.dir, "best").unwrap();
            }
            if smooth > best_smoothed {
                best_smoothed = smooth;
                best_smoothed_episode = episode;
            }

            if eval.win_rate >= options.stop.target_win_rate {
                reason = StopReason::TargetReached;
                break;
            }
            if episode >= options.stop.min_episodes
                && episode - best_smoothed_episode >= options.stop.patience
            {
                reason = StopReason::Plateaued;
                break;
            }
        }

        println!(
            "stopped: {reason:?} after {episodes_run} episodes; best margin {:+.1} (win {:.0}%) at episode {best_episode}",
            best.margin,
            100.0 * best.win_rate
        );

        (
            ppo,
            TrainSummary {
                episodes_run,
                best,
                best_episode,
                reason,
                explained_variance: last_ev,
            },
        )
    }
}

/// Whether a rollout samples the policy or takes its best move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selection {
    Sampled,
    Greedy,
}

/// One step of a game, as plain data.
///
/// Deliberately not tensors: collection is inference only, and keeping it plain
/// means a game carries no autodiff graph and can cross thread boundaries.
#[derive(Debug, Default)]
struct GameResult {
    states: Vec<f32>,
    masks: Vec<f32>,
    /// log pi_old for the action actually taken.
    log_probs: Vec<f32>,
    values: Vec<f32>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    score: [u8; 2],
}

impl GameResult {
    fn steps(&self) -> usize {
        self.actions.len()
    }
}

/// Collected rollouts, flattened ready to batch.
#[derive(Debug, Default)]
struct Batch {
    states: Vec<f32>,
    masks: Vec<f32>,
    log_probs: Vec<f32>,
    returns: Vec<f32>,
    values: Vec<f32>,
    advantages: Vec<f32>,
    actions: Vec<i32>,
}

impl Batch {
    fn len(&self) -> usize {
        self.actions.len()
    }

    fn extend(&mut self, result: &GameResult, returns: Vec<f32>) {
        self.states.extend_from_slice(&result.states);
        self.masks.extend_from_slice(&result.masks);
        self.log_probs.extend_from_slice(&result.log_probs);
        self.actions.extend_from_slice(&result.actions);
        self.values.extend_from_slice(&result.values);
        self.returns.extend(returns);
    }

    /// Standardise the critic's targets across the episode.
    fn normalise_returns(&mut self) {
        let n = self.returns.len() as f32;
        if n == 0.0 {
            return;
        }
        let mean = self.returns.iter().sum::<f32>() / n;
        let var = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt() + 1e-8;
        for r in &mut self.returns {
            *r = (*r - mean) / std;
        }
    }

    /// Advantage is the return the critic failed to predict.
    fn compute_advantages(&mut self) {
        self.advantages = self
            .returns
            .iter()
            .zip(&self.values)
            .map(|(r, v)| r - v)
            .collect();
    }

    /// Normalise advantages across the whole episode, once.
    fn normalise_advantages(&mut self) {
        let n = self.advantages.len() as f32;
        if n == 0.0 {
            return;
        }
        let mean = self.advantages.iter().sum::<f32>() / n;
        let var = self
            .advantages
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>()
            / n;
        let std = var.sqrt() + 1e-8;
        for a in &mut self.advantages {
            *a = (*a - mean) / std;
        }
    }

    /// Policy and critic loss for one slice, as a single batched forward pass.
    fn losses<B: AutodiffBackend>(
        &self,
        ppo: &PPOMoveSelector<B>,
        idx: &[usize],
        device: &B::Device,
        options: &TrainOptions,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let n = idx.len();
        let mut flat_states = Vec::with_capacity(n * STATE_SIZE);
        let mut flat_masks = Vec::with_capacity(n * ACTION_SIZE);
        let mut acts = Vec::with_capacity(n);
        let mut olp = Vec::with_capacity(n);
        let mut adv = Vec::with_capacity(n);
        let mut rets = Vec::with_capacity(n);
        for &i in idx {
            flat_states.extend_from_slice(&self.states[i * STATE_SIZE..(i + 1) * STATE_SIZE]);
            flat_masks.extend_from_slice(&self.masks[i * ACTION_SIZE..(i + 1) * ACTION_SIZE]);
            acts.push(self.actions[i]);
            olp.push(self.log_probs[i]);
            adv.push(self.advantages[i]);
            rets.push(self.returns[i]);
        }

        let states =
            Tensor::<B, 2>::from_data(TensorData::new(flat_states, [n, STATE_SIZE]), device);
        let masks =
            Tensor::<B, 2>::from_data(TensorData::new(flat_masks, [n, ACTION_SIZE]), device);
        let actions = Tensor::<B, 2, Int>::from_data(TensorData::new(acts, [n, 1]), device);
        let old_log_probs = Tensor::<B, 2>::from_data(TensorData::new(olp, [n, 1]), device);
        let advantages = Tensor::<B, 2>::from_data(TensorData::new(adv, [n, 1]), device);
        let returns = Tensor::<B, 2>::from_data(TensorData::new(rets, [n, 1]), device);

        // One forward pass for the whole slice, rather than n passes of one row
        let log_probs = log_softmax(ppo.policy.action(states.clone()) + masks, 1);
        let value_preds = ppo.value.value(states);

        let chosen = log_probs.clone().gather(1, actions);
        let ratio = (chosen - old_log_probs).exp();
        let unclipped = ratio.clone() * advantages.clone();
        let clipped = ratio.clamp(1.0 - options.epsilon, 1.0 + options.epsilon) * advantages;
        let surrogate = unclipped.min_pair(clipped).mean();

        // Entropy bonus, to stop the policy collapsing onto one move early on.
        // Illegal actions contribute nothing: their probability is exactly 0.
        let entropy = -(log_probs.clone().exp() * log_probs).sum_dim(1).mean();

        let policy_loss = -surrogate - entropy.mul_scalar(options.entropy_coeff);

        let huber = HuberLoss {
            delta: options.huber_delta,
            lin_bias: 0.0,
        };
        let critic_loss = huber.forward(value_preds, returns, Reduction::Mean);

        (policy_loss, critic_loss)
    }
}

/// Fraction of the return variance the critic accounts for.
///
/// `1.0` is a perfect baseline; `0.0` means the critic is no better than
/// predicting the mean return; negative means it is *worse* than the mean, in
/// which case subtracting it adds variance to the policy gradient rather than
/// removing it.
///
/// `residuals` must be `returns - values`, i.e. advantages before
/// normalisation.
///
/// This is the diagnostic that separates "the reward signal is too sparse"
/// from "the critic never fit", which look identical from the win rate alone.
fn explained_variance(returns: &[f32], residuals: &[f32]) -> f32 {
    let n = returns.len() as f32;
    if n == 0.0 {
        return f32::NAN;
    }
    let mean = returns.iter().sum::<f32>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
    if var <= 1e-12 {
        return f32::NAN;
    }
    let res_mean = residuals.iter().sum::<f32>() / n;
    let res_var = residuals
        .iter()
        .map(|r| (r - res_mean).powi(2))
        .sum::<f32>()
        / n;
    1.0 - res_var / var
}

/// Discounted returns for one game.
fn returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut out = vec![0.0; rewards.len()];
    let mut cumulative = 0.0;
    for (reward, discounted) in rewards.iter().zip(out.iter_mut()).rev() {
        cumulative = *reward + gamma * cumulative;
        *discounted = cumulative;
    }
    // Deliberately left unnormalised. The critic is trained to predict these,
    // so normalising per game would give it a target whose scale shifts from
    // game to game. Advantages get normalised instead, across the episode.
    out
}

/// Play `games` games spread across the thread pool.
///
/// Games are independent and seeded, so this is a straight fan-out. The agent
/// and the opponent are cloned once per worker *before* entering the parallel
/// region: burn's `Param` holds a `OnceCell`, so a module is `Send` but not
/// `Sync`, and picking a move takes `&mut self` besides. Results are restored
/// to seed order so that a run stays reproducible.
fn play_games<B: Backend>(
    ppo: &PPOMoveSelector<B>,
    opponent: &dyn Player<2, 6>,
    games: usize,
    seed_base: u64,
    rollout: Rollout,
) -> Vec<GameResult> {
    if games == 0 {
        return Vec::new();
    }
    let workers = rayon::current_num_threads().clamp(1, games);
    let owned: Vec<_> = (0..workers)
        .map(|w| (w, ppo.clone(), dyn_clone::clone_box(opponent)))
        .collect();

    let mut results: Vec<(usize, GameResult)> = owned
        .into_par_iter()
        .flat_map(|(w, ppo, mut opponent)| {
            (w..games)
                .step_by(workers)
                .map(|i| {
                    (
                        i,
                        play_game(
                            &ppo,
                            opponent.as_mut(),
                            seed_base + i as u64,
                            match rollout.seats {
                                // The agent has to be able to play second:
                                // that is where the GUI puts it.
                                SeatMode::Alternate => (i % 2) as u8,
                                SeatMode::First => 0,
                            },
                            rollout,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    results.sort_unstable_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, r)| r).collect()
}

/// Play the greedy policy on held-out seeds and report how it did.
fn evaluate<B: Backend>(
    ppo: &PPOMoveSelector<B>,
    opponent: &dyn Player<2, 6>,
    games: usize,
    seed_base: u64,
    seats: SeatMode,
) -> EvalResult {
    let results = play_games(
        ppo,
        opponent,
        games,
        seed_base,
        Rollout {
            selection: Selection::Greedy,
            seats,
            reward_mode: RewardMode::Immediate,
            terminal_reward: 0.0,
        },
    );
    let n = results.len().max(1) as f32;
    let wins = results.iter().filter(|r| r.score[0] > r.score[1]).count() as f32;
    let margin: f32 = results
        .iter()
        .map(|r| r.score[0] as f32 - r.score[1] as f32)
        .sum::<f32>()
        / n;
    let mean_score: f32 = results.iter().map(|r| r.score[0] as f32).sum::<f32>() / n;
    EvalResult {
        win_rate: wins / n,
        margin,
        mean_score,
    }
}

/// Play a game and collect the results
fn play_game<B: Backend>(
    ppo: &PPOMoveSelector<B>,
    opponent: &mut dyn Player<2, 6>,
    seed: u64,
    agent_seat: u8,
    rollout: Rollout,
) -> GameResult {
    let selection = rollout.selection;
    let sampled = selection == Selection::Sampled;
    let mut result = GameResult::default();
    let mut gs = Gamestate::new_2_player_with_seed(seed, 0);
    let seat = agent_seat as usize;
    // Score difference from the agent's seat, whichever seat that is.
    let differential = |gs: &Gamestate<2, 6>| {
        gs.boards()[seat].predicted_score as f32 - gs.boards()[1 - seat].predicted_score as f32
    };
    // Differential before the agent's last move, held until the opponent has
    // replied. A move's consequences are only visible after the reply, so the
    // reward for it cannot be read off immediately.
    let mut pending: Option<f32> = None;
    // Differential at the start of the current round, for RoundMargin.
    let mut round_start = 0.0f32;

    loop {
        let moves = gs.get_moves();
        let state = if gs.current_player() == agent_seat {
            {
                if let Some(before) = pending.take() {
                    result.rewards.push((differential(&gs) - before) / 10.0);
                }

                let picked = match selection {
                    Selection::Greedy => {
                        // No bookkeeping needed: evaluation never trains.
                        ppo.pick_move_greedy(&gs, &moves)
                    }
                    Selection::Sampled => {
                        let pick = ppo.pick_move_sampled(&gs, &moves);
                        result.states.extend_from_slice(&pick.state);
                        result.masks.extend_from_slice(&pick.mask);
                        result.log_probs.push(pick.log_prob);
                        result.values.push(pick.value);
                        result.actions.push(pick.action as i32);
                        if rollout.reward_mode == RewardMode::DifferentialDelayed {
                            pending = Some(differential(&gs));
                        }
                        pick.picked_move
                    }
                };

                let own_before = gs.boards()[seat].predicted_score as f32;
                let floor_before = gs.boards()[seat].floor_penalty() as f32;
                let state = gs.play_move(picked);
                if sampled {
                    match rollout.reward_mode {
                        RewardMode::Immediate => {
                            let own_after = gs.boards()[seat].predicted_score as f32;
                            let delta = (own_after - own_before) / 10.0;
                            result.rewards.push(if own_after == 0.0 {
                                delta.min(-1.0)
                            } else {
                                delta
                            });
                        }
                        RewardMode::Shaped => {
                            let own_after = gs.boards()[seat].predicted_score as f32;
                            let floor_after = gs.boards()[seat].floor_penalty() as f32;
                            result.rewards.push(
                                ((own_after - own_before) - (floor_after - floor_before)) / 10.0,
                            );
                        }
                        // Settled at the next agent turn, below.
                        RewardMode::DifferentialDelayed => {}
                        // Credit arrives at a round end or at the game end;
                        // push a placeholder so rewards stay 1:1 with actions.
                        RewardMode::TerminalOnly | RewardMode::RoundMargin => {
                            result.rewards.push(0.0)
                        }
                    }
                }
                state
            }
        } else {
            gs.play_move(opponent.pick_move(&gs, moves))
        };
        if state == State::RoundEnd {
            trace!("Round ended");
            let game_over = gs.end_round() == State::GameEnd;
            if sampled && rollout.reward_mode == RewardMode::RoundMargin {
                // Credit the round's swing to the last move played in it.
                let now = differential(&gs);
                if let Some(last) = result.rewards.last_mut() {
                    *last += (now - round_start) / 10.0;
                }
                round_start = now;
            }
            if game_over {
                trace!("Game ended");
                break;
            }
        }
    }
    if let Some(before) = pending.take() {
        result.rewards.push((differential(&gs) - before) / 10.0);
    }
    // Ordered [agent, opponent], not [seat 0, seat 1].
    let scores = gs.scores();
    result.score = [scores[seat], scores[1 - seat]];

    if sampled {
        // The Monte Carlo modes get the outcome itself here; `returns` then
        // walks it back through every state and action in the game.
        if matches!(
            rollout.reward_mode,
            RewardMode::TerminalOnly | RewardMode::RoundMargin
        ) {
            let margin = result.score[0] as f32 - result.score[1] as f32;
            if let Some(last) = result.rewards.last_mut() {
                *last += margin / 10.0;
            }
        }
    }

    // Tell the agent whether it actually won.
    if sampled {
        if let Some(last) = result.rewards.last_mut() {
            *last += match result.score[0].cmp(&result.score[1]) {
                std::cmp::Ordering::Greater => rollout.terminal_reward,
                std::cmp::Ordering::Less => -rollout.terminal_reward,
                std::cmp::Ordering::Equal => 0.0,
            };
        }
    }

    debug_assert!(selection == Selection::Greedy || result.steps() == result.rewards.len());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::minimax::{Minimaxer, ScoreEvaluator};
    use crate::players::ppo::{PPOConfig, PolicyConfig, ValueConfig};
    use burn::backend::{Autodiff, NdArray};
    use minimaxer::negamax::SearchOptions;

    type B = Autodiff<NdArray>;

    fn tiny_agent(device: &<B as Backend>::Device) -> PPOMoveSelector<B> {
        PPOMoveSelector::new(
            PPOConfig::new(
                PolicyConfig::new(STATE_SIZE, 32),
                ValueConfig::new(STATE_SIZE, 32),
            ),
            device,
        )
    }

    fn depth1() -> Box<dyn Player<2, 6>> {
        Box::new(Minimaxer::new(
            SearchOptions {
                max_depth: Some(1),
                ..Default::default()
            },
            "Depth1",
            ScoreEvaluator,
        ))
    }

    /// A short end-to-end run on the CPU backend.
    ///
    /// Not a check that the agent learns anything -- two episodes cannot show
    /// that. It checks the loop runs start to finish, that the budget stop
    /// fires, and that the best checkpoint can be read back with the right
    /// shapes.
    #[test]
    fn short_run_completes_and_round_trips_its_best_checkpoint() {
        let device = Default::default();
        let dir = std::env::temp_dir().join("azul_ppo_short_run");
        let _ = std::fs::remove_dir_all(&dir);

        let (trained, summary) = PPOTrainer::new(tiny_agent(&device), depth1(), &device)
            .with_options(TrainOptions {
                epochs: 1,
                batch_size: 32,
                games_per_episode: 2,
                eval_games: 2,
                dir: dir.clone(),
                stop: StopCondition {
                    // Unreachable target and generous patience, so the budget
                    // is what ends the run.
                    target_win_rate: 2.0,
                    patience: 100,
                    min_episodes: 0,
                    max_episodes: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .train();

        assert_eq!(summary.episodes_run, 2);
        assert_eq!(summary.reason, StopReason::BudgetExhausted);
        assert_eq!(trained.config().policy.hidden_size, 32);

        let reloaded = PPOMoveSelector::<B>::from_checkpoint(&dir, "best", &device).unwrap();
        assert_eq!(reloaded.config().policy.hidden_size, 32);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Training stops on its own once the agent stops improving.
    #[test]
    fn plateau_ends_the_run_before_the_budget() {
        let device = Default::default();
        let dir = std::env::temp_dir().join("azul_ppo_plateau");
        let _ = std::fs::remove_dir_all(&dir);

        let (_, summary) = PPOTrainer::new(tiny_agent(&device), depth1(), &device)
            .with_options(TrainOptions {
                epochs: 1,
                batch_size: 32,
                games_per_episode: 1,
                eval_games: 1,
                dir: dir.clone(),
                stop: StopCondition {
                    target_win_rate: 2.0,
                    patience: 1,
                    min_episodes: 0,
                    max_episodes: 50,
                    ..Default::default()
                },
                ..Default::default()
            })
            .train();

        assert_eq!(summary.reason, StopReason::Plateaued);
        assert!(summary.episodes_run < 50);

        std::fs::remove_dir_all(&dir).ok();
    }
}
