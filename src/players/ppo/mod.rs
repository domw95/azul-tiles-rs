use burn::{
    config::{Config, ConfigError},
    module::AutodiffModule,
    tensor::backend::AutodiffBackend,
    nn::{Linear, LinearConfig, Relu},
    prelude::{Backend, Module},
    record::{DefaultFileRecorder, FullPrecisionSettings, RecorderError},
    tensor::{activation, cast::ToElement, Tensor},
};
use rand_distr::{Distribution, WeightedIndex};

use crate::{
    gamestate::{Gamestate, Move},
    players::{
        nn::{gs_to_array_for, index_to_move},
        Player,
    },
};

pub mod pretrain;
pub mod train;

/// Length of the encoded gamestate produced by [`gs_to_array_for`].
pub const STATE_SIZE: usize = 150;

/// Size of the action space: every (source, tile, destination) combination.
pub const ACTION_SIZE: usize = 180;

/// Shapes of the two networks.
///
/// Saved alongside the weights rather than restated at each call site. The
/// hidden size is a property of a given set of weights, so a checkpoint that
/// does not carry it can only be loaded by guessing -- which is exactly how
/// the GUI and the trainer ended up disagreeing (240 against 320).
#[derive(Config, Debug)]
pub struct PPOConfig {
    pub policy: PolicyConfig,
    pub value: ValueConfig,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            policy: PolicyConfig::new(STATE_SIZE, 320),
            value: ValueConfig::new(STATE_SIZE, 320),
        }
    }
}

/// Failure to read or write a checkpoint.
#[derive(Debug)]
pub enum CheckpointError {
    Config(ConfigError),
    Record(RecorderError),
    Io(std::io::Error),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "checkpoint config: {e}"),
            Self::Record(e) => write!(f, "checkpoint weights: {e}"),
            Self::Io(e) => write!(f, "checkpoint io: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<ConfigError> for CheckpointError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}

impl From<RecorderError> for CheckpointError {
    fn from(e: RecorderError) -> Self {
        Self::Record(e)
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Everything training needs to record about one sampled move.
///
/// Plain `f32` rather than tensors: rollout is inference only, so carrying an
/// autodiff graph out of it would be wasted work, and plain data crosses thread
/// boundaries without fuss.
#[derive(Debug, Clone)]
pub struct Pick {
    /// The encoded gamestate, [`STATE_SIZE`] long.
    pub state: Vec<f32>,
    /// Additive mask over the action space, [`ACTION_SIZE`] long.
    pub mask: Vec<f32>,
    /// Index of the action chosen.
    pub action: usize,
    /// `log pi_old` for the action chosen.
    ///
    /// Log, not raw, probability: PPO's importance ratio is
    /// `exp(log pi_new - log pi_old)`, and only the taken action is needed.
    pub log_prob: f32,
    /// Value estimate from the critic.
    pub value: f32,
    /// The move that was picked.
    pub picked_move: Move,
}

/// Player that can select a move and evaluate a gamestate using a policy network
#[derive(Debug, Clone)]
pub struct PPOMoveSelector<B: Backend> {
    device: B::Device,
    config: PPOConfig,
    policy: Policy<B>,
    value: Value<B>,
}

impl<B: Backend> PPOMoveSelector<B> {
    pub fn new(config: PPOConfig, device: &B::Device) -> Self {
        Self {
            device: device.clone(),
            policy: config.policy.init(device),
            value: config.value.init(device),
            config,
        }
    }

    /// The network shapes these weights were built with.
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Write both networks into `dir`, tagged with `tag`.
    ///
    /// The critic is saved alongside the policy. Without it, resuming a run
    /// restarts from a randomly initialised value function, which throws away
    /// the advantage estimates the policy was trained against.
    pub fn save(
        &self,
        dir: &std::path::Path,
        tag: impl std::fmt::Display,
    ) -> Result<(), CheckpointError> {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::default();
        self.config
            .save(dir.join(format!("checkpoint_{tag}_config.json")))?;
        self.policy
            .clone()
            .save_file(dir.join(format!("checkpoint_{tag}_policy")), &recorder)?;
        self.value
            .clone()
            .save_file(dir.join(format!("checkpoint_{tag}_value")), &recorder)?;
        Ok(())
    }

    /// Load both networks previously written by [`Self::save`].
    /// Load both networks, taking their shapes from the checkpoint's own config.
    pub fn from_checkpoint(
        dir: &std::path::Path,
        tag: impl std::fmt::Display,
        device: &B::Device,
    ) -> Result<Self, CheckpointError> {
        let config = PPOConfig::load(dir.join(format!("checkpoint_{tag}_config.json")))?;
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::default();
        let policy = config.policy.init(device).load_file(
            dir.join(format!("checkpoint_{tag}_policy")),
            &recorder,
            device,
        )?;
        let value = config.value.init(device).load_file(
            dir.join(format!("checkpoint_{tag}_value")),
            &recorder,
            device,
        )?;
        Ok(Self {
            device: device.clone(),
            config,
            policy,
            value,
        })
    }

    pub fn action(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        self.policy.action(state)
    }

    pub fn value(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        self.value.value(state)
    }

    /// Run the policy for `gamestate`, returning the encoded state, the action
    /// mask, and the masked log probabilities over the action space.
    fn policy_log_probs(
        &self,
        gamestate: &Gamestate<2, 6>,
        moves: &[Move],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Encode from the acting player's seat, not always seat 0.
        let state = gs_to_array_for(gamestate, gamestate.current_player() as usize)
            .as_slice()
            .to_vec();
        let state_tensor = Tensor::<B, 1>::from_data(state.as_slice(), &self.device);

        // Mask the illegal moves so they cannot be selected. -1e8 rather than
        // -inf keeps the entropy term finite: exp(-1e8) is exactly 0 in f32,
        // so 0 * -1e8 is 0 rather than NaN.
        let mut mask = vec![-1e8f32; ACTION_SIZE];
        for m in moves {
            mask[m.to_index()] = 0.0;
        }
        let mask_tensor = Tensor::<B, 1>::from_data(mask.as_slice(), &self.device);

        let log_probs = activation::log_softmax(self.policy.action(state_tensor) + mask_tensor, 0)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        (state, mask, log_probs)
    }

    /// Map an index in the action space back to the matching entry of `moves`.
    fn move_from_index(moves: &[Move], index: usize) -> Move {
        let (source, tile, destination) = index_to_move(index);
        moves
            .iter()
            .find(|m| {
                usize::from(m.source) == source
                    && usize::from(m.tile) == tile
                    && usize::from(m.destination) == destination
            })
            .cloned()
            .expect("selected action was not one of the legal moves")
    }

    /// Pick a move by sampling the policy, returning what training needs.
    pub fn pick_move_sampled(&self, gamestate: &Gamestate<2, 6>, moves: &[Move]) -> Pick {
        let (state, mask, log_probs) = self.policy_log_probs(gamestate, moves);

        // Sample only over the legal indices. The additive -1e8 mask alone is
        // not enough: it suppresses an action only while the raw logits stay
        // well under 1e8, so once the network starts diverging an illegal move
        // can outscore the mask and get picked. Restricting the distribution to
        // legal moves makes that structurally impossible, and is cheaper too --
        // a handful of candidates rather than all 180.
        let legal: Vec<usize> = moves.iter().map(|m| m.to_index()).collect();
        let probs: Vec<f32> = legal
            .iter()
            .map(|&i| {
                let p = log_probs[i].exp();
                if p.is_finite() { p } else { 0.0 }
            })
            .collect();
        // If every legal move underflowed, or the network produced garbage,
        // fall back to uniform rather than taking the whole run down.
        let action = match WeightedIndex::new(&probs) {
            Ok(dist) => legal[dist.sample(&mut rand::thread_rng())],
            Err(_) => legal[rand::random::<usize>() % legal.len()],
        };

        let state_tensor = Tensor::<B, 1>::from_data(state.as_slice(), &self.device);
        let value = self.value.value(state_tensor).into_scalar().to_f32();

        Pick {
            picked_move: Self::move_from_index(moves, action),
            log_prob: log_probs[action],
            state,
            mask,
            action,
            value,
        }
    }

    /// Pick the highest probability legal move, with no sampling.
    ///
    /// This is what [`Player::pick_move`] uses. Sampling is right while
    /// training, but when the agent is being measured it only adds noise and
    /// understates how strong the policy actually is.
    pub fn pick_move_greedy(&self, gamestate: &Gamestate<2, 6>, moves: &[Move]) -> Move {
        let (_, _, log_probs) = self.policy_log_probs(gamestate, moves);
        // Argmax over legal indices only, for the same reason as sampling.
        let action = moves
            .iter()
            .map(|m| m.to_index())
            .max_by(|&a, &b| log_probs[a].total_cmp(&log_probs[b]))
            .expect("a player always has at least one legal move");
        Self::move_from_index(moves, action)
    }
}

impl<B: AutodiffBackend> PPOMoveSelector<B> {
    /// A copy of this agent on the inner, non-autodiff backend.
    ///
    /// Rollout and evaluation are inference only; running them through the
    /// autodiff backend would record a graph that is immediately thrown away.
    pub fn valid(&self) -> PPOMoveSelector<B::InnerBackend> {
        PPOMoveSelector {
            device: self.device.clone(),
            config: self.config.clone(),
            policy: self.policy.valid(),
            value: self.value.valid(),
        }
    }
}

impl<B: Backend> Player<2, 6> for PPOMoveSelector<B> {
    fn pick_move(
        &mut self,
        gamestate: &crate::gamestate::Gamestate<2, 6>,
        moves: Vec<crate::gamestate::Move>,
    ) -> crate::gamestate::Move {
        self.pick_move_greedy(gamestate, &moves)
    }

    fn name(&self) -> String {
        "PPOMoveSelector".into()
    }
}

#[derive(Config, Debug)]
pub struct PolicyConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    /// Number of hidden_size -> hidden_size layers after the input layer.
    ///
    /// Defaults to 1, which is the two-hidden-layer network everything before
    /// this was trained with; the default also keeps older checkpoints, whose
    /// config JSON has no such field, loadable.
    #[config(default = 1)]
    pub hidden_layers: usize,
}

impl PolicyConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Policy<B> {
        Policy {
            input: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            hidden: (0..self.hidden_layers)
                .map(|_| LinearConfig::new(self.hidden_size, self.hidden_size).init(device))
                .collect(),
            output: LinearConfig::new(self.hidden_size, ACTION_SIZE).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Policy<B: Backend> {
    input: Linear<B>,
    hidden: Vec<Linear<B>>,
    output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Policy<B> {
    /// Run the policy network without normalising the result.
    ///
    /// Generic over rank so the same path serves a single state during rollout
    /// and a whole `[batch, STATE_SIZE]` slice during training.
    fn action<const D: usize>(&self, state: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = self.activation.forward(self.input.forward(state));
        for layer in &self.hidden {
            x = self.activation.forward(layer.forward(x));
        }
        self.output.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ValueConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    /// See [`PolicyConfig::hidden_layers`].
    #[config(default = 1)]
    pub hidden_layers: usize,
}

impl ValueConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Value<B> {
        Value {
            input: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            hidden: (0..self.hidden_layers)
                .map(|_| LinearConfig::new(self.hidden_size, self.hidden_size).init(device))
                .collect(),
            output: LinearConfig::new(self.hidden_size, 1).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
struct Value<B: Backend> {
    input: Linear<B>,
    hidden: Vec<Linear<B>>,
    output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Value<B> {
    fn value<const D: usize>(&self, state: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = self.activation.forward(self.input.forward(state));
        for layer in &self.hidden {
            x = self.activation.forward(layer.forward(x));
        }
        self.output.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    /// The masked distribution must be a real distribution, and both selection
    /// paths must return a move that was actually offered.
    ///
    /// Worth pinning down because the mask is applied in log space: if the
    /// masking or the log_softmax were wrong, training would still run, just
    /// silently on nonsense.
    #[test]
    fn masked_policy_is_a_distribution_over_legal_moves() {
        let device = Default::default();
        let ppo = PPOMoveSelector::<NdArray>::new(
            PPOConfig::new(
                PolicyConfig::new(STATE_SIZE, 32),
                ValueConfig::new(STATE_SIZE, 32),
            ),
            &device,
        );

        let gs = Gamestate::<2, 6>::new_2_player_with_seed(0, 0);
        let moves = gs.get_moves();
        assert!(!moves.is_empty());

        let (state, mask, log_probs) = ppo.policy_log_probs(&gs, &moves);
        assert_eq!(state.len(), STATE_SIZE);
        assert_eq!(mask.len(), ACTION_SIZE);
        let probs: Vec<f32> = log_probs.iter().map(|l| l.exp()).collect();

        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "probabilities summed to {total}");

        // Every legal move holds some probability mass, every illegal slot none.
        let legal: Vec<usize> = moves.iter().map(|m| m.to_index()).collect();
        for (i, &p) in probs.iter().enumerate() {
            if legal.contains(&i) {
                assert!(p > 0.0, "legal action {i} had zero probability");
            } else {
                assert_eq!(p, 0.0, "illegal action {i} had probability {p}");
            }
        }

        assert!(moves.contains(&ppo.pick_move_greedy(&gs, &moves)));
        assert!(moves.contains(&ppo.pick_move_sampled(&gs, &moves).picked_move));
    }
}
