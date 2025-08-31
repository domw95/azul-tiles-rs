use burn::{
    config::Config,
    nn::{Linear, LinearConfig, Relu},
    prelude::{Backend, Module},
    record::{self, DefaultFileRecorder, FullPrecisionSettings},
    tensor::{activation, cast::ToElement, Tensor},
};
use rand_distr::{Distribution, WeightedIndex};

use crate::{
    gamestate::{Gamestate, Move},
    players::{
        nn::{gs_to_array, index_to_move},
        Player,
    },
};

pub mod train;

pub struct PickReturn<B: Backend> {
    /// The state converted from gamestate
    pub state: Tensor<B, 1>,
    /// Action chosen
    pub action: usize,
    /// Action probabilities from policy network
    pub action_probs: Tensor<B, 1>,
    /// Action mask
    pub action_mask: Tensor<B, 1>,
    /// Value estimate from critic network
    pub value: Tensor<B, 1>,
    /// The move that was picked
    pub picked_move: Move,
}

/// Player that can select a move and evaluate a gamestate using a policy network
#[derive(Debug, Clone)]
pub struct PPOMoveSelector<B: Backend> {
    device: B::Device,
    policy: Policy<B>,
    value: Value<B>,
}

impl<B: Backend> PPOMoveSelector<B> {
    pub fn new(policy: PolicyConfig, value: ValueConfig, device: &B::Device) -> Self {
        Self {
            device: device.clone(),
            policy: policy.init(device),
            value: value.init(device),
        }
    }

    pub fn from_file(
        policy: PolicyConfig,
        value: ValueConfig,
        path: &std::path::Path,
        device: &B::Device,
    ) -> Self {
        let policy = policy.init(device);
        let value = value.init(device);

        let mut recorder = DefaultFileRecorder::<FullPrecisionSettings>::default();
        let policy = policy.load_file(path, &recorder, device).unwrap();
        Self {
            device: device.clone(),
            policy,
            value,
        }
    }

    pub fn action(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        self.policy.action(state)
    }

    pub fn value(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        self.value.value(state)
    }

    /// Pick a move and return all the other useful info that is required for training
    pub fn pick_move_train(
        &mut self,
        gamestate: &Gamestate<2, 6>,
        moves: Vec<Move>,
    ) -> PickReturn<B> {
        // Convert the gamestate into a tensor
        let state = Tensor::from_data(gs_to_array(gamestate).as_slice(), &self.device);
        // Get action vector and value
        let action = self.policy.action(state.clone());
        let value = self.value.value(state.clone());

        // Convert the moves into a vec of booleans to mask invalid
        let indices = moves.iter().map(|m| m.to_index()).collect::<Vec<_>>();
        let mut mask = [-1e8f32; 180];
        for &i in &indices {
            mask[i] = 0.0;
        }
        let masked_action = action.clone() + Tensor::from_data(mask.as_slice(), &self.device);

        let action_probs = activation::softmax(masked_action, 0);
        let action_probs_vec = action_probs.to_data().to_vec::<f32>().unwrap();

        // Choose from the actions
        let dist = WeightedIndex::new(action_probs_vec).unwrap();
        let choice = dist.sample(&mut rand::thread_rng());
        // Find the move with the corresponding value
        let (source, tile, destination) = index_to_move(choice);
        // println!("Moves: {:?}", moves);
        let m = moves
            .iter()
            .find(|m| {
                usize::from(m.source) == source
                    && usize::from(m.tile) == tile
                    && usize::from(m.destination) == destination
            })
            .cloned()
            .unwrap();
        PickReturn {
            state,
            action: choice,
            action_probs,
            action_mask: Tensor::from_data(mask.as_slice(), &self.device),
            value,
            picked_move: m,
        }
    }
}

impl<B: Backend> Player<2, 6> for PPOMoveSelector<B> {
    fn pick_move(
        &mut self,
        gamestate: &crate::gamestate::Gamestate<2, 6>,
        moves: Vec<crate::gamestate::Move>,
    ) -> crate::gamestate::Move {
        let pick = self.pick_move_train(gamestate, moves);
        pick.picked_move
    }

    fn name(&self) -> String {
        "PPOMoveSelector".into()
    }
}

#[derive(Config, Debug)]
pub struct PolicyConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl PolicyConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Policy<B> {
        let input = LinearConfig::new(self.input_size, self.hidden_size).init(device);
        let hidden = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let output = LinearConfig::new(self.hidden_size, 180).init(device);

        Policy {
            input,
            hidden,
            output,
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Policy<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Policy<B> {
    /// Run the policy network without normalising the result
    fn action(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = self.input.forward(state);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ValueConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl ValueConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Value<B> {
        let input = LinearConfig::new(self.input_size, self.hidden_size).init(device);
        let hidden = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let output = LinearConfig::new(self.hidden_size, 1).init(device);

        Value {
            input,
            hidden,
            output,
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
struct Value<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Value<B> {
    fn value(&self, state: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = self.input.forward(state);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
}
