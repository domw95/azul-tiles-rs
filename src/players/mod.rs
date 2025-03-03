use dyn_clone::DynClone;
use nalgebra::{SMatrix, Vector6};
use rand::{Rng, SeedableRng};
use rand_distr::{Bernoulli, Distribution, StandardNormal};

use crate::gamestate::{Destination, Gamestate, Move};

pub mod minimax;
pub mod nn;

/// Required implementation for a player
/// Main function is [Player::pick_move]
/// Gives read access to current gamestate
/// and a list of possible moves
pub trait Player<const P: usize, const F: usize>: DynClone {
    fn pick_move(&mut self, gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move;
}

#[derive(Debug, Clone)]
pub struct RandomPlayer(rand::prelude::SmallRng);

impl RandomPlayer {
    pub fn new() -> Self {
        Self(rand::prelude::SmallRng::from_entropy())
    }
}

impl Default for RandomPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const F: usize> Player<P, F> for RandomPlayer {
    fn pick_move(&mut self, _gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move {
        moves[self.0.gen_range(0..moves.len())]
    }
}

/// Picks first move
#[derive(Default, Clone)]
pub struct FirstMovePlayer;

impl<const P: usize, const F: usize> Player<P, F> for FirstMovePlayer {
    fn pick_move(&mut self, _gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move {
        moves[0]
    }
}

/// Picks moves based on a simple move ranking
#[derive(Default, Clone)]
pub struct MoveRankPlayer;

impl MoveRankPlayer {
    pub fn new() -> Self {
        Self
    }

    fn compare_move<'a>(&self, a: &'a Move, b: &'a Move) -> &'a Move {
        match (a.destination, b.destination) {
            (Destination::Row(_), Destination::Floor) => a,
            (Destination::Floor, Destination::Row(_)) => b,
            (Destination::Floor, Destination::Floor) => b,
            (Destination::Row(a_r), Destination::Row(b_r)) => {
                match (a.fills_row(), b.fills_row()) {
                    (true, false) => a,
                    (false, true) => b,
                    (true, true) => a,
                    (false, false) => a,
                }
            }
        }
    }
}

impl<const P: usize, const F: usize> Player<P, F> for MoveRankPlayer {
    fn pick_move(&mut self, _gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move {
        *moves.iter().reduce(|a, b| self.compare_move(a, b)).unwrap()
    }
}

#[derive(Default, Clone)]
pub struct MoveRankPlayer2;

impl MoveRankPlayer2 {
    pub fn new() -> Self {
        Self
    }

    fn compare_move<'a>(
        &self,
        a: &'a (i8, bool, Move),
        b: &'a (i8, bool, Move),
    ) -> &'a (i8, bool, Move) {
        if a.0 > b.0 {
            return a;
        } else if a.0 < b.0 {
            return b;
        }
        if a.1 && !b.1 {
            return a;
        } else if !a.1 && b.1 {
            return b;
        }

        match (a.2.destination, b.2.destination) {
            (Destination::Row(_), Destination::Floor) => a,
            (Destination::Floor, Destination::Row(_)) => b,
            _ => a,
        }
    }
}

impl<const P: usize, const F: usize> Player<P, F> for MoveRankPlayer2 {
    fn pick_move(&mut self, gs: &Gamestate<P, F>, moves: Vec<Move>) -> Move {
        let moves = moves
            .into_iter()
            .map(|m| (gs.predict_score(m).1, gs.takes_fp(&m), m))
            .collect::<Vec<_>>();
        moves
            .iter()
            .reduce(|a, b| self.compare_move(a, b))
            .unwrap()
            .2
    }
}

pub trait EvolvingPlayer {
    /// Create a new random player
    fn birth() -> Self;
    /// Mutate the player with a given probability (prob)
    /// and a random number generator for the new value
    fn mutate(&self, prob: Bernoulli, rng: &mut rand::rngs::SmallRng) -> Self;
    /// Crossover with another player
    ///
    /// Select each player feature with a coin flip
    fn crossover(&self, other: &Self, prob: Bernoulli) -> Self;
}

#[derive(Debug, Clone)]
pub struct MoveWeightPlayer {
    weights: nalgebra::SMatrix<f32, 8, 1>,
}

impl MoveWeightPlayer {
    pub fn new(weights: [f32; 8]) -> Self {
        Self {
            weights: weights.into(),
        }
    }

    pub fn new_random() -> Self {
        let d = StandardNormal;
        let mut rng = rand::thread_rng();
        let weights: SMatrix<f32, 8, 1> = SMatrix::from_distribution(&d, &mut rng);
        Self {
            weights: weights.normalize(),
        }
    }

    fn score_move(&self, move_: &Move, gs: &Gamestate<2, 6>) -> f32 {
        let (score, delta) = gs.predict_score(*move_);
        [
            move_.count as f32,
            move_.floor_tiles() as f32,
            move_.row_capacity() as f32,
            move_.fills_row() as u8 as f32,
            delta as f32,
            move_.perfect_move() as u8 as f32,
            gs.takes_fp(move_) as u8 as f32,
            move_.no_floor_tiles() as u8 as f32,
        ]
        .iter()
        .zip(self.weights.iter())
        .map(|(a, b)| a * b)
        .sum()
    }
}

impl Player<2, 6> for MoveWeightPlayer {
    fn pick_move(&mut self, gamestate: &Gamestate<2, 6>, moves: Vec<Move>) -> Move {
        moves
            .into_iter()
            .map(|m| (m, self.score_move(&m, gamestate)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl EvolvingPlayer for MoveWeightPlayer {
    fn mutate(&self, prob: Bernoulli, rng: &mut rand::rngs::SmallRng) -> Self {
        let weights = self
            .weights
            .map(|w| {
                if prob.sample(rng) {
                    let a: f32 = rand_distr::StandardNormal.sample(rng);
                    w + a / 10.0
                } else {
                    w
                }
            })
            .normalize();

        Self { weights }
    }

    fn crossover(&self, other: &Self, prob: Bernoulli) -> Self {
        let weights = self
            .weights
            .map_with_location(|r, c, a| {
                if prob.sample(&mut rand::thread_rng()) {
                    a
                } else {
                    other.weights[(r, c)]
                }
            })
            .normalize();

        Self { weights }
    }

    fn birth() -> Self {
        Self::new_random()
    }
}

// Single layer neural network
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SLNNPlayer {
    weights1: nalgebra::SMatrix<f32, 16, 8>,
    weights2: nalgebra::SMatrix<f32, 1, 16>,
}

impl SLNNPlayer {
    pub fn new_random() -> Self {
        let d = StandardNormal;
        let mut rng = rand::thread_rng();
        let weights1: SMatrix<f32, 16, 8> = SMatrix::from_distribution(&d, &mut rng);
        let weights2: SMatrix<f32, 1, 16> = SMatrix::from_distribution(&d, &mut rng);
        Self {
            weights1: weights1.normalize(),
            weights2: weights2.normalize(),
        }
    }

    fn score_move(&self, move_: &Move, gs: &Gamestate<2, 6>) -> f32 {
        let (score, delta) = gs.predict_score(*move_);
        let input: SMatrix<f32, 8, 1> = [
            move_.count as f32,
            move_.floor_tiles() as f32,
            move_.row_capacity() as f32,
            move_.fills_row() as u8 as f32,
            delta as f32,
            move_.perfect_move() as u8 as f32,
            gs.takes_fp(move_) as u8 as f32,
            move_.no_floor_tiles() as u8 as f32,
        ]
        .into();
        let hidden = self.weights1 * input;
        let output = self.weights2 * hidden.map(|x| x.tanh());
        output[0]
    }
}

impl Player<2, 6> for SLNNPlayer {
    fn pick_move(&mut self, gamestate: &Gamestate<2, 6>, moves: Vec<Move>) -> Move {
        moves
            .into_iter()
            .map(|m| (m, self.score_move(&m, gamestate)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl EvolvingPlayer for SLNNPlayer {
    fn mutate(&self, prob: Bernoulli, rng: &mut rand::rngs::SmallRng) -> Self {
        let weights1 = self.weights1.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });

        let weights2 = self.weights2.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });
        Self { weights1, weights2 }
    }

    fn crossover(&self, other: &Self, prob: Bernoulli) -> Self {
        let weights1 = self.weights1.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.weights1[(r, c)]
            }
        });
        let weights2 = self.weights2.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.weights2[(r, c)]
            }
        });

        Self { weights1, weights2 }
    }

    fn birth() -> Self {
        Self::new_random()
    }
}
