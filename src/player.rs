use rand::{Rng, SeedableRng};

use crate::{
    gamestate::{Destination, Gamestate, Move},
    playerboard::Row,
};

/// Required implementation for a player
/// Main function is [Player::pick_move]
/// Gives read access to current gamestate
/// and a list of possible moves
pub trait Player<const P: usize, const F: usize> {
    fn pick_move(&mut self, gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move;
}

#[derive(Debug)]
pub struct RandomPlayer(rand::prelude::SmallRng);

impl RandomPlayer {
    pub fn new() -> Self {
        Self(rand::prelude::SmallRng::from_os_rng())
    }
}

impl Default for RandomPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const F: usize> Player<P, F> for RandomPlayer {
    fn pick_move(&mut self, _gamestate: &Gamestate<P, F>, moves: Vec<Move>) -> Move {
        moves[self.0.random_range(0..moves.len())]
    }
}

/// Picks moves based on a simple move ranking
#[derive(Default)]
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
