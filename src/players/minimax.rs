use crate::gamestate;
use log::debug;
use minimaxer::{self, negamax::SearchOptions, node::Node, Evaluate};

use super::Player;

impl minimaxer::Gamestate<gamestate::Move> for gamestate::Gamestate<2, 6> {
    fn get_moves(&mut self) -> Vec<gamestate::Move> {
        gamestate::Gamestate::get_moves(self)
    }

    fn play_move(&mut self, m: &gamestate::Move) {
        gamestate::Gamestate::play_move(self, *m);
    }

    fn player_aim(&self) -> minimaxer::NodeAim {
        match self.current_player() {
            0 => minimaxer::NodeAim::Maximise,
            1 => minimaxer::NodeAim::Minimise,
            _ => panic!("Invalid player"),
        }
    }
}

impl minimaxer::Move for gamestate::Move {}

#[derive(Debug, Clone)]
pub struct ScoreEvaluator;

impl minimaxer::Evaluate<gamestate::Gamestate<2, 6>> for ScoreEvaluator {
    fn evaluate(&mut self, g: &gamestate::Gamestate<2, 6>) -> f32 {
        g.differential_predicted_score()
    }
}

// Evaluate based on score and other heuristics
#[derive(Debug, Clone)]
pub struct HeuristicEvaluator {
    fp_weight: f32,
    wall_weight: [[f32; 5]; 5], // Weight for each position on the wall
}

impl HeuristicEvaluator {
    pub fn new_no_wall_weight(fp_weight: f32) -> Self {
        Self {
            fp_weight,
            wall_weight: [[0.0; 5]; 5],
        }
    }
}

impl Default for HeuristicEvaluator {
    fn default() -> Self {
        Self {
            fp_weight: 0.5,
            wall_weight: [
                [0.9, 0.95, 0.97, 0.95, 0.9],
                [0.95, 0.97, 1.0, 0.97, 0.95],
                [0.9, 0.95, 0.97, 0.95, 0.9],
                [0.85, 0.9, 0.95, 0.9, 0.85],
                [0.8, 0.85, 0.9, 0.85, 0.8],
            ],
        }
    }
}

impl minimaxer::Evaluate<gamestate::Gamestate<2, 6>> for HeuristicEvaluator {
    fn evaluate(&mut self, g: &gamestate::Gamestate<2, 6>) -> f32 {
        // Combine various heuristics to evaluate the game state
        let mut score = g.differential_predicted_score();
        // Check who has the first tile marker
        score += if g.boards()[0].first_player_tile {
            self.fp_weight
        } else if g.boards()[1].first_player_tile {
            -self.fp_weight
        } else {
            0.0
        };
        let wall = g.boards()[0].simulate_wall();
        for (row, weight) in wall.iter().zip(self.wall_weight.iter()) {
            for (tile, &w) in row.iter().zip(weight.iter()) {
                if tile.is_some() {
                    score += w;
                }
            }
        }
        score
    }
}

#[derive(Debug, Clone)]
pub struct Minimaxer<E> {
    pub opts: minimaxer::negamax::SearchOptions,
    pub name: String,
    pub evaluator: E,
}

impl<E> Minimaxer<E> {
    pub fn new(
        opts: minimaxer::negamax::SearchOptions,
        name: impl Into<String>,
        evaluator: E,
    ) -> Self {
        Self {
            opts,
            name: name.into(),
            evaluator,
        }
    }
}

impl<E: Evaluate<gamestate::Gamestate<2, 6>>> Player<2, 6> for Minimaxer<E> {
    fn pick_move(
        &mut self,
        gamestate: &gamestate::Gamestate<2, 6>,
        moves: Vec<gamestate::Move>,
    ) -> gamestate::Move {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(gamestate.clone()),
            self.evaluator.clone(),
            self.opts,
        );
        let result = n.search();
        debug!("Minimax search result: {:?}", result);
        result.best
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
