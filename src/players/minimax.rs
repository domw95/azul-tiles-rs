use crate::gamestate;
use log::debug;
use minimaxer::{self, node::Node};

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

struct ScoreEvaluator;

impl minimaxer::Evaluate<gamestate::Gamestate<2, 6>> for ScoreEvaluator {
    fn evaluate(&mut self, g: &gamestate::Gamestate<2, 6>) -> f32 {
        g.differential_predicted_score()
    }
}

#[derive(Debug, Clone)]
pub struct Minimaxer {
    pub opts: minimaxer::negamax::SearchOptions,
    pub name: String,
}

impl Minimaxer {
    pub fn new(opts: minimaxer::negamax::SearchOptions, name: impl Into<String>) -> Self {
        Self {
            opts,
            name: name.into(),
        }
    }
}

impl Player<2, 6> for Minimaxer {
    fn pick_move(
        &mut self,
        gamestate: &gamestate::Gamestate<2, 6>,
        moves: Vec<gamestate::Move>,
    ) -> gamestate::Move {
        let evaluator = ScoreEvaluator;
        let mut n =
            minimaxer::negamax::Negamax::new(Node::new(gamestate.clone()), evaluator, self.opts);
        let result = n.search();
        debug!("Minimax search result: {:?}", result);
        result.best
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
