use crate::gamestate;
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
pub struct Minimaxer {}

impl Player<2, 6> for Minimaxer {
    fn pick_move(
        &mut self,
        gamestate: &gamestate::Gamestate<2, 6>,
        moves: Vec<gamestate::Move>,
    ) -> gamestate::Move {
        let evaluator = ScoreEvaluator;
        let mut n = minimaxer::negamax::Negamax::new(Node::new(gamestate.clone()), evaluator);
        n.alpha_beta = true;
        n.max_depth = Some(5);
        let result = n.search();
        dbg!(&result);
        result.best
    }
}
