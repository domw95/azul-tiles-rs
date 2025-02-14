//! Module for running (multiple) games from start to finish with players

use crate::{
    gamestate::{Gamestate, State},
    player::Player,
};

/// Game runner
struct Runner<const P: usize, const F: usize> {
    gamestate: Gamestate<P, F>,
    players: [Box<dyn Player<P, F>>; P],
}

impl Runner<2, 6> {
    pub fn new_2_player(players: [Box<dyn Player<2, 6>>; 2]) -> Self {
        Self {
            gamestate: Gamestate::new_2_player(),
            players,
        }
    }

    pub fn play_round(&mut self) {
        loop {
            let moves = self.gamestate.get_moves();
            let move_ = self.players[self.gamestate.current_player() as usize]
                .pick_move(&self.gamestate, moves);
            if self.gamestate.play_move(move_) == State::RoundEnd {
                break;
            }
        }
    }
}
