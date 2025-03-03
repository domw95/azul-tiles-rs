use azul_tiles_rs::players::Player;
use azul_tiles_rs::{gamestate::Gamestate, players::minimax::Minimaxer};

fn main() {
    // Create game
    let mut game = Gamestate::<2, 6>::new_2_player();
    let mut player = Minimaxer {};
    player.pick_move(&game, game.get_moves());
}
