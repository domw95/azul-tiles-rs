//! Reproduces the worker's ponder-in-slices loop natively, to get a readable
//! panic out of what the wasm build can only report as `unreachable`.

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use std::time::Duration;

fn options() -> SearchOptions {
    SearchOptions {
        alpha_beta: true,
        iterative: true,
        pre_sort: true,
        tt_bits: 20,
        retain_depth: 4,
        initial_depth: u8::MAX,
        ..Default::default()
    }
}

fn main() {
    let slice: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(250);
    let mut game = Gamestate::<2, 6>::new_2_player_with_seed(11, 0);
    let mut search = Negamax::new(Node::new(game.clone()), ScoreEvaluator, options());
    let mut moves = game.get_moves();
    let mut rng = 12345u64;

    for step in 0..400 {
        if moves.is_empty() {
            if game.end_round() == State::GameEnd {
                println!("game end at step {step}");
                return;
            }
            search = Negamax::new(Node::new(game.clone()), ScoreEvaluator, options());
            moves = game.get_moves();
            continue;
        }
        // Two ponder slices, as the worker would run between messages.
        for _ in 0..2 {
            search.options.max_time = Some(Duration::from_millis(slice));
            let r = search.search();
            println!(
                "step {step} depth {} exit {:?} nodes {} tree {}",
                r.depth,
                r.exit,
                r.nodes,
                search.tree_size()
            );
        }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let m: Move = moves[(rng >> 33) as usize % moves.len()];
        game.play_move(m);
        if !search.play_move(&m) {
            search = Negamax::new(Node::new(game.clone()), ScoreEvaluator, options());
        }
        moves = game.get_moves();
    }
}
