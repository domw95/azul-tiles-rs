//! Does pondering in slices actually make progress?
//!
//! A pass cut short by the time limit is discarded, so if every slice is
//! shorter than one pass the search could in principle spin forever at the
//! same depth. The transposition table is what should stop that: a redone
//! pass reads the values the discarded one stored. This measures which of the
//! two happens.

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use std::time::{Duration, Instant};

fn main() {
    let slice: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(250);
    let slices: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    // A fresh round start, the position the worker would begin pondering from.
    let mut game = Gamestate::<2, 6>::new_2_player_with_seed(11, 0);
    let mut rng = 999u64;
    for _ in 0..3 {
        let moves = game.get_moves();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let m: Move = moves[(rng >> 33) as usize % moves.len()];
        if game.play_move(m) == State::GameEnd {
            break;
        }
    }

    let mut search = Negamax::new(
        Node::new(game),
        ScoreEvaluator,
        SearchOptions {
            alpha_beta: true,
            iterative: true,
            pre_sort: true,
            tt_bits: 20,
            retain_depth: 4,
            initial_depth: u8::MAX,
            max_time: Some(Duration::from_millis(slice)),
            ..Default::default()
        },
    );

    println!("slice={slice}ms  {slices} slices on one position\n");
    println!("{:>6} {:>8} {:>8} {:>12} {:>10}", "slice", "ms", "depth", "exit", "tree");
    for i in 1..=slices {
        let t = Instant::now();
        let r = search.search();
        println!(
            "{:>6} {:>8.0} {:>8} {:>12} {:>10}",
            i,
            t.elapsed().as_secs_f64() * 1000.0,
            r.depth,
            format!("{:?}", r.exit),
            search.tree_size()
        );
        if r.exit == minimaxer::SearchExit::Exhaustive {
            println!("\nsolved after {i} slices");
            return;
        }
    }
}
