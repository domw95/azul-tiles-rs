//! Minimax against minimax, to calibrate what a win rate means.
//! Args: <depth_a> <depth_b> [games]
//!
//! A win rate against a fixed opponent saturates: Azul's tile draws decide
//! close games, so even a far stronger player cannot approach 100%. Without
//! knowing what a *stronger searcher* scores against the same opponent there
//! is no way to tell "the policy is near the ceiling" from "the policy has
//! plenty left to gain", and the depth-2 yardstick may already be exhausted.
//!
//! Reports the margin's standard deviation as well as the mean, since that is
//! what converts a score advantage into a win rate.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::Player;
use minimaxer::negamax::SearchOptions;

fn searcher(depth: u8) -> Minimaxer<ScoreEvaluator> {
    Minimaxer::new(
        SearchOptions {
            max_depth: Some(depth),
            alpha_beta: true,
            sort_on_create: true,
            sort_on_create_min_depth: 1,
            tt_bits: 20,
            ..Default::default()
        },
        "mm",
        ScoreEvaluator,
    )
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let depth_a: u8 = a[1].parse().unwrap();
    let depth_b: u8 = a[2].parse().unwrap();
    let games: u64 = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(200);

    // Same seed range the policy holdouts use, so the numbers are comparable.
    for seat in [0usize, 1] {
        let (mut wins, mut ours, mut theirs) = (0u32, 0u32, 0u32);
        let mut margins: Vec<f64> = Vec::new();
        let mut pa = searcher(depth_a);
        let mut pb = searcher(depth_b);
        for seed in 9_000_000..9_000_000 + games {
            let mut gs = Gamestate::new_2_player_with_seed(seed, 0);
            loop {
                let moves = gs.get_moves();
                let state = if gs.current_player() as usize == seat {
                    gs.play_move(pa.pick_move(&gs, moves))
                } else {
                    gs.play_move(pb.pick_move(&gs, moves))
                };
                if state == State::RoundEnd && gs.end_round() == State::GameEnd {
                    break;
                }
            }
            let s = gs.scores();
            ours += s[seat] as u32;
            theirs += s[1 - seat] as u32;
            margins.push(s[seat] as f64 - s[1 - seat] as f64);
            if s[seat] > s[1 - seat] {
                wins += 1;
            }
        }
        let n = games as f64;
        let mean = margins.iter().sum::<f64>() / n;
        let sd = (margins.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n).sqrt();
        println!(
            "depth{depth_a} vs depth{depth_b} seat{seat}: {wins}/{games} ({:.1}%) mean {:.1} v {:.1} | margin {:+.2} sd {:.2}",
            100.0 * wins as f64 / n,
            ours as f64 / n,
            theirs as f64 / n,
            mean,
            sd
        );
    }
}
