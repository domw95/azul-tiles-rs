//! Head to head between two evaluation weight vectors.
//!
//! Paired seeds, both seatings of each deal, so deal luck and whatever going
//! first is worth both cancel.
//!
//! Fixed depth isolates evaluation quality by holding the tree constant. Fixed
//! time is the question that actually matters, because a dearer evaluation buys
//! fewer nodes and has to pay for them. Timed mode therefore reports the nodes
//! and plies each side achieved alongside the win rate: a win rate alone cannot
//! distinguish "this term is not worth its cost" from "the box was busy and the
//! expensive side dropped a ply".
//!
//! Usage: `cargo run --release --bin evalmatch -- [pairs] [depth] [time_ms]`

use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{HeuristicEvaluator, Minimaxer, Weights, N_FEATURES};
use azul_tiles_rs::players::Player;
use minimaxer::negamax::SearchOptions;

#[derive(Clone, Copy)]
struct Budget {
    depth: u8,
    time_ms: u64,
}

fn opts(b: Budget) -> SearchOptions {
    if b.time_ms > 0 {
        SearchOptions {
            iterative: true,
            alpha_beta: true,
            max_time: Some(std::time::Duration::from_millis(b.time_ms)),
            ..Default::default()
        }
    } else {
        SearchOptions {
            max_depth: Some(b.depth),
            alpha_beta: true,
            ..Default::default()
        }
    }
}

/// What one game cost each seat, and who won it.
struct GameStats {
    /// Final margin from the point of view of seat 0.
    margin: i32,
    nodes: [u64; 2],
    plies: [u64; 2],
    searches: [u64; 2],
}

fn play(seed: u64, first_player: u8, weights: [Weights; 2], budget: Budget) -> GameStats {
    let mut players = [
        Minimaxer::new(opts(budget), "a", HeuristicEvaluator::new(weights[0])),
        Minimaxer::new(opts(budget), "b", HeuristicEvaluator::new(weights[1])),
    ];
    let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, first_player);
    loop {
        let moves = gs.get_moves();
        let seat = gs.current_player() as usize;
        let move_ = players[seat].pick_move(&gs, moves);
        if gs.play_move(move_) == State::RoundEnd && gs.end_round() == State::GameEnd {
            break;
        }
    }
    let s = gs.scores();
    GameStats {
        margin: i32::from(s[0]) - i32::from(s[1]),
        nodes: [players[0].nodes, players[1].nodes],
        plies: [players[0].depth_total, players[1].depth_total],
        searches: [players[0].searches, players[1].searches],
    }
}

fn matchup(name: &str, challenger: Weights, baseline: Weights, pairs: usize, budget: Budget) {
    let (mut wins, mut draws, mut losses) = (0u32, 0u32, 0u32);
    let mut margin_total = 0i64;
    let mut margin_sq = 0i64;
    let mut games = 0i64;
    // [challenger, baseline]
    let mut nodes = [0u64; 2];
    let mut plies = [0u64; 2];
    let mut searches = [0u64; 2];

    for i in 0..pairs {
        let seed = 0xc0ffee_0000 + i as u64;
        for first in 0..2u8 {
            for seat in 0..2usize {
                let w = if seat == 0 {
                    [challenger, baseline]
                } else {
                    [baseline, challenger]
                };
                let g = play(seed, first, w, budget);
                let margin = if seat == 0 { g.margin } else { -g.margin };
                // Index 0 is always the challenger, whichever seat it took.
                let c = seat;
                let b = 1 - seat;
                nodes[0] += g.nodes[c];
                nodes[1] += g.nodes[b];
                plies[0] += g.plies[c];
                plies[1] += g.plies[b];
                searches[0] += g.searches[c];
                searches[1] += g.searches[b];

                match margin.cmp(&0) {
                    std::cmp::Ordering::Greater => wins += 1,
                    std::cmp::Ordering::Equal => draws += 1,
                    std::cmp::Ordering::Less => losses += 1,
                }
                margin_total += i64::from(margin);
                margin_sq += i64::from(margin) * i64::from(margin);
                games += 1;
            }
        }
    }

    let n = games as f64;
    let score = (f64::from(wins) + 0.5 * f64::from(draws)) / n;
    let se = (score * (1.0 - score) / n).sqrt();
    let mean_margin = margin_total as f64 / n;
    let var = margin_sq as f64 / n - mean_margin * mean_margin;
    let margin_se = (var / n).sqrt();

    println!(
        "{name}\n  {games} games  +{wins} ={draws} -{losses}\n  \
         score {:.1}% +/- {:.1}   mean margin {:+.2} +/- {:.2}",
        100.0 * score,
        100.0 * 1.96 * se,
        mean_margin,
        1.96 * margin_se,
    );
    if budget.time_ms > 0 {
        let per = |i: usize| {
            (
                nodes[i] as f64 / searches[i] as f64,
                plies[i] as f64 / searches[i] as f64,
            )
        };
        let (cn, cd) = per(0);
        let (bn, bd) = per(1);
        println!(
            "  challenger {cn:.0} nodes/move, depth {cd:.2}   \
             baseline {bn:.0} nodes/move, depth {bd:.2}   ratio {:.3}",
            cn / bn,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(250);
    let depth: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let time_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let budget = Budget { depth, time_ms };

    let fitted: Weights = serde_json::from_reader(
        std::fs::File::open("eval_weights.json").expect("run the tune binary first"),
    )
    .unwrap();
    // The default has the forecast term off, so it is the cheap side of the
    // ablation. HeuristicEvaluator::new sees the zeros and skips the work, so
    // this really is the cheaper evaluator and not just a muted one.
    let no_forecast = Weights::default();
    let forecast_on = no_forecast.with_ts_forecast();
    let mut score_only = Weights([0.0; N_FEATURES]);
    score_only.0[0] = 1.0;

    if time_ms > 0 {
        println!("{time_ms}ms per move, {} games per matchup\n", pairs * 4);
    } else {
        println!("depth {depth}, {} games per matchup\n", pairs * 4);
    }

    // Identical weights both sides must read 50%. This detects asymmetry, which
    // is the class the one sided wall term belonged to. It cannot detect load
    // biasing a comparison between evaluators of different cost, because it is
    // symmetric by construction and that comparison is not: see the nodes/move
    // ratio for that.
    matchup("null control", no_forecast, no_forecast, pairs, budget);
    matchup("forecast on vs off", forecast_on, no_forecast, pairs, budget);
    if time_ms > 0 {
        return;
    }
    matchup("no forecast vs score only", no_forecast, score_only, pairs, budget);
    matchup("fitted vs score only", fitted, score_only, pairs, budget);
    matchup("fitted vs default", fitted, no_forecast, pairs, budget);
    matchup("forecast on vs score only", forecast_on, score_only, pairs, budget);
}
