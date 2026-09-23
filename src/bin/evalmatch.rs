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
    time_us: u64,
}

fn opts(b: Budget) -> SearchOptions {
    if b.time_us > 0 {
        SearchOptions {
            iterative: true,
            alpha_beta: true,
            max_time: Some(std::time::Duration::from_micros(b.time_us)),
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

fn play(seed: u64, first_player: u8, weights: [Weights; 2], budgets: [Budget; 2]) -> GameStats {
    let mut players = [
        Minimaxer::new(opts(budgets[0]), "a", HeuristicEvaluator::new(weights[0])),
        Minimaxer::new(opts(budgets[1]), "b", HeuristicEvaluator::new(weights[1])),
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

fn matchup(
    name: &str,
    challenger: Weights,
    baseline: Weights,
    pairs: usize,
    budget: Budget,
) {
    matchup_handicap(name, challenger, baseline, pairs, [budget, budget]);
}

/// As [`matchup`], but each side gets its own budget, so one can be handicapped.
fn matchup_handicap(
    name: &str,
    challenger: Weights,
    baseline: Weights,
    pairs: usize,
    budgets: [Budget; 2],
) {
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
                let bud = if seat == 0 {
                    [budgets[0], budgets[1]]
                } else {
                    [budgets[1], budgets[0]]
                };
                let g = play(seed, first, w, bud);
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
    if budgets[0].time_us > 0 {
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

/// Measure what a node deficit is worth in win rate.
///
/// The obvious experiment, handicapping one side by the ~10% a real feature
/// costs, is not affordable: theory puts that at roughly a point of win rate,
/// and resolving a point needs on the order of ten thousand games. So measure
/// where the signal is strong, at large handicaps, and interpolate down.
///
/// Strength is roughly linear in log time, so the fit is over log2 of the
/// achieved node ratio rather than the ratio itself. The achieved ratio is what
/// gets reported and fitted, not the nominal one: iterative deepening is a
/// staircase, and a side given 15% less clock may well search the same depth
/// and give up nothing.
fn calibrate(pairs: usize, base_us: u64) {
    let w = Weights::default();
    let base = Budget { depth: 0, time_us: base_us };
    println!(
        "calibration: identical evaluators, one side handicapped\n\
         base {:.2}ms per move, {} games per level\n",
        base_us as f64 / 1000.0,
        pairs * 4
    );
    // 1.0 is the null control and anchors the fit at 50%.
    for frac in [1.0, 0.85, 0.7, 0.5] {
        let handicapped = Budget {
            depth: 0,
            time_us: (base_us as f64 * frac).round() as u64,
        };
        matchup_handicap(
            &format!("handicap {frac:.2}"),
            w,
            w,
            pairs,
            [handicapped, base],
        );
    }
    println!(
        "\nfit points-per-log2(node ratio) from the rows above; a feature whose\n\
         node ratio is r can then be expected to lose that rate * -log2(r)\n\
         points from its fixed depth margin."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("screen");
    let pairs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    if mode == "calibrate" {
        let base_ms: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        calibrate(pairs, (base_ms * 1000.0).round() as u64);
        return;
    }

    // A probe runs only the null control and the candidate, for when the full
    // set is too dear: depth 5 costs roughly sixteen times depth 3.
    let probe = mode == "probe";
    let budget = if mode == "timed" || (probe && args.get(4).map_or(false, |v| v != "0")) {
        let idx = if probe { 4 } else { 3 };
        let ms: f64 = args.get(idx).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        Budget { depth: 0, time_us: (ms * 1000.0).round() as u64 }
    } else {
        // Screen at the depth the timed test actually reaches, otherwise the
        // screen and the confirmation differ in depth as well as in cost and
        // neither tells you which one moved the result.
        let depth: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        Budget { depth, time_us: 0 }
    };

    let default = Weights::default();
    let hand_set = Weights::hand_set();
    let mut score_only = Weights([0.0; N_FEATURES]);
    score_only.0[0] = 1.0;
    // A freshly fitted vector, if one has been written, measured against
    // whatever currently ships. This is the comparison that decides the default.
    let candidate: Option<Weights> = std::fs::File::open("eval_weights.json")
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok())
        .filter(|c: &Weights| *c != default);

    if budget.time_us > 0 {
        println!("{:.2}ms per move, {} games per matchup\n", budget.time_us as f64 / 1000.0, pairs * 4);
    } else {
        println!("depth {}, {} games per matchup\n", budget.depth, pairs * 4);
    }

    // Identical weights both sides must read 50%. This detects asymmetry, which
    // is the class the one sided wall term belonged to. It cannot detect load
    // biasing a comparison between evaluators of different cost, because it is
    // symmetric by construction and that comparison is not: see the nodes/move
    // ratio for that, and the calibrate mode for what a ratio costs.
    matchup("null control", default, default, pairs, budget);
    if probe {
        match &candidate {
            Some(c) => matchup("candidate vs default", *c, default, pairs, budget),
            None => println!("no distinct eval_weights.json to probe"),
        }
        return;
    }
    matchup("default vs hand set", default, hand_set, pairs, budget);
    matchup("default vs score only", default, score_only, pairs, budget);
    matchup("default-no-forecast vs default", default.without_forecast(), default, pairs, budget);

    if let Some(c) = candidate {
        matchup("eval_weights.json vs default", c, default, pairs, budget);
    } else {
        println!("no distinct eval_weights.json to compare");
    }

    if budget.time_us > 0 {
        // Timed runs cost ~10x a screen and have a +/-5 run to run spread on
        // this machine, so under a clock we ask only the decisive questions.
        return;
    }
    matchup("hand set vs score only", hand_set, score_only, pairs, budget);
    matchup("ts-forecast vs default", hand_set.with_ts_forecast(), default, pairs, budget);

}
