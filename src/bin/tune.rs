//! Fit the heuristic evaluation weights by least squares on round end
//! positions.
//!
//! The search treats the end of a round as terminal, so that is the only kind
//! of position the evaluation is ever asked to judge. This plays out games,
//! records the feature vector at every round end, labels each one with the
//! final score margin of the game it came from, and solves for the weights that
//! best predict that margin.
//!
//! Every position is a training example, rather than every game being one bit
//! as it is when ranking players by win rate, which is what makes this cheap
//! enough to be worth doing.
//!
//! Usage: `cargo run --release --bin tune -- [games] [depth] [ridge]`

use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{
    features, HeuristicEvaluator, Minimaxer, Weights, FEATURE_NAMES, N_FEATURES,
};
use azul_tiles_rs::players::Player;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use nalgebra::{DMatrix, DVector};

fn opts(depth: u8) -> SearchOptions {
    SearchOptions {
        max_depth: Some(depth),
        alpha_beta: true,
        ..Default::default()
    }
}

/// One game's round end positions: the features, the position itself so a
/// deeper label can be computed later, and the final margin of the game.
struct Sample {
    features: Vec<[f32; N_FEATURES]>,
    states: Vec<Gamestate<2, 6>>,
    margin: f32,
    /// One label per recorded position, filled once the target is known.
    labels: Vec<f32>,
}

/// What to regress against.
#[derive(Clone, Copy, PartialEq)]
enum Target {
    /// The final score margin of the game the position came from.
    ///
    /// Simple, but the label sits up to nine rounds of dealing downstream, so
    /// most of its variance is future chance that no evaluation could predict.
    /// It also teaches correlation rather than cause: a board is not losing
    /// because of the feature that happens to mark losing boards.
    Final,
    /// What a search says the position is worth, looking across the round
    /// boundary.
    ///
    /// Deals the next round, searches it with the current weights at the
    /// leaves, and averages over several deals. That is one Bellman backup, so
    /// refitting to it pushes the horizon a round further out each pass, and
    /// the label carries one round of averaged chance instead of nine.
    Backup,
}

/// Value of a round end position, one round deeper than the evaluation can see.
fn backup_value(
    state: &Gamestate<2, 6>,
    weights: Weights,
    depth: u8,
    samples: u32,
    seed: u64,
) -> f32 {
    let mut total = 0.0;
    for k in 0..samples {
        let mut g = state.clone();
        // Each sample needs its own deal, or they are all the same round.
        g.reseed(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(u64::from(k)));
        if g.end_round() == State::GameEnd {
            // Nothing left to search: the game's own result is the value.
            let s = g.scores();
            total += f32::from(s[0]) - f32::from(s[1]);
            continue;
        }
        let mut search = Negamax::new(
            Node::new(g.clone()),
            HeuristicEvaluator::new(weights),
            opts(depth),
        );
        let result = search.search();
        // The root's value is from the perspective of the player to move, and
        // the features are player 0's lead, so flip when player 1 is on move.
        let sign = if g.current_player() == 0 { 1.0 } else { -1.0 };
        total += result.value * sign;
    }
    total / samples as f32
}

/// Play a game out, capturing the feature vector at each round end.
fn play_and_record(seed: u64, first_player: u8, players: &mut [Box<dyn Player<2, 6>>; 2]) -> Sample {
    let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, first_player);
    let mut collected = Vec::new();
    let mut states = Vec::new();
    loop {
        let moves = gs.get_moves();
        let move_ = players[gs.current_player() as usize].pick_move(&gs, moves);
        if gs.play_move(move_) == State::RoundEnd {
            // Record before scoring the round: this is the position the search
            // would have evaluated as a leaf.
            collected.push(features(&gs));
            states.push(gs.clone());
            if gs.end_round() == State::GameEnd {
                break;
            }
        }
    }
    let scores = gs.scores();
    Sample {
        features: collected,
        states,
        margin: f32::from(scores[0]) - f32::from(scores[1]),
        labels: Vec::new(),
    }
}

/// R squared of a weight vector against the labels.
fn r_squared(a: &DMatrix<f64>, b: &DVector<f64>, w: &DVector<f64>) -> f64 {
    let mean = b.mean();
    let ss_tot: f64 = b.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res = (b - a * w).norm_squared();
    1.0 - ss_res / ss_tot
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let depth: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let ridge: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let target = match args.get(4).map(|s| s.as_str()) {
        Some("backup") => Target::Backup,
        _ => Target::Final,
    };
    let backup_depth: u8 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
    let backup_samples: u32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(3);
    // Starting weights: these both generate the games and sit at the leaves of
    // every backup search, so iterating the value iteration means passing the
    // previous pass's output back in here.
    let start: Weights = match args.get(7) {
        Some(path) => serde_json::from_reader(
            std::fs::File::open(path).expect("cannot open start weights"),
        )
        .expect("cannot parse start weights"),
        None => Weights::default(),
    };

    println!(
        "generating {games} games at depth {depth}, target {}",
        if target == Target::Backup { "backup" } else { "final margin" }
    );

    let mut samples = Vec::new();
    for i in 0..games {
        let seed = 0x5eed_0000 + i as u64;
        // Both seatings of the same deal, so the data is not biased towards
        // whatever going first is worth.
        for first in 0..2u8 {
            let mut players: [Box<dyn Player<2, 6>>; 2] = [
                Box::new(Minimaxer::new(
                    opts(depth),
                    "p0",
                    HeuristicEvaluator::new(start),
                )),
                Box::new(Minimaxer::new(
                    opts(depth),
                    "p1",
                    HeuristicEvaluator::new(start),
                )),
            ];
            samples.push(play_and_record(seed, first, &mut players));
        }
        if (i + 1) % 25 == 0 {
            println!("  {} games, {} positions", i + 1, samples.iter().map(|s| s.features.len()).sum::<usize>());
        }
    }

    // Attach labels. The backup target costs a search per sample, so report
    // what it is doing rather than appearing to hang.
    match target {
        Target::Final => {
            for s in &mut samples {
                s.labels = vec![s.margin; s.features.len()];
            }
        }
        Target::Backup => {
            let total: usize = samples.iter().map(|s| s.features.len()).sum();
            println!(
                "backing up {total} positions: {backup_samples} deals each, depth {backup_depth}"
            );
            let mut done = 0usize;
            for (gi, s) in samples.iter_mut().enumerate() {
                let states = std::mem::take(&mut s.states);
                s.labels = states
                    .iter()
                    .enumerate()
                    .map(|(pi, st)| {
                        backup_value(
                            st,
                            start,
                            backup_depth,
                            backup_samples,
                            (gi as u64) << 20 | pi as u64,
                        )
                    })
                    .collect();
                done += s.labels.len();
                if gi % 200 == 0 && gi > 0 {
                    println!("  {done}/{total}");
                }
            }
        }
    }

    // Split by game, not by position. Positions from one game share a label,
    // so splitting by position would leak the answer into the holdout.
    let split = samples.len() * 4 / 5;
    let design = |batch: &[Sample]| {
        let rows: Vec<(&[f32; N_FEATURES], f32)> = batch
            .iter()
            .flat_map(|s| s.features.iter().zip(s.labels.iter().copied()))
            .collect();
        let n = rows.len();
        let a = DMatrix::<f64>::from_fn(n, N_FEATURES, |i, j| f64::from(rows[i].0[j]));
        let b = DVector::<f64>::from_fn(n, |i, _| f64::from(rows[i].1));
        (a, b)
    };
    let (a, b) = design(&samples[..split]);
    let (a_test, b_test) = design(&samples[split..]);
    let n = a.nrows();
    println!("\n{n} training positions, {} held out", a_test.nrows());

    // How often each term actually fires, which says how much to trust it.
    println!("\n{:<20} {:>12}", "feature", "nonzero %");
    for j in 0..N_FEATURES {
        let hits = (0..n).filter(|&i| a[(i, j)].abs() > 1e-6).count();
        println!("{:<20} {:>11.1}%", FEATURE_NAMES[j], 100.0 * hits as f64 / n as f64);
    }

    // Ridge keeps the solve well conditioned; the forecast buckets for lines
    // needing three or four tiles are rare and nearly collinear with each other.
    let mut normal = a.transpose() * &a;
    for i in 0..N_FEATURES {
        normal[(i, i)] += ridge;
    }
    let atb = a.transpose() * &b;
    let solved = normal.lu().solve(&atb).expect("normal equations singular");

    let mut fitted = Weights([0.0; N_FEATURES]);
    for i in 0..N_FEATURES {
        fitted.0[i] = solved[i] as f32;
    }
    let fitted = fitted.normalised();

    // Compare against the hand set weights and against using the score
    // differential alone, which is what ScoreEvaluator does.
    let as_vec = |w: &Weights| DVector::<f64>::from_fn(N_FEATURES, |i, _| f64::from(w.0[i]));
    let mut score_only = Weights([0.0; N_FEATURES]);
    score_only.0[0] = 1.0;

    println!("\n{:<20} {:>10} {:>10}", "feature", "current", "fitted");
    for i in 0..N_FEATURES {
        println!(
            "{:<20} {:>10.3} {:>10.3}",
            FEATURE_NAMES[i], start.0[i], fitted.0[i]
        );
    }

    println!(
        "\nR^2 against {}{:>14}{:>10}",
        if target == Target::Backup { "backed up value" } else { "final margin" },
        "train",
        "holdout"
    );
    for (name, w) in [
        ("score only", score_only),
        ("hand set", Weights::hand_set()),
        ("current", start),
        ("fitted", fitted),
    ] {
        println!(
            "  {:<12} {:>21.4} {:>9.4}",
            name,
            r_squared(&a, &b, &as_vec(&w)),
            r_squared(&a_test, &b_test, &as_vec(&w)),
        );
    }

    let path = "eval_weights.json";
    serde_json::to_writer_pretty(std::fs::File::create(path).unwrap(), &fitted).unwrap();
    println!("\nwrote {path}");
}
