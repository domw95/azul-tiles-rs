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
use minimaxer::negamax::SearchOptions;
use nalgebra::{DMatrix, DVector};

fn opts(depth: u8) -> SearchOptions {
    SearchOptions {
        max_depth: Some(depth),
        alpha_beta: true,
        ..Default::default()
    }
}

/// One game's round end positions, and the final margin they are labelled with.
struct Sample {
    features: Vec<[f32; N_FEATURES]>,
    margin: f32,
}

/// Play a game out, capturing the feature vector at each round end.
fn play_and_record(seed: u64, first_player: u8, players: &mut [Box<dyn Player<2, 6>>; 2]) -> Sample {
    let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, first_player);
    let mut collected = Vec::new();
    loop {
        let moves = gs.get_moves();
        let move_ = players[gs.current_player() as usize].pick_move(&gs, moves);
        if gs.play_move(move_) == State::RoundEnd {
            // Record before scoring the round: this is the position the search
            // would have evaluated as a leaf.
            collected.push(features(&gs));
            if gs.end_round() == State::GameEnd {
                break;
            }
        }
    }
    let scores = gs.scores();
    Sample {
        features: collected,
        margin: f32::from(scores[0]) - f32::from(scores[1]),
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

    // Generate with whatever currently ships, so the position distribution
    // matches where the fitted weights will actually be used.
    let start = Weights::default();
    println!("generating {games} games at depth {depth}");

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

    // Split by game, not by position. Positions from one game share a label,
    // so splitting by position would leak the answer into the holdout.
    let split = samples.len() * 4 / 5;
    let design = |batch: &[Sample]| {
        let rows: Vec<(&[f32; N_FEATURES], f32)> = batch
            .iter()
            .flat_map(|s| s.features.iter().map(move |f| (f, s.margin)))
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

    println!("\nR^2 against final margin{:>14}{:>10}", "train", "holdout");
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
