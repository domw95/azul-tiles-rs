#![recursion_limit = "512"]
//! What one call to `evaluate` costs, per evaluator.
//! Args: <eval> [<eval> ...] with `--positions <n>` and `--reps <n>` optional.
//!
//!   eval: score | heuristic | handset | nn:<dir>[:<tag>]
//!
//! This is the number the NN-evaluation argument turns on. Alpha-beta calls
//! `evaluate` once per leaf and thousands of times per move, so the ratio
//! between two evaluators' per-call costs *is* the depth they can afford
//! relative to each other -- and one ply is worth about twelve points of
//! margin on this game. A race under a clock gives the answer that matters,
//! but it gives it as a win rate, which cannot say whether a loss was quality
//! or speed. This says which.
//!
//! Measured on real positions from real games rather than one position over
//! and over: an evaluator whose cost depends on how full the boards are (the
//! heuristic's wall scan does) would otherwise be measured at whatever a
//! single position happens to cost, and the network's input encoding is built
//! from scratch every call regardless.
//!
//! Every evaluator sees the same positions and the passes are interleaved, so
//! a machine whose load drifts during the run perturbs them together rather
//! than handing the last one measured a different box. That matters here: this
//! is a shared box, and a contended timing is worthless as an absolute. The
//! ratio between evaluators measured in one interleaved run survives it.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{HeuristicEvaluator, ScoreEvaluator, Weights};
use azul_tiles_rs::players::nn_eval::NnEvaluator;
use burn::backend::NdArray;
use minimaxer::Evaluate;
use std::time::Instant;

/// One timed pass over every position.
///
/// Generic, and called once per pass rather than once per position, so the
/// inner loop is monomorphised and the dispatch is amortised over thousands of
/// evaluations. A `Box<dyn ...>` here would put a virtual call inside the loop,
/// which costs a couple of nanoseconds -- nothing against a network, but tens
/// of percent against `ScoreEvaluator`, and it is precisely their ratio being
/// measured. (`Evaluate` requires `Clone`, so it is not dyn compatible anyway.)
fn timed_pass<E: Evaluate<Gamestate<2, 6>>>(e: &mut E, states: &[Gamestate<2, 6>]) -> u128 {
    let t0 = Instant::now();
    for gs in states {
        // Black box, or the whole loop is dead code for the cheap evaluators
        // and the comparison becomes "arithmetic the optimiser deleted"
        // against "a network it could not".
        std::hint::black_box(e.evaluate(std::hint::black_box(gs)));
    }
    t0.elapsed().as_nanos()
}

enum AnyEvaluator {
    Score(ScoreEvaluator),
    Heuristic(HeuristicEvaluator),
    Nn(NnEvaluator<NdArray>),
}

impl AnyEvaluator {
    fn pass(&mut self, states: &[Gamestate<2, 6>]) -> u128 {
        match self {
            Self::Score(e) => timed_pass(e, states),
            Self::Heuristic(e) => timed_pass(e, states),
            Self::Nn(e) => timed_pass(e, states),
        }
    }
}

fn evaluator(spec: &str) -> AnyEvaluator {
    let mut parts = spec.split(':');
    match parts.next().expect("evaluator") {
        "score" => AnyEvaluator::Score(ScoreEvaluator),
        "heuristic" => AnyEvaluator::Heuristic(HeuristicEvaluator::default()),
        "handset" => AnyEvaluator::Heuristic(HeuristicEvaluator::new(Weights::hand_set())),
        "nn" => {
            let dir = parts.next().expect("nn:<dir>[:<tag>]");
            let tag = parts.next().unwrap_or("best");
            let device = Default::default();
            AnyEvaluator::Nn(
                NnEvaluator::<NdArray>::from_checkpoint(std::path::Path::new(dir), tag, &device)
                    .expect("checkpoint"),
            )
        }
        other => panic!("unknown evaluator {other:?}"),
    }
}

/// Positions taken from self-play, spread across the whole game rather than
/// bunched at the opening, where the boards are empty and every evaluator is
/// at its cheapest.
fn positions(want: usize) -> Vec<Gamestate<2, 6>> {
    let mut out = Vec::with_capacity(want);
    let mut seed = 9_500_000u64;
    while out.len() < want {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        loop {
            let moves = gs.get_moves();
            // Greedy on the static evaluation: cheap, and it keeps the boards
            // in shapes a search would actually reach.
            let mut best = moves[0];
            let mut best_v = f32::NEG_INFINITY;
            let sign = if gs.current_player() == 0 { 1.0 } else { -1.0 };
            for m in &moves {
                let mut next = gs.clone();
                next.play_move(*m);
                let v = sign * next.differential_predicted_score();
                if v > best_v {
                    best_v = v;
                    best = *m;
                }
            }
            out.push(gs.clone());
            let s = gs.play_move(best);
            if s == State::RoundEnd && gs.end_round() == State::GameEnd {
                break;
            }
            if out.len() >= want {
                break;
            }
        }
        seed += 1;
    }
    out.truncate(want);
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut specs: Vec<String> = Vec::new();
    let mut n_positions = 2000usize;
    let mut reps = 20usize;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--positions" => {
                n_positions = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--reps" => {
                reps = argv[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                specs.push(argv[i].clone());
                i += 1;
            }
        }
    }
    assert!(!specs.is_empty(), "give at least one evaluator");

    let states = positions(n_positions);
    let mut evals: Vec<_> = specs.iter().map(|s| evaluator(s)).collect();
    println!(
        "{} positions, {reps} interleaved passes each, {} evaluations per evaluator",
        states.len(),
        states.len() * reps
    );

    // Warm up every evaluator before timing any of them. Burn initialises its
    // parameters lazily on first use, which would otherwise land entirely on
    // the network's first timed pass.
    let warm: Vec<_> = states.iter().take(64).cloned().collect();
    for e in &mut evals {
        e.pass(&warm);
    }

    let mut totals = vec![0u128; evals.len()];
    for _ in 0..reps {
        for (k, e) in evals.iter_mut().enumerate() {
            totals[k] += e.pass(&states);
        }
    }

    let calls = (states.len() * reps) as f64;
    let base = totals[0] as f64 / calls;
    for (spec, total) in specs.iter().zip(&totals) {
        let ns = *total as f64 / calls;
        println!(
            "{spec}: {ns:.0} ns/eval, {:.2}M evals/s, {:.0}x {}",
            1000.0 / ns,
            ns / base,
            specs[0]
        );
    }
}
