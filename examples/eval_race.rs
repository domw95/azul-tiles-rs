#![recursion_limit = "512"]
//! Race two evaluators against each other under one budget.
//! Args: <eval_a> <eval_b> <budget> [games] [sort_on_create 1|0] [first_seed]
//!
//!   eval:   score | heuristic | handset | nn:<dir>[:<tag>]
//!   budget: depth:<n> | time:<ms>, or two of those comma separated to give
//!           each side its own -- `depth:3,depth:2` is the ladder that
//!           calibrates what a margin is worth, and it is the same harness,
//!           so the numbers are directly comparable to an equal-budget race.
//!
//! **`time:` is the measurement that means anything; `depth:` is a control.**
//! An evaluator that is better per leaf but slower per leaf buys its quality
//! with depth, and one ply of depth is worth about twelve points of margin on
//! this game -- so at equal depth a dear evaluator looks like a clear win right
//! up until it is played under a clock, where it loses. That failure is silent
//! at fixed depth, which is why `depth:` is here only to separate "the eval is
//! bad" from "the eval is too slow": run both and read the pair.
//!
//! Reports margin and its standard deviation, not just the win rate. The
//! measured margin sd is about 15.6 points, so at 300 games per seat the
//! standard error is +/-0.9 and a win rate alone cannot resolve anything under
//! a few points. Nodes and achieved depth per side are reported too, because
//! under a clock a dearer evaluation buys fewer nodes and the two sides do not
//! give up the same amount -- that column is what separates "this evaluation is
//! not worth its cost" from "the machine was busy".
//!
//! Dispatch is once per move, through `Contender`, not once per leaf: an enum
//! or a trait object inside `evaluate` would tax the cheap evaluator in exactly
//! the comparison that is trying to measure it.
use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::{HeuristicEvaluator, Minimaxer, ScoreEvaluator, Weights};
use azul_tiles_rs::players::nn_eval::NnEvaluator;
use azul_tiles_rs::players::Player;
use burn::backend::NdArray;
use minimaxer::negamax::SearchOptions;
use minimaxer::Evaluate;
use std::time::Duration;

/// A searcher whose per-move statistics can be read back.
///
/// `Player` alone would do for playing, but it cannot report nodes or depth,
/// and under a time budget those are half the result.
trait Contender {
    fn pick(&mut self, gs: &Gamestate<2, 6>, moves: Vec<Move>) -> Move;
    /// Nodes, summed plies, searches.
    fn stats(&self) -> (u64, u64, u64);
}

impl<E: Evaluate<Gamestate<2, 6>>> Contender for Minimaxer<E> {
    fn pick(&mut self, gs: &Gamestate<2, 6>, moves: Vec<Move>) -> Move {
        self.pick_move(gs, moves)
    }

    fn stats(&self) -> (u64, u64, u64) {
        (self.nodes, self.depth_total, self.searches)
    }
}

/// Parse one or two budgets. One means both sides get it, which is the case
/// worth measuring: the only difference between them is then the evaluation
/// itself.
fn budgets(spec: &str, sort_on_create: bool) -> (SearchOptions, SearchOptions) {
    match spec.split_once(',') {
        Some((a, b)) => (budget(a, sort_on_create), budget(b, sort_on_create)),
        None => {
            let o = budget(spec, sort_on_create);
            (o, o)
        }
    }
}

/// Parse `depth:<n>` or `time:<ms>` into search options.
fn budget(spec: &str, sort_on_create: bool) -> SearchOptions {
    let base = SearchOptions {
        alpha_beta: true,
        sort_on_create,
        sort_on_create_min_depth: 1,
        tt_bits: 20,
        ..Default::default()
    };
    match spec.split_once(':') {
        Some(("depth", n)) => SearchOptions {
            max_depth: Some(n.parse().expect("depth")),
            ..base
        },
        Some(("time", ms)) => SearchOptions {
            // Iterative deepening is what makes a clock usable: the search has
            // an answer to return at every ply, so it can be stopped anywhere.
            iterative: true,
            max_time: Some(Duration::from_millis(ms.parse().expect("time in ms"))),
            // A cap, not a target. The tree bottoms out at the round end
            // anyway, so this only stops a runaway.
            max_depth: Some(16),
            ..base
        },
        _ => panic!("budget must be depth:<n> or time:<ms>, got {spec:?}"),
    }
}

fn contender(spec: &str, opts: SearchOptions) -> Box<dyn Contender> {
    let mut parts = spec.split(':');
    match parts.next().expect("evaluator") {
        "score" => Box::new(Minimaxer::new(opts, spec, ScoreEvaluator)),
        "heuristic" => Box::new(Minimaxer::new(opts, spec, HeuristicEvaluator::default())),
        "handset" => Box::new(Minimaxer::new(
            opts,
            spec,
            HeuristicEvaluator::new(Weights::hand_set()),
        )),
        "nn" => {
            let dir = parts.next().expect("nn:<dir>[:<tag>]");
            let tag = parts.next().unwrap_or("best");
            let device = Default::default();
            let eval = NnEvaluator::<NdArray>::from_checkpoint(
                std::path::Path::new(dir),
                tag,
                &device,
            )
            .expect("checkpoint");
            Box::new(Minimaxer::new(opts, spec, eval))
        }
        other => panic!("unknown evaluator {other:?}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let spec_a = a[1].clone();
    let spec_b = a[2].clone();
    let budget_spec = a[3].clone();
    let games: u64 = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(200);
    let sort: bool = a.get(5).map(|v| v != "0").unwrap_or(true);
    // The seed range the policy holdouts and mm_vs_mm use, so the numbers line
    // up with everything already recorded.
    let first_seed: u64 = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(9_000_000);

    let (opts_a, opts_b) = budgets(&budget_spec, sort);
    println!(
        "{spec_a} vs {spec_b}, budget {budget_spec}, {games} games/seat, seeds {first_seed}.., sort_on_create {sort}"
    );

    let started = std::time::Instant::now();
    for seat in [0usize, 1] {
        let (mut wins, mut draws) = (0u32, 0u32);
        let mut margins: Vec<f64> = Vec::new();
        // Fresh searchers per seat so the node counts are per-seat too.
        let mut pa = contender(&spec_a, opts_a);
        let mut pb = contender(&spec_b, opts_b);
        for seed in first_seed..first_seed + games {
            let mut gs = Gamestate::new_2_player_with_seed(seed, 0);
            loop {
                let moves = gs.get_moves();
                let state = if gs.current_player() as usize == seat {
                    gs.play_move(pa.pick(&gs, moves))
                } else {
                    gs.play_move(pb.pick(&gs, moves))
                };
                if state == State::RoundEnd && gs.end_round() == State::GameEnd {
                    break;
                }
            }
            let s = gs.scores();
            margins.push(s[seat] as f64 - s[1 - seat] as f64);
            match s[seat].cmp(&s[1 - seat]) {
                std::cmp::Ordering::Greater => wins += 1,
                std::cmp::Ordering::Equal => draws += 1,
                std::cmp::Ordering::Less => {}
            }
        }
        let n = games as f64;
        let mean = margins.iter().sum::<f64>() / n;
        let sd = (margins.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n).sqrt();
        // Standard error of the mean margin: the number that says whether a
        // difference is resolvable at this many games.
        let se = sd / n.sqrt();
        println!(
            "seat{seat}: {wins}/{games} ({:.1}%) {draws} drawn | margin {mean:+.2} +/-{se:.2} sd {sd:.2}",
            100.0 * wins as f64 / n
        );
        for (who, c) in [(&spec_a, &pa), (&spec_b, &pb)] {
            let (nodes, plies, searches) = c.stats();
            let s = searches.max(1) as f64;
            println!(
                "    {who}: {:.0} nodes/move, mean depth {:.2}, {searches} moves",
                nodes as f64 / s,
                plies as f64 / s
            );
        }
    }
    println!("done in {:.0}s", started.elapsed().as_secs_f32());
}
