#![recursion_limit = "512"]
//! What fraction of the leaves a search evaluates are round ends?
//! Args: <budget> [<budget> ...] [--games <n>]
//!
//! This is the feasibility number for "expensive evaluation at round end
//! only". `is_terminal` is `is_round_over` here, so a search bottoms out at
//! either the depth limit or the end of the round, and `evaluate` is called at
//! both. If round ends are a small share of those calls, an evaluator that
//! costs 300 us at a round end and 11 ns everywhere else is affordable, and
//! the share is exactly the factor by which it is discounted.
//!
//! The share is not a constant, which is the point of measuring it per budget
//! rather than assuming one. A round runs about ten to twelve plies, so a
//! depth-5 search launched at the start of a round reaches no round end at
//! all, while one launched near the end reaches almost nothing else. What
//! comes out is the average over the positions a real game actually visits.
//!
//! Counting is done in the evaluator rather than from `SearchResult.terminals`
//! because what is being costed is calls to `evaluate`, and those are not the
//! same as terminal nodes: the counter sits exactly where the expense would.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::Player;
use minimaxer::negamax::SearchOptions;
use minimaxer::Evaluate;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Wraps an evaluator and counts where it is called.
///
/// The counters are shared rather than copied, because the search clones the
/// evaluator per node and per thread; a plain field would count one clone's
/// calls and silently discard the rest.
#[derive(Clone)]
struct Counting<E> {
    inner: E,
    total: Arc<AtomicU64>,
    round_over: Arc<AtomicU64>,
}

impl<E: Evaluate<Gamestate<2, 6>>> Evaluate<Gamestate<2, 6>> for Counting<E> {
    fn evaluate(&mut self, g: &Gamestate<2, 6>) -> f32 {
        self.total.fetch_add(1, Ordering::Relaxed);
        if g.is_round_over() {
            self.round_over.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.evaluate(g)
    }
}

fn budget(spec: &str) -> SearchOptions {
    let base = SearchOptions {
        alpha_beta: true,
        sort_on_create: true,
        sort_on_create_min_depth: 1,
        tt_bits: 20,
        ..Default::default()
    };
    match spec.split_once(':') {
        Some(("depth", n)) => SearchOptions { max_depth: Some(n.parse().unwrap()), ..base },
        Some(("time", ms)) => SearchOptions {
            iterative: true,
            max_time: Some(Duration::from_millis(ms.parse().unwrap())),
            max_depth: Some(16),
            ..base
        },
        _ => panic!("budget must be depth:<n> or time:<ms>, got {spec:?}"),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut specs = Vec::new();
    let mut games = 30u64;
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--games" {
            games = argv[i + 1].parse().unwrap();
            i += 2;
        } else {
            specs.push(argv[i].clone());
            i += 1;
        }
    }

    println!(
        "{:<12} {:>14} {:>12} {:>8} {:>10}",
        "budget", "evals", "round-end", "share", "depth"
    );
    for spec in &specs {
        let total = Arc::new(AtomicU64::new(0));
        let round_over = Arc::new(AtomicU64::new(0));
        let ev = Counting {
            inner: ScoreEvaluator,
            total: total.clone(),
            round_over: round_over.clone(),
        };
        let mut p = Minimaxer::new(budget(spec), "count", ev);

        for seed in 9_000_000..9_000_000 + games {
            let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
            loop {
                let moves = gs.get_moves();
                let s = gs.play_move(p.pick_move(&gs, moves));
                if s == State::RoundEnd && gs.end_round() == State::GameEnd {
                    break;
                }
            }
        }
        let t = total.load(Ordering::Relaxed);
        let r = round_over.load(Ordering::Relaxed);
        println!(
            "{spec:<12} {t:>14} {r:>12} {:>7.2}% {:>10.2}",
            100.0 * r as f64 / t.max(1) as f64,
            p.depth_total as f64 / p.searches.max(1) as f64
        );
    }
}
