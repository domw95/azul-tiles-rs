//! How often does a fixed-depth teacher pick the exact move?
//!
//! `gen_labels` labels positions with a depth-capped search. Those labels are
//! the training target, so wherever the capped search disagrees with the exact
//! answer the target is simply wrong, and no amount of capacity or extra rows
//! can push agreement past that. This measures the size of that ceiling
//! directly: for each position it runs the fixed depths and a search that
//! deepens until the round is solved, and reports how often each depth agrees
//! with the exact answer.
//!
//! The comparison is only meaningful on positions the exhaustive search
//! actually solved, so positions that hit the node budget are counted and
//! excluded rather than silently treated as exact.
//!
//! Args: <games> [first_seed]
//! Env: `DEPTHS` (default 1,2,3,4,6), `NODE_BUDGET` (3000000), `TT_BITS` (21),
//! `EVAL` (heuristic).
//!
//! Note on ties: the search reports one best move, and at these depths many
//! moves are genuinely equal. Disagreement therefore over-counts real error --
//! two moves of identical value score as a miss. `value_gap` is the honest
//! figure alongside it: the exact evaluation lost by playing the shallow
//! search's move instead of the exact one, which is zero for a tie.
use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::{HeuristicEvaluator, ScoreEvaluator};
use azul_tiles_rs::players::nn::FactoryOrder;
use std::collections::HashMap;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use minimaxer::{Evaluate, SearchExit};
use rand::Rng;

#[derive(Clone)]
enum Eval {
    Score(ScoreEvaluator),
    Heuristic(HeuristicEvaluator),
}

impl Evaluate<Gamestate<2, 6>> for Eval {
    fn evaluate(&mut self, g: &Gamestate<2, 6>) -> f32 {
        match self {
            Eval::Score(e) => e.evaluate(g),
            Eval::Heuristic(e) => e.evaluate(g),
        }
    }
}

/// How many canonically-distinct moves share the teacher's own top value.
///
/// This, not agreement-with-exact, is what bounds how well a net can imitate a
/// fixed-depth teacher. The net is not trying to be exact -- it is trying to
/// reproduce the teacher's argmax -- so a weak teacher is imitable as long as
/// its choices are a function of the encoded state. Where `k` moves tie at the
/// teacher's best value, the teacher picks between them on search-internal
/// move ordering, which the encoder cannot see; a net that ranks perfectly
/// still scores `1/k`. Turning `random_best` off makes that tie-break
/// deterministic but no less arbitrary.
///
/// Counted over *canonical* action indices, because the policy is indexed that
/// way: two interchangeable factories collapse to one label, so "blue from
/// factory 1" and "blue from identical factory 3" are not a tie the net can
/// lose. Counting raw moves would overstate the ceiling badly.
///
/// Root values at depth `d` come from searching each child at `d - 1` and
/// negating, which is exact and cheap next to the exhaustive search: the whole
/// root vector costs about as much as one depth-`d` search.
fn tie_count<E: Evaluate<Gamestate<2, 6>> + Clone>(
    gs: &Gamestate<2, 6>,
    moves: &[Move],
    depth: u8,
    eval: &E,
    tt: u8,
) -> usize {
    debug_assert!(depth >= 2, "root values need a d-1 search on each child");
    let order = FactoryOrder::canonical(gs);
    let mut best: HashMap<usize, f32> = HashMap::new();
    for &m in moves {
        let mut child = gs.clone();
        child.play_move(m);
        let v = if child.is_round_over() {
            // Terminal for the search: its value is the static evaluation,
            // which a d-1 search would return anyway.
            let mut e = eval.clone();
            -e.evaluate(&child)
        } else {
            let mut n = Negamax::new(
                Node::new(child),
                eval.clone(),
                SearchOptions { max_depth: Some(depth - 1), ..base_opts(tt) },
            );
            -n.search().value
        };
        let slot = best.entry(order.canonical_index(&m)).or_insert(f32::NEG_INFINITY);
        if v > *slot {
            *slot = v;
        }
    }
    let top = best.values().copied().fold(f32::NEG_INFINITY, f32::max);
    // Values are sums of small integers times weights, so exact ties really
    // are exact; the epsilon only guards float accumulation order.
    best.values().filter(|v| (**v - top).abs() < 1e-6).count().max(1)
}

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn base_opts(tt: u8) -> SearchOptions {
    SearchOptions {
        alpha_beta: true,
        iterative: true,
        pre_sort: true,
        sort_on_create: true,
        sort_on_create_min_depth: 1,
        tt_bits: tt,
        retain_depth: 3,
        // Deliberately off. A tie broken at random would show up as
        // disagreement that is really a coin flip, inflating the very number
        // this is trying to measure.
        random_best: false,
        ..Default::default()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: u64 = a.get(1).map(|v| v.parse().unwrap()).unwrap_or(50);
    let first: u64 = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(500_000);
    let depths: Vec<u8> = std::env::var("DEPTHS")
        .unwrap_or_else(|_| "1,2,3,4,6".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    // Raised from 3M: at 3M, 11.6% of positions were excluded for hitting the
    // budget, and those are the large-tree ones. They are excluded from the
    // learnable ceiling as well as from agreement, and large trees plausibly
    // carry more near-equal moves, so the ceiling is biased upward by exactly
    // the positions it drops.
    let budget: u32 = env("NODE_BUDGET", 20_000_000u32);
    let tt: u8 = env("TT_BITS", 21u8);
    // Which depths get a full root value vector. Each one costs a d-1 search
    // per legal move, so it is the expensive column: depth 2 is one depth-1
    // search per move, depth 6 is a depth-5 search per move. Defaults to 2
    // alone, which is the teacher the published agreement figure was measured
    // against; the rest of the table is agreement and value lost, which are
    // cheap and are the part this measurement uniquely answers.
    let tie_depths: Vec<u8> = std::env::var("TIE_DEPTHS")
        .unwrap_or_else(|_| "2".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let evaluator = match std::env::var("EVAL").unwrap_or_else(|_| "heuristic".into()).as_str() {
        "score" => Eval::Score(ScoreEvaluator),
        _ => Eval::Heuristic(HeuristicEvaluator::default()),
    };

    println!("depths {depths:?}, {games} games from seed {first}, budget {budget} nodes");
    let mut agree = vec![0u64; depths.len()];
    let mut gap = vec![0f64; depths.len()];
    // Sum of 1/k: its mean is the best agreement any net could reach.
    let mut learnable = vec![0f64; depths.len()];
    let mut tied_positions = vec![0u64; depths.len()];
    // The same figures split by how hard the position was to solve, so the
    // bias from dropping unsolvable positions can be read off a trend rather
    // than assumed. If the ceiling is flat across buckets, excluding the
    // hardest positions costs little; if it falls, the true ceiling is below
    // the headline and by roughly the slope.
    const BUCKETS: [(&str, u32); 4] =
        [("<100k", 100_000), ("<1M", 1_000_000), ("<10M", 10_000_000), (">=10M", u32::MAX)];
    let mut b_n = [0u64; 4];
    let mut b_learnable = [0f64; 4];
    let mut b_agree = [0u64; 4];
    let mut b_gap = [0f64; 4];
    let (mut exact, mut skipped) = (0u64, 0u64);
    let mut rng = rand::thread_rng();
    let started = std::time::Instant::now();

    for seed in first..first + games {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        loop {
            if gs.is_round_over() {
                if gs.end_round() == State::GameEnd {
                    break;
                }
                continue;
            }
            let moves = gs.get_moves();

            // The exact answer: deepen until the round is solved, or give up
            // on this position entirely.
            let mut truth = Negamax::new(
                Node::new(gs.clone()),
                evaluator.clone(),
                SearchOptions { max_depth: None, ..base_opts(tt) },
            );
            let (mut d, mut best) = (1u8, None);
            let r = loop {
                truth.options.max_depth = Some(d);
                let r = truth.search();
                if r.exit != SearchExit::Depth {
                    break r;
                }
                if r.nodes >= budget || d == u8::MAX {
                    break r;
                }
                best = Some(r);
                d = r.depth.max(d).saturating_add(1);
            };
            let _ = best;

            if r.exit == SearchExit::Exhaustive {
                exact += 1;
                let bucket = BUCKETS.iter().position(|&(_, hi)| r.nodes < hi).unwrap_or(3);
                b_n[bucket] += 1;
                for (i, &depth) in depths.iter().enumerate() {
                    let mut n = Negamax::new(
                        Node::new(gs.clone()),
                        evaluator.clone(),
                        SearchOptions { max_depth: Some(depth), ..base_opts(tt) },
                    );
                    let s = n.search();
                    if depth >= 2 && tie_depths.contains(&depth) {
                        let k = tie_count(&gs, &moves, depth, &evaluator, tt);
                        learnable[i] += 1.0 / k as f64;
                        if k > 1 {
                            tied_positions[i] += 1;
                        }
                    }
                    if depth == 2 && tie_depths.contains(&depth) {
                        b_learnable[bucket] += 1.0 / tie_count(&gs, &moves, depth, &evaluator, tt) as f64;
                    }
                    if s.best == r.best {
                        agree[i] += 1;
                        if depth == 2 {
                            b_agree[bucket] += 1;
                        }
                    } else {
                        // What the shallow choice actually costs, measured by
                        // the exact search: play its move, then solve the
                        // resulting position exactly. A tie costs zero.
                        let mut after = gs.clone();
                        after.play_move(s.best);
                        let mut v = Negamax::new(
                            Node::new(after),
                            evaluator.clone(),
                            SearchOptions { max_depth: Some(d), ..base_opts(tt) },
                        );
                        let got = v.search().value;
                        // Both values are from the mover's point of view, and
                        // the position has changed hands, so negate.
                        let lost = (r.value - -got).abs() as f64;
                        gap[i] += lost;
                        if depth == 2 {
                            b_gap[bucket] += lost;
                        }
                    }
                }
            } else {
                skipped += 1;
            }

            let played = if rng.gen_bool(0.25) {
                moves[rng.gen_range(0..moves.len())]
            } else {
                r.best
            };
            gs.play_move(played);
        }

        if exact > 0 && (seed - first + 1) % 5 == 0 {
            println!(
                "  {} games, {exact} exact positions ({skipped} skipped), {:.0}s",
                seed - first + 1,
                started.elapsed().as_secs_f32()
            );
            for (i, &depth) in depths.iter().enumerate() {
                println!(
                    "    depth {depth}: agrees with exact {:.1}%  value lost {:.3}{}",
                    100.0 * agree[i] as f64 / exact as f64,
                    gap[i] / exact as f64,
                    if tie_depths.contains(&depth) {
                        format!(
                            "  learnable ceiling {:.1}% ({:.0}% tied)",
                            100.0 * learnable[i] / exact as f64,
                            100.0 * tied_positions[i] as f64 / exact as f64,
                        )
                    } else {
                        String::new()
                    },
                );
            }
        }
    }

    println!("\nfinal: {exact} exact positions, {skipped} skipped (budget)");
    println!("\nby solve cost (depth 2), to expose the bias from dropping hard positions:");
    println!("  {:>7} {:>8} {:>12} {:>11} {:>11}", "nodes", "n", "ceiling", "agreement", "value lost");
    for (b, (name, _)) in BUCKETS.iter().enumerate() {
        if b_n[b] == 0 {
            continue;
        }
        let n = b_n[b] as f64;
        println!(
            "  {name:>7} {:>8} {:>11.1}% {:>10.1}% {:>11.3}",
            b_n[b],
            100.0 * b_learnable[b] / n,
            100.0 * b_agree[b] as f64 / n,
            b_gap[b] / n,
        );
    }
    for (i, &depth) in depths.iter().enumerate() {
        println!(
            "  depth {depth}: {:.2}% agreement with exact, mean value lost {:.4}{}",
            100.0 * agree[i] as f64 / exact.max(1) as f64,
            gap[i] / exact.max(1) as f64,
            if tie_depths.contains(&depth) {
                format!(
                    ", learnable ceiling {:.2}% ({:.1}% of positions have a tie)",
                    100.0 * learnable[i] / exact.max(1) as f64,
                    100.0 * tied_positions[i] as f64 / exact.max(1) as f64,
                )
            } else {
                String::new()
            },
        );
    }
}

