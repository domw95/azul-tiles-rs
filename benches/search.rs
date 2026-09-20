//! Search benchmarks for the Azul minimax player.
//!
//! Reports node counts next to wall time. Node counts are deterministic, so a
//! pruning or ordering change shows up exactly, while wall time on a shared
//! machine does not. The checksum over returned values proves a change left
//! the search's answers alone.
//!
//! `cargo bench --bench search` runs a default sweep. For a single
//! measurement:
//!
//!     cargo bench --bench search -- <mode> <depth> <reps>
//!
//! modes: fixed | iter | iter_sorted | iter_path | iter_rand | iter_w5
//!        | timed | div | size

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use minimaxer::negamax::{NegamaxAim, SearchOptions};
use minimaxer::node::Node;
use std::time::{Duration, Instant};



/// Deterministic LCG so position selection never depends on rand's version.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Build a reproducible mid-game position by playing `plies` seeded moves.
fn position(seed: u64, plies: usize) -> Gamestate<2, 6> {
    let mut g = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
    let mut rng = Lcg(seed ^ 0x9E3779B97F4A7C15);
    for _ in 0..plies {
        let moves = g.get_moves();
        if moves.is_empty() {
            if g.end_round() == State::GameEnd {
                break;
            }
            continue;
        }
        let m: Move = moves[rng.next() as usize % moves.len()];
        if g.play_move(m) == State::GameEnd {
            break;
        }
    }
    // Never hand back a position with no legal moves.
    if g.get_moves().is_empty() {
        g.end_round();
    }
    g
}

fn positions() -> Vec<Gamestate<2, 6>> {
    (0..8u64).map(|s| position(s * 7919 + 13, 4 + (s as usize % 5) * 3)).collect()
}

/// Fixed-depth alpha-beta, no iterative deepening. Node count is identical
/// across variants that only change per-node cost, so time is the signal.
fn fixed_depth(depth: u8) -> (u64, Duration, u64) {
    let mut nodes = 0u64;
    let mut elapsed = Duration::ZERO;
    let mut sum = 0u64;
    for g in positions() {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions {
                max_depth: Some(depth),
                alpha_beta: true,
                tt_bits: std::env::var("TT_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
                sort_on_create: std::env::var("SORT_ON_CREATE").is_ok(),
                sort_on_create_min_depth: std::env::var("SOC_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                ..Default::default()
            },
        );
        let t = Instant::now();
        let r = n.search();
        elapsed += t.elapsed();
        nodes += r.nodes as u64;
        sum = sum.wrapping_mul(31).wrapping_add(r.value.to_bits() as u64);
        std::hint::black_box(r.best);
    }
    (nodes, elapsed, sum)
}

/// Iterative deepening to a fixed depth cap. Node count *drops* when move
/// ordering improves, which is the noise-free signal for the pre_sort work.
fn iterative(depth: u8, pre_sort: bool) -> (u64, Duration, u64) {
    let mut nodes = 0u64;
    let mut elapsed = Duration::ZERO;
    let mut sum = 0u64;
    for g in positions() {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions {
                max_depth: Some(depth),
                alpha_beta: true,
                iterative: true,
                pre_sort,
                tt_bits: std::env::var("TT_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
                sort_on_create: std::env::var("SORT_ON_CREATE").is_ok(),
                sort_on_create_min_depth: std::env::var("SOC_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                ..Default::default()
            },
        );
        let t = Instant::now();
        let r = n.search();
        elapsed += t.elapsed();
        nodes += r.nodes as u64;
        sum = sum.wrapping_mul(31).wrapping_add(r.value.to_bits() as u64);
        std::hint::black_box(r.best);
    }
    (nodes, elapsed, sum)
}

/// Iterative deepening with the TS-parity options turned on.
fn iterative_opts(depth: u8, path_len: bool, rand_best: bool, rand_w: f32) -> (u64, Duration, u64) {
    let mut nodes = 0u64;
    let mut elapsed = Duration::ZERO;
    let mut sum = 0u64;
    for g in positions() {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions {
                max_depth: Some(depth),
                alpha_beta: true,
                iterative: true,
                pre_sort: true,
                prune_by_path_length: path_len,
                sort_on_create: std::env::var("SORT_ON_CREATE").is_ok(),
                sort_on_create_min_depth: std::env::var("SOC_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                random_best: rand_best,
                random_weight: rand_w,
                keep_equal_siblings: path_len,
                ..Default::default()
            },
        );
        let t = Instant::now();
        let r = n.search();
        elapsed += t.elapsed();
        nodes += r.nodes as u64;
        sum = sum.wrapping_mul(31).wrapping_add(r.value.to_bits() as u64);
        std::hint::black_box(r.best);
    }
    (nodes, elapsed, sum)
}

/// Fixed depth, but with a time limit so large it never fires. Isolates the
/// cost of the per-node clock check: node counts stay identical.
fn fixed_timed(depth: u8, pre_sort: bool) -> (u64, Duration, u64) {
    let mut nodes = 0u64;
    let mut elapsed = Duration::ZERO;
    let mut sum = 0u64;
    for g in positions() {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions {
                max_depth: Some(depth),
                alpha_beta: true,
                iterative: true,
                pre_sort,
                max_time: Some(Duration::from_secs(3600)),
                ..Default::default()
            },
        );
        let t = Instant::now();
        let r = n.search();
        elapsed += t.elapsed();
        nodes += r.nodes as u64;
        sum = sum.wrapping_mul(31).wrapping_add(r.value.to_bits() as u64);
        std::hint::black_box(r.best);
    }
    (nodes, elapsed, sum)
}

/// How many distinct moves does the engine actually pick, over repeated
/// searches of the same positions? This is what a training opponent needs.
fn diversity(depth: u8, path_len: bool, rand_best: bool, rand_w: f32, reps: usize) {
    let mut total_distinct = 0usize;
    let mut total_ms = 0.0;
    let mut nodes = 0u64;
    for g in positions() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..reps {
            let mut n = minimaxer::negamax::Negamax::new(
                Node::new(g.clone()),
                ScoreEvaluator,
                SearchOptions {
                    max_depth: Some(depth),
                    alpha_beta: true,
                    iterative: true,
                    pre_sort: true,
                    prune_by_path_length: path_len,
                    random_best: rand_best,
                    random_weight: rand_w,
                    keep_equal_siblings: path_len,
                    ..Default::default()
                },
            );
            let t = Instant::now();
            let r = n.search();
            total_ms += t.elapsed().as_secs_f64() * 1000.0;
            nodes += r.nodes as u64;
            seen.insert(format!("{:?}", r.best));
        }
        total_distinct += seen.len();
    }
    let n_pos = positions().len();
    println!(
        "distinct_moves/position={:.2} (of {reps} tries)  total_ms={total_ms:.0}  nodes={nodes}",
        total_distinct as f64 / n_pos as f64
    );
}

/// What `cargo bench --bench search` runs with no arguments: the configurations
/// that matter, at a depth that finishes quickly, with node counts so a
/// regression is visible without trusting the clock.
fn default_sweep() {
    println!("Azul search sweep over {} seeded positions\n", positions().len());
    println!(
        "{:<24} {:>6} {:>12} {:>10}  {}",
        "config", "depth", "nodes", "ms", "checksum"
    );
    for depth in [4u8, 5, 6] {
        for (name, f) in [
            ("alpha_beta", 0u8),
            ("alpha_beta+presort", 1),
            ("presort+random_best", 2),
        ] {
            let (nodes, dur, sum) = match f {
                0 => fixed_depth(depth),
                1 => iterative(depth, true),
                _ => iterative_opts(depth, false, true, 0.0),
            };
            println!(
                "{:<24} {:>6} {:>12} {:>10.1}  {:016x}",
                name,
                depth,
                nodes,
                dur.as_secs_f64() * 1000.0,
                sum
            );
        }
    }
}

fn main() {
    // cargo passes flags like `--bench`; drop them so they are not read as a mode.
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with('-')).collect();
    if args.len() <= 1 {
        default_sweep();
        return;
    }
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("fixed");
    let depth: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    if mode == "div" {
        let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
        print!("{:<26}", "deterministic:");
        diversity(depth, false, false, 0.0, reps);
        print!("{:<26}", "random_best:");
        diversity(depth, false, true, 0.0, reps);
        print!("{:<26}", "random_weight=5:");
        diversity(depth, false, false, 5.0, reps);
        print!("{:<26}", "random_weight=1.5:");
        diversity(depth, false, false, 1.5, reps);
        return;
    }
    if mode == "engine" {
        engine_micro();
        return;
    }
    if mode == "trans" {
        transposition_rate(depth, args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3));
        return;
    }
    if mode == "size" {
        println!("gamestate_bytes={}", std::mem::size_of::<Gamestate<2, 6>>());
        println!("node_bytes={}", std::mem::size_of::<Node<Gamestate<2, 6>, Move>>());
        return;
    }

    // Warm caches/branch predictors; discarded.
    let _ = fixed_depth(2);

    let mut best = Duration::MAX;
    let mut nodes = 0u64;
    let mut sum = 0u64;
    for _ in 0..reps {
        let (n, d, c) = match mode {
            "iter" => iterative(depth, false),
            "iter_sorted" => iterative(depth, true),
            "iter_path" => iterative_opts(depth, true, false, 0.0),
            "iter_rand" => iterative_opts(depth, false, true, 0.0),
            "iter_w5" => iterative_opts(depth, false, false, 5.0),
            "timed" => fixed_timed(depth, true),
            "timed_nosort" => fixed_timed(depth, false),
            _ => fixed_depth(depth),
        };
        nodes = n;
        sum = c;
        // Minimum across reps: the run least disturbed by neighbours.
        if d < best {
            best = d;
        }
    }
    println!("mode={mode} depth={depth} nodes={nodes} checksum={sum:016x} min_ms={:.3}", best.as_secs_f64() * 1000.0);
}

/// How often does the Azul search reach the same position twice?
/// This decides whether a transposition table is worth anything here.
/// Keyed on the derived Debug output: slow, but unambiguous for a one-off.
fn transposition_rate(depth: u8, n_pos: usize) {
    fn walk(
        g: &Gamestate<2, 6>,
        depth: u8,
        seen: &mut std::collections::HashSet<String>,
        total: &mut u64,
    ) {
        *total += 1;
        seen.insert(format!("{:?}", g));
        if depth == 0 || g.is_round_over() {
            return;
        }
        for m in g.get_moves() {
            let mut c = g.clone();
            c.play_move(m);
            walk(&c, depth - 1, seen, total);
        }
    }

    println!("{:>5} {:>12} {:>12} {:>10}", "depth", "nodes", "distinct", "repeats");
    for g in positions().into_iter().take(n_pos) {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0u64;
        walk(&g, depth, &mut seen, &mut total);
        let distinct = seen.len() as u64;
        println!(
            "{:>5} {:>12} {:>12} {:>9.1}%",
            depth,
            total,
            distinct,
            100.0 * (total - distinct) as f64 / total as f64
        );
    }
}

/// Times the game engine primitives in isolation. These sit under both the
/// minimax search and PPO rollouts, so a win here helps both.
fn engine_micro() {
    let states: Vec<Gamestate<2, 6>> = positions();
    let reps = 20_000;

    // clone
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..reps {
        for g in &states {
            let c = std::hint::black_box(g.clone());
            acc += c.current_player() as usize;
        }
    }
    let n = (reps * states.len()) as f64;
    println!("clone                {:>8.1} ns", t.elapsed().as_nanos() as f64 / n);

    // get_moves (allocates a Vec every call)
    let t = Instant::now();
    let mut total_moves = 0usize;
    let mut max_moves = 0usize;
    for _ in 0..reps {
        for g in &states {
            let m = std::hint::black_box(g.get_moves());
            total_moves += m.len();
            max_moves = max_moves.max(m.len());
        }
    }
    println!("get_moves            {:>8.1} ns   (avg {:.1} moves, max {})",
        t.elapsed().as_nanos() as f64 / n, total_moves as f64 / n, max_moves);

    // play_move on a fresh clone, which is what the search actually does
    let movesets: Vec<(Gamestate<2, 6>, Move)> =
        states.iter().filter_map(|g| g.get_moves().first().map(|m| (g.clone(), *m))).collect();
    let t = Instant::now();
    for _ in 0..reps {
        for (g, m) in &movesets {
            let mut c = g.clone();
            c.play_move(*m);
            std::hint::black_box(c.current_player());
        }
    }
    let n2 = (reps * movesets.len()) as f64;
    println!("clone + play_move    {:>8.1} ns", t.elapsed().as_nanos() as f64 / n2);

    // the terminal check and the evaluator
    let t = Instant::now();
    for _ in 0..reps {
        for g in &states {
            std::hint::black_box(g.is_round_over());
        }
    }
    println!("is_round_over        {:>8.1} ns", t.elapsed().as_nanos() as f64 / n);

    let t = Instant::now();
    for _ in 0..reps {
        for g in &states {
            std::hint::black_box(g.differential_predicted_score());
        }
    }
    println!("evaluate (score)     {:>8.1} ns", t.elapsed().as_nanos() as f64 / n);

    // The richer evaluator, to see what better evaluation actually costs
    // relative to move generation.
    use azul_tiles_rs::players::minimax::HeuristicEvaluator;
    use minimaxer::Evaluate;
    let mut heur = HeuristicEvaluator::default();
    let t = Instant::now();
    for _ in 0..reps {
        for g in &states {
            std::hint::black_box(heur.evaluate(g));
        }
    }
    println!("evaluate (heuristic) {:>8.1} ns", t.elapsed().as_nanos() as f64 / n);

    let t = Instant::now();
    for _ in 0..reps {
        for g in &states {
            std::hint::black_box(g.position_key());
        }
    }
    println!("position_key         {:>8.1} ns", t.elapsed().as_nanos() as f64 / n);
    println!("\n(acc {acc})");
}
