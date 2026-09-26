//! Cross-language benchmark: the Rust half.
//!
//! Mirrors `benchmarks/xlang.mjs` in the azul-tiles TypeScript repo move for
//! move, so the two sets of numbers can be put side by side. Identical
//! positions are not possible across the two implementations -- the shufflers
//! differ -- so the harness draws from the same distribution instead: the same
//! seeds, the same number of random plies, the same move-selection rule, and
//! enough positions that the mean is stable.
//!
//!     cargo bench --bench xlang -- [playout|fixed|deepen|timed|best|all]

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use std::time::{Duration, Instant};

/// The same LCG as the TypeScript side, so both draw the same ply counts and
/// indices.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// A reproducible mid-game position, reached by playing `plies` random moves.
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
    if g.get_moves().is_empty() {
        g.end_round();
    }
    g
}

fn positions() -> Vec<Gamestate<2, 6>> {
    let n: u64 = std::env::var("N_POS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    (0..n).map(|s| position(s * 7919 + 13, 4 + (s as usize % 5) * 3)).collect()
}

/// Full random games: move generation, move application and round handling.
fn playout(n_games: u64) {
    let mut moves_played = 0u64;
    let mut offered = 0u64;
    let t = Instant::now();
    for seed in 1..=n_games {
        let mut g = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        let mut rng = Lcg(seed ^ 0x9E3779B97F4A7C15);
        loop {
            let moves = g.get_moves();
            if moves.is_empty() {
                if g.end_round() == State::GameEnd {
                    break;
                }
                continue;
            }
            offered += moves.len() as u64;
            let m = moves[rng.next() as usize % moves.len()];
            moves_played += 1;
            if g.play_move(m) == State::GameEnd {
                break;
            }
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "playout    games={n_games} moves={moves_played} ms={ms:.1} moves_per_sec={:.0} branching={:.2}",
        moves_played as f64 / ms * 1000.0,
        offered as f64 / moves_played as f64
    );
}

/// One search from `g`, timed. Returns (nodes, ms, value, depth).
fn search(g: Gamestate<2, 6>, options: SearchOptions) -> (u64, f64, f32, u8) {
    let mut n = Negamax::new(Node::new(g), ScoreEvaluator, options);
    let t = Instant::now();
    let r = n.search();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(r.best);
    (r.nodes as u64, ms, r.value, r.depth)
}

fn sweep(label: &str, depths: &[u8], opts: impl Fn(u8) -> SearchOptions) {
    for &d in depths {
        let (mut nodes, mut ms, mut checksum) = (0u64, 0.0f64, 0i64);
        for g in positions() {
            let (n, t, v, _) = search(g, opts(d));
            nodes += n;
            ms += t;
            checksum = (checksum.wrapping_mul(31) + (v * 1000.0).round() as i64) % 1_000_000_000_000;
        }
        println!(
            "{label:<10} depth={d} nodes={nodes:>10} ms={ms:>9.1} knodes_per_sec={:>6.0} checksum={checksum}",
            nodes as f64 / ms
        );
    }
}

/// Fixed wall-clock budget per position: how deep does each implementation get?
fn timed(budget_ms: u64, options: impl Fn() -> SearchOptions) {
    let (mut nodes, mut ms, mut depth_sum) = (0u64, 0.0f64, 0u32);
    let ps = positions();
    let n_pos = ps.len();
    for g in ps {
        let mut o = options();
        o.max_time = Some(Duration::from_millis(budget_ms));
        let (n, t, _, d) = search(g, o);
        nodes += n;
        ms += t;
        depth_sum += u32::from(d);
    }
    println!(
        "timed      budget={budget_ms}ms mean_depth={:.2} nodes={nodes} ms={ms:.1} knodes_per_sec={:.0}",
        f64::from(depth_sum) / n_pos as f64,
        nodes as f64 / ms
    );
}

fn base() -> SearchOptions {
    SearchOptions { alpha_beta: true, ..Default::default() }
}

/// Everything the Rust engine has that the TypeScript one does not.
fn best() -> SearchOptions {
    SearchOptions {
        alpha_beta: true,
        iterative: true,
        pre_sort: true,
        tt_bits: 20,
        sort_on_create: true,
        sort_on_create_min_depth: 2,
        ..Default::default()
    }
}

/// Warm the caches and the allocator before anything is timed. The TypeScript
/// half needs this for the JIT; running the same thing here keeps the two
/// harnesses line for line.
fn warmup() {
    for seed in 9001..9021u64 {
        let mut g = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        let mut rng = Lcg(seed);
        loop {
            let moves = g.get_moves();
            if moves.is_empty() {
                if g.end_round() == State::GameEnd {
                    break;
                }
                continue;
            }
            let m = moves[rng.next() as usize % moves.len()];
            if g.play_move(m) == State::GameEnd {
                break;
            }
        }
    }
    for s in 0..4u64 {
        let g = position(s * 104729 + 7, 6);
        search(g.clone(), SearchOptions { max_depth: Some(3), ..base() });
        search(
            g,
            SearchOptions { max_depth: Some(3), iterative: true, pre_sort: true, ..base() },
        );
    }
}

fn main() {
    warmup();
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with('-')).collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let all = mode == "all";

    if all || mode == "playout" {
        playout(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200));
    }
    if all || mode == "fixed" {
        sweep("fixed", &[3, 4, 5], |d| SearchOptions { max_depth: Some(d), ..base() });
    }
    if all || mode == "deepen" {
        sweep("deepen", &[3, 4, 5], |d| SearchOptions {
            max_depth: Some(d),
            iterative: true,
            pre_sort: true,
            ..base()
        });
    }
    if all || mode == "timed" {
        timed(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100), || SearchOptions {
            iterative: true,
            pre_sort: true,
            ..base()
        });
    }
    if all || mode == "best" {
        sweep("best", &[3, 4, 5, 6], |d| SearchOptions { max_depth: Some(d), ..best() });
        timed(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100), best);
    }
}
