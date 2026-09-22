//! The `xlang` benchmark, compiled to wasm.
//!
//! Mirrors `benches/xlang.rs` exactly -- same LCG, same seeds, same ply
//! counts, same search options, same 32 positions -- so the wasm number can be
//! put next to the native one and the TypeScript one without any argument
//! about what was measured. Timing is taken inside the module with the same
//! clock the search uses, so it covers the searches only, not the position
//! setup, exactly as the native harness does.
//!
//! No wasm-bindgen: the only import is `env.now_ms`, which keeps the module
//! readable as plain WebAssembly and the host glue to three lines.

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use core::sync::atomic::{AtomicU64, Ordering};
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use minimaxer::time::Instant;
use std::time::Duration;

/// Node count from the last benchmark call. `extern "C"` returns one value,
/// and the interesting runs return two.
static LAST_NODES: AtomicU64 = AtomicU64::new(0);
/// Summed search depth from the last `bench_timed`, times 100.
static LAST_DEPTH_X100: AtomicU64 = AtomicU64::new(0);

#[no_mangle]
pub extern "C" fn last_nodes() -> f64 {
    LAST_NODES.load(Ordering::Relaxed) as f64
}

#[no_mangle]
pub extern "C" fn last_mean_depth() -> f64 {
    LAST_DEPTH_X100.load(Ordering::Relaxed) as f64 / 100.0
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

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

const N_POSITIONS: u64 = 32;
fn positions() -> Vec<Gamestate<2, 6>> {
    (0..N_POSITIONS).map(|s| position(s * 7919 + 13, 4 + (s as usize % 5) * 3)).collect()
}

fn base() -> SearchOptions {
    SearchOptions { alpha_beta: true, ..Default::default() }
}

fn run(options: impl Fn() -> SearchOptions) -> (f64, u64, u64) {
    let (mut nodes, mut ms, mut depth_x100) = (0u64, 0.0f64, 0u64);
    for g in positions() {
        let mut n = Negamax::new(Node::new(g), ScoreEvaluator, options());
        let t = Instant::now();
        let r = n.search();
        ms += t.elapsed().as_secs_f64() * 1000.0;
        nodes += u64::from(r.nodes);
        depth_x100 += u64::from(r.depth) * 100;
        core::hint::black_box(r.best);
    }
    (ms, nodes, depth_x100 / N_POSITIONS)
}

fn record(out: (f64, u64, u64)) -> f64 {
    LAST_NODES.store(out.1, Ordering::Relaxed);
    LAST_DEPTH_X100.store(out.2, Ordering::Relaxed);
    out.0
}

/// Full random games. Returns milliseconds; `last_nodes` holds moves played.
#[no_mangle]
pub extern "C" fn bench_playout(n_games: u32) -> f64 {
    let mut moves_played = 0u64;
    let t = Instant::now();
    for seed in 1..=u64::from(n_games) {
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
            let m = moves[rng.next() as usize % moves.len()];
            moves_played += 1;
            if g.play_move(m) == State::GameEnd {
                break;
            }
        }
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    LAST_NODES.store(moves_played, Ordering::Relaxed);
    ms
}

#[no_mangle]
pub extern "C" fn bench_fixed(depth: u32) -> f64 {
    record(run(|| SearchOptions { max_depth: Some(depth as u8), ..base() }))
}

#[no_mangle]
pub extern "C" fn bench_deepen(depth: u32) -> f64 {
    record(run(|| SearchOptions {
        max_depth: Some(depth as u8),
        iterative: true,
        pre_sort: true,
        ..base()
    }))
}

#[no_mangle]
pub extern "C" fn bench_timed(budget_ms: u32) -> f64 {
    record(run(|| SearchOptions {
        max_time: Some(Duration::from_millis(u64::from(budget_ms))),
        iterative: true,
        pre_sort: true,
        ..base()
    }))
}

/// Warm-up, matching the native and TypeScript harnesses.
#[no_mangle]
pub extern "C" fn warmup() {
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
        let mut n = Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions { max_depth: Some(3), iterative: true, pre_sort: true, ..base() },
        );
        core::hint::black_box(n.search().best);
    }
}
