//! A pondering Azul engine, for a Web Worker.
//!
//! The point of this shape, as against the one-search-per-call shape the
//! benchmark uses: the `Negamax` lives across calls, so the tree built while
//! the opponent is thinking is still there when it is our turn. The host
//! drives it in slices -- `ponder(budget_ms)` returns after roughly that long
//! -- which is what lets a worker stay responsive to its message queue without
//! `SharedArrayBuffer` or cross-origin isolation.
//!
//! The 1 second guarantee falls out of iterative deepening rather than out of
//! interrupting anything: there is always a completed pass to answer from, so
//! `best_index` is O(1) and always valid. Cancellation only decides when to
//! stop *spending*, never when an answer becomes available.
//!
//! Moves cross the boundary as indices into `get_moves()`, which is
//! deterministic for a position, so the host never has to know the move
//! representation.

use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use minimaxer::SearchExit;
use std::cell::RefCell;
use std::time::Duration;

/// Bounds the tree to the top plies and lets the table carry the ordering
/// below. wasm linear memory never shrinks, so an unbounded ponder would make
/// its high-water mark permanent for the life of the tab.
const RETAIN_DEPTH: u8 = 4;
/// 2^20 entries.
const TT_BITS: u8 = 20;

struct Engine {
    game: Gamestate<2, 6>,
    search: Negamax<Gamestate<2, 6>, Move, ScoreEvaluator>,
    /// Moves of the current position, in the order the indices refer to.
    moves: Vec<Move>,
    /// Set once a pass has searched the round out to its end.
    solved: bool,
    depth: u8,
    value: f32,
    nodes: u64,
    best: Option<usize>,
}

fn options() -> SearchOptions {
    SearchOptions {
        alpha_beta: true,
        iterative: true,
        pre_sort: true,
        tt_bits: TT_BITS,
        retain_depth: RETAIN_DEPTH,
        // Resume one ply past whatever the retained tree already reached
        // instead of replaying the early passes on every slice.
        initial_depth: u8::MAX,
        ..Default::default()
    }
}

impl Engine {
    fn new(seed: u64) -> Self {
        let game = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        Self::from_game(game)
    }

    fn from_game(game: Gamestate<2, 6>) -> Self {
        let moves = game.get_moves();
        let search = Negamax::new(Node::new(game.clone()), ScoreEvaluator, options());
        Engine { game, search, moves, solved: false, depth: 0, value: 0.0, nodes: 0, best: None }
    }

    /// Search for about `budget_ms`, then hand control back so the worker can
    /// read its messages. Returns whether the round is now solved.
    fn ponder(&mut self, budget_ms: u32) -> bool {
        if self.solved || self.moves.is_empty() {
            return self.solved;
        }
        self.search.options.max_time = Some(Duration::from_millis(u64::from(budget_ms)));
        let r = self.search.search();
        self.nodes += u64::from(r.nodes);
        self.depth = r.depth;
        self.value = r.value;
        self.best = self.moves.iter().position(|m| *m == r.best);
        if r.exit == SearchExit::Exhaustive {
            self.solved = true;
        }
        self.solved
    }

    /// Play a move and re-root onto it, keeping the subtree already searched.
    fn play(&mut self, index: usize) -> State {
        let m = self.moves[index];
        let state = self.game.play_move(m);
        // A re-rooted tree has been searched to one ply shallower than it was,
        // and a solved round stays solved only if the subtree kept covers it.
        if !self.search.play_move(&m) {
            // The move was never expanded -- start again from the new position.
            self.search = Negamax::new(Node::new(self.game.clone()), ScoreEvaluator, options());
            self.solved = false;
        }
        self.depth = self.search.root().searched_depth();
        self.moves = self.game.get_moves();
        // The re-rooted tree already carries the move the previous search
        // settled on, so there is an answer before the next slice even starts.
        self.best = self
            .search
            .root()
            .best_move()
            .and_then(|m| self.moves.iter().position(|c| c == m));
        if self.moves.is_empty() {
            self.solved = false;
        }
        state
    }

    /// End the round and start a fresh search on the new one. This is the
    /// "spawn the search at round start" point.
    fn end_round(&mut self) -> State {
        let state = self.game.end_round();
        let game = self.game.clone();
        *self = Engine::from_game(game);
        state
    }
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

fn with<T>(f: impl FnOnce(&mut Engine) -> T, default: T) -> T {
    ENGINE.with(|e| match e.borrow_mut().as_mut() {
        Some(engine) => f(engine),
        None => default,
    })
}

fn state_code(state: State) -> u32 {
    match state {
        State::RoundActive => 0,
        State::RoundEnd => 1,
        State::GameEnd => 2,
    }
}

/// Start a new game. `seed` is an f64 because that is what a JS number is.
#[no_mangle]
pub extern "C" fn new_game(seed: f64) {
    ENGINE.with(|e| *e.borrow_mut() = Some(Engine::new(seed as u64)));
}

/// Search for about `budget_ms`. Returns 1 once the round is solved.
#[no_mangle]
pub extern "C" fn ponder(budget_ms: u32) -> u32 {
    with(|e| u32::from(e.ponder(budget_ms)), 0)
}

/// Search until there is a move to give, for up to `budget_ms`.
///
/// Only reachable on a cold root -- the opening position, or a re-root onto a
/// move the previous search never expanded. Everywhere else the retained tree
/// already holds an answer and this returns immediately.
#[no_mangle]
pub extern "C" fn ensure_move(budget_ms: u32) -> i32 {
    with(
        |e| {
            if e.best.is_none() && !e.moves.is_empty() {
                e.ponder(budget_ms);
            }
            e.best.map_or(-1, |i| i as i32)
        },
        -1,
    )
}

/// Index of the best move found so far, or -1 if no pass has finished.
#[no_mangle]
pub extern "C" fn best_index() -> i32 {
    with(|e| e.best.map_or(-1, |i| i as i32), -1)
}

#[no_mangle]
pub extern "C" fn best_value() -> f32 {
    with(|e| e.value, 0.0)
}

#[no_mangle]
pub extern "C" fn depth() -> u32 {
    with(|e| u32::from(e.depth), 0)
}

#[no_mangle]
pub extern "C" fn solved() -> u32 {
    with(|e| u32::from(e.solved), 0)
}

#[no_mangle]
pub extern "C" fn nodes() -> f64 {
    with(|e| e.nodes as f64, 0.0)
}

#[no_mangle]
pub extern "C" fn tree_size() -> f64 {
    with(|e| e.search.tree_size() as f64, 0.0)
}

#[no_mangle]
pub extern "C" fn move_count() -> u32 {
    with(|e| e.moves.len() as u32, 0)
}

#[no_mangle]
pub extern "C" fn current_player() -> u32 {
    with(|e| u32::from(e.game.current_player()), 0)
}

#[no_mangle]
pub extern "C" fn round() -> u32 {
    with(|e| u32::from(e.game.round()), 0)
}

#[no_mangle]
pub extern "C" fn score(player: u32) -> u32 {
    with(|e| u32::from(e.game.scores()[player as usize]), 0)
}

#[no_mangle]
pub extern "C" fn play_index(index: u32) -> u32 {
    with(|e| state_code(e.play(index as usize)), 0)
}

#[no_mangle]
pub extern "C" fn end_round() -> u32 {
    with(|e| state_code(e.end_round()), 0)
}
