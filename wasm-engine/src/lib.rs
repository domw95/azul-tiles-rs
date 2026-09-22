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

// ---------------------------------------------------------------------------
// One-shot player, for a host that owns the game itself.
//
// The pondering engine above owns its gamestate, which is what makes it a
// clean prototype and what makes it awkward to drop into a browser: there the
// TypeScript `GameState` is authoritative and this side has to be told the
// position. These entry points take one, search it, and hand back a move in
// the `(factory, tile, line)` form the web UI already thinks in.
//
// No worker, no retained tree, no re-rooting: one call in, one move out. It
// blocks whatever thread calls it, which is no worse than the TypeScript AI it
// replaces, and it is the half of the wasm player that needs no changes to the
// host's game loop.

/// Bytes of the position buffer. See `set_position` for the layout.
const POSITION_BYTES: usize = 117;
const PLAYER_BYTES: usize = 42;

static mut POSITION_BUF: [u8; POSITION_BYTES] = [0; POSITION_BYTES];

thread_local! {
    static POSITION: RefCell<Option<Gamestate<2, 6>>> = const { RefCell::new(None) };
}

/// Where the host writes the position before calling `set_position`.
#[no_mangle]
pub extern "C" fn position_buffer() -> *mut u8 {
    &raw mut POSITION_BUF as *mut u8
}

#[no_mangle]
pub extern "C" fn position_buffer_len() -> u32 {
    POSITION_BYTES as u32
}

fn tile_from_index(i: u8) -> Option<azul_tiles_rs::tiles::Tile> {
    use strum::IntoEnumIterator;
    azul_tiles_rs::tiles::Tile::iter().nth(usize::from(i))
}

/// Read the position out of the buffer. Returns the number of legal moves,
/// which is 0 if the round is over and -1 if the bytes did not parse.
///
/// Layout, all `u8`:
///
/// ```text
///   0        player to move
///   1        first-player token still in the centre
///   2        round
///   3..33    six factories of five colour counts, factory 0 the centre
///   33..75   player 0, then 33+42..117 player 1, each:
///     +0..25   wall occupancy, row major
///     +25..35  five pattern lines as (tile index or 255, count)
///     +35..40  floor colour counts
///     +40      floor holds the first-player token
///     +41      score
/// ```
#[no_mangle]
pub extern "C" fn set_position() -> i32 {
    use azul_tiles_rs::playerboard::PlayerBoard;
    use azul_tiles_rs::tiles::{Tile, TileGroup};

    let buf: [u8; POSITION_BYTES] = unsafe { POSITION_BUF };

    let current_player = buf[0];
    if current_player > 1 {
        return -1;
    }
    let first_player_tile = buf[1] != 0;
    let round = u16::from(buf[2]);

    let mut factories = [TileGroup::new_empty(); 6];
    for (f, factory) in factories.iter_mut().enumerate() {
        for colour in 0..5u8 {
            let count = buf[3 + f * 5 + usize::from(colour)];
            if count > 0 {
                match tile_from_index(colour) {
                    Some(tile) => factory.add_tiles(tile, count),
                    None => return -1,
                }
            }
        }
    }

    let mut boards = [PlayerBoard::default(); 2];
    for (p, board) in boards.iter_mut().enumerate() {
        let base = 33 + p * PLAYER_BYTES;
        let mut wall = [[false; 5]; 5];
        for (r, row) in wall.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = buf[base + r * 5 + c] != 0;
            }
        }
        let mut rows: [(Option<Tile>, u8); 5] = [(None, 0); 5];
        for (r, slot) in rows.iter_mut().enumerate() {
            let tile_index = buf[base + 25 + r * 2];
            let count = buf[base + 25 + r * 2 + 1];
            *slot = if tile_index == 255 || count == 0 {
                (None, 0)
            } else {
                match tile_from_index(tile_index) {
                    Some(tile) => (Some(tile), count),
                    None => return -1,
                }
            };
        }
        let mut floor = TileGroup::new_empty();
        for colour in 0..5u8 {
            let count = buf[base + 35 + usize::from(colour)];
            if count > 0 {
                match tile_from_index(colour) {
                    Some(tile) => floor.add_tiles(tile, count),
                    None => return -1,
                }
            }
        }
        *board = PlayerBoard::from_parts(
            &wall,
            &rows,
            floor,
            buf[base + 40] != 0,
            buf[base + 41],
        );
    }

    let g = Gamestate::from_parts(factories, first_player_tile, boards, current_player, round);
    let count = g.get_moves().len() as i32;
    POSITION.with(|p| *p.borrow_mut() = Some(g));
    count
}

/// Pack a move the way the web UI names one: factory, colour, line, where
/// line 5 is the floor.
fn pack(m: &Move) -> i32 {
    let factory = usize::from(m.source) as i32;
    let tile = usize::from(m.tile) as i32;
    let line = usize::from(m.destination) as i32;
    (factory << 8) | (tile << 4) | line
}

/// Search the loaded position and return the best move packed by [`pack`], or
/// -1 if there is nothing to play.
///
/// `budget_ms` of 0 means no time limit; `max_depth` of 0 means no depth cap.
/// Giving neither is a way to hang a browser tab, so the host should set one.
#[no_mangle]
pub extern "C" fn search_move(budget_ms: u32, max_depth: u32) -> i32 {
    POSITION.with(|p| {
        let Some(g) = p.borrow().as_ref().cloned() else { return -1 };
        if g.get_moves().is_empty() {
            return -1;
        }
        let mut n = Negamax::new(
            Node::new(g),
            ScoreEvaluator,
            SearchOptions {
                alpha_beta: true,
                iterative: true,
                pre_sort: true,
                tt_bits: TT_BITS,
                retain_depth: RETAIN_DEPTH,
                max_time: (budget_ms > 0).then(|| Duration::from_millis(u64::from(budget_ms))),
                max_depth: (max_depth > 0).then_some(max_depth as u8),
                ..Default::default()
            },
        );
        pack(&n.search().best)
    })
}

/// Depth the last `search_move` reached. Diagnostic.
#[no_mangle]
pub extern "C" fn position_moves() -> i32 {
    POSITION.with(|p| p.borrow().as_ref().map_or(-1, |g| g.get_moves().len() as i32))
}

/// Play a move in the loaded position, naming it as the web UI does. Returns
/// the resulting state, or -1 if no such move is legal here.
#[no_mangle]
pub extern "C" fn position_play(factory: u32, tile: u32, line: u32) -> i32 {
    POSITION.with(|p| {
        let mut borrow = p.borrow_mut();
        let Some(g) = borrow.as_mut() else { return -1 };
        let wanted = g.get_moves().into_iter().find(|m| {
            usize::from(m.source) == factory as usize
                && usize::from(m.tile) == tile as usize
                && usize::from(m.destination) == line as usize
        });
        match wanted {
            Some(m) => state_code(g.play_move(m)) as i32,
            None => -1,
        }
    })
}

/// End the round in the loaded position. Scores the boards; does not deal,
/// for the reason `Gamestate::from_parts` gives.
#[no_mangle]
pub extern "C" fn position_end_round() -> i32 {
    POSITION.with(|p| {
        let mut borrow = p.borrow_mut();
        borrow.as_mut().map_or(-1, |g| state_code(g.end_round()) as i32)
    })
}

#[no_mangle]
pub extern "C" fn position_score(player: u32) -> i32 {
    POSITION.with(|p| {
        p.borrow().as_ref().map_or(-1, |g| i32::from(g.scores()[player as usize]))
    })
}

/// Tiles on a player's floor, the first-player token included. Diagnostic:
/// the floor is what decides the end-of-round penalty, and it is the part of
/// a loaded position that cannot be read back off the board.
#[no_mangle]
pub extern "C" fn position_floor_total(player: u32) -> i32 {
    POSITION.with(|p| {
        p.borrow().as_ref().map_or(-1, |g| {
            i32::from(g.boards()[player as usize].floor_total())
        })
    })
}
