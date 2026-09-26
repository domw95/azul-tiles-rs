//! Full-depth label generation on one core, sharded as it goes.
//!
//! Every position is labelled by a search with **no depth cap**: iterative
//! deepening runs until the tree is exhausted, which in Azul means every line
//! reached the end of the round (the search treats a round's end as terminal).
//! The label is then the game-theoretic best move for the round, not the best
//! move a depth budget could find, and `exit == Exhaustive` records that it
//! really was exact rather than merely deep.
//!
//! `NODE_BUDGET` buys throughput back from the few positions that cost
//! everything -- see [`search_budgeted`]. Positions inside the budget are
//! still exact; the ones that exceed it fall back to the deepest pass they
//! could afford and are flagged inexact in `meta_*.bin`.
//!
//! Two things make that affordable from the opening move, where the tree is
//! largest. `retain_depth` bounds memory by keeping only the top few plies of
//! the tree, so resident size is flat in the nodes searched instead of linear
//! in them -- measured from the opening, 399 MiB flat to depth 12 against 9.8
//! GiB by depth 8 without it. And a large transposition table pays for itself
//! because Azul reaches the same position down many orders of taking: 56% of
//! nodes were table hits by depth 12.
//!
//! Usage:
//!
//!     gen_full <out_dir> [positions_per_shard] [first_seed]
//!
//! Env knobs: `EVAL` (heuristic | score | fitted), `NODE_BUDGET` (1000000, 0
//! for no cap), `TT_BITS` (24), `RETAIN` (3), `MAX_SECS` (900), `EPSILON`
//! (0.25), `RANDOM_BEST` (1), `TAG` (""), `SEED_STRIDE` (1).
//!
//! `EVAL` is the one that changes what the labels *mean* rather than how fast
//! they arrive -- see [`Eval`].
//!
//! **Running a fleet.** One process per core, all writing to the same
//! directory: give each a distinct `TAG`, a `SEED_STRIDE` equal to the number
//! of workers, and a starting seed equal to its index. Each worker then owns
//! its own files and its own stripe of the seed space, and the directory loads
//! as a single dataset. The processes share nothing and need no coordination;
//! one `STOP` file stops all of them.
//!
//! **Stopping it.** `touch <out_dir>/STOP`. The position being searched
//! finishes, everything held in memory is written out -- including a partial
//! game, which stays replayable because a seed plus the moves played up to
//! that point rebuilds exactly those positions -- and the run exits 0. Stop
//! latency is one search, so at most `MAX_SECS`. A hard kill instead loses
//! only the shard in progress, which by default is the game in progress.
//!
//! Re-running the same command resumes: the next shard index and the next
//! seed come from `state.json`, so no seed is ever labelled twice.
use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::{HeuristicEvaluator, ScoreEvaluator, Weights};
use azul_tiles_rs::players::nn::{gs_to_array_ordered, FactoryOrder};
use azul_tiles_rs::players::ppo::pretrain::{MultiDataset, Replay};
use azul_tiles_rs::players::ppo::ACTION_SIZE;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::Evaluate;
use minimaxer::node::Node;
use minimaxer::SearchExit;
use rand::Rng;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The depth slot these labels are filed under in [`MultiDataset`].
///
/// The container indexes labels by the depth that produced them, and a
/// full-depth search has no single depth -- it stops wherever the round ends,
/// which varies by position. 255 is the sentinel for "searched to exhaustion";
/// it is also literally `u8::MAX`, the cap the search runs under. Pass `255`
/// as the depth argument to `pipeline` to train on these.
const FULL: u8 = u8::MAX;

fn env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The evaluator, chosen at run time by `EVAL`.
///
/// Worth a branch per call because of what the evaluator *is* in this run.
/// `is_terminal` is `is_round_over`, so an exhaustive search bottoms out on
/// round end positions and takes their value from `evaluate`. The search is
/// then exact by construction and the evaluator carries the whole of the
/// question "which move is best" -- it is not a tie-breaker or an ordering
/// hint here, it is the objective function the labels encode. Recorded in
/// `status.json` for the same reason: a shard is only interpretable next to
/// the evaluator that produced it.
///
/// Measured head to head at depth 2, 1000 paired games each (`evalmatch`,
/// null control 50.0% +/- 3.1):
///
/// - `heuristic` vs `score`: 59.1% +/- 3.0, +3.00 points per game
/// - `fitted` vs `score`:    56.9% +/- 3.1
/// - `fitted` vs `heuristic`: 52.8% +/- 3.1, which spans 50%
///
/// So `heuristic` is the default: clearly better than score alone, and the
/// least squares fit in `eval_weights.json` does not beat it despite being
/// fitted on round end positions, because it carries the forecast term that
/// measured actively harmful (47.0% +/- 3.1 against the same weights without).
#[derive(Clone)]
enum Eval {
    Score(ScoreEvaluator),
    Heuristic(HeuristicEvaluator),
}

impl Eval {
    fn from_env() -> (Self, &'static str) {
        match std::env::var("EVAL").unwrap_or_else(|_| "heuristic".into()).as_str() {
            "score" => (Eval::Score(ScoreEvaluator), "score"),
            "fitted" => {
                let w: Weights = serde_json::from_reader(
                    std::fs::File::open("eval_weights.json").expect("eval_weights.json"),
                )
                .expect("eval_weights.json");
                (Eval::Heuristic(HeuristicEvaluator::new(w)), "fitted")
            }
            _ => (Eval::Heuristic(HeuristicEvaluator::default()), "heuristic"),
        }
    }
}

impl minimaxer::Evaluate<Gamestate<2, 6>> for Eval {
    fn evaluate(&mut self, g: &Gamestate<2, 6>) -> f32 {
        match self {
            Eval::Score(e) => e.evaluate(g),
            Eval::Heuristic(e) => e.evaluate(g),
        }
    }
}

fn opts() -> SearchOptions {
    SearchOptions {
        // No cap: deepen until the round is solved.
        max_depth: None,
        // A safety valve, not the plan. A position that cannot be solved
        // inside it yields the deepest completed pass and is recorded as
        // inexact, rather than stalling the run for ever.
        max_time: Some(std::time::Duration::from_secs(env("MAX_SECS", 900u64))),
        iterative: true,
        alpha_beta: true,
        pre_sort: true,
        sort_on_create: true,
        sort_on_create_min_depth: 1,
        tt_bits: env("TT_BITS", 24u8),
        retain_depth: env("RETAIN", 3u8),
        // Break ties between equally good moves at random, as the fixed-depth
        // generator does. At full depth ties are genuine -- Azul's scores are
        // coarse and many lines transpose -- so taking the first every time
        // would teach the policy move-generation order rather than the game.
        random_best: env("RANDOM_BEST", 1u8) != 0,
        ..Default::default()
    }
}

/// Deepen until the round is solved, or until the node budget is spent.
///
/// Exhausting a round is affordable everywhere except the first ply or two of
/// it, and the imbalance is not close: over 2,123 exhaustively searched
/// positions, 1% of them accounted for 58.5% of all nodes and 5% for 89.5%.
/// Spending the whole machine on that 5% buys the exact answer to the hardest
/// positions and almost no data. Capping them and exhausting everything else
/// is most of the quality for a fraction of the cost -- at a million nodes,
/// 87.5% of positions still come back exact for 18% of the work.
///
/// **The budget is in nodes, not seconds, and that is the point.** Under a
/// clock, how hard a position is searched depends on what else the box was
/// doing. A fleet of twelve workers sharing six physical cores would search
/// every position perhaps three times less deeply than one worker with the
/// machine to itself, so the dataset would carry a quality gradient tracking
/// machine load -- deeper labels from quiet nights, shallower ones from busy
/// afternoons, and nothing in the data saying which is which. Worse, it is a
/// gradient a model can neither see nor correct for.
///
/// Node counts are a property of the search, not of the machine, so a budget
/// in nodes spends the same effort on the same position however loaded the box
/// is. That makes the labels homogeneous, which is the property that matters
/// for training. It does not make them bit-reproducible -- `random_best`
/// breaks ties with a coin and the epsilon walk picks different positions each
/// run -- but the effort behind every label is the same.
///
/// Budget 0 means no cap: deepen until exhausted, whatever it costs.
fn search_budgeted<E: Evaluate<Gamestate<2, 6>>>(
    search: &mut Negamax<Gamestate<2, 6>, Move, E>,
    budget: u32,
) -> minimaxer::SearchResult<Move> {
    if budget == 0 {
        search.options.max_depth = None;
        return search.search();
    }
    let mut depth = 1u8;
    // The tree is retained across plies, so the root's descendant count starts
    // at whatever the previous position left behind. Budgeting the raw count
    // would charge this position for work already done and stop it dead on the
    // first pass; the increment from this position's first pass is what it
    // actually costs.
    let mut base: Option<u32> = None;
    loop {
        search.options.max_depth = Some(depth);
        let r = search.search();
        let b = *base.get_or_insert(r.nodes);
        // Exhaustive, or out of moves: the answer is exact and deepening is
        // meaningless.
        if r.exit != SearchExit::Depth {
            return r;
        }
        if r.nodes.saturating_sub(b) >= budget || depth == u8::MAX {
            return r;
        }
        // Jump to what the retained tree already covers rather than walking up
        // from 1 through passes that are already done.
        depth = r.depth.max(depth).saturating_add(1);
    }
}

/// Mask in *canonical* action space -- the space the policy is indexed by.
///
/// Must use the same `FactoryOrder` as the state encoding, or the mask and the
/// label name different displays than the state shows, and nothing reports it.
fn mask_for(moves: &[Move], order: &FactoryOrder) -> Vec<f32> {
    let mut m = vec![-1e8f32; ACTION_SIZE];
    for mv in moves {
        m[order.canonical_index(mv)] = 0.0;
    }
    m
}

/// Per-position provenance, written beside each shard.
///
/// Kept out of the shard itself so the `MultiDataset` format is untouched and
/// existing loaders keep working -- `load_dir` selects on the `shard_` prefix,
/// so these files are invisible to it. Without them a later reader cannot tell
/// an exact label from one that ran out of time, which is the difference
/// between a target and a guess.
#[derive(Default)]
struct Meta {
    depth: Vec<u8>,
    exhaustive: Vec<u8>,
    nodes: Vec<u32>,
    secs: Vec<f32>,
}

impl Meta {
    fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.depth.len() as u64).to_le_bytes())?;
        f.write_all(&self.depth)?;
        f.write_all(&self.exhaustive)?;
        for v in &self.nodes {
            f.write_all(&v.to_le_bytes())?;
        }
        for v in &self.secs {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()
    }
}

/// Everything accumulated since the last flush.
struct Shard {
    data: MultiDataset,
    replay: Replay,
    meta: Meta,
}

impl Shard {
    fn new() -> Self {
        Shard {
            data: MultiDataset {
                depths: vec![FULL],
                targets: vec![Vec::new()],
                values: vec![Vec::new()],
                ..Default::default()
            },
            replay: Replay::default(),
            meta: Meta::default(),
        }
    }
}

/// Bytes actually written so far, which is the number the operator wants when
/// deciding whether to let it keep running.
fn bytes_in(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Totals over the whole run, including anything a previous run left behind.
struct Totals {
    games: u64,
    positions: u64,
    exact: u64,
    depth_sum: u64,
    search_secs: f64,
    started: Instant,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: gen_full <out_dir> [positions_per_shard] [first_seed]");
        std::process::exit(2);
    }
    let out = PathBuf::from(&a[1]);
    // Shards close on a game boundary, so this is a floor rather than a size:
    // a shard that has reached it finishes the game it is in first. Splitting
    // mid-game would leave the continuation unreplayable, since rebuilding a
    // position needs the seed *and* every move before it.
    let per_shard: usize = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(1);
    std::fs::create_dir_all(&out).unwrap();

    // Resume where a previous run stopped, so restarting after a stop neither
    // repeats a seed nor overwrites a shard.
    // One process per core writes into one directory, so every file it owns
    // carries its tag. `load_dir` selects on the `shard_` prefix and the tag
    // sits after it, so a whole fleet's output loads as one dataset with no
    // merge step -- which is the point of sharing the directory rather than
    // giving each worker its own.
    let tag = std::env::var("TAG").unwrap_or_default();
    // Workers stripe the seed space rather than splitting it into blocks, so
    // no worker can run out of its allocation while another still has years of
    // seeds left, and adding or removing a worker does not renumber anything.
    let stride: u64 = env("SEED_STRIDE", 1u64);
    let state_path = out.join(format!("state_{tag}.json"));
    let (mut next_shard, mut seed) = std::fs::read_to_string(&state_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .map(|v| {
            (
                v["next_shard"].as_u64().unwrap_or(0) as usize,
                v["next_seed"].as_u64().unwrap_or(0),
            )
        })
        .unwrap_or((0, a.get(3).map(|v| v.parse().unwrap()).unwrap_or(0)));

    let stop_path = out.join("STOP");
    if stop_path.exists() {
        eprintln!("{} exists; remove it before starting", stop_path.display());
        std::process::exit(2);
    }

    let epsilon: f64 = env("EPSILON", 0.25f64);
    let (evaluator, eval_name) = Eval::from_env();
    // 0 disables the cap. The default trades 12.5% of positions losing their
    // exactness for 5.5x the throughput; `meta_*.bin` records which.
    let node_budget: u32 = env("NODE_BUDGET", 1_000_000u32);
    let o = opts();
    println!(
        "full-depth labels -> {}\n  eval {eval_name}, tt 2^{}, retain {}, cap {:?}, epsilon {epsilon}, random_best {}\n  resuming at seed {seed}, shard {next_shard}\n  stop with: touch {}",
        out.display(),
        o.tt_bits,
        o.retain_depth,
        o.max_time.unwrap(),
        o.random_best,
        stop_path.display(),
    );

    let mut t = Totals {
        games: 0,
        positions: 0,
        exact: 0,
        depth_sum: 0,
        search_secs: 0.0,
        started: Instant::now(),
    };
    let mut shard = Shard::new();
    let mut rng = rand::thread_rng();
    let mut stopping = false;

    'outer: while !stopping {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        // One search object per game, re-rooted onto each move played. The
        // subtree below that move is already searched and already ordered, and
        // the transposition table stays warm across the whole game.
        let mut search = Negamax::new(Node::new(gs.clone()), evaluator.clone(), opts());
        // Replay indices stay in REAL action space: they are decoded back into
        // moves to rebuild positions, and the canonical order is a property of
        // a position, not of a move.
        let mut played_idx: Vec<i32> = Vec::new();
        let game_started = Instant::now();
        let mut game_positions = 0u32;

        loop {
            if gs.is_round_over() {
                if gs.end_round() == State::GameEnd {
                    break;
                }
                // A new deal invalidates the retained tree, so start it over.
                search.replace_gamestate(gs.clone());
                continue;
            }
            // Checked between positions, which is the only point where the
            // held state is consistent enough to write out.
            if stop_path.exists() {
                stopping = true;
                break;
            }

            let moves = gs.get_moves();
            let seat = gs.current_player() as usize;
            // One order per position, shared by the state, the mask and the
            // label below.
            let order = FactoryOrder::canonical(&gs);
            shard
                .data
                .states
                .extend_from_slice(gs_to_array_ordered(&gs, seat, &order).as_slice());
            shard.data.masks.extend_from_slice(&mask_for(&moves, &order));

            let r = search_budgeted(&mut search, node_budget);
            let exact = r.exit == SearchExit::Exhaustive;
            shard.data.targets[0].push(order.canonical_index(&r.best) as i32);
            shard.data.values[0].push(r.value);
            shard.meta.depth.push(r.depth);
            shard.meta.exhaustive.push(exact as u8);
            shard.meta.nodes.push(r.nodes);
            shard.meta.secs.push(r.time.as_secs_f32());

            t.positions += 1;
            t.exact += exact as u64;
            t.depth_sum += r.depth as u64;
            t.search_secs += r.time.as_secs_f64();
            game_positions += 1;
            println!(
                "  seed {seed} ply {game_positions:>2}: depth {:>2} {:>11} nodes {:>7.1}s {:?} value {:.1}",
                r.depth,
                r.nodes,
                r.time.as_secs_f32(),
                r.exit,
                r.value,
            );

            // Explore off the principal variation, or every game from the same
            // seed would relabel one line.
            let played = if rng.gen_bool(epsilon) {
                moves[rng.gen_range(0..moves.len())]
            } else {
                r.best
            };
            played_idx.push(played.to_index() as i32);
            gs.play_move(played);
            search.play_move(&played);
        }

        // A partial game is still replayable: its seed and the moves recorded
        // rebuild exactly the positions labelled.
        let partial = !played_idx.is_empty();
        if partial {
            shard.replay.seeds.push(seed);
            shard.replay.plies.push(played_idx.len() as u32);
            shard.replay.played.extend(played_idx);
        }
        if !stopping {
            t.games += 1;
            println!(
                "game {seed} done: {game_positions} positions in {:.0}s",
                game_started.elapsed().as_secs_f32()
            );
        }
        // Past the seed either way. A game cut short by a stop has already had
        // its positions written, so resuming onto the same seed would label
        // that opening a second time and hand the trainer duplicate rows --
        // correlated duplicates, which is the kind a shuffle cannot undo.
        // Abandoning the rest of one game costs nothing: the dataset is a pile
        // of positions, not a set of complete games.
        if partial || !stopping {
            seed += stride;
        }

        if shard.data.len() >= per_shard || stopping {
            if shard.data.is_empty() {
                break 'outer;
            }
            let n = shard.data.len();
            shard.data.save_shard(&out.join(format!("shard_{tag}{next_shard:04}.bin"))).unwrap();
            shard.replay.save(&out.join(format!("replay_{tag}{next_shard:04}.bin"))).unwrap();
            shard.meta.save(&out.join(format!("meta_{tag}{next_shard:04}.bin"))).unwrap();
            next_shard += 1;
            shard = Shard::new();

            let bytes = bytes_in(&out);
            let hours = t.started.elapsed().as_secs_f64() / 3600.0;
            let status = serde_json::json!({
                "eval": eval_name,
                "node_budget": node_budget,
                "shards": next_shard,
                "games": t.games,
                "positions": t.positions,
                "bytes": bytes,
                "mib": (bytes as f64 / 1048576.0 * 10.0).round() / 10.0,
                "exhaustive_frac": t.exact as f64 / t.positions.max(1) as f64,
                "mean_depth": t.depth_sum as f64 / t.positions.max(1) as f64,
                "mean_search_s": t.search_secs / t.positions.max(1) as f64,
                "elapsed_s": t.started.elapsed().as_secs(),
                "positions_per_hour": t.positions as f64 / hours.max(1e-9),
                "mib_per_hour": bytes as f64 / 1048576.0 / hours.max(1e-9),
                "next_seed": seed,
                "next_shard": next_shard,
                "stopped": stopping,
            });
            std::fs::write(out.join(format!("status_{tag}.json")), format!("{status:#}\n")).unwrap();
            // Appended rather than overwritten, so the shape of the run over
            // time survives; status.json only ever shows the latest.
            {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(out.join(format!("progress_{tag}.jsonl")))
                    .unwrap();
                writeln!(f, "{status}").unwrap();
            }
            // state.json is what a restart reads. Written last, so a crash
            // mid-flush resumes onto a shard index that is free rather than
            // one already holding data.
            std::fs::write(
                &state_path,
                format!("{{\"next_seed\": {seed}, \"next_shard\": {next_shard}}}\n"),
            )
            .unwrap();

            println!(
                "shard {:04}: {n} positions | total {} positions, {} games, {:.1} MiB, {:.0}% exact, mean depth {:.1}, {:.0} positions/hour",
                next_shard - 1,
                t.positions,
                t.games,
                bytes as f64 / 1048576.0,
                100.0 * t.exact as f64 / t.positions.max(1) as f64,
                t.depth_sum as f64 / t.positions.max(1) as f64,
                t.positions as f64 / hours.max(1e-9),
            );
        }
    }

    println!(
        "stopped after {} positions in {} games, {:.1} MiB in {} shards, {:.0}s elapsed",
        t.positions,
        t.games,
        bytes_in(&out) as f64 / 1048576.0,
        next_shard,
        t.started.elapsed().as_secs_f32(),
    );
}
