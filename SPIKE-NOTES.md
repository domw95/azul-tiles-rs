# wasm spike: what is done, what is not

The engine compiled to wasm, measured against the native and TypeScript
builds, and wired into `tiles-web` as a playable opponent. This file is the
cross-repo picture, because the work spans four repositories and no single
pull request shows all of it.

Written 2026-09-26. Numbers are from this box: 12 logical / 6 physical cores,
work pinned to cores 2,3,8,9 at nice 19, interleaved against whatever else was
running.

## Branches

| repo | branch | what is on it |
|---|---|---|
| `minimaxer-rs` | `master` | the two search fixes, **landed** |
| `minimaxer-rs` | `wasm/spike` | wasm build, `principal_variation` |
| `azul-tiles-rs` | `wasm/spike` | feature gating, position loading, benchmark, pondering engine |
| `azul-tiles-rs` | `bench/xlang` | the Rust benchmark body |
| `azul-tiles` | `bench/xlang` | the TypeScript benchmark body |
| `tiles-web` | `feat/wasm-ai` | the player, the analysis readout, the debug panel |
| `azul-bench` | `master` | the benchmark spec, runner and results |

## What was measured

- **Rust is ~20x TypeScript per node.** Deepening to depth 5: 157 ms against
  3,435 ms, on trees within 1% of the same size. Not an algorithm gap.
- **The cost is representation.** Native gamestate is a 176-byte memcpy into
  an arena, ~250 bytes per node. TypeScript allocates ~15 objects and ~25
  `Move` instances per node, ~4.5 KB, and a depth-5 search exhausts a 2 GB
  heap cap.
- **wasm runs at native speed**: 0.97-1.09x per node. The module is 86 KB.
- **The fitted evaluation is worth 25 points of win rate** over the plain
  score differential, on identical code against the same opponent.
- **Against the Master personality**, 40 games, seats alternating: 36-2-2 on
  equal time, 34-6 with the engine on a tenth of the time.
- **Rules conformance**: 500 games, 3,322 rounds, 35,307 moves replayed
  through both implementations, agreeing on every move and round score.

## What is not done

### Nothing has been tested in a browser

There is no browser on this box. The Worker booting, `instantiateStreaming`,
and both panels rendering are unverified. Everything below is downstream of
settling this. The analysis panel is the quickest tell: if it reads
"thinking..." forever, the worker did not start and the console will say why.

The player degrades quietly rather than loudly if the worker fails -- it falls
back to a synchronous search every move -- so watch the panel, not the moves.

### Memory

Pondering a full game grows wasm linear memory from 1.1 MB to **164.6 MB**,
peak tree 210,081 nodes. wasm memory never shrinks, so that is permanent for
the life of the tab. Fine on a desktop, marginal on a phone.

`retain_depth` is the lever, currently 4. It does not bound a wide root much,
because the top four plies of a 96-move position are large on their own.
Lowering it to 2 or 3 trades principal variation length for memory.

### The pondering worker cannot be interrupted mid-slice

A deepening pass either completes or is discarded, so the slice budget is both
the responsiveness knob and the depth ceiling. It adapts -- 150 ms, doubling
when a slice gains no depth, capped at 2 s -- which is why a fresh 96-move
round reaches depth 6-7 rather than sitting at depth 4 forever, as it did with
a fixed 150 ms slice.

The cap is therefore also the worst case for noticing a new position, and why
the player keeps its synchronous fallback. The proper fix is a cancellation
flag polled where the clock is polled in `negamax.rs`; that would also stop a
cut-short pass being thrown away, which is pure waste today.

Consequence: **autoplay does not exercise pondering.** Moves land faster than
the worker can re-point, so the fallback handles most of them. It is built for
a human taking a few seconds.

### Blockers for merging

- `azul-tiles-rs/Cargo.toml`, `wasm-bench/Cargo.toml` and
  `wasm-engine/Cargo.toml` carry `[patch]` sections pointing at a sibling
  `minimaxer-rs` worktree by relative path. Spike convenience. They want
  replacing with a git revision, now that the fixes are on `minimaxer-rs`
  master.
- `tiles-web` does not build from a clean clone: `setup.ts` imports
  `ParanoidAI`, which the published `azul-tiles` 3.3.0 does not export. This
  predates the spike, but it blocks a deploy from a fresh checkout. Either
  publish `azul-tiles` or pin a file dependency.
- `azul-bench`'s runner defaults point at worktree paths, which go stale once
  those are cleaned up. Repoint at the main checkouts when `bench/xlang`
  merges.

### Known divergence, not a bug

Rust force-ends a game at its round 10; TypeScript has no cap, and neither
does Azul. Past that the two are playing different games by design. It shows
up in roughly 1 game in 250 and cannot affect a search, which never crosses a
round boundary. The conformance check classifies it rather than failing on it.

Note the counting: this engine deals in its constructor, so its first playable
round is 1 where the TypeScript implementation's is 0. The evaluation crosses
every term with rounds remaining, so a host handing over the wrong number
skews the whole evaluation. `position_round` exists to make that checkable.

## Things considered and rejected

- **Rewriting the TypeScript engine** with typed arrays and a node pool. Would
  plausibly recover 3-6x of the 20x, for a rewrite of the azul-tiles core. The
  wasm port reuses code that is already correct and already measured.
- **The NN evaluator** (`players::nn_eval`, merged to master separately) for
  the browser player. Its own documentation puts a burn forward at batch 1 at
  three to four orders of magnitude more than `ScoreEvaluator`, per leaf, and
  alpha-beta calls the evaluator at every leaf. It is not a candidate for a
  search that visits 200k nodes in a second.
- **Threads.** `rayon` is behind a default-on `parallel` feature and off for
  wasm. Enabling it needs `SharedArrayBuffer` and cross-origin isolation
  headers, which is a server change for a single-digit speedup on a search
  that already solves most positions outright.
