# Depth-2 results: clone, capacity, depth, fine-tune

Everything below was measured against **depth-2 minimax** (`ScoreEvaluator`,
alpha-beta, TT 2^20) on held-out seeds `9_000_000+`, which never appear in
training: `gen_labels` used seeds `0..40000` and the in-loop eval uses
`EVAL_SEED_BASE = 1_000_000`.

Dataset throughout: 2,129,455 positions labelled at search depths 1, 2 and 3,
encoded with the 321-float encoder, depth-2 labels used for cloning.

## Headline

| model | depth-2 win rate |
|---|---|
| Best before this work | 43.4% |
| Best clone alone (240x1) | 40.7% |
| **Best fine-tuned (320x1)** | **65.3%** |

`320x1` fine-tuned scores 47.1 v 40.4 on average. Reproduce with
`holdout <dir> 1000` under `AZUL_DEPTH=2`.

## Read this first: 300 games cannot resolve these differences

`holdout` defaults to 300 games per seat. At these win rates that is **±2.8
points of standard error per seat**, so two measurements of the *same weights*
land up to 8 points apart. Measured instances:

- A 320x1 clone read **32.9%** on 300 games and **37.3%** on 1000.
- A fine-tuned 160x1 read **63.6%** on 300 and **60.0%** on 1000.
- Two fine-tunes looked 3.7 points apart on 300 games and 0.2 apart on 1000.

**Use 1000 games per seat, minimum.** Four conclusions drawn during this work
were overturned by re-measuring, all of them from trusting 300 games.

This also revises the note that run-to-run variance on an identical config is
~6 points. That is mostly the ruler, not the training: two independently
trained 320x1 clones, different stopping epochs and different weights, agreed
to **within 0.3 points** at 1000 games.

Two further measurement notes:

- **The in-loop 40-game eval overreads by ~1.3-1.5x** and must not be used to
  rank models. Worst case seen: it reported 320x1 and 320x2 as equivalent
  (in-loop margin +10.875 vs +10.675) where 1000 games separates them at 6.7
  sigma. It is fit for checkpoint selection only.
- **Score on margin, not win rate.** `examples/compare.rs` reports the mean
  margin difference with a standard error; over identical games that gave
  z = 2.45 where the win-rate difference gave 1.93, about 27% more power for
  free, because win/loss discards the magnitude of each result.

## Calibration: what a win rate means here

From `examples/mm_vs_mm.rs`, 200 games/seat. Per-game margin **sd = 15.6
points**, and `P(win) = Phi(margin / 15.6)` fits every row closely:

| matchup | win rate | margin |
|---|---|---|
| depth2 vs depth2 | ~50% | 0.0 |
| best policy vs depth2 | 65.3% | +6.5 |
| depth3 vs depth2 | 78.8% | +12.2 |
| best policy vs depth3 | 44.2% | -2.2 |

Win rate is therefore a *compressive* measure of skill — +6.7 buys 65%, +11.5
buys 75%, +17.5 buys 85% — and 100% is unreachable at any skill because tile
draws decide close games. Prefer margin as the primary signal.

Two consequences:

- **Depth 2 is not an exhausted yardstick.** A depth-3 search extracts nearly
  double our margin from the same opponent, so ~14 points of win rate remain
  available before we would even match depth-3 search.
- **Depth 3 is a good next training opponent and costs little.** The policy is
  about three quarters of the way from depth 2 to depth 3, losing to depth 3 by
  only 2.2 points where depth 2 loses by 12.2. A near-even match is where a
  margin reward carries most information, and a depth-3 opponent costs only
  ~1.27x the search time (round boundaries cap effective depth), so perhaps
  10-15% of episode throughput.

## Capacity, cloning: the peak is ~180k parameters

`examples/bc_sweep.rs`, batch 256, lr 1e-3, every cell run to its own
validation plateau (patience 12, min_delta 0.02pp) rather than a fixed epoch
budget. Validation agreement over the held-out 10% (~213k positions):

| net | params | train | val | gap | best epoch | clone win |
|---|---|---|---|---|---|---|
| 80x1 | 46,820 | 54.01 | 53.61 | 0.40 | 103 | 32.3% |
| 160x1 | 106,260 | 58.05 | 56.89 | 1.16 | 133 | 39.4% |
| **240x1** | 178,500 | 59.52 | **56.98** | 2.54 | 79 | **40.7%** |
| 240x2 | 236,340 | 59.75 | 56.98 | 2.77 | 73 | 37.3% |
| 320x1 | 263,540 | 59.71 | 56.41 | 3.29 | 46 | 37.6% |
| 320x2 | 366,260 | 59.30 | 56.06 | 3.23 | 25 | 34.4% |
| 640x1 | 731,700 | 60.60 | 54.53 | 6.07 | 15 | 36.1% |
| 640x2 | 1,141,940 | 58.30 | 54.61 | 3.68 | 9 | 31.4% |
| 1024x2 | 2,613,428 | 58.70 | 54.26 | 4.44 | 8 | 33.8% |
| 1024x2 @ lr 3e-4 | 2,613,428 | 60.83 | 54.99 | 5.84 | 6 | 36.5% |
| 2048x2 | 9,420,980 | 57.82 | 54.17 | 3.65 | 6 | 33.3% |

**The train/val gap rises monotonically with size** — 0.40, 1.16, 2.54, 3.29,
3.68, 4.44 — so the data constrains the model, not the reverse. Larger nets
peak earlier and then decay for as long as they are allowed to run (2048x2 at
epoch 6, 160x1 at epoch 133), which is why a fixed epoch budget mismeasures
them: at a 22-epoch cap, 160x1 records ~54% and looks too small when it is
actually the joint best.

Capacity was never the limit on *fitting*: 1024x2 reached train 71.4% while its
validation fell to 53.3%. Lowering its learning rate to 3e-4 gained 0.7 points
of validation but made it memorise faster, so the falling-validation signature
is genuine overfitting and not an untuned step size.

## Capacity, fine-tuning: the peak moves, but only a little

`examples/long_train.rs`, 6 hours each, lr 1e-4, decay 0.7,
`RewardMode::Immediate`, resumed from the clone, one core pair each. All at
1000 games/seat:

| net | params | clone win | episodes | fine-tuned win |
|---|---|---|---|---|
| 160x1 | 106k | 39.4% | 2558 | 60.0% |
| 240x1 | 178k | 40.7% | 1562 | 60.2% |
| **320x1** | 264k | 37.6% | 1759 | **65.3%** |
| 240x2 | 236k | 37.3% | 1876 | 55.4% |
| 320x2 | 366k | 34.4% | 2313 | 56.0% |
| 640x1 | 732k | 36.1% | 1673 | ~44% |

**Clone quality does not predict fine-tuned strength.** Among 160/240/320 the
ranking inverts — the worst clone gives the best fine-tune, by 5.1 points and
with 800 fewer episodes than the net it beats. It does not keep inverting:
640x1 has both the worst clone and by far the worst fine-tune. So architecture
must be chosen by measuring the fine-tune. Do not extrapolate "wider is better"
past 320x1; that was predicted here and measured false at 640x1, which loses to
320x1 by 8.6 margin points (z = -15.3).

RL supplies most of the strength: +20 to +28 points over the clone it started
from.

## Depth: one wide layer beats two narrow ones

| fine-tuned | 1 layer | 2 layers |
|---|---|---|
| 240 wide | 60.2% | 55.4% |
| 320 wide | **65.3%** | 56.0% |

There is an **interaction**, not just a main effect:

- Widening 240 -> 320 is worth +5.1pp with one hidden layer (4.6 sigma) and
  **nothing** with two (+0.38 margin points, z = 0.72, CI [-0.66, +1.42]).
- The depth penalty grows with width: -1.7 margin points at 240, -3.6 at 320
  (z = -6.7).

So a second layer both costs strength and removes the benefit of width. The
320x2 result is conservative: it ran 2313 episodes against 320x1's 1759 and
still lost by 9.3 points.

Depth costs clone *strength* too (~3.3pp at both widths) while barely touching
agreement — neutral at 240, -0.35 at 320 — so no amount of clone-side sweeping
would have found it. It only shows up by playing games.

## Reproducing

```bash
# clone a candidate (GPU; needs --features gpu)
cargo build --release --features gpu --example bc_sweep
./target/release/examples/bc_sweep /tmp/labels_v3 2 300 256 /tmp/sweep_320x1 "320x1" 12 0.0002 0.001

# fine-tune it
AZUL_DEPTH=2 ./target/release/examples/long_train 320 1 6.0 /tmp/ft_320x1 0.7 /tmp/sweep_320x1/h320x1 0 0.0001

# score it: 1000 games/seat, and against another checkpoint on margin
AZUL_DEPTH=2 ./target/release/examples/holdout /tmp/ft_320x1 1000
AZUL_DEPTH=2 ./target/release/examples/compare /tmp/ft_320x1 /tmp/ft_240x1 1000

# calibrate what a win rate means
./target/release/examples/mm_vs_mm 3 2 200
```

Two operational notes for long runs on a 32G box with no swap: run **one
training job at a time** (three concurrent fine-tunes plus another session's
workers hit the ceiling twice), and launch each in **its own cgroup** with
`systemd-run --user --scope -p MemoryMax=8G`, because `nohup` and `setsid` both
stay inside the launching pane's scope and die with it.

## Open, in priority order

1. **Extend the winner.** Every fine-tune stopped on `OutOfTime`, not a
   plateau, and 320x1's best checkpoint landed at episode 1695 of 1759 — its
   final 4%. None has finished improving. Note the LR schedule decays by
   `0.7^(episode/500)`, so a naive continuation starts at 3.43e-5 and collapses;
   restart the schedule rather than continuing it.
2. **Train against depth 3.** Near-even match, ~10-15% throughput cost,
   sharper reward signal. Started but not finished.
3. **Re-measure the 43.4% historical reference at 1000 games**, so the baseline
   is sound rather than a 300-game figure on a superseded encoding.
4. **`long_train` needs a `max_episodes` argument.** Matching core counts does
   not match episode counts when background load differs: 320x2 got 2313
   episodes against 320x1's 1759 under nominally identical settings, purely
   because the box was quieter.
5. **Teacher label quality, not dataset size,** is the leading explanation for
   the ~57% agreement ceiling. Exhaustive-to-round-end labels exist for a small
   set and would measure teacher noise directly; more rows of equally noisy
   labels would not fix it.
