# Depth-2 results: clone, capacity, depth, fine-tune

Everything below was measured against **depth-2 minimax** (`ScoreEvaluator`,
alpha-beta, TT 2^20) on held-out seeds `9_000_000+`, which never appear in
training: `gen_labels` used seeds `0..40000` and the in-loop eval uses
`EVAL_SEED_BASE = 1_000_000`.

**What that opponent is and is not.** It is *fixed* and deterministic at fixed
depth, which is what makes every comparison here sound — all of them share it.
It is not the strongest opponent available: `HeuristicEvaluator` with fitted
weights has been measured beating `ScoreEvaluator` by roughly 70/30 under a 5ms
clock, though that is one timed run on a loaded box and should be treated as
approximate rather than as a figure to re-baseline against. So "beats depth-2
minimax at 65.3%" is a narrower claim than it sounds, and a stronger
*reproducible* opponent exists if one is wanted:
`HeuristicEvaluator::new(Weights::hand_set())` at fixed depth is deterministic
and sits between the two.

Verified across master's evaluator rework (through e013a5a): `ScoreEvaluator`'s
impl is byte-identical, and — since an unchanged body proves nothing about
changed inputs, and 111 lines moved under `gamestate.rs` and `playerboard/` —
`holdout ft_320x1 300` reproduces exactly, 69.0%/62.0% with means 47.4v39.8 and
45.8v39.9 on both seats.

Dataset throughout: 2,129,455 positions labelled at search depths 1, 2 and 3,
encoded with the 321-float encoder, depth-2 labels used for cloning.

## Headline

| model | vs depth 2 | vs depth 3 |
|---|---|---|
| Best before this work | 43.4% | — |
| Best clone alone (240x1) | 40.7% | — |
| depth-2 search | 50% | 21.2% |
| `ft_320x1`, 1,759 episodes | 65.3% | 43.3% |
| `ext2`, ~4,785 episodes | ~74% | 52.3% |
| **`ft_320x1_d3b`, ~7,004 episodes** | **77.0%** | **57.9%** |
| depth-3 search | 78.8% | 50% |

The best model scores 52.0 v 39.4, a margin of +12.2 — the same margin depth-3
*search* achieves against depth 2 — and it beats depth-3 search itself. Reproduce
with `holdout <dir> 1000` under `AZUL_DEPTH=2`.

**Training length is the lever.** Every architecture and opponent variation tested
became a null once length was controlled; the three models above differ mainly in
episodes. Two of the stop conditions were silently truncating runs (see the noisy
maximum note below), which made this harder to see than it should have been.

**Training against a stronger opponent is a null.** `d3b` trained against depth 3
and leads `ext2` (depth 2, same lineage) by +1.94 margin points at depth 2 and
+2.18 at depth 3 — **flat**, where a genuinely better training signal should widen
against harder opposition. It also carries 46% more episodes, which is enough to
explain the whole difference. Beware the win-rate view of this same pair: the gap
reads 3 points at depth 2 and 8 at depth 3, apparent confirmation, but that is
`Phi(margin/15.6)` being steepest near 50% rather than a real widening.

## The most repeated mistake here: a maximum over noisy draws

Four instances in one day, all the same shape — a threshold or an argmax taken
over many noisy samples and read as a measurement:

1. **Checkpoint selection** compared single 40-game evals. One lucky +15.35 at
   episode 110 locked out 1,845 further episodes with nothing written to disk.
2. **The 0.90 win-rate target** compared a single eval. At ~480 evals per run and
   ±8 points of standard error, crossing 90% is near-certain at a true 65%; it
   ended a 12-hour run at episode 2,391.
3. **A train-agreement column** read at the best-validation epoch was taken for
   agreement at convergence, producing a confident and wrong underfitting
   diagnosis.
4. **The EMA introduced to fix 1 and 2** is seeded from its first sample, so at
   eval 1 "smoothed" is one 40-game draw. A lucky first draw set a high-water mark
   the smoothed series never beat: a 5-hour run stopped as `Plateaued` with
   `best_episode: 0`.

All four are fixed (e1ee756, dec43a2, 2dd3964). The general rule: any threshold
tested repeatedly against a 40-game statistic will be crossed by noise, and
smoothing only helps once it has samples to smooth.

## How to read these numbers

**Every measurement here carries the regime it was taken in.** Three conclusions
on this project were later found to be scoped rather than general — a capacity
sweep measured at 307k positions, a seat default measured at 7-17% win rates, a
tie hypothesis measured on 237 positions. None was wrong when taken; each was
carried forward without its conditions attached, and a bare number reads as
general. So: model strength, epoch-selection rule, which teacher, and what the
metric counts belong next to the figure, not in a paragraph elsewhere.

Two entries below are **provisional** for that reason, and say so inline.

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

> **PROVISIONAL, but quantified.** Every number in this section was measured
> with a `behaviour_clone` that never shuffled its training order (fixed in
> e95451a). Re-measured at 320x1 on the identical split, shuffling is worth
> **+1.28 points** (56.41 -> 57.69), confirmed independently by the learning
> curve's full-data cell on a different validation set (57.72). Training also
> runs ~50% longer before plateauing (46 -> 70 epochs), consistent with more
> diverse gradients. Only 320x1 has been re-measured, so the peak *location*
> is still untested — though a level shift of this kind would not move it.
>
> **Regime:** batch 256, lr 1e-3, depth-2 labels, 2.13M positions / 40,000
> games, validation = tail 10%, `train` column = train agreement **at the
> best-validation epoch**, not at convergence. That last point matters: large
> nets peak on validation by epoch 6-8, so their `train` figure is six epochs
> in, and reading it as a fitting ceiling is wrong — 2048x2 reaches train 65.7%
> by epoch 18 and 1024x2 reaches 77.4%, both still rising.

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
3.68, 4.44 — which says the nets overfit rather than underfit, and is consistent
with data being the binding constraint. Consistent with, not proof of: the gap
alone cannot distinguish "more data would help" from "the target is only
partly predictable from this encoding", and validation sits ~32 points below the
88.2% learnable ceiling measured below. A learning curve over dataset fraction
(`examples/learning_curve.rs`, fixed validation set, fractions sampled by whole
game) is the direct test and is not yet run. Larger nets
peak earlier and then decay for as long as they are allowed to run (2048x2 at
epoch 6, 160x1 at epoch 133), which is why a fixed epoch budget mismeasures
them: at a 22-epoch cap, 160x1 records ~54% and looks too small when it is
actually the joint best.

Capacity was never the limit on *fitting*: 1024x2 reached train 71.4% while its
validation fell to 53.3%. Lowering its learning rate to 3e-4 gained 0.7 points
of validation but made it memorise faster, so the falling-validation signature
is genuine overfitting and not an untuned step size.

## What is learnable from a depth-2 teacher

Measured independently by the exhaustive-search generator (azul-eval-ce), two
runs at different node budgets. **Regime:** ~260 exact positions from 5 games
each, positions within a game correlated so true intervals are wider than the
binomial ones shown; `random_best` off, so ties are broken deterministically
rather than by coin flip.

| depth | agrees with exact | mean value lost per decision |
|---|---|---|
| 1 | 39% | 1.67 |
| 2 | 45% | 1.30 |
| 3 | 51% | 1.16 |
| 4 | 59% | 0.82 |
| 6 | 72% | 0.45 |

**Learnable ceiling against depth-2 targets: 88.2% [84.3, 92.1]** — 22% of
positions carry a tie at depth 2's own top value, and a net that ranks perfectly
still scores 1/k on those. So the 56.41 validation agreement above is **31.8
points** below what is learnable, and ties do not explain it.

That ceiling is robust to the one bias we knew about. Raising the node budget
from 3M to 20M cut excluded (budget-exceeded) positions from 11.6% to 4.0%, and
the ceiling did *not* fall — it rose slightly, 86.5% to 88.2%, well inside both
intervals. So large-tree positions do not carry more tie mass, contrary to the
obvious guess; plausibly a large tree means a tactically live position where
exact search finds real distinctions, while a small tree means a forced one.

Note this bounds *imitability*, not quality: **depth 2 concedes ~1.3 points of
value per decision** against a solved round, over ~50 decisions a game. A net
can imitate it faithfully and still inherit a policy that gives away real value,
which is consistent with fine-tuning being where the strength comes from.

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

## Is the clone data-limited? Yes, at a known and slow exchange rate

`examples/learning_curve.rs`, 320x1, fractions sampled by whole game against one
**fixed** validation set of 4,000 games, shuffled loop:

| fraction | games | positions | train | val | gap | epochs |
|---|---|---|---|---|---|---|
| 0.10 | 3,600 | 191,658 | 51.85 | 44.46 | 7.39 | 7 |
| 0.25 | 9,000 | 478,982 | 57.95 | 49.35 | 8.60 | 14 |
| 0.50 | 18,000 | 958,386 | 58.94 | 53.26 | 5.68 | 24 |
| 0.75 | 27,000 | 1,437,262 | 60.92 | 55.88 | 5.05 | 53 |
| 1.00 | 36,000 | 1,916,558 | 61.64 | **57.72** | 3.91 | 87 |

**The curve does not flatten.** Slopes per doubling of games: 3.70, 3.91, 4.48,
**4.39** — the top segment is as steep as the middle, so there is no data ceiling
in sight.

**But the exchange rate makes data a slow lever.** At +4.39 points per doubling,
closing the 30.5-point gap to the 88.2% learnable ceiling needs **7 doublings —
124x the current 36,000 games, about 4.5M of them.** Concretely: an available
74,141-game set is ~1.04 doublings and should be worth **~+4.6 points**. Real,
and worth ingesting; nowhere near sufficient alone.

So "the data constrains the model" is right in direction. Read as "get more data
and you are done" it is badly wrong about magnitude.

Note the epoch column: 7 -> 87 as data grows. A fixed epoch budget mismeasures
every cell of a curve like this, and the same is true of the capacity sweep.

### What explains the gap to the learnable ceiling

| hypothesis | verdict |
|---|---|
| Ties in the teacher's targets | **No.** The 88.2% ceiling already accounts for them; 22% incidence, mean k~2.2 |
| Correlated batches (no shuffling) | **Small.** +1.28 points, ~4% of the gap |
| Data volume | **Yes, but slow.** +4.39/doubling; 124x more games to close it |

## The trainer specialises to seat 0

`TrainOptions::default()` uses `SeatMode::First` — "always seat 0, always moving
first" — and it shows, at 1000 games/seat:

| model | seat 0 | seat 1 | gap |
|---|---|---|---|
| clone 320x1 | 36.7 | 38.5 | -1.8 |
| clone 2048x2 | 30.3 | 36.2 | -5.9 |
| ft_320x1 | 66.8 | 63.7 | **+3.1** |
| ft_240x1 | 62.8 | 57.5 | **+5.3** |
| ft_160x1 | 62.2 | 57.8 | **+4.4** |

Clones carry a seat-0 *disadvantage* of about the size the game's own asymmetry
predicts (`mm_vs_mm` depth2-vs-depth2: seat0 margin -1.50). Fine-tuning flips
the sign, so PPO adds a seat-0 advantage of its own — a +6 to +10 point swing.
The cause is visitation, not coverage: the encoding is mover-relative and the
seat index is never encoded, so the labels cover both seats' position classes
(verified — the opening position is bit-identical between seat views), but an
agent that only ever occupies seat 0 only ever visits seat-0-reachable states.

The default's in-code justification cites `Immediate + First` 17% against
`Immediate + Alternate` 7%. **Regime: those were measured at 7-17% win rates**,
where splitting a scarce episode budget across seats is clearly bad. At 65% it
is not clearly bad, and "transfers to seat 1 on its own" was a 2-point leak on a
weak model against 3-5 points on a strong one.

All comparisons in this document used `SeatMode::First`, so rankings are
unaffected; the absolute numbers leave roughly 1.5-2.5 points of mean win rate
unclaimed. `SeatMode::Alternate` is a one-line change.

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
