use crate::gamestate;
use crate::playerboard::wall::{cell_index, score_tile_mask};
use log::debug;
use minimaxer::{self, negamax::SearchOptions, node::Node, Evaluate};

use super::Player;

impl minimaxer::Gamestate<gamestate::Move> for gamestate::Gamestate<2, 6> {
    fn get_moves(&mut self) -> Vec<gamestate::Move> {
        gamestate::Gamestate::get_moves(self)
    }

    fn play_move(&mut self, m: &gamestate::Move) {
        gamestate::Gamestate::play_move(self, *m);
    }

    fn position_key(&self) -> u64 {
        gamestate::Gamestate::position_key(self)
    }

    fn is_terminal(&mut self) -> bool {
        gamestate::Gamestate::is_round_over(self)
    }

    fn player_aim(&self) -> minimaxer::NodeAim {
        match self.current_player() {
            0 => minimaxer::NodeAim::Maximise,
            1 => minimaxer::NodeAim::Minimise,
            _ => panic!("Invalid player"),
        }
    }
}

impl minimaxer::Move for gamestate::Move {}

#[derive(Debug, Clone)]
pub struct ScoreEvaluator;

impl minimaxer::Evaluate<gamestate::Gamestate<2, 6>> for ScoreEvaluator {
    fn evaluate(&mut self, g: &gamestate::Gamestate<2, 6>) -> f32 {
        g.differential_predicted_score()
    }
}

/// Fixed basis for the centre weighting feature. Tuning scales this whole
/// table by one scalar rather than moving 25 numbers independently.
const CENTRE_WEIGHTS: [[f32; 5]; 5] = [
    [0.9, 0.95, 0.97, 0.95, 0.9],
    [0.95, 0.97, 1.0, 0.97, 0.95],
    [0.9, 0.95, 0.97, 0.95, 0.9],
    [0.85, 0.9, 0.95, 0.9, 0.85],
    [0.8, 0.85, 0.9, 0.85, 0.8],
];

/// Number of terms in the evaluation feature vector.
/// Base terms, before they are crossed with rounds remaining.
pub const N_BASE: usize = 7;

/// Every base term appears twice: on its own, and multiplied by the fraction of
/// the game still to play. One weight vector otherwise has to serve round one
/// and round ten alike, and no vector can, because the truth differs: a
/// partially filled line in the last round is worth exactly nothing, there
/// being no next round to complete it in.
///
/// Crossing keeps the evaluation linear in its parameters, so least squares
/// still fits it, and costs almost nothing: the base terms are computed once
/// and the dot product goes from 7 multiply-adds to 14.
pub const N_FEATURES: usize = N_BASE * 2;

/// Index of the first forecast bucket. The four buckets hold lines needing one
/// more tile through lines needing four.
const FORECAST_BASE: usize = 3;
const FORECAST_END: usize = 7;

/// Most future rounds there can be, which is what rounds remaining is scaled
/// by. A wall row completing ends the game, and no row starts with any tiles,
/// so five is the cap however many rounds the counter has left.
const MAX_FUTURE_ROUNDS: f32 = 5.0;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "score",
    "first_player",
    "centre",
    "forecast_missing_1",
    "forecast_missing_2",
    "forecast_missing_3",
    "forecast_missing_4",
    "score_x_rounds",
    "first_player_x_rounds",
    "centre_x_rounds",
    "forecast_missing_1_x_rounds",
    "forecast_missing_2_x_rounds",
    "forecast_missing_3_x_rounds",
    "forecast_missing_4_x_rounds",
];

/// Differential feature vector for a position, player 0 minus player 1.
///
/// Every term is a lead rather than a standing, because negamax negates the
/// evaluation wholesale for the minimising player. The evaluation itself is the
/// dot product of this with a weight vector, which keeps it linear in its
/// parameters and so fittable by least squares.
pub fn features(g: &gamestate::Gamestate<2, 6>) -> [f32; N_FEATURES] {
    features_with(g, true, true, true)
}

/// As [`features`], but skipping terms whose weights are zero.
///
/// Zeroing a weight leaves its work in place, which is invisible at fixed depth
/// and wrong at fixed time: an ablation has to charge a term for the nodes it
/// costs, not just neutralise its output.
pub fn features_with(
    g: &gamestate::Gamestate<2, 6>,
    centre: bool,
    forecast: bool,
    rounds: bool,
) -> [f32; N_FEATURES] {
    let mut f = [0.0; N_FEATURES];
    f[0] = g.differential_predicted_score();
    f[1] = if g.boards()[0].first_player_tile {
        1.0
    } else if g.boards()[1].first_player_tile {
        -1.0
    } else {
        0.0
    };

    // How full the fullest wall row will be once this round's lines are
    // placed. A completed row ends the game, so this bounds how much game is
    // left just as surely as the round counter does.
    let mut fullest_row = 0u8;

    // Everything below reads the wall, so if nothing wants it there is
    // nothing left to do. Without this a score-only evaluator pays for wall
    // work per leaf that it never looks at, inflating its node cost and
    // flattering whatever is measured against it.
    if !centre && !forecast && !rounds {
        return f;
    }

    for (i, board) in g.boards().iter().enumerate() {
        let sign = if i == 0 { 1.0 } else { -1.0 };
        // Occupancy the wall will have once this round's full lines land.
        // Nothing here needs the colours, only which cells are taken, so this
        // replaces a copy of the whole wall with one integer.
        let mut mask = board.projected_mask();

        if centre || rounds {
            for r in 0..5usize {
                let bits = (mask >> (r * 5)) & 0x1f;
                if centre {
                    // Cell by cell in the original order: a table of per row
                    // sums would be quicker still, but re-associating the
                    // addition changes the last bits of the result and with
                    // them, occasionally, the move chosen.
                    for c in 0..5usize {
                        if (bits >> c) & 1 == 1 {
                            f[2] += sign * CENTRE_WEIGHTS[r][c];
                        }
                    }
                }
                if rounds {
                    fullest_row = fullest_row.max(bits.count_ones() as u8);
                }
            }
        }
        if !forecast {
            continue;
        }

        // Partially filled lines are the asset carried into the next round, and
        // nothing else in the evaluation sees them. Credit each with what it
        // would score once completed, bucketed by how many tiles it still
        // needs, so tuning can learn the discount curve instead of us guessing
        // a formula for it.
        //
        // Placement is cumulative, as in the TypeScript original: two lines
        // that would land next to each other each see the other, which is
        // optimistic but does capture the pairing.
        for (row_ind, line) in board.row_iter() {
            let capacity = row_ind.capacity();
            let count = line.count();
            if count == 0 || count >= capacity {
                continue;
            }
            let Some(tile) = line.tile() else { continue };
            let missing = usize::from(capacity - count);
            let bit = cell_index(row_ind, tile);
            f[FORECAST_BASE + missing - 1] += sign * f32::from(score_tile_mask(mask, bit));
            mask |= 1 << bit;
        }
    }

    if rounds {
        let by_counter = 10i32 - i32::from(g.round());
        let by_row = 5i32 - i32::from(fullest_row);
        let rounds_left = by_counter.min(by_row).max(0) as f32 / MAX_FUTURE_ROUNDS;
        for i in 0..N_BASE {
            f[N_BASE + i] = f[i] * rounds_left;
        }
    }
    f
}

/// Weights applied to [`features`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Weights(pub [f32; N_FEATURES]);

impl Default for Weights {
    /// Fitted by least squares on 15,000 round end positions labelled with the
    /// final score margin of the game they came from, over the base terms and
    /// their crossings with rounds remaining.
    ///
    /// Beats the previous round-blind vector by 56.9% at depth 3 and 59.5%
    /// under a 5ms clock, for a node ratio of 0.983: the crossings cost about
    /// 0.2 points at the measured 8.0 points per halving, which is as close to
    /// free as a feature gets.
    ///
    /// The crossed weights are the interesting part. Centre weighting is worth
    /// 6.74 with the game ahead of it and 0.45 in the last round, a fifteenfold
    /// spread that a single weight had been averaging over. Every forecast
    /// bucket changes sign across the game, positive early and slightly
    /// negative at the end, because a partial line is dead weight once there is
    /// no round left to finish it in. And the score differential itself is
    /// worth 1.0 in the last round against 0.5 in the first, early leads being
    /// the less decisive.
    fn default() -> Self {
        Self([
            1.0, // score
            0.79598224, // first_player
            0.44542772, // centre
            -0.063497774, // forecast_missing_1
            -0.05389085, // forecast_missing_2
            -0.20389807, // forecast_missing_3
            -0.29097763, // forecast_missing_4
            -0.5007711, // score x rounds
            0.29502016, // first_player x rounds
            6.2921076, // centre x rounds
            1.5160196, // forecast_missing_1 x rounds
            0.841121, // forecast_missing_2 x rounds
            1.0340607, // forecast_missing_3 x rounds
            0.393935, // forecast_missing_4 x rounds
        ])
    }
}

impl Weights {
    /// The hand set vector this started from: score in points, first player
    /// tile at 0.5, centre weighting at 1.0, no forecast term.
    ///
    /// Kept as the baseline the fitted default is measured against, and so that
    /// constructors wanting the original term set do not silently follow
    /// [`Default`] when it is retuned.
    pub fn hand_set() -> Self {
        Self([1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// The forecast bucket values implied by the TypeScript evaluation at round
    /// one, which is where the ported term started.
    ///
    /// Kept for ablation, not because they are good: two of the four have the
    /// wrong sign, and 1/(missing+0.5) cannot express a negative. With these
    /// values the term measured dead neutral at depth 3 while costing 10% of
    /// nodes; with the fitted values it is worth about 3 points.
    pub const TS_FORECAST: [f32; 4] = [0.4, 0.24, 0.17, 0.13];

    /// Turn the forecast term on with its original hand set values.
    pub fn with_ts_forecast(mut self) -> Self {
        self.0[FORECAST_BASE..FORECAST_END].copy_from_slice(&Self::TS_FORECAST);
        self
    }

    /// Zero the forecast buckets. [`HeuristicEvaluator::new`] then skips the
    /// partial line scan rather than merely muting it, so this really is the
    /// cheaper evaluator.
    pub fn without_forecast(mut self) -> Self {
        for w in &mut self.0[FORECAST_BASE..FORECAST_END] {
            *w = 0.0;
        }
        for w in &mut self.0[N_BASE + FORECAST_BASE..N_BASE + FORECAST_END] {
            *w = 0.0;
        }
        self
    }

    /// Rescale so the score term weighs exactly one point.
    ///
    /// Only the ratios between weights affect move choice, so this is free, and
    /// it leaves the evaluation readable in points.
    pub fn normalised(mut self) -> Self {
        let score = self.0[0];
        if score.abs() > f32::EPSILON {
            for w in &mut self.0 {
                *w /= score;
            }
        }
        self
    }
}

// Evaluate based on score and other heuristics
#[derive(Debug, Clone)]
pub struct HeuristicEvaluator {
    weights: Weights,
    centre: bool,
    forecast: bool,
    rounds: bool,
}

impl HeuristicEvaluator {
    pub fn new(weights: Weights) -> Self {
        Self {
            centre: weights.0[2] != 0.0 || weights.0[N_BASE + 2] != 0.0,
            rounds: weights.0[N_BASE..].iter().any(|&w| w != 0.0),
            forecast: weights.0[FORECAST_BASE..FORECAST_END]
                .iter()
                .chain(weights.0[N_BASE + FORECAST_BASE..N_BASE + FORECAST_END].iter())
                .any(|&w| w != 0.0),
            weights,
        }
    }

    pub fn new_no_wall_weight(fp_weight: f32) -> Self {
        let mut weights = Weights::hand_set();
        weights.0[1] = fp_weight;
        weights.0[2] = 0.0;
        Self::new(weights)
    }

    /// The term set as it stood before the forecast buckets were added: score,
    /// first player tile and centre weighting, with the wall term differential
    /// rather than one sided.
    ///
    /// Exists so a before/after on the wall term bug measures only that bug,
    /// rather than the bug fix, the forecast term and the retuned weights
    /// summed together.
    pub fn new_pre_forecast() -> Self {
        Self::new(Weights::hand_set())
    }

    pub fn weights(&self) -> Weights {
        self.weights
    }
}

impl Default for HeuristicEvaluator {
    fn default() -> Self {
        Self::new(Weights::default())
    }
}

impl minimaxer::Evaluate<gamestate::Gamestate<2, 6>> for HeuristicEvaluator {
    fn evaluate(&mut self, g: &gamestate::Gamestate<2, 6>) -> f32 {
        features_with(g, self.centre, self.forecast, self.rounds)
            .iter()
            .zip(self.weights.0.iter())
            .map(|(x, w)| x * w)
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct Minimaxer<E> {
    pub opts: minimaxer::negamax::SearchOptions,
    pub name: String,
    pub evaluator: E,
    /// Nodes and plies actually achieved, summed over every move picked.
    ///
    /// Under a clock a dearer evaluation buys fewer nodes, and the two sides of
    /// a match do not give up the same amount. Reporting this alongside the win
    /// rate is what distinguishes "this term is not worth its cost" from "the
    /// machine was busy".
    pub nodes: u64,
    pub searches: u64,
    pub depth_total: u64,
}

impl<E> Minimaxer<E> {
    pub fn new(
        opts: minimaxer::negamax::SearchOptions,
        name: impl Into<String>,
        evaluator: E,
    ) -> Self {
        Self {
            opts,
            name: name.into(),
            evaluator,
            nodes: 0,
            searches: 0,
            depth_total: 0,
        }
    }
}

impl<E: Evaluate<gamestate::Gamestate<2, 6>>> Player<2, 6> for Minimaxer<E> {
    fn pick_move(
        &mut self,
        gamestate: &gamestate::Gamestate<2, 6>,
        moves: Vec<gamestate::Move>,
    ) -> gamestate::Move {
        let mut n = minimaxer::negamax::Negamax::new(
            Node::new(gamestate.clone()),
            self.evaluator.clone(),
            self.opts,
        );
        let result = n.search();
        debug!("Minimax search result: {:?}", result);
        self.nodes += u64::from(result.nodes);
        self.depth_total += u64::from(result.depth);
        self.searches += 1;
        result.best
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
