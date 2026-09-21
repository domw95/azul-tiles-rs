use crate::gamestate;
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
pub const N_FEATURES: usize = 7;

/// Index of the first forecast bucket. The four buckets hold lines needing one
/// more tile through lines needing four.
const FORECAST_BASE: usize = 3;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "score",
    "first_player",
    "centre",
    "forecast_missing_1",
    "forecast_missing_2",
    "forecast_missing_3",
    "forecast_missing_4",
];

/// Differential feature vector for a position, player 0 minus player 1.
///
/// Every term is a lead rather than a standing, because negamax negates the
/// evaluation wholesale for the minimising player. The evaluation itself is the
/// dot product of this with a weight vector, which keeps it linear in its
/// parameters and so fittable by least squares.
pub fn features(g: &gamestate::Gamestate<2, 6>) -> [f32; N_FEATURES] {
    features_with(g, true, true)
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

    // Both remaining terms need the simulated wall, so if neither is wanted
    // there is nothing left to do.
    if !centre && !forecast {
        return f;
    }

    for (i, board) in g.boards().iter().enumerate() {
        let sign = if i == 0 { 1.0 } else { -1.0 };
        // The wall as it will stand once this round's full lines are placed.
        let mut wall = board.simulate_wall();
        if centre {
            for (row, weight) in wall.iter().zip(CENTRE_WEIGHTS.iter()) {
                for (tile, &w) in row.iter().zip(weight.iter()) {
                    if tile.is_some() {
                        f[2] += sign * w;
                    }
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
            let scored = f32::from(wall.place_and_score_tile(row_ind, tile));
            f[FORECAST_BASE + missing - 1] += sign * scored;
        }
    }
    f
}

/// Weights applied to [`features`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Weights(pub [f32; N_FEATURES]);

impl Default for Weights {
    /// Fitted by least squares on 15,000 round end positions labelled with the
    /// final score margin of the game they came from, then rescaled so the
    /// score term weighs one point.
    ///
    /// Beats the hand set vector by 6.2 points over 1800 games at depth 3, two
    /// independent runs agreeing to 0.5, and by a predicted 5.4 under a clock
    /// once its node cost is charged at the measured 8.0 points per halving.
    /// Fixed time runs on the machine this was measured on have a run to run
    /// spread of +/-5, so they could neither confirm nor refute that; the depth
    /// 3 screen reproduces to 0.6 and is what this rests on.
    ///
    /// Two of the forecast buckets are negative, which the hand set values
    /// could not express: a line still needing three or four tiles is a
    /// liability, not a discounted asset.
    fn default() -> Self {
        Self([
            1.0,
            1.4792728,
            2.1410458,
            0.3069138,
            0.10572213,
            -0.11700917,
            -0.29516673,
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
        Self([1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
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
        self.0[FORECAST_BASE..].copy_from_slice(&Self::TS_FORECAST);
        self
    }

    /// Zero the forecast buckets. [`HeuristicEvaluator::new`] then skips the
    /// partial line scan rather than merely muting it, so this really is the
    /// cheaper evaluator.
    pub fn without_forecast(mut self) -> Self {
        for w in &mut self.0[FORECAST_BASE..] {
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
}

impl HeuristicEvaluator {
    pub fn new(weights: Weights) -> Self {
        Self {
            centre: weights.0[2] != 0.0,
            forecast: weights.0[FORECAST_BASE..].iter().any(|&w| w != 0.0),
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
        features_with(g, self.centre, self.forecast)
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
