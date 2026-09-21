use std::cmp::Reverse;

use nalgebra::SMatrix;
use rand_distr::{Distribution, StandardNormal};
use strum::IntoEnumIterator;

use crate::{
    gamestate::{Gamestate, Move},
    playerboard::{wall::Wall, PlayerBoard},
    tiles::{Tile, TileGroup},
};

use super::{EvolvingPlayer, Player};

/// Number of factory slots, including the centre pile at index 0.
pub const FACTORY_SLOTS: usize = 6;

/// Number of factory displays, i.e. the slots that are interchangeable.
pub const DISPLAYS: usize = FACTORY_SLOTS - 1;

/// Length of one player's board block. See [`pb_to_array`].
///
/// ```text
/// pattern rows        5 * 6 = 30   colour one-hot + fill fraction
/// tiles still needed  5     =  5   absolute, not a fraction
/// wall occupancy      5 * 5 = 25
/// horizontal runs     5 * 5 = 25
/// vertical runs       5 * 5 = 25
/// legal colours       5 * 5 = 25
/// floor                      1
/// first player tile          1
/// score                      1
/// predicted score            1
///                          ---
///                          139
/// ```
pub const BOARD_SIZE: usize = 139;

/// Length of the vector produced by [`gs_to_array_for`].
///
/// ```text
/// boards    139 * 2 = 278   acting player first
/// centre               5    factories[0], on its own scale
/// displays    5 * 5 = 25    factories[1..], in canonical order
/// bag                  5
/// available            5    every factory and the centre, per colour
/// score lead           1    from the acting player's seat
/// fp tile              1
/// round                1
///                    ---
///                    321
/// ```
pub const ENCODED_SIZE: usize = 2 * BOARD_SIZE + 43;

/// The encoded gamestate.
pub type StateVector = SMatrix<f32, ENCODED_SIZE, 1>;

/// One player's board.
pub type BoardVector = SMatrix<f32, BOARD_SIZE, 1>;

/// Tiles dealt to a factory display, and so the largest count one can hold.
const TILES_PER_DISPLAY: f32 = 4.0;

/// Capacity of the longest pattern row, and the length of a full wall line.
const ROW_SCALE: f32 = 5.0;

/// Spread of a predicted score, and of a lead over the opponent.
const SCORE_SCALE: f32 = 100.0;

/// Plausible bound for one colour in the centre pile.
///
/// The centre is not a display: it accumulates everything not taken during a
/// round, so a single colour there can reach double figures. A round deals 20
/// tiles in a two player game and the bag holds only 20 of each colour, so 20
/// is the hard ceiling and 10 is a typical busy centre. Dividing the centre by
/// a display's capacity, as the encoding used to, let one feature run to ~5
/// while the identically shaped slots next to it stayed under 1.
const CENTRE_SCALE: f32 = 10.0;

/// Tiles of each colour in a full bag.
const TILES_PER_COLOUR: f32 = 20.0;

/// Bijection between real factory slots and the canonical order they are
/// encoded in.
///
/// Azul's factory displays are interchangeable: permuting them yields an
/// isomorphic position. Encoded in raw slot order, the network has to learn
/// five separate copies of the same concept and cannot transfer anything it
/// learns about display 1 to display 3. Sorting the displays by their contents
/// before encoding collapses those five copies into one.
///
/// The centre pile stays pinned at slot 0. It is not interchangeable with a
/// display: it is the only source that grows during a round, and it is the
/// only one that can carry the first player tile.
///
/// Reordering the inputs moves the action space with them -- an action index
/// `source * 30 + tile * 6 + destination` names a *canonical* source once the
/// state is canonicalised. Every crossing of that boundary must go through
/// [`Self::canonical_index`] (real to canonical, when building an action mask
/// or a training label) or [`Self::real_move_parts`] (canonical to real, when
/// decoding a chosen action). Get one direction wrong and the agent plays a
/// legal-looking move from the wrong display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactoryOrder {
    /// `canon_to_real[c]` is the real slot encoded at canonical position `c`.
    canon_to_real: [u8; FACTORY_SLOTS],
    /// `real_to_canon[s]` is the canonical position of real slot `s`.
    real_to_canon: [u8; FACTORY_SLOTS],
}

impl FactoryOrder {
    /// The canonical order for `gs`: displays sorted by tile counts, descending.
    pub fn canonical(gs: &Gamestate<2, FACTORY_SLOTS>) -> Self {
        Self::from_factories(gs.factories())
    }

    fn from_factories(factories: &[Option<TileGroup>; FACTORY_SLOTS]) -> Self {
        // Descending by the five counts, lexicographically, with the slot index
        // as a tie break so the result is a deterministic function of the
        // position alone. Two displays holding the same tiles are genuinely
        // interchangeable, so which way a tie falls cannot change the encoding;
        // it only decides which of two identical slots a decoded action names,
        // and either is legal and plays the same tiles.
        let mut displays = [1usize, 2, 3, 4, 5];
        displays.sort_by_key(|&slot| (Reverse(slot_counts(&factories[slot])), slot));

        let mut canon_to_real = [0u8; FACTORY_SLOTS];
        let mut real_to_canon = [0u8; FACTORY_SLOTS];
        for (canonical, &real) in displays.iter().enumerate() {
            canon_to_real[canonical + 1] = real as u8;
            real_to_canon[real] = canonical as u8 + 1;
        }
        Self {
            canon_to_real,
            real_to_canon,
        }
    }

    /// The mapping that reorders nothing.
    pub fn identity() -> Self {
        let slots = [0, 1, 2, 3, 4, 5];
        Self {
            canon_to_real: slots,
            real_to_canon: slots,
        }
    }

    /// The real factory slot encoded at canonical position `canonical`.
    pub fn real_slot(&self, canonical: usize) -> usize {
        self.canon_to_real[canonical] as usize
    }

    /// The canonical position of real factory slot `real`.
    pub fn canonical_slot(&self, real: usize) -> usize {
        self.real_to_canon[real] as usize
    }

    /// The action index `m` occupies once the state has been canonicalised.
    ///
    /// The canonical counterpart of [`Move::to_index`].
    pub fn canonical_index(&self, m: &Move) -> usize {
        self.canonical_slot(usize::from(m.source)) * 30
            + (m.tile as usize) * 6
            + usize::from(m.destination)
    }

    /// Decode a canonical action index into real `(source, tile, destination)`.
    pub fn real_move_parts(&self, index: usize) -> (usize, usize, usize) {
        let (canonical_source, tile, destination) = index_to_move(index);
        (self.real_slot(canonical_source), tile, destination)
    }
}

/// Tile counts of a factory slot, treating an emptied slot as no tiles.
fn slot_counts(factory: &Option<TileGroup>) -> [u8; 5] {
    factory.map_or([0; 5], |f| *f.counts())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoveSelectNN {
    weights_1: SMatrix<f32, 180, ENCODED_SIZE>,
    bias_1: SMatrix<f32, 180, 1>,
    weights_2: SMatrix<f32, 180, 180>,
    bias_2: SMatrix<f32, 180, 1>,
}

impl MoveSelectNN {
    pub fn new_random() -> Self {
        let d = StandardNormal;
        let mut rng = rand::thread_rng();
        let weights_1: SMatrix<f32, 180, ENCODED_SIZE> =
            SMatrix::from_distribution(&d, &mut rng);
        let bias_1: SMatrix<f32, 180, 1> = SMatrix::from_distribution(&d, &mut rng);
        let weights_2: SMatrix<f32, 180, 180> = SMatrix::from_distribution(&d, &mut rng);
        let bias_2: SMatrix<f32, 180, 1> = SMatrix::from_distribution(&d, &mut rng);

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
        }
    }
}

impl Player<2, 6> for MoveSelectNN {
    fn pick_move(&mut self, gamestate: &Gamestate<2, 6>, moves: Vec<Move>) -> Move {
        // The encoding sorts the factory displays, so the output index names a
        // canonical source. Decode through the same ordering that encoded it.
        let order = FactoryOrder::canonical(gamestate);
        let input = gs_to_array_ordered(gamestate, 0, &order);
        // calculate hidden layer
        let hidden = self.weights_1 * input + self.bias_1;
        // calculate output layer
        let hidden = hidden.map(|x| x.tanh());
        let output = self.weights_2 * hidden + self.bias_2;

        // find the best move
        // sort output with index
        let mut output = output.into_iter().enumerate().collect::<Vec<_>>();
        output.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        // convert moves to hashmap
        let moves = moves
            .into_iter()
            .map(|m| {
                (
                    (
                        usize::from(m.source),
                        usize::from(m.tile),
                        usize::from(m.destination),
                    ),
                    m,
                )
            })
            .collect::<fxhash::FxHashMap<_, _>>();
        // find the first move that is valid
        for (i, _) in output {
            // construct move source -> tile -> destination

            if let Some(m) = moves.get(&order.real_move_parts(i)) {
                return *m;
            }
        }

        unreachable!()
    }

    fn name(&self) -> String {
        "MoveSelectNN".into()
    }
}

pub fn index_to_move(index: usize) -> (usize, usize, usize) {
    let source = index / 30;
    let tile = (index % 30) / 6;
    let dest = index % 6;
    (source, tile, dest)
}

impl EvolvingPlayer for MoveSelectNN {
    fn birth() -> Self {
        Self::new_random()
    }

    fn mutate(&self, prob: rand_distr::Bernoulli, rng: &mut rand::rngs::SmallRng) -> Self {
        let weights_1 = self.weights_1.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });

        let bias_1 = self.bias_1.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });
        let weights_2 = self.weights_2.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });

        let bias_2 = self.bias_2.map(|w| {
            if prob.sample(rng) {
                let a: f32 = rand_distr::StandardNormal.sample(rng);
                w + a / 5.0
            } else {
                w
            }
        });

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
        }
    }

    fn crossover(&self, other: &Self, prob: rand_distr::Bernoulli) -> Self {
        let weights_1 = self.weights_1.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.weights_1[(r, c)]
            }
        });
        let bias_1 = self.bias_1.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.bias_1[(r, c)]
            }
        });

        let weights_2 = self.weights_2.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.weights_2[(r, c)]
            }
        });

        let bias_2 = self.bias_2.map_with_location(|r, c, a| {
            if prob.sample(&mut rand::thread_rng()) {
                a
            } else {
                other.bias_2[(r, c)]
            }
        });

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
        }
    }
}

/// Encode the gamestate from player 0's point of view.
pub fn gs_to_array(gs: &Gamestate<2, 6>) -> StateVector {
    gs_to_array_for(gs, 0)
}

/// Encode the gamestate from `player`'s point of view.
///
/// The acting player's board always goes first. Without this the network reads
/// board 0 as "mine" whoever is actually to move, so a model trained in seat 0
/// and then seated at index 1 -- which is exactly what the GUI does -- plans
/// using its opponent's board.
pub fn gs_to_array_for(gs: &Gamestate<2, 6>, player: usize) -> StateVector {
    gs_to_array_ordered(gs, player, &FactoryOrder::canonical(gs))
}

/// Encode the gamestate with an explicit factory ordering.
///
/// Callers that also have to map action indices build the ordering once and
/// pass it in, rather than sorting twice and trusting the two to agree.
pub fn gs_to_array_ordered(
    gs: &Gamestate<2, 6>,
    player: usize,
    order: &FactoryOrder,
) -> StateVector {
    let mut arr = SMatrix::zeros();
    let a = pb_to_array(&gs.boards()[player]);
    let b = pb_to_array(&gs.boards()[1 - player]);
    let factories = gs.factories();

    // See ENCODED_SIZE for the layout. The centre is encoded separately from
    // the displays: same shape, different range, so it gets its own scale.
    let displays = (1..FACTORY_SLOTS)
        .flat_map(|canonical| display_to_array(&factories[order.real_slot(canonical)]));

    for (i, v) in a
        .into_iter()
        .copied()
        .chain(b.into_iter().copied())
        .chain(centre_to_array(&factories[order.real_slot(0)]))
        .chain(displays)
        .chain(bag_to_array(gs.tilebag()))
        .chain(available_to_array(factories))
        .chain([
            score_lead(gs, player),
            gs.first_player_tile() as u8 as f32,
            gs.round() as f32 / 5.0,
        ])
        .enumerate()
    {
        arr[(i, 0)] = v;
    }
    arr
}

/// The predicted score lead, from `player`'s point of view.
///
/// This is exactly the quantity the minimax teacher maximises, so it is the
/// single most relevant scalar in the position. Both predicted scores are
/// already encoded, but only as two numbers the network has to learn to
/// subtract.
///
/// [`Gamestate::differential_predicted_score`] is hardcoded as board 0 minus
/// board 1, while this encoding is always written from the acting player's
/// seat. Seat 1 therefore has to see it negated. Getting this backwards would
/// be silent -- the model would simply learn to play for its opponent -- so
/// `score_lead_is_from_the_players_own_seat` pins it down.
fn score_lead(gs: &Gamestate<2, 6>, player: usize) -> f32 {
    let lead = gs.differential_predicted_score();
    let from_seat = if player == 0 { lead } else { -lead };
    from_seat / SCORE_SCALE
}

/// Tiles of each colour available anywhere this turn: every display plus the
/// centre.
///
/// Derivable by summing six separate groups, which is precisely why it is
/// worth handing over. Bounded by the 20 tiles of each colour in a full bag.
fn available_to_array(factories: &[Option<TileGroup>; FACTORY_SLOTS]) -> [f32; 5] {
    let mut totals = [0u16; 5];
    for factory in factories {
        for (colour, &count) in slot_counts(factory).iter().enumerate() {
            totals[colour] += u16::from(count);
        }
    }
    totals.map(|t| f32::from(t) / TILES_PER_COLOUR)
}

/// A display holds at most four tiles, so these features top out at 1.0.
fn display_to_array(factory: &Option<TileGroup>) -> [f32; 5] {
    slot_counts(factory).map(|v| f32::from(v) / TILES_PER_DISPLAY)
}

/// The centre pile, on its own scale. See [`CENTRE_SCALE`].
fn centre_to_array(centre: &Option<TileGroup>) -> [f32; 5] {
    slot_counts(centre).map(|v| f32::from(v) / CENTRE_SCALE)
}

/// What is left to be drawn, one feature per colour.
///
/// Without this the agent cannot tell a colour that is exhausted from one that
/// is about to flood the next round, which is most of what makes holding a
/// partially filled row a good or a terrible idea.
fn bag_to_array(bag: &TileGroup) -> [f32; 5] {
    bag.counts().map(|v| f32::from(v) / TILES_PER_COLOUR)
}

/// Encode one player's board. See [`BOARD_SIZE`] for the layout.
fn pb_to_array(pb: &PlayerBoard) -> BoardVector {
    let mut arr = BoardVector::zeros();
    for (i, v) in pb
        .row_iter()
        .flat_map(|(ind, row)| {
            let mut arr = [0.0; 6];

            if let Some(tile) = row.tile() {
                arr[tile as usize] = 1.0;
                arr[5] = row.count() as f32 / (ind.capacity() as f32);
            }
            arr
        })
        .chain(rows_remaining(pb))
        .chain(wall_to_array(&pb.wall).into_iter().copied())
        .chain(wall_runs(&pb.wall))
        .chain(legal_colours(pb))
        .chain([
            pb.floor.total().min(7) as f32 / 7.0,
            pb.first_player_tile as u8 as f32,
            pb.score as f32 / 100.0,
            // predicted_eval, not predicted_score: the latter saturates at
            // zero, so every board that is underwater encodes identically and
            // the network cannot tell a slight deficit from a disastrous one.
            // Master added the unclamped variant for exactly that reason.
            f32::from(pb.predicted_eval) / SCORE_SCALE,
        ])
        .enumerate()
    {
        arr[(i, 0)] = v;
    }

    arr
}

/// Tiles each pattern row still needs to complete, as an absolute count.
///
/// The fill fraction next to it normalises away the quantity planning actually
/// turns on: "this row needs exactly two more" is a different proposition in
/// row 2 and row 5, but both read 0.5. Dividing by a fixed constant rather
/// than the row's own capacity keeps the five rows on one scale.
fn rows_remaining(pb: &PlayerBoard) -> [f32; 5] {
    let mut out = [0.0; 5];
    for (i, (ind, row)) in pb.row_iter().enumerate() {
        out[i] = f32::from(ind.capacity() - row.count()) / ROW_SCALE;
    }
    out
}

/// Which colours may legally be placed in which pattern row, row major.
///
/// A hard constraint gating every move, and one the network otherwise has to
/// reconstruct: a colour's wall column is `(row + colour) % 5`, so deriving
/// this from the 25 occupancy bits means learning modular arithmetic first.
///
/// This encodes full legality, [`PlayerBoard::can_play_tile`], rather than
/// just the wall rule [`Wall::cell_available`]: it also covers a row already
/// holding a different colour, or already full. That subsumes the wall rule
/// and matches exactly what move generation will allow.
fn legal_colours(pb: &PlayerBoard) -> [f32; 25] {
    let mut out = [0.0; 25];
    for (i, (ind, _)) in pb.row_iter().enumerate() {
        for tile in Tile::iter() {
            if pb.can_play_tile(ind, tile, 1).is_some() {
                out[i * 5 + tile as usize] = 1.0;
            }
        }
    }
    out
}

/// Length of the horizontal and vertical run each filled wall cell belongs to,
/// row major: 25 horizontal then 25 vertical, 0 for an empty cell.
///
/// Azul scores contiguous runs, and a flat occupancy vector hides adjacency
/// completely -- two tiles side by side and two at opposite ends of a row look
/// identical to a linear layer. Kept per cell rather than collapsed to a
/// per-row and per-column maximum because where a run sits decides what
/// extends it; the model is underfitting, so input width is not the binding
/// constraint.
fn wall_runs(wall: &Wall) -> [f32; 50] {
    let mut filled = [[false; 5]; 5];
    for (r, row) in wall.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            filled[r][c] = cell.is_some();
        }
    }

    let mut out = [0.0; 50];
    for r in 0..5 {
        for c in 0..5 {
            if !filled[r][c] {
                continue;
            }
            let mut horizontal = 1;
            let mut i = c;
            while i > 0 && filled[r][i - 1] {
                horizontal += 1;
                i -= 1;
            }
            let mut i = c;
            while i < 4 && filled[r][i + 1] {
                horizontal += 1;
                i += 1;
            }

            let mut vertical = 1;
            let mut i = r;
            while i > 0 && filled[i - 1][c] {
                vertical += 1;
                i -= 1;
            }
            let mut i = r;
            while i < 4 && filled[i + 1][c] {
                vertical += 1;
                i += 1;
            }

            out[r * 5 + c] = horizontal as f32 / ROW_SCALE;
            out[25 + r * 5 + c] = vertical as f32 / ROW_SCALE;
        }
    }
    out
}

fn wall_to_array(wall: &Wall) -> SMatrix<f32, 25, 1> {
    let mut arr = SMatrix::zeros();
    for (i, row) in wall.iter().enumerate() {
        for (j, tile) in row.iter().enumerate() {
            arr[(i * 5 + j, 0)] = if tile.is_some() { 1.0 } else { 0.0 };
        }
    }
    arr
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::*;
    use crate::playerboard::wall::{RowIndex, WALL_COLOURS};

    /// Offsets within one board block. See [`BOARD_SIZE`].
    const REMAINING_AT: usize = 30;
    const WALL_AT: usize = REMAINING_AT + 5;
    const HRUN_AT: usize = WALL_AT + 25;
    const VRUN_AT: usize = HRUN_AT + 25;
    const LEGAL_AT: usize = VRUN_AT + 25;
    const PREDICTED_AT: usize = BOARD_SIZE - 1;

    /// Offsets of the global blocks. See [`ENCODED_SIZE`].
    const CENTRE_AT: usize = 2 * BOARD_SIZE;
    const DISPLAYS_AT: usize = CENTRE_AT + 5;
    const BAG_AT: usize = DISPLAYS_AT + 25;
    const AVAILABLE_AT: usize = BAG_AT + 5;
    const LEAD_AT: usize = AVAILABLE_AT + 5;
    const FP_AT: usize = LEAD_AT + 1;
    const ROUND_AT: usize = FP_AT + 1;

    /// A position part way through a round: the centre has filled up and some
    /// displays have been emptied, so the slots are genuinely unalike.
    ///
    /// Stops early if the round runs out rather than asserting, since how many
    /// plies a round lasts depends on the deal.
    fn mid_round(seed: u64, plies: usize) -> Gamestate<2, 6> {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        for _ in 0..plies {
            let moves = gs.get_moves();
            if moves.is_empty() {
                break;
            }
            gs.play_move(moves[moves.len() / 2]);
        }
        gs
    }

    /// A copy of `gs` whose displays have been rearranged: slot `i + 1` takes
    /// the contents of real slot `arrangement[i]`. The centre is untouched.
    fn rearranged(gs: &Gamestate<2, 6>, arrangement: [usize; DISPLAYS]) -> Gamestate<2, 6> {
        let mut sorted = arrangement;
        sorted.sort_unstable();
        assert_eq!(sorted, [1, 2, 3, 4, 5], "not a permutation of the displays");

        let original = *gs.factories();
        let mut out = gs.clone();
        for (slot, &from) in arrangement.iter().enumerate() {
            out.factories_mut()[slot + 1] = original[from];
        }
        out
    }

    #[test]
    fn move_from_index() {
        for i in 0..180 {
            let (s, t, d) = index_to_move(i);
            assert!(s < 6, "source {s} out of range at index {i}");
            assert!(t < 5, "tile {t} out of range at index {i}");
            assert!(d < 6, "destination {d} out of range at index {i}");
            // Must invert Move::to_index exactly, or the action mask and the
            // move it selects refer to different moves.
            assert_eq!(s * 30 + t * 6 + d, i);
        }
    }

    /// The encoding must not be able to tell two arrangements of the same
    /// displays apart. This is the whole point of canonicalising them.
    #[test]
    fn permuting_the_displays_leaves_the_encoding_unchanged() {
        // A position whose displays already all look the same would pass with
        // no canonicalisation at all, so count how many are worth something.
        let mut informative = 0;

        for (seed, plies) in [(7u64, 0usize), (7, 2), (11, 3), (23, 4), (31, 5)] {
            let gs = mid_round(seed, plies);
            let slots: Vec<[u8; 5]> = (1..FACTORY_SLOTS)
                .map(|s| slot_counts(&gs.factories()[s]))
                .collect();
            if slots.iter().any(|c| *c != slots[0]) {
                informative += 1;
            }

            let base = gs_to_array_for(&gs, 0);
            let base_seat_1 = gs_to_array_for(&gs, 1);
            for arrangement in [
                [5, 4, 3, 2, 1],
                [2, 3, 1, 5, 4],
                [3, 1, 4, 5, 2],
                [1, 2, 3, 5, 4],
            ] {
                let other = rearranged(&gs, arrangement);
                assert_eq!(
                    gs_to_array_for(&other, 0),
                    base,
                    "seed {seed}/{plies} plies, arrangement {arrangement:?}"
                );
                // The other seat is a separate path through the encoder.
                assert_eq!(gs_to_array_for(&other, 1), base_seat_1);
            }
        }

        assert!(
            informative >= 4,
            "only {informative} positions had displays that differ; the test \
             would pass without canonicalising"
        );
    }

    /// The centre is not one of the interchangeable slots: it is the only
    /// source that grows, and the only one carrying the first player tile.
    /// Swapping it with a display is a different position.
    #[test]
    fn the_centre_is_not_permutable() {
        let mut gs = mid_round(13, 4);
        let mut centre = TileGroup::new_empty();
        centre.add_tiles(Tile::Red, 6);
        let mut display = TileGroup::new_empty();
        display.add_tiles(Tile::Blue, 4);
        {
            let f = gs.factories_mut();
            f[0] = Some(centre);
            f[1] = Some(display);
        }

        let base = gs_to_array_for(&gs, 0);
        let mut swapped = gs.clone();
        swapped.factories_mut().swap(0, 1);
        assert_ne!(gs_to_array_for(&swapped, 0), base);
    }

    /// Real slot to canonical position and back is a bijection, with the
    /// centre pinned. Everything else here depends on that.
    #[test]
    fn the_factory_order_is_a_bijection() {
        for (seed, plies) in [(7u64, 0usize), (11, 3), (23, 5), (31, 9), (41, 14)] {
            let gs = mid_round(seed, plies);
            let order = FactoryOrder::canonical(&gs);
            assert_eq!(order.canonical_slot(0), 0, "the centre must stay put");
            assert_eq!(order.real_slot(0), 0, "the centre must stay put");
            for slot in 0..FACTORY_SLOTS {
                assert_eq!(order.real_slot(order.canonical_slot(slot)), slot);
                assert_eq!(order.canonical_slot(order.real_slot(slot)), slot);
            }
        }
        for slot in 0..FACTORY_SLOTS {
            assert_eq!(FactoryOrder::identity().real_slot(slot), slot);
            assert_eq!(FactoryOrder::identity().canonical_slot(slot), slot);
        }
    }

    /// Encode a position, take every legal move's canonical index, decode it,
    /// and get the original move back. If this is wrong, the agent plays a
    /// different move than the one the network chose.
    #[test]
    fn canonical_action_indices_round_trip() {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(17, 0);
        let mut checked = 0;

        for ply in 0..60 {
            let moves = gs.get_moves();
            if moves.is_empty() {
                if gs.end_round() == crate::gamestate::State::GameEnd {
                    break;
                }
                continue;
            }
            let order = FactoryOrder::canonical(&gs);
            let mut seen = HashSet::new();

            for m in &moves {
                let index = order.canonical_index(m);
                assert!(index < 180, "ply {ply}: index {index} out of range");
                // Injective, or two legal moves would share a mask slot and
                // one of them could never be selected.
                assert!(
                    seen.insert(index),
                    "ply {ply}: two legal moves share canonical index {index}"
                );

                let (source, tile, destination) = order.real_move_parts(index);
                assert_eq!(source, usize::from(m.source), "ply {ply}");
                assert_eq!(tile, m.tile as usize, "ply {ply}");
                assert_eq!(destination, usize::from(m.destination), "ply {ply}");
                checked += 1;
            }

            gs.play_move(moves[ply % moves.len()]);
        }

        assert!(checked > 500, "only {checked} moves round tripped");
    }

    /// The canonical index of a move must be the plain index that same move
    /// would have if the displays were physically sorted. That is the property
    /// the network relies on: the action slot it scores is the slot it saw.
    #[test]
    fn canonical_indices_match_the_physically_sorted_position() {
        for (seed, plies) in [(7u64, 0usize), (11, 3), (23, 5), (31, 9)] {
            let gs = mid_round(seed, plies);
            let order = FactoryOrder::canonical(&gs);

            let arrangement = [
                order.real_slot(1),
                order.real_slot(2),
                order.real_slot(3),
                order.real_slot(4),
                order.real_slot(5),
            ];
            let sorted = rearranged(&gs, arrangement);

            // Sorting is idempotent, so the sorted position needs no reorder.
            assert_eq!(FactoryOrder::canonical(&sorted), FactoryOrder::identity());
            // And it encodes identically, which is what canonicalising means.
            assert_eq!(gs_to_array_for(&sorted, 0), gs_to_array_for(&gs, 0));

            let mut canonical: Vec<usize> = gs
                .get_moves()
                .iter()
                .map(|m| order.canonical_index(m))
                .collect();
            let mut plain: Vec<usize> = sorted.get_moves().iter().map(Move::to_index).collect();
            canonical.sort_unstable();
            plain.sort_unstable();
            assert_eq!(canonical, plain, "seed {seed}/{plies} plies");
        }
    }

    /// The encoded vector is the length everything downstream assumes, and the
    /// global blocks land where the layout comment says.
    #[test]
    fn the_global_layout_is_what_the_consts_claim() {
        let mut gs = mid_round(5, 6);

        let mut centre = TileGroup::new_empty();
        centre.add_tiles(Tile::Blue, 20);
        let mut display = TileGroup::new_empty();
        display.add_tiles(Tile::Blue, 4);
        {
            let f = gs.factories_mut();
            f[0] = Some(centre);
            for slot in 1..FACTORY_SLOTS {
                f[slot] = Some(display);
            }
        }

        let arr = gs_to_array(&gs);
        assert_eq!(arr.len(), ENCODED_SIZE);

        // The centre's own scale: a stuffed centre reads 2.0, not the 4.0 it
        // used to when it shared the displays' divisor, and a full display
        // reads exactly 1.0 rather than 0.8.
        assert_eq!(arr[(CENTRE_AT + Tile::Blue as usize, 0)], 2.0);
        for canonical in 0..DISPLAYS {
            assert_eq!(
                arr[(DISPLAYS_AT + canonical * 5 + Tile::Blue as usize, 0)],
                1.0
            );
        }

        assert!(gs.tilebag().total() > 0, "bag block would be all zeroes");
        for (colour, &count) in gs.tilebag().counts().iter().enumerate() {
            assert_eq!(arr[(BAG_AT + colour, 0)], f32::from(count) / 20.0);
        }

        // 20 in the centre plus 4 on each of five displays.
        assert_eq!(arr[(AVAILABLE_AT + Tile::Blue as usize, 0)], 40.0 / 20.0);
        for colour in [Tile::Yellow, Tile::Red, Tile::Black, Tile::White] {
            assert_eq!(arr[(AVAILABLE_AT + colour as usize, 0)], 0.0);
        }

        assert_eq!(arr[(FP_AT, 0)], gs.first_player_tile() as u8 as f32);
        assert_eq!(arr[(ROUND_AT, 0)], gs.round() as f32 / 5.0);
    }

    /// The score lead is written from the acting player's seat, but
    /// `differential_predicted_score` is hardcoded board 0 minus board 1. A
    /// sign error here is silent: the model would just learn to play for its
    /// opponent, and every other test would still pass.
    #[test]
    fn score_lead_is_from_the_players_own_seat() {
        // Build an asymmetric position: seat 0 ahead on predicted score.
        let mut gs = mid_round(29, 7);
        while gs.differential_predicted_score() == 0.0 {
            let moves = gs.get_moves();
            assert!(!moves.is_empty(), "ran out of moves before a lead appeared");
            gs.play_move(moves[0]);
        }

        let lead = gs.differential_predicted_score();
        let seat_0 = gs_to_array_for(&gs, 0);
        let seat_1 = gs_to_array_for(&gs, 1);

        assert_eq!(seat_0[(LEAD_AT, 0)], lead / 100.0);
        assert_eq!(seat_1[(LEAD_AT, 0)], -lead / 100.0);
        assert_ne!(seat_0[(LEAD_AT, 0)], seat_1[(LEAD_AT, 0)]);

        // The sign must agree with the predicted scores the same vector
        // carries: the acting player's board goes first, so a positive lead
        // means the first board's predicted score is the larger one.
        for player in [0usize, 1] {
            let arr = gs_to_array_for(&gs, player);
            let mine = arr[(PREDICTED_AT, 0)];
            let theirs = arr[(BOARD_SIZE + PREDICTED_AT, 0)];
            assert!(
                (arr[(LEAD_AT, 0)] - (mine - theirs)).abs() < 1e-6,
                "seat {player}: lead disagrees with the encoded scores"
            );
            assert_eq!(
                mine,
                f32::from(gs.boards()[player].predicted_eval) / 100.0,
                "seat {player}: the acting player's board is not first"
            );
        }
    }

    /// The legal-colour mask must agree, cell for cell, with the rule move
    /// generation actually applies.
    #[test]
    fn the_legal_colour_mask_matches_move_generation() {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(37, 0);
        let mut ones = 0;
        let mut zeroes = 0;

        for ply in 0..80 {
            let moves = gs.get_moves();
            if moves.is_empty() {
                if gs.end_round() == crate::gamestate::State::GameEnd {
                    break;
                }
                continue;
            }

            for player in [0usize, 1] {
                let arr = gs_to_array_for(&gs, player);
                let board = &gs.boards()[player];
                for (row_i, (row, _)) in board.row_iter().enumerate() {
                    for tile in Tile::iter() {
                        let encoded = arr[(LEGAL_AT + row_i * 5 + tile as usize, 0)];
                        let legal = board.can_play_tile(row, tile, 1).is_some();
                        assert_eq!(
                            encoded,
                            if legal { 1.0 } else { 0.0 },
                            "ply {ply} seat {player} row {row_i} {tile:?}"
                        );
                        // It also has to imply the wall rule it subsumes.
                        if legal {
                            assert!(board.wall.cell_available(row, &tile));
                            ones += 1;
                        } else {
                            zeroes += 1;
                        }
                    }
                }
            }

            gs.play_move(moves[ply % moves.len()]);
        }

        assert!(ones > 100 && zeroes > 100, "mask was near constant");
    }

    /// Remaining capacity is absolute, so two rows needing the same number of
    /// tiles read the same however long they are.
    #[test]
    fn remaining_capacity_is_absolute_not_a_fraction() {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(43, 0);
        // Row 2 (capacity 2) and row 5 (capacity 5) both one short.
        {
            let board = &mut gs.boards_mut()[0];
            board.place_tiles_in_row(RowIndex::Two, Tile::Blue, 1);
            board.place_tiles_in_row(RowIndex::Five, Tile::Red, 4);
        }
        let arr = gs_to_array_for(&gs, 0);

        assert_eq!(arr[(REMAINING_AT, 0)], 1.0 / 5.0, "empty row 1 needs 1");
        assert_eq!(arr[(REMAINING_AT + 1, 0)], 1.0 / 5.0, "row 2 needs 1 more");
        assert_eq!(arr[(REMAINING_AT + 2, 0)], 3.0 / 5.0, "empty row 3 needs 3");
        assert_eq!(arr[(REMAINING_AT + 4, 0)], 1.0 / 5.0, "row 5 needs 1 more");
        // The fraction block cannot tell those two apart; this block can.
        assert_eq!(arr[(REMAINING_AT + 1, 0)], arr[(REMAINING_AT + 4, 0)]);
    }

    /// Run lengths have to see adjacency, which the occupancy bits cannot.
    #[test]
    fn wall_runs_count_contiguous_tiles() {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(47, 0);
        {
            let wall = &mut gs.boards_mut()[0].wall;
            // Row 0: columns 0, 1, 2 filled -- a run of three.
            wall.place_tile(RowIndex::One, WALL_COLOURS[0][0]);
            wall.place_tile(RowIndex::One, WALL_COLOURS[0][1]);
            wall.place_tile(RowIndex::One, WALL_COLOURS[0][2]);
            // Column 0: rows 1 and 2 as well, making a vertical run of three.
            wall.place_tile(RowIndex::Two, WALL_COLOURS[1][0]);
            wall.place_tile(RowIndex::Three, WALL_COLOURS[2][0]);
            // Row 0 column 4, isolated: a run of one, with a gap at column 3.
            wall.place_tile(RowIndex::One, WALL_COLOURS[0][4]);
        }
        let arr = gs_to_array_for(&gs, 0);
        let h = |r: usize, c: usize| arr[(HRUN_AT + r * 5 + c, 0)];
        let v = |r: usize, c: usize| arr[(VRUN_AT + r * 5 + c, 0)];

        for c in 0..3 {
            assert_eq!(h(0, c), 3.0 / 5.0, "row 0 col {c} horizontal run");
        }
        assert_eq!(h(0, 3), 0.0, "empty cells stay at zero");
        assert_eq!(v(0, 3), 0.0, "empty cells stay at zero");
        assert_eq!(h(0, 4), 1.0 / 5.0, "isolated tile is a run of one");

        for r in 0..3 {
            assert_eq!(v(r, 0), 3.0 / 5.0, "row {r} col 0 vertical run");
        }
        assert_eq!(v(0, 1), 1.0 / 5.0, "no tile below row 0 column 1");
        assert_eq!(h(1, 0), 1.0 / 5.0, "row 1 has one tile");

        // The occupancy block is blind to all of this: same bits either way.
        assert_eq!(arr[(WALL_AT, 0)], 1.0);
        assert_eq!(arr[(WALL_AT + 3, 0)], 0.0);
    }

    /// The encoding must still work for the genetic player, which feeds it to
    /// a fixed size matrix, and must be seat 0's view.
    #[test]
    fn gs_to_array_is_seat_zero() {
        let gs = mid_round(53, 6);
        assert_eq!(gs_to_array(&gs), gs_to_array_for(&gs, 0));
        assert_eq!(gs_to_array(&gs).len(), ENCODED_SIZE);
    }
}
