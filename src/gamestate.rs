use std::ops::AddAssign;

use rand::SeedableRng;
use strum::IntoEnumIterator;

use crate::{
    playerboard::{PlayerBoard, RowIndex},
    tiles::{Tile, TileGroup},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gamestate<const P: usize, const F: usize> {
    /// List of boards for each player
    boards: [PlayerBoard; P],
    /// Contains tiles that are not in play
    tilebag: TileGroup,
    /// Factories from which tiles are chosen
    factories: [Option<TileGroup>; F],
    /// First player token
    first_player_tile: bool,
    /// rng for picking tiles from bag
    rng: rand::prelude::SmallRng,
    /// Current player
    current_player: u8,
    /// Round number
    round: u16,
    /// State tracking
    state: State,
}

impl<const P: usize, const F: usize> Default for Gamestate<P, F> {
    fn default() -> Self {
        Self::new(rand::random(), 0)
    }
}

impl Gamestate<2, 6> {
    pub fn new_2_player() -> Self {
        Self::default()
    }

    pub fn new_2_player_with_seed(seed: u64, first_player: u8) -> Self {
        Self::new(seed, first_player)
    }

    pub fn differential_predicted_score(&self) -> f32 {
        self.boards[0].predicted_score as f32 - self.boards[1].predicted_score as f32
    }
}

impl Gamestate<3, 8> {
    pub fn new_3_player() -> Self {
        Self::default()
    }
}

impl Gamestate<4, 10> {
    pub fn new_4_player() -> Self {
        Self::default()
    }
}

impl<const P: usize, const F: usize> Gamestate<P, F> {
    /// Initialiser
    pub fn new(seed: u64, first_player: u8) -> Self {
        let mut gs = Self {
            boards: [PlayerBoard::default(); P],
            tilebag: TileGroup::new_bag(),
            factories: [None; F],
            first_player_tile: true,
            rng: rand::prelude::SmallRng::seed_from_u64(seed),
            current_player: first_player,
            round: 0,
            state: State::GameEnd,
        };
        gs.deal();
        gs
    }

    /// Get current game state
    pub fn state(&self) -> State {
        self.state
    }

    /// Get current round
    pub fn round(&self) -> u16 {
        self.round
    }

    /// Get game scores
    pub fn scores(&self) -> [u8; P] {
        let mut scores = [0; P];
        for (i, b) in self.boards.iter().enumerate() {
            scores[i] = b.score;
        }
        scores
    }

    /// Get tile bag
    pub fn tilebag(&self) -> &TileGroup {
        &self.tilebag
    }

    /// Get the current player index
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// Get the first_player tile state
    pub fn first_player_tile(&self) -> bool {
        self.first_player_tile
    }

    /// Get access to the player boards
    pub fn boards(&self) -> &[PlayerBoard; P] {
        &self.boards
    }

    /// Get access to factories
    pub fn factories(&self) -> &[Option<TileGroup>; F] {
        &self.factories
    }

    /// Get access to centre
    pub fn centre(&self) -> TileGroup {
        self.factories[0].unwrap_or_default()
    }

    fn deal(&mut self) {
        // Deal tiles to factories
        for factory in self.factories[1..].iter_mut() {
            let mut f = TileGroup::new_empty();
            for _ in 0..4 {
                if let Some(tile) = self.tilebag.random_tile(&mut self.rng) {
                    f.add_tile(tile);
                }
            }
            *factory = Some(f);
        }
        self.state = State::RoundActive;
        self.round += 1;
    }

    /// get a list of possible moves to play
    pub fn get_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        for (source, factory) in self
            .factories
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.as_ref().map(|f| (Source(i as u8), f)))
        {
            // for each tile that factory contains
            for (&count, tile) in factory.into_iter().filter(|(&c, _)| c > 0) {
                // for each row in the current player's board
                // Check if can play how many will be played
                for row in RowIndex::iter() {
                    if let Some((play_count, row_count)) =
                        self.boards[self.current_player as usize].can_play_tile(row, tile, count)
                    {
                        moves.push(Move::new(
                            source,
                            tile,
                            count,
                            play_count,
                            row_count,
                            row.into(),
                        ));
                    }
                }
                // add the floor as a destination
                moves.push(Move::new_to_floor(source, tile, count));
            }
        }
        moves
    }

    pub fn play_move(&mut self, move_: Move) -> State {
        // Get tiles from factory
        let mut factory = self.factories[move_.source.0 as usize].take().unwrap();
        let tile = move_.tile;
        let count = factory.take_tile(tile);
        let fp = self.first_player_tile && move_.source.is_centre();

        // Place on board
        self.boards[self.current_player as usize].place_tiles(move_.destination, tile, count, fp);

        // Remove first player tile if used
        if fp {
            self.first_player_tile = false;
        }

        // Move remaining tiles to centre
        if let Some(centre) = &mut self.factories[0] {
            centre.add_assign(factory);
        } else {
            self.factories[0] = Some(factory);
        }

        // Check for end of round
        if self
            .factories
            .iter()
            .all(|f| f.map_or(true, |f| f.total() == 0))
        {
            self.state = State::RoundEnd;
        } else {
            // next players turn
            self.current_player = (self.current_player + 1) % P as u8;
        }
        self.state
    }

    /// Get the predicted score if this move were to be played
    /// Helps players evaluate each move
    /// Returns the score and the change in predicted score
    pub fn predict_score(&self, move_: Move) -> (u8, i8) {
        // Clone the board
        let mut board = self.boards[self.current_player as usize].clone();
        // record previous predicted score
        let prev_score = board.predicted_score;

        // Place on board
        board.place_tiles(
            move_.destination,
            move_.tile,
            move_.count,
            move_.source.is_centre() && self.first_player_tile,
        );

        (
            board.predicted_score,
            board.predicted_score as i8 - prev_score as i8,
        )
    }

    /// Check if this move will take the first player tile
    pub fn takes_fp(&self, move_: &Move) -> bool {
        move_.source.is_centre() && self.first_player_tile
    }

    /// End the round, add up scores and check for game end conditions
    pub fn end_round(&mut self) -> State {
        // Get first player tile from boards
        for (i, b) in self.boards.iter().enumerate() {
            if b.first_player_tile {
                self.current_player = i as u8;
            }
        }
        self.first_player_tile = true;

        // Move tiles on game board, calc scores and return to bag
        if self
            .boards
            .iter_mut()
            .map(|b| b.end_round())
            .map(|(t, g)| {
                self.tilebag.add_assign(t);
                g
            })
            .collect::<Vec<_>>()
            .into_iter()
            .any(|g| g)
            || self.round == 10
        {
            // game over, calculate final scores
            for b in &mut self.boards {
                b.end_game();
            }
            self.state = State::GameEnd;
        } else {
            // Set up for next round
            self.deal();
        }

        self.state
    }

    /// Count up the tiles in play
    /// Used for testing to validate logic
    fn tile_count(&self) -> u8 {
        self.boards.iter().map(|b| b.tile_count()).sum::<u8>()
            + self.tilebag.total()
            + self
                .factories
                .iter()
                .filter_map(|f| f.as_ref())
                .map(|f| f.total())
                .sum::<u8>()
    }

    /// Check number of first player tiles in play
    /// Used for testing to validate logic
    fn fp_count(&self) -> usize {
        self.boards.iter().filter(|b| b.first_player_tile).count()
            + if self.first_player_tile { 1 } else { 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Move {
    /// Where the tiles will be taken from
    pub source: Source,
    /// Which tile will be played
    pub tile: Tile,
    /// How many tiles will be played
    pub count: u8,
    /// How many will end up in the row
    pub play_count: u8,
    /// How many tiles will be in the row after
    pub row_count: u8,
    /// Where the tiles will be placed
    pub destination: Destination,
}

impl Move {
    pub fn new(
        source: Source,
        tile: Tile,
        count: u8,
        play_count: u8,
        row_count: u8,
        destination: Destination,
    ) -> Self {
        Self {
            source,
            tile,
            count,
            play_count,
            row_count,
            destination,
        }
    }

    pub fn new_to_floor(source: Source, tile: Tile, count: u8) -> Self {
        Self {
            source,
            tile,
            count,
            play_count: 0,
            row_count: 0,
            destination: Destination::Floor,
        }
    }

    pub fn fills_row(&self) -> bool {
        match self.destination {
            Destination::Row(row) => self.row_count == row.capacity(),
            Destination::Floor => false,
        }
    }

    pub fn no_floor_tiles(&self) -> bool {
        match self.destination {
            Destination::Row(row) => self.count == self.play_count,
            Destination::Floor => false,
        }
    }

    pub fn perfect_move(&self) -> bool {
        self.fills_row() && self.no_floor_tiles()
    }

    pub fn floor_tiles(&self) -> u8 {
        match self.destination {
            Destination::Row(_) => self.count - self.play_count,
            Destination::Floor => self.count,
        }
    }

    pub fn row_capacity(&self) -> u8 {
        match self.destination {
            Destination::Row(row) => row.capacity(),
            Destination::Floor => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct MoveDetailed {
    move_: Move,
    count: u8,
    fp: bool,
    row: RowState,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum RowState {
    Partial,
    Full,
    Overfull,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum State {
    RoundActive,
    RoundEnd,
    GameEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Source(pub u8);

impl From<Source> for usize {
    fn from(value: Source) -> Self {
        value.0 as usize
    }
}

impl Source {
    fn is_centre(&self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Destination {
    Row(RowIndex),
    Floor,
}

impl From<Destination> for usize {
    fn from(value: Destination) -> Self {
        match value {
            Destination::Row(r) => r as usize,
            Destination::Floor => 5,
        }
    }
}

impl From<RowIndex> for Destination {
    fn from(value: RowIndex) -> Self {
        Self::Row(value)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn gamestate() {
        let mut g = super::Gamestate::new_2_player();
        // sanity checks
        assert_eq!(g.boards.len(), 2);
        assert_eq!(g.factories.len(), 6);
        assert!(g.first_player_tile);
        assert_eq!(g.round, 1);
        assert_eq!(g.tilebag.total(), 80);
        assert_eq!(g.factories[0], None);
        for f in &g.factories[1..] {
            assert_eq!(f.as_ref().unwrap().total(), 4);
        }
        assert_eq!(g.tile_count(), 100);
        assert_eq!(g.fp_count(), 1);

        let moves = g.get_moves();
        assert_eq!(g.play_move(moves[0]), super::State::RoundActive);
        assert_eq!(g.current_player, 1);
        assert_eq!(g.tile_count(), 100);
        assert_eq!(g.fp_count(), 1);

        // Play a full game
        loop {
            loop {
                let moves = g.get_moves();
                match g.play_move(moves[0]) {
                    crate::gamestate::State::RoundActive => (),
                    crate::gamestate::State::RoundEnd => break,
                    crate::gamestate::State::GameEnd => panic!("Game should not end"),
                }
                assert_eq!(g.tile_count(), 100);
                assert_eq!(g.fp_count(), 1);
            }
            assert_eq!(g.tile_count(), 100);
            assert_eq!(g.fp_count(), 1);
            if g.end_round() == super::State::GameEnd {
                break;
            }
            // dbg!(&g);
            assert_eq!(g.tile_count(), 100);
            assert_eq!(g.fp_count(), 1);
        }
    }
}
