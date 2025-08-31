pub mod wall;

pub use wall::RowIndex;

use core::panic;
use std::{iter::Zip, mem};

use strum::IntoEnumIterator;
use wall::{RowIndexIter, Wall};

use crate::{
    gamestate::Destination,
    tiles::{Tile, TileGroup},
};

/// Line of tiles on board
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Row(Option<(Tile, u8)>);

impl Row {
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn tile(&self) -> Option<Tile> {
        if let Some((tile, _)) = self.0 {
            Some(tile)
        } else {
            None
        }
    }

    pub fn count(&self) -> u8 {
        if let Some((_, count)) = self.0 {
            count
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlayerBoard {
    /// Wall of tiles
    pub wall: Wall,
    /// Floor of tiles
    pub floor: TileGroup,
    /// First player tile
    pub first_player_tile: bool,
    /// Pattern lines
    pub rows: [Row; 5],
    /// Score
    pub score: u8,
    /// Predicted score if rows were moved to wall
    pub predicted_score: u8,
}

impl PlayerBoard {
    /// Iterate over the rows of the board with their indices
    pub fn row_iter(&self) -> Zip<RowIndexIter, core::slice::Iter<'_, Row>> {
        RowIndex::iter().zip(self.rows.iter())
    }

    /// Check if tile can be played in this row
    /// Returns the number of tiles that can be played
    /// and how many tiles will be on the row after
    pub fn can_play_tile(&self, row: RowIndex, tile: Tile, count: u8) -> Option<(u8, u8)> {
        if let Some((row_tile, row_count)) = self.rows[usize::from(row)].0 {
            if row_tile == tile {
                // Check if row is full
                if row_count < row.row_capacity() {
                    let total = (row_count + count).min(row.row_capacity());
                    Some((total - row_count, total))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            // Check the wall
            if self.wall.cell_available(row, &tile) {
                Some((count, count))
            } else {
                None
            }
        }
    }

    /// Place tiles in a row or on the floor
    /// Does not check that the move is valid
    /// Updates predicted score
    pub fn place_tiles(
        &mut self,
        dest: Destination,
        tile: Tile,
        count: u8,
        first_player_tile: bool,
    ) {
        if first_player_tile {
            self.first_player_tile = true;
        }
        match dest {
            Destination::Row(row) => self.place_tiles_in_row(row, tile, count),
            Destination::Floor => self.floor.add_tiles(tile, count),
        }
        // update predicted score
        self.predict_score();
    }

    /// Place tiles in a row
    /// Does not check that the move is valid
    pub fn place_tiles_in_row(&mut self, row_ind: RowIndex, tile: Tile, count: u8) {
        // Get access to row
        let row = &mut self.rows[row_ind as usize];
        // Get row capacity
        let capacity = row_ind.row_capacity();

        let leftover = if let Some((row_tile, row_count)) = &mut row.0 {
            // If row is empty or matches factory tile, is valid move

            let total = *row_count + count;
            *row_count = if total > capacity { capacity } else { total };
            total - *row_count
        } else {
            let total = if count > capacity { capacity } else { count };
            row.0 = Some((tile, total));
            count - total
        };
        // If there are leftover tiles, add them to the floor
        self.floor.add_tiles(tile, leftover);
    }

    /// Fake move the full rows to the wall to calculate score
    /// Does not actually move the tiles
    /// Assigns the new score to predicted_score and returns it
    pub fn predict_score(&mut self) -> u8 {
        // Copy the wall
        let mut wall = self.wall;
        let mut score = 0;
        for row_ind in RowIndex::iter() {
            if let Some((tile, count)) = self.rows[usize::from(row_ind)].0 {
                if count == row_ind.row_capacity() {
                    score += wall.place_and_score_tile(row_ind, tile);
                }
            }
        }
        self.predicted_score = self.score + score + wall.score();
        // cap the score depending on floor
        let floor_score = floor_score(&self.floor, self.first_player_tile);
        if self.predicted_score < floor_score {
            self.predicted_score = 0;
        } else {
            self.predicted_score -= floor_score;
        }
        self.predicted_score
    }

    /// Return a copy of the wall with all tiles moved to where they will be at the end
    /// of the round
    pub fn simulate_wall(&self) -> Wall {
        let mut wall = self.wall.clone();
        for row_ind in RowIndex::iter() {
            if let Some((tile, count)) = self.rows[usize::from(row_ind)].0 {
                if count == row_ind.row_capacity() {
                    wall.place_tile(row_ind, tile);
                }
            }
        }
        wall
    }

    /// Move tiles from rows to wall
    /// Score as it goes
    /// Return tiles that are to be returned
    /// Calculate floor score and empty
    /// Set things up for next round
    /// returns true if the game is over
    pub fn end_round(&mut self) -> (TileGroup, bool) {
        // Store tiles that are to be returned
        let mut tile_return = TileGroup::new_empty();
        // Count score as it goes
        let mut score = 0;
        // Go through rows in order
        for row_ind in RowIndex::iter() {
            // if row contains any tiles
            if let Some((tile, count)) = self.rows[usize::from(row_ind)].0 {
                // if row is at capacity, move single tile to wall
                // otherwise leave tiles as they are
                if count == row_ind.row_capacity() {
                    // Get score from placing this tile
                    score += self.wall.score_tile(row_ind, tile);
                    // Assume that wall is empty in this cell
                    // Tile will disappear otherwise and is previous logic error
                    // in move generation
                    self.wall.place_tile(row_ind, tile);
                    // add remaining tiles to return
                    tile_return.add_tiles(tile, count - 1);
                    // clear the row
                    self.rows[usize::from(row_ind)] = Row::default();
                }
            }
        }
        // Empty the floor
        let floor = self.floor.empty();
        // Calculate floor score
        let floor_score = floor_score(&floor, self.first_player_tile);
        let total = self.score + score;
        // Add up scores, can't go below zero
        if total < floor_score {
            self.score = 0;
        } else {
            self.score = total - floor_score;
        }
        // remove first player tile
        self.first_player_tile = false;

        // Return tiles that are to be put back in bag
        tile_return += floor;
        (tile_return, self.wall.has_full_row())
    }

    pub fn end_game(&mut self) {
        // row score
        self.score += self.wall.score();
    }

    /// Count tiles on the board for testing
    pub(crate) fn tile_count(&self) -> u8 {
        let mut count = 0;
        for row in &self.rows {
            if let Some((_, c)) = row.0 {
                count += c;
            }
        }
        count += self.floor.total() + self.wall.tile_count();
        count
    }
}

fn floor_score(tiles: &TileGroup, fp: bool) -> u8 {
    let total = tiles.total() + if fp { 1 } else { 0 };
    match total {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        4 => 6,
        5 => 8,
        6 => 11,
        _ => 14,
    }
}

impl RowIndex {
    fn row_capacity(&self) -> u8 {
        match self {
            RowIndex::One => 1,
            RowIndex::Two => 2,
            RowIndex::Three => 3,
            RowIndex::Four => 4,
            RowIndex::Five => 5,
        }
    }
}
