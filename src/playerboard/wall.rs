//! Contains player board wall
//! Responsible for tracking correct placement of tiles in wall
//! and counting points at end of round and end of game

use std::ops::Index;

use strum::IntoEnumIterator;

use crate::tiles::Tile;

pub const WALL_COLOURS: [[Tile; 5]; 5] = [
    [
        Tile::Blue,
        Tile::Yellow,
        Tile::Red,
        Tile::Black,
        Tile::White,
    ],
    [
        Tile::White,
        Tile::Blue,
        Tile::Yellow,
        Tile::Red,
        Tile::Black,
    ],
    [
        Tile::Black,
        Tile::White,
        Tile::Blue,
        Tile::Yellow,
        Tile::Red,
    ],
    [
        Tile::Red,
        Tile::Black,
        Tile::White,
        Tile::Blue,
        Tile::Yellow,
    ],
    [
        Tile::Yellow,
        Tile::Red,
        Tile::Black,
        Tile::White,
        Tile::Blue,
    ],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Wall([[Option<Tile>; 5]; 5], u32);

impl Wall {
    /// Occupancy as a 25 bit mask, bit `5 * row + column`.
    ///
    /// Maintained by [`Wall::place_tile`], which is the only way a tile gets
    /// onto the wall. Every cell's colour is fixed by its position, so this
    /// describes the wall completely, and evaluation can work on it without
    /// touching the cells at all.
    pub fn mask(&self) -> u32 {
        self.1
    }
}

impl std::hash::Hash for Wall {
    /// Every cell's colour is fixed by its position, so occupancy alone
    /// describes the wall. Hashing the maintained mask rather than deriving it
    /// from 25 `Option<Tile>` makes building a position key cheaper again, and
    /// two walls are equal exactly when their occupancy matches.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.1);
    }
}

impl Index<(RowIndex, ColumnIndex)> for Wall {
    type Output = Option<Tile>;

    fn index(&self, index: (RowIndex, ColumnIndex)) -> &Self::Output {
        &self.0[usize::from(&index.0)][usize::from(&index.1)]
    }
}

impl Wall {
    /// Read access to inner array
    pub fn iter(&self) -> impl Iterator<Item = &[Option<Tile>; 5]> {
        self.0.iter()
    }
    /// Checks if a tile can be placed in this row
    /// Used for move generation
    pub fn cell_available(&self, row: RowIndex, tile: &Tile) -> bool {
        self[(row, row.tile_column(tile))].is_none()
    }

    /// Place tile on the wall and return the score
    pub fn place_and_score_tile(&mut self, row: RowIndex, tile: Tile) -> u8 {
        let score = self.score_tile(row, tile);
        self.place_tile(row, tile);
        score
    }

    /// Place a tile in the wall
    /// Does not check if the move is valid
    /// Should have been previously checked with cell_available
    pub fn place_tile(&mut self, row: RowIndex, tile: Tile) {
        let col = row.tile_column(&tile);
        let (r, c) = (usize::from(&row), usize::from(&col));
        self.0[r][c] = Some(tile);
        self.1 |= 1 << (r * 5 + c);
    }

    /// Calculate score of placing tile
    pub fn score_tile(&self, row: RowIndex, tile: Tile) -> u8 {
        let col: usize = (&row.tile_column(&tile)).into();
        let row: usize = (&row).into();

        let mut col_score = 0;
        // Check up
        for i in (0..row).rev() {
            if self.0[i][col].is_none() {
                break;
            }
            col_score += 1;
        }
        // Check down
        for i in row + 1..5 {
            if self.0[i][col].is_none() {
                break;
            }
            col_score += 1;
        }
        if col_score > 0 {
            col_score += 1;
        }
        let mut row_score = 0;
        // Check left
        for i in (0..col).rev() {
            if self.0[row][i].is_none() {
                break;
            }
            row_score += 1;
        }
        // Check right
        for i in col + 1..5 {
            if self.0[row][i].is_none() {
                break;
            }
            row_score += 1;
        }
        if row_score > 0 {
            row_score += 1;
        }
        let mut score = col_score + row_score;
        if score == 0 {
            score = 1;
        }
        score
    }

    /// Calculate the score of the wall
    /// Includes row, column and colours
    pub fn score(&self) -> u8 {
        let mut score = 0;
        // Row
        score += 2 * self
            .0
            .iter()
            .filter(|row| row.iter().all(|t| t.is_some()))
            .count() as u8;
        // Column
        score += 7 * ColumnIndex::iter()
            .filter(|col| RowIndex::iter().all(|row| self[(row, *col)].is_some()))
            .count() as u8;
        // Colours
        score += 10
            * Tile::iter()
                .filter(|tile| {
                    RowIndex::iter().all(|row| {
                        let col = row.tile_column(tile);
                        self[(row, col)].is_some()
                    })
                })
                .count() as u8;
        score
    }

    /// Check for full row as game ending condition
    pub fn has_full_row(&self) -> bool {
        (0..5).any(|r| (self.1 >> (r * 5)) & 0x1f == 0x1f)
    }

    pub(crate) fn tile_count(&self) -> u8 {
        self.1.count_ones() as u8
    }
}

/// Bit position of the cell a tile occupies in a row, for the occupancy mask.
pub fn cell_index(row: RowIndex, tile: Tile) -> u32 {
    let col = row.tile_column(&tile);
    (usize::from(&row) * 5 + usize::from(&col)) as u32
}

/// Score for placing a tile at `bit`, read off an occupancy mask.
///
/// Same rule as [`Wall::score_tile`]: the contiguous run through the cell
/// horizontally plus the run vertically, counting the cell once in each
/// direction it extends, and one point if it touches nothing.
pub fn score_tile_mask(mask: u32, bit: u32) -> u8 {
    let (r, c) = ((bit / 5) as i32, (bit % 5) as i32);
    let filled = |rr: i32, cc: i32| (mask >> (rr * 5 + cc)) & 1 == 1;

    let mut row_score = 0u8;
    let mut i = c - 1;
    while i >= 0 && filled(r, i) {
        row_score += 1;
        i -= 1;
    }
    let mut i = c + 1;
    while i < 5 && filled(r, i) {
        row_score += 1;
        i += 1;
    }
    if row_score > 0 {
        row_score += 1;
    }

    let mut col_score = 0u8;
    let mut i = r - 1;
    while i >= 0 && filled(i, c) {
        col_score += 1;
        i -= 1;
    }
    let mut i = r + 1;
    while i < 5 && filled(i, c) {
        col_score += 1;
        i += 1;
    }
    if col_score > 0 {
        col_score += 1;
    }

    let score = row_score + col_score;
    if score == 0 {
        1
    } else {
        score
    }
}

/// For indexing into wall
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter, PartialOrd, Ord)]
pub enum RowIndex {
    One,
    Two,
    Three,
    Four,
    Five,
}

impl RowIndex {
    /// Returns column index of tile in row
    pub(crate) fn tile_column(&self, tile: &Tile) -> ColumnIndex {
        ((u8::from(self) + u8::from(tile)) % 5).into()
    }

    /// Returns how many tiles can fit in this row
    pub fn capacity(&self) -> u8 {
        match self {
            RowIndex::One => 1,
            RowIndex::Two => 2,
            RowIndex::Three => 3,
            RowIndex::Four => 4,
            RowIndex::Five => 5,
        }
    }
}

impl From<&RowIndex> for u8 {
    fn from(value: &RowIndex) -> Self {
        *value as u8
    }
}

impl From<&RowIndex> for usize {
    fn from(value: &RowIndex) -> Self {
        *value as usize
    }
}

impl From<RowIndex> for usize {
    fn from(value: RowIndex) -> Self {
        value as usize
    }
}

impl From<u8> for RowIndex {
    fn from(value: u8) -> Self {
        (value as usize).into()
    }
}

impl From<usize> for RowIndex {
    fn from(value: usize) -> Self {
        match value {
            0 => RowIndex::One,
            1 => RowIndex::Two,
            2 => RowIndex::Three,
            3 => RowIndex::Four,
            4 => RowIndex::Five,
            _ => panic!("Invalid row index"),
        }
    }
}

/// For indexing into wall
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
pub enum ColumnIndex {
    One,
    Two,
    Three,
    Four,
    Five,
}

impl From<&ColumnIndex> for u8 {
    fn from(value: &ColumnIndex) -> Self {
        *value as u8
    }
}

impl From<&ColumnIndex> for usize {
    fn from(value: &ColumnIndex) -> Self {
        *value as usize
    }
}

impl From<usize> for ColumnIndex {
    fn from(value: usize) -> Self {
        match value {
            0 => ColumnIndex::One,
            1 => ColumnIndex::Two,
            2 => ColumnIndex::Three,
            3 => ColumnIndex::Four,
            4 => ColumnIndex::Five,
            _ => panic!("Invalid column index"),
        }
    }
}

impl From<u8> for ColumnIndex {
    fn from(value: u8) -> Self {
        (value as usize).into()
    }
}
#[cfg(test)]
mod test {
    use strum::IntoEnumIterator;

    use crate::{playerboard::wall::WALL_COLOURS, tiles::Tile};

    use super::{RowIndex, Wall};

    #[test]
    fn tile_column() {
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let col = row.tile_column(&tile);
                assert_eq!(tile, WALL_COLOURS[row as usize][col as usize]);
            }
        }
    }

    #[test]
    fn single_tile_score() {
        let wal = Wall::default();
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let score = wal.score_tile(row, tile);
                assert_eq!(score, 1);
            }
        }
    }

    #[test]
    fn wall_colours() {
        let mut wall = Wall::default();
        assert!(
            wall.iter().flat_map(|r| r.iter()).all(|t| t.is_none()),
            "a fresh wall should be empty"
        );

        // A tile lands in the column its colour occupies for that row.
        let column = RowIndex::Two.tile_column(&Tile::Black);
        assert_eq!(
            Tile::Black,
            WALL_COLOURS[RowIndex::Two as usize][column as usize]
        );

        wall.place_tile(RowIndex::One, Tile::Black);
        let placed = RowIndex::One.tile_column(&Tile::Black);
        assert_eq!(wall[(RowIndex::One, placed)], Some(Tile::Black));
        assert_eq!(
            wall.iter().flat_map(|r| r.iter()).filter(|t| t.is_some()).count(),
            1,
            "placing one tile should occupy exactly one cell"
        );
    }

    #[test]
    fn tile_scores() {
        // put tile in top left
        let mut wall = Wall::default();
        wall.place_tile(RowIndex::One, Tile::Blue);
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let expected = match (row, tile) {
                    (RowIndex::One, Tile::Yellow) => 2,
                    (RowIndex::Two, Tile::White) => 2,
                    _ => 1,
                };

                assert_eq!(wall.score_tile(row, tile), expected);
            }
        }

        // put tile in centre
        let mut wall = Wall::default();
        wall.place_tile(RowIndex::Three, Tile::Blue);
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let expected = match (row, tile) {
                    (RowIndex::Two, Tile::Yellow) => 2,
                    (RowIndex::Three, Tile::White) => 2,
                    (RowIndex::Three, Tile::Yellow) => 2,
                    (RowIndex::Four, Tile::White) => 2,
                    _ => 1,
                };

                assert_eq!(wall.score_tile(row, tile), expected);
            }
        }

        // Add a tile to top right
        wall.place_tile(RowIndex::Two, Tile::Red);
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let expected = match (row, tile) {
                    (RowIndex::Two, Tile::Yellow) => 4,
                    (RowIndex::Three, Tile::White) => 2,
                    (RowIndex::Three, Tile::Yellow) => 4,
                    (RowIndex::Four, Tile::White) => 2,
                    (RowIndex::One, Tile::Black) => 2,
                    (RowIndex::Two, Tile::Black) => 2,
                    _ => 1,
                };

                assert_eq!(wall.score_tile(row, tile), expected);
            }
        }

        // Add more for complexity
        wall.place_tile(RowIndex::Three, Tile::Red);
        wall.place_tile(RowIndex::Four, Tile::Blue);
        for row in RowIndex::iter() {
            for tile in Tile::iter() {
                let expected = match (row, tile) {
                    (RowIndex::Two, Tile::Yellow) => 4,
                    (RowIndex::Three, Tile::White) => 2,
                    (RowIndex::Three, Tile::Yellow) => 6,
                    (RowIndex::Four, Tile::White) => 4,
                    (RowIndex::One, Tile::Black) => 2,
                    (RowIndex::Two, Tile::Black) => 4,
                    (RowIndex::Four, Tile::Yellow) => 4,
                    (RowIndex::Five, Tile::White) => 2,
                    _ => 1,
                };

                assert_eq!(wall.score_tile(row, tile), expected);
            }
        }
    }

    #[test]
    fn wall_scores() {
        let mut wall = Wall::default();
        wall.place_tile(RowIndex::Five, Tile::Blue);
        assert_eq!(wall.score(), 0);
        wall.place_tile(RowIndex::Four, Tile::Yellow);
        assert_eq!(wall.score(), 0);
        wall.place_tile(RowIndex::Three, Tile::Red);
        assert_eq!(wall.score(), 0);
        wall.place_tile(RowIndex::Two, Tile::Black);
        assert_eq!(wall.score(), 0);
        wall.place_tile(RowIndex::One, Tile::White);
        assert_eq!(wall.score(), 7);
        wall.place_tile(RowIndex::One, Tile::Blue);
        assert_eq!(wall.score(), 7);
        wall.place_tile(RowIndex::One, Tile::Yellow);
        assert_eq!(wall.score(), 7);
        wall.place_tile(RowIndex::One, Tile::Red);
        assert_eq!(wall.score(), 7);
        wall.place_tile(RowIndex::One, Tile::Black);
        assert_eq!(wall.score(), 9);
        wall.place_tile(RowIndex::Two, Tile::Blue);
        assert_eq!(wall.score(), 9);
        wall.place_tile(RowIndex::Three, Tile::Blue);
        assert_eq!(wall.score(), 9);
        wall.place_tile(RowIndex::Four, Tile::Blue);
        assert_eq!(wall.score(), 19);
        wall.place_tile(RowIndex::Two, Tile::White);
        assert_eq!(wall.score(), 19);
        wall.place_tile(RowIndex::Two, Tile::Yellow);
        assert_eq!(wall.score(), 19);
        wall.place_tile(RowIndex::Two, Tile::Red);
        assert_eq!(wall.score(), 21);
        wall.place_tile(RowIndex::Three, Tile::Yellow);
        assert_eq!(wall.score(), 21);
        wall.place_tile(RowIndex::Five, Tile::White);
        assert_eq!(wall.score(), 28);
        wall.place_tile(RowIndex::Five, Tile::Yellow);
        assert_eq!(wall.score(), 38);
    }
}
