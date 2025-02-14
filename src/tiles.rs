use std::{
    iter::Zip,
    ops::{Add, AddAssign},
    path::Iter,
};

use rand::Rng;
use strum::IntoEnumIterator;

/// Types of tiles
/// These are in the order as they appear on the first row of the wall
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
pub enum Tile {
    Blue,
    Yellow,
    Red,
    Black,
    White,
}

impl From<&Tile> for u8 {
    fn from(value: &Tile) -> Self {
        *value as u8
    }
}
/// Stores a selection of tiles for bag or centre factory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileGroup {
    counts: [u8; 5],
}

impl AddAssign for TileGroup {
    fn add_assign(&mut self, other: Self) {
        for (count, tile) in other.into_iter() {
            self.counts[tile as usize] += count;
        }
    }
}

impl TileGroup {
    /// Create a new bag of tiles
    pub fn new_bag() -> Self {
        Self {
            counts: [20, 20, 20, 20, 20],
        }
    }

    /// Create a new centre factory
    pub fn new_empty() -> Self {
        Self::default()
    }

    /// Empty and return the tiles in the group
    pub fn empty(&mut self) -> Self {
        let counts = self.counts;
        self.counts = [0; 5];
        Self { counts }
    }

    /// total number of tiles in the group
    pub fn total(&self) -> u8 {
        self.counts.iter().sum()
    }

    /// Take all tiles of a certain type from the group
    pub fn take_tile(&mut self, tile: Tile) -> u8 {
        let count = self.counts[tile as usize];
        self.counts[tile as usize] = 0;
        count
    }

    /// Select a random tile from the group
    /// Returns None if the group is empty
    pub fn random_tile(&mut self, rng: &mut rand::prelude::SmallRng) -> Option<Tile> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        let n = rng.random_range(0..total);
        let mut sum = 0;
        for (count, tile) in self.into_iter() {
            sum += count;
            if n < sum {
                self.counts[tile as usize] -= 1;
                return Some(tile);
            }
        }
        unreachable!()
    }

    /// Add a tile to the group
    pub fn add_tile(&mut self, tile: Tile) {
        self.counts[tile as usize] += 1;
    }

    /// Add multiple tiles to the group
    pub fn add_tiles(&mut self, tile: Tile, count: u8) {
        self.counts[tile as usize] += count;
    }

    /// Vec of each tile in group in [Tile] order
    pub fn tile_vec(&self) -> Vec<Tile> {
        self.into_iter()
            .flat_map(|(c, t)| std::iter::repeat(t).take(*c as usize))
            .collect()
    }
}

impl<'a> IntoIterator for &'a TileGroup {
    type Item = (&'a u8, Tile);
    type IntoIter = Zip<std::slice::Iter<'a, u8>, TileIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.counts.iter().zip(Tile::iter())
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn take_tiles() {
        let mut tg = TileGroup::new_bag();
        let mut tg_2 = TileGroup::new_empty();
        assert_eq!(tg.total(), 100);
        let mut rng = rand::prelude::SmallRng::from_os_rng();
        for _ in 0..100 {
            let tile = tg.random_tile(&mut rng).unwrap();
            tg_2.add_tile(tile);
        }
        assert_eq!(tg.total(), 0);
        assert!(tg.random_tile(&mut rng).is_none());
        assert_eq!(tg_2.total(), 100);
        // assert_eq!(tg_2.black, 20);
        // assert_eq!(tg_2.blue, 20);
        // assert_eq!(tg_2.yellow, 20);
        // assert_eq!(tg_2.red, 20);
        // assert_eq!(tg_2.white, 20);
    }
}
