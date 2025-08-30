use nalgebra::SMatrix;
use rand_distr::{Distribution, StandardNormal};

use crate::{
    gamestate::{Gamestate, Move},
    playerboard::{wall::Wall, PlayerBoard},
    tiles::TileGroup,
};

use super::{EvolvingPlayer, Player};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoveSelectNN {
    weights_1: SMatrix<f32, 180, 150>,
    bias_1: SMatrix<f32, 180, 1>,
    weights_2: SMatrix<f32, 180, 180>,
    bias_2: SMatrix<f32, 180, 1>,
}

impl MoveSelectNN {
    pub fn new_random() -> Self {
        let d = StandardNormal;
        let mut rng = rand::thread_rng();
        let weights_1: SMatrix<f32, 180, 150> = SMatrix::from_distribution(&d, &mut rng);
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
        // convert game state to input vector
        let input = gs_to_array(gamestate);
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

            if let Some(m) = moves.get(&index_to_move(i)) {
                return *m;
            }
        }

        unreachable!()
    }

    fn name(&self) -> String {
        "MoveSelectNN".into()
    }
}

fn index_to_move(index: usize) -> (usize, usize, usize) {
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

fn gs_to_array(gs: &Gamestate<2, 6>) -> SMatrix<f32, 150, 1> {
    let mut arr = SMatrix::zeros();
    let a = pb_to_array(&gs.boards()[0]);
    let b = pb_to_array(&gs.boards()[1]);
    // board = 59 * 2 = 118
    // factories = 5 * 6 = 30
    // bag = 5
    // fp tile = 1
    // round = 1
    for (i, v) in a
        .into_iter()
        .copied()
        .chain(b.into_iter().copied())
        .chain(gs.factories().iter().flat_map(|f| match f {
            Some(f) => factory_to_array(f),
            None => [0.0; 5],
        }))
        .chain([gs.first_player_tile() as u8 as f32, gs.round() as f32 / 5.0])
        .enumerate()
    {
        arr[(i, 0)] = v;
    }
    arr
}

fn factory_to_array(factory: &TileGroup) -> [f32; 5] {
    factory.counts().map(|v| f32::from(v) / 5.0)
}

fn pb_to_array(pb: &PlayerBoard) -> SMatrix<f32, 59, 1> {
    // rows = 5 * 6 = 30
    // wall = 5 * 5 = 25
    // floor = 1
    // fp tile = 1
    // score = 1
    // prediction = 1
    // total = 59
    let mut arr = SMatrix::zeros();
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
        .chain(wall_to_array(&pb.wall).into_iter().copied())
        .chain([
            pb.floor.total().max(7) as f32 / 7.0,
            pb.first_player_tile as u8 as f32,
            pb.score as f32 / 100.0,
            pb.predicted_score as f32 / 100.0,
        ])
        .enumerate()
    {
        arr[(i, 0)] = v;
    }

    arr
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
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn move_from_index() {
        for i in 0..180 {
            let (s, t, d) = index_to_move(i);
            println!("{} -> ({}, {}, {})", i, s, t, d);
        }
    }
}
