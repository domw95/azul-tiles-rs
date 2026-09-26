#![recursion_limit = "512"]
//! Where the time in one network evaluation actually goes.
//! Args: <checkpoint_dir> [tag]
//!
//! Throwaway diagnostic, but a necessary one: a per-evaluation cost only means
//! something once it is known which part of the call it is, because the design
//! it argues for -- an incremental accumulator, hand-rolled inference --
//! addresses the matmul and not the encoding, and would be worthless if the
//! cost were somewhere else entirely.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::nn::gs_to_array_for;
use azul_tiles_rs::players::ppo::{PPOMoveSelector, STATE_SIZE};
use burn::backend::NdArray;
use burn::tensor::cast::ToElement as _;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

type B = NdArray;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let tag = a.get(2).cloned().unwrap_or_else(|| "best".into());
    let device = Default::default();
    let net = PPOMoveSelector::<B>::from_checkpoint(&dir, &tag, &device).expect("checkpoint");

    // A handful of real positions, played out greedily.
    let mut states = Vec::new();
    let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(9_500_000, 0);
    while states.len() < 300 {
        let moves = gs.get_moves();
        states.push(gs.clone());
        let s = gs.play_move(moves[0]);
        if s == State::RoundEnd && gs.end_round() == State::GameEnd {
            gs = Gamestate::new_2_player_with_seed(9_500_001, 0);
        }
    }

    let reps = 20;
    let calls = (states.len() * reps) as f64;

    let t0 = Instant::now();
    for _ in 0..reps {
        for g in &states {
            std::hint::black_box(gs_to_array_for(g, g.current_player() as usize));
        }
    }
    let encode = t0.elapsed().as_nanos() as f64 / calls;

    let vecs: Vec<Vec<f32>> = states
        .iter()
        .map(|g| gs_to_array_for(g, g.current_player() as usize).as_slice().to_vec())
        .collect();

    let t0 = Instant::now();
    for _ in 0..reps {
        for v in &vecs {
            std::hint::black_box(Tensor::<B, 1>::from_data(
                TensorData::new(v.clone(), [STATE_SIZE]),
                &device,
            ));
        }
    }
    let make_1d = t0.elapsed().as_nanos() as f64 / calls;

    // Warm, then the 1-D forward the evaluator actually uses.
    for v in vecs.iter().take(16) {
        let t = Tensor::<B, 1>::from_data(TensorData::new(v.clone(), [STATE_SIZE]), &device);
        std::hint::black_box(net.value(t).into_scalar().to_f32());
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        for v in &vecs {
            let t = Tensor::<B, 1>::from_data(TensorData::new(v.clone(), [STATE_SIZE]), &device);
            std::hint::black_box(net.value(t).into_scalar().to_f32());
        }
    }
    let forward_1d = t0.elapsed().as_nanos() as f64 / calls;

    // The same forward with the input shaped [1, STATE_SIZE] instead. If the
    // rank is what costs, this is where it shows.
    for v in vecs.iter().take(16) {
        let t = Tensor::<B, 2>::from_data(TensorData::new(v.clone(), [1, STATE_SIZE]), &device);
        std::hint::black_box(net.value_batched(t).into_scalar().to_f32());
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        for v in &vecs {
            let t = Tensor::<B, 2>::from_data(TensorData::new(v.clone(), [1, STATE_SIZE]), &device);
            std::hint::black_box(net.value_batched(t).into_scalar().to_f32());
        }
    }
    let forward_2d = t0.elapsed().as_nanos() as f64 / calls;

    // And a real batch, to see what the per-row cost looks like once the
    // fixed overhead is amortised.
    let batch = vecs.len().min(256);
    let flat: Vec<f32> = vecs.iter().take(batch).flat_map(|v| v.iter().copied()).collect();
    let t0 = Instant::now();
    for _ in 0..reps {
        let t = Tensor::<B, 2>::from_data(
            TensorData::new(flat.clone(), [batch, STATE_SIZE]),
            &device,
        );
        std::hint::black_box(net.value_batched(t).into_data());
    }
    let per_row = t0.elapsed().as_nanos() as f64 / (reps * batch) as f64;

    println!("encode only        : {encode:>10.0} ns");
    println!("build 1-D tensor   : {make_1d:>10.0} ns");
    println!("forward, rank 1    : {forward_1d:>10.0} ns");
    println!("forward, rank 2 [1]: {forward_2d:>10.0} ns");
    println!("forward, batch {batch:<3} : {per_row:>10.0} ns/row");
}
