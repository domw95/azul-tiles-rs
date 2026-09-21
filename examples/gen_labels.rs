//! Label positions at several search depths at once.
//! Args: <max_depth> <games> <out_dir> [chunk]
use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::ScoreEvaluator;
use azul_tiles_rs::players::nn::gs_to_array_for;
use azul_tiles_rs::players::ppo::pretrain::{MultiDataset, Replay};
use azul_tiles_rs::players::ppo::ACTION_SIZE;
use minimaxer::negamax::{Negamax, SearchOptions};
use minimaxer::node::Node;
use rand::Rng;
use rayon::prelude::*;

fn opts(depth: u8) -> SearchOptions {
    SearchOptions {
        max_depth: Some(depth),
        alpha_beta: true,
        sort_on_create: true,
        sort_on_create_min_depth: 1,
        tt_bits: 20,
        random_best: true,
        ..Default::default()
    }
}

fn mask_for(moves: &[Move]) -> Vec<f32> {
    let mut m = vec![-1e8f32; ACTION_SIZE];
    for mv in moves {
        m[mv.to_index()] = 0.0;
    }
    m
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let max_depth: u8 = a[1].parse().unwrap();
    let games: u64 = a[2].parse().unwrap();
    let out = std::path::PathBuf::from(&a[3]);
    let chunk: u64 = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(2000);
    std::fs::create_dir_all(&out).unwrap();
    let depths: Vec<u8> = (1..=max_depth).collect();
    let epsilon = 0.25f64;

    println!("depths {depths:?}, {games} games, chunks of {chunk} -> {}", out.display());
    let started = std::time::Instant::now();

    for (shard, base) in (0..games).step_by(chunk as usize).enumerate() {
        let end = (base + chunk).min(games);
        let parts: Vec<(MultiDataset, u64, Vec<i32>)> = (base..end)
            .into_par_iter()
            .map(|seed| {
                let mut rng = rand::thread_rng();
                let mut d = MultiDataset {
                    depths: depths.clone(),
                    targets: vec![Vec::new(); depths.len()],
                    values: vec![Vec::new(); depths.len()],
                    ..Default::default()
                };
                let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
                // Record what was actually played, so these positions can be
                // rebuilt and re-encoded later without repeating the searches.
                let mut played_idx: Vec<i32> = Vec::new();
                loop {
                    let moves = gs.get_moves();
                    let seat = gs.current_player() as usize;
                    d.states.extend_from_slice(gs_to_array_for(&gs, seat).as_slice());
                    d.masks.extend_from_slice(&mask_for(&moves));

                    // One search per depth. The deepest dominates the cost, so
                    // the shallower labels are nearly free.
                    let mut deepest = moves[0];
                    for (i, &depth) in depths.iter().enumerate() {
                        let mut n =
                            Negamax::new(Node::new(gs.clone()), ScoreEvaluator, opts(depth));
                        let r = n.search();
                        d.targets[i].push(r.best.to_index() as i32);
                        d.values[i].push(r.value);
                        deepest = r.best;
                    }

                    let played = if rng.gen_bool(epsilon) {
                        moves[rng.gen_range(0..moves.len())]
                    } else {
                        deepest
                    };
                    played_idx.push(played.to_index() as i32);
                    let s = gs.play_move(played);
                    if s == State::RoundEnd && gs.end_round() == State::GameEnd {
                        break;
                    }
                }
                (d, seed, played_idx)
            })
            .collect();

        let mut merged = MultiDataset {
            depths: depths.clone(),
            targets: vec![Vec::new(); depths.len()],
            values: vec![Vec::new(); depths.len()],
            ..Default::default()
        };
        let mut replay = Replay::default();
        for (p, seed, played) in parts {
            replay.seeds.push(seed);
            replay.plies.push(played.len() as u32);
            replay.played.extend(played);
            merged.states.extend(p.states);
            merged.masks.extend(p.masks);
            for i in 0..depths.len() {
                merged.targets[i].extend(&p.targets[i]);
                merged.values[i].extend(&p.values[i]);
            }
        }
        let path = out.join(format!("shard_{shard:04}.bin"));
        merged.save_shard(&path).unwrap();
        replay.save(&out.join(format!("replay_{shard:04}.bin"))).unwrap();
        println!(
            "shard {shard}: games {base}..{end}, {} positions, {:.0}s elapsed",
            merged.len(),
            started.elapsed().as_secs_f32()
        );
    }
}
