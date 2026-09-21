//! Behaviour-clone a minimax teacher. Args: <depth> [games] [epochs]
use azul_tiles_rs::gamestate::{Gamestate, Move, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::nn::gs_to_array_for;
use azul_tiles_rs::players::ppo::pretrain::{behaviour_clone, Dataset};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector, ACTION_SIZE};
use azul_tiles_rs::players::Player;
use burn::backend::{Autodiff, NdArray};
use minimaxer::negamax::SearchOptions;
use rand::Rng;
use rayon::prelude::*;

type B = Autodiff<NdArray>;

/// The teacher. `random` breaks ties between equally-rated root moves, which
/// widens the state distribution at no cost in strength.
pub fn teacher(depth: u8, random: bool) -> Minimaxer<ScoreEvaluator> {
    Minimaxer::new(
        SearchOptions {
            max_depth: Some(depth),
            alpha_beta: true,
            // Order children by static eval on first visit: alpha-beta only
            // cuts once a good move is found, and at fixed depth `pre_sort`
            // has no previous iteration to learn an ordering from.
            sort_on_create: true,
            sort_on_create_min_depth: 1,
            tt_bits: 20,
            random_best: random,
            ..Default::default()
        },
        format!("Depth{depth}"),
        ScoreEvaluator,
    )
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
    let depth: u8 = a[1].parse().unwrap();
    let games: u64 = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(4000);
    let epochs: usize = a.get(3).map(|v| v.parse().unwrap()).unwrap_or(12);
    let epsilon: f64 = 0.25;
    let device = Default::default();
    // Explicit cache path so a big generation run does not clobber a file a
    // sweep is currently reading.
    let cache = std::path::PathBuf::from(
        a.get(4)
            .cloned()
            .unwrap_or_else(|| format!("/tmp/bc{depth}_data.bin")),
    );

    let start = std::time::Instant::now();
    if let Ok(cached) = Dataset::load(&cache) {
        println!("loaded {} cached positions for depth {depth}", cached.len());
        let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
        let trained = behaviour_clone(ppo, &cached, epochs, 256, 0.001, &device);
        let dir = std::path::PathBuf::from(format!("/tmp/bc{depth}"));
        std::fs::create_dir_all(&dir).unwrap();
        trained.save(&dir, "best").unwrap();
        println!("saved to {}", dir.display());
        return;
    }
    let parts: Vec<(Vec<f32>, Vec<f32>, Vec<i32>)> = (0..games)
        .into_par_iter()
        .map(|seed| {
            let mut t = teacher(depth, true);
            let mut rng = rand::thread_rng();
            let (mut st, mut mk, mut tg) = (Vec::new(), Vec::new(), Vec::new());
            let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
            loop {
                let moves = gs.get_moves();
                let seat = gs.current_player() as usize;
                let best = t.pick_move(&gs, moves.clone());
                st.extend_from_slice(gs_to_array_for(&gs, seat).as_slice());
                mk.extend_from_slice(&mask_for(&moves));
                tg.push(best.to_index() as i32);

                let played = if rng.gen_bool(epsilon) {
                    moves[rng.gen_range(0..moves.len())]
                } else {
                    best
                };
                let s = gs.play_move(played);
                if s == State::RoundEnd && gs.end_round() == State::GameEnd {
                    break;
                }
            }
            (st, mk, tg)
        })
        .collect();

    let mut data = Dataset::default();
    for (st, mk, tg) in parts {
        data.states.extend(st);
        data.masks.extend(mk);
        data.targets.extend(tg);
    }
    println!(
        "depth {depth}: {} positions from {games} games in {:.1}s",
        data.len(),
        start.elapsed().as_secs_f32()
    );

    data.save(&cache).unwrap();
    println!("cached dataset to {}", cache.display());

    let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
    let trained = behaviour_clone(ppo, &data, epochs, 256, 0.001, &device);
    let dir = std::path::PathBuf::from(format!("/tmp/bc{depth}"));
    std::fs::create_dir_all(&dir).unwrap();
    trained.save(&dir, "best").unwrap();
    println!("saved to {}", dir.display());
}
