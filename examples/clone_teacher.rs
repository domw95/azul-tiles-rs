//! Behaviour-clone depth-1 minimax, then measure the clone against it.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::pretrain::{behaviour_clone, Dataset};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector};
use azul_tiles_rs::players::Player;
use burn::backend::{Autodiff, NdArray};
use minimaxer::negamax::SearchOptions;
use rand::Rng;

type B = Autodiff<NdArray>;

fn teacher() -> Minimaxer<ScoreEvaluator> {
    Minimaxer::new(
        SearchOptions { max_depth: Some(1), ..Default::default() },
        "Depth1",
        ScoreEvaluator,
    )
}

fn main() {
    let games: u64 = std::env::args().nth(1).map(|v| v.parse().unwrap()).unwrap_or(4000);
    let epsilon: f64 = 0.25; // random moves, purely to widen the state distribution
    let device = Default::default();

    // --- collect (state, teacher's move) over both seats ---
    let mut data = Dataset::default();
    let mut t = teacher();
    let mut rng = rand::thread_rng();
    for seed in 0..games {
        let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        loop {
            let moves = gs.get_moves();
            // Label every position with the teacher's choice, whoever is to move.
            let best = t.pick_move(&gs, moves.clone());
            // push_position encodes the state and the label together. Doing it
            // by hand here is how the two came apart before: the encoder sorts
            // the factory displays, so a label built from `Move::to_index`
            // names a different display than the one the network is shown, and
            // nothing anywhere would report it -- the run would simply not
            // learn.
            data.push_position(&gs, &moves, &best);

            // Act randomly some of the time so the dataset is not confined to
            // the narrow trajectory the teacher plays against itself.
            let played = if rng.gen_bool(epsilon) {
                moves[rng.gen_range(0..moves.len())]
            } else {
                best
            };
            let st = gs.play_move(played);
            if st == State::RoundEnd && gs.end_round() == State::GameEnd {
                break;
            }
        }
    }
    println!("collected {} labelled positions from {games} games", data.len());

    let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
    let trained = behaviour_clone(ppo, &data, 12, 256, 0.001, &device);
    let dir = std::path::PathBuf::from("/tmp/bc");
    std::fs::create_dir_all(&dir).unwrap();
    trained.save(&dir, "best").unwrap();
    println!("saved to {}", dir.display());
}
