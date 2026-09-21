//! Held-out measurement of a saved checkpoint, per seat. Args: <dir> [games]
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::PPOMoveSelector;
use azul_tiles_rs::players::Player;
use burn::backend::NdArray;
use minimaxer::negamax::SearchOptions;

fn main() {
    type B = NdArray;
    let dir = std::env::args().nth(1).unwrap();
    let games: u64 = std::env::args().nth(2).map(|v| v.parse().unwrap()).unwrap_or(300);
    let device = Default::default();
    let ppo = match PPOMoveSelector::<B>::from_checkpoint(std::path::Path::new(&dir), "best", &device) {
        Ok(p) => p,
        Err(e) => { println!("{dir}: load failed: {e}"); return; }
    };
    for seat in [0usize, 1] {
        let mut opponent = Minimaxer::new(
            SearchOptions { max_depth: Some(1), ..Default::default() },
            "Depth1",
            ScoreEvaluator,
        );
        let (mut wins, mut us, mut them) = (0u32, 0u32, 0u32);
        for seed in 9_000_000..9_000_000 + games {
            let mut gs = Gamestate::new_2_player_with_seed(seed, 0);
            loop {
                let moves = gs.get_moves();
                let state = if gs.current_player() as usize == seat {
                    let m = ppo.pick_move_greedy(&gs, &moves);
                    gs.play_move(m)
                } else {
                    gs.play_move(opponent.pick_move(&gs, moves))
                };
                if state == State::RoundEnd && gs.end_round() == State::GameEnd { break; }
            }
            let s = gs.scores();
            us += s[seat] as u32; them += s[1 - seat] as u32;
            if s[seat] > s[1 - seat] { wins += 1; }
        }
        let n = games as f32;
        println!("{dir} seat{seat}: {wins}/{games} ({:.1}%) mean {:.1} v {:.1}",
            100.0 * wins as f32 / n, us as f32 / n, them as f32 / n);
    }
}
