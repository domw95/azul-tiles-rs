//! One long, time-budgeted training run. Args: <hidden> <layers> <hours> <dir>
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::train::{PPOTrainer, StopCondition, TrainOptions};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector, PolicyConfig, ValueConfig, STATE_SIZE};
use azul_tiles_rs::players::Player;
use burn::backend::{Autodiff, NdArray};
use minimaxer::negamax::SearchOptions;
use std::time::Duration;

type B = Autodiff<NdArray>;

/// Opponent search depth, from AZUL_DEPTH.
fn opponent_depth() -> u8 {
    std::env::var("AZUL_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn depth1() -> Minimaxer<ScoreEvaluator> {
    Minimaxer::new(
        SearchOptions { max_depth: Some(opponent_depth()), alpha_beta: true, sort_on_create: true, sort_on_create_min_depth: 1, tt_bits: 20, ..Default::default() },
        "Depth1",
        ScoreEvaluator,
    )
}

/// Greedy play on deals neither training nor the in-loop eval ever sees.
fn holdout(ppo: &PPOMoveSelector<B>, seat: usize, games: u64) -> (f32, f32, f32) {
    let mut opponent = depth1();
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
    (100.0 * wins as f32 / n, us as f32 / n, them as f32 / n)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let hidden: usize = a[1].parse().unwrap();
    let layers: usize = a[2].parse().unwrap();
    let hours: f64 = a[3].parse().unwrap();
    let dir = std::path::PathBuf::from(&a[4]);
    let lr_decay: f64 = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(1.0);
    // Optional: continue an existing checkpoint, picking the LR schedule up
    // where it stopped rather than resetting to the full rate.
    let resume: Option<String> = a.get(6).cloned();
    let lr_start_episode: usize = a.get(7).map(|v| v.parse().unwrap()).unwrap_or(0);
    // Fine-tuning a supervised policy needs a gentler rate than training from
    // scratch: the critic starts untrained, so early advantages are noise and
    // full-size steps would undo the cloning.
    let learning_rate: f64 = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(0.001);

    let device = Default::default();
    let config = PPOConfig::new(
        PolicyConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
        ValueConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
    );
    println!("hidden={hidden} layers={layers} hours={hours} lr_decay={lr_decay} dir={}", dir.display());

    let ppo = match &resume {
        Some(from) => {
            let p = PPOMoveSelector::<B>::from_checkpoint(std::path::Path::new(from), "best", &device)
                .expect("resume checkpoint should load");
            println!("resumed from {from} at lr_start_episode={lr_start_episode}");
            p
        }
        None => PPOMoveSelector::<B>::new(config, &device),
    };
    let (trained, summary) = PPOTrainer::new(ppo, Box::new(depth1()), &device)
        .with_options(TrainOptions {
            dir: dir.clone(),
            eval_every: 5,
            lr_decay,
            lr_start_episode,
            // Restore Adam moments from the same directory as the weights.
            resume_optimiser: resume.as_ref().map(std::path::PathBuf::from),
            learning_rate,
            stop: StopCondition {
                // The 40-game in-loop eval reads ~1.4-2x higher than a
                // 300-game holdout: the previous fine-tune stopped here at
                // 60% in-loop while actually playing at ~43%. Set well above
                // parity so a run is not ended by its own optimism.
                target_win_rate: 0.90,
                // Long runs: never stop early on a noisy flat stretch.
                // Stop when it genuinely stops improving, not on a clock.
                patience: 2000,
                min_episodes: 1000,
                max_episodes: 10_000_000,
                max_duration: Some(Duration::from_secs_f64(hours * 3600.0)),
                smoothing: 0.2,
            },
            ..Default::default()
        })
        .train();

    println!("SUMMARY {summary:?}");
    // The saved best, not whatever was current when the clock ran out.
    let best = PPOMoveSelector::<B>::from_checkpoint(&dir, "best", &device).unwrap_or(trained);
    let (w0, u0, t0) = holdout(&best, 0, 300);
    let (w1, u1, t1) = holdout(&best, 1, 300);
    println!("FINAL hidden={hidden} layers={layers} episodes={} | seat0 {w0:.1}% {u0:.1}v{t0:.1} | seat1 {w1:.1}% {u1:.1}v{t1:.1}",
        summary.episodes_run);
}
