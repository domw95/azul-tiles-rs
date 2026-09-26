#![recursion_limit = "512"]
//! Is the clone limited by data volume? Args: <labels_dir> <depth> <hidden> <layers> <max_epochs> <batch> [fractions]
//!
//! The clone plateaus near 56-57% validation agreement against a depth-2
//! teacher whose learnable ceiling measures ~86.5%, and the nets overfit rather
//! than underfit, so the 30-point gap is a generalisation gap. The question
//! this answers is whether more data would close it.
//!
//! Two design points that the obvious version of this experiment gets wrong:
//!
//! - **The validation set is fixed across every fraction.** Scoring each
//!   fraction against its own tail would compare numbers measured on different
//!   problems.
//! - **Fractions are sampled by whole game, uniformly at random.** Positions
//!   arrive ply by ply, so taking the first N rows takes the first N games and
//!   inherits whatever ordering the seed sequence has. Sampling by game also
//!   keeps every game wholly inside one side of the split.
//!
//! Games rather than positions are the unit because a position is a poor unit
//! of independence: plies within a round differ by one move. Rounds are better
//! (each refills the factories from the bag) and games better still, so the
//! curve is reported against all three counts and the reader can pick.
use azul_tiles_rs::players::ppo::pretrain::{
    behaviour_clone_indexed, CloneStop, MultiDataset, Replay,
};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector, PolicyConfig, ValueConfig, STATE_SIZE};
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[cfg(feature = "gpu")]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let depth: u8 = a[2].parse().unwrap();
    let hidden: usize = a[3].parse().unwrap();
    let layers: usize = a[4].parse().unwrap();
    let max_epochs: usize = a[5].parse().unwrap();
    let batch: usize = a[6].parse().unwrap();
    let fractions: Vec<f64> = a
        .get(7)
        .map(|f| f.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0.1, 0.25, 0.5, 1.0]);

    let multi = MultiDataset::load_dir(&dir, "shard_").expect("labels");
    let data = multi.view_depth(depth).unwrap();
    // Game boundaries come from the replay files: plies[i] is the number of
    // labelled positions game i contributed, in the same order the shards were
    // concatenated.
    let replay = Replay::load_dir(&dir, "replay_").expect("replays");
    let mut games: Vec<(usize, usize)> = Vec::with_capacity(replay.plies.len());
    let mut at = 0usize;
    for &p in &replay.plies {
        games.push((at, at + p as usize));
        at += p as usize;
    }
    assert_eq!(
        at,
        data.len(),
        "replay plies ({at}) must account for every labelled position ({})",
        data.len()
    );
    println!(
        "{} positions across {} games ({:.1} per game), depth {depth}",
        data.len(),
        games.len(),
        data.len() as f64 / games.len() as f64
    );

    // Shuffle games once, then hold the last tenth as the fixed validation set.
    // Every fraction below is drawn from the remaining games and scored against
    // this one set.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1EA57);
    games.shuffle(&mut rng);
    let val_games = games.len() / 10;
    let val_idx: Vec<usize> = games[..val_games].iter().flat_map(|&(s, e)| s..e).collect();
    let pool = &games[val_games..];
    println!(
        "validation fixed at {} games / {} positions; training pool {} games\n",
        val_games,
        val_idx.len(),
        pool.len()
    );

    let device = Default::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    println!("{:>6} {:>8} {:>10} {:>8} {:>8} {:>7} {:>7} {:>6}", "frac", "games", "positions", "rounds~", "train", "val", "gap", "ep");
    for f in fractions {
        let take = ((pool.len() as f64 * f).round() as usize).max(1);
        let train_idx: Vec<usize> = pool[..take].iter().flat_map(|&(s, e)| s..e).collect();
        let ppo = PPOMoveSelector::<B>::new(
            PPOConfig::new(
                PolicyConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
                ValueConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
            ),
            &device,
        );
        let (_, s) = behaviour_clone_indexed(
            ppo,
            data,
            &train_idx,
            &val_idx,
            CloneStop { max_epochs, patience: 12, min_delta: 0.0002 },
            batch,
            0.001,
            &device,
        );
        // A round refills every factory, so it is the coarsest unit that is not
        // one move from its neighbour. Azul is 5 rounds in a 2-player game.
        let rounds = take * 5;
        println!(
            "CURVE {:>5.2} {:>8} {:>10} {:>8} {:>8.2} {:>7.2} {:>7.2} {:>6}",
            f,
            take,
            train_idx.len(),
            rounds,
            100.0 * s.best_train,
            100.0 * s.best_val,
            100.0 * (s.best_train - s.best_val),
            s.best_epoch
        );
    }
}
