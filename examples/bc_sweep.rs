#![recursion_limit = "512"]
//! Capacity sweep at fixed data: is the depth-2 policy too small, or out of data?
//!
//! Args: <labels_dir> <depth> <max_epochs> <batch> <out_dir> [grid] [patience] [min_delta]
//! where grid is "320x1,640x2,1536x2" (hidden x hidden_layers), default below.
//!
//! Every cell must reach its own validation plateau, so max_epochs is a safety
//! net rather than the budget: a cell stopped by the cap is a lower bound, and
//! a lower bound on the largest nets is exactly where the trend would be
//! misread, since capacity that has not finished fitting looks like capacity
//! that does not help. Patience and min_delta are arguments because the bigger
//! nets improve in smaller per-epoch steps, and a threshold tuned on the small
//! net cuts them off while they are still climbing.
//!
//! An earlier sweep at 307k positions said capacity does not help, but there
//! every net overfit. At 2.1M the same 320x1 net underfits, so that finding is
//! scoped to the old data volume rather than general, and the frontier has to
//! be located again. Batch is held at one value across the whole grid: capacity
//! is then the only thing that varies, and the cells stay comparable to each
//! other (not to a run at a different batch, since the rate was not retuned).
use azul_tiles_rs::players::ppo::pretrain::{behaviour_clone, CloneStop, MultiDataset};
use azul_tiles_rs::players::ppo::{
    PPOConfig, PPOMoveSelector, PolicyConfig, ValueConfig, ACTION_SIZE, STATE_SIZE,
};
use std::io::Write;

#[cfg(feature = "gpu")]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

/// Trainable parameters in the policy head, which is the only part cloning
/// touches. Two Linears plus `layers` square ones, each with a bias.
fn policy_params(hidden: usize, layers: usize) -> usize {
    (STATE_SIZE * hidden + hidden) + layers * (hidden * hidden + hidden) + (hidden * ACTION_SIZE + ACTION_SIZE)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let depth: u8 = a[2].parse().unwrap();
    let max_epochs: usize = a[3].parse().unwrap();
    let batch: usize = a[4].parse().unwrap();
    let out = std::path::PathBuf::from(&a[5]);
    let patience: usize = a.get(7).map(|v| v.parse().unwrap()).unwrap_or(10);
    let min_delta: f32 = a.get(8).map(|v| v.parse().unwrap()).unwrap_or(0.0002);
    let grid: Vec<(usize, usize)> = a
        .get(6)
        .map(|g| {
            g.split(',')
                .map(|c| {
                    let (h, l) = c.trim().split_once('x').expect("grid cell is HIDDENxLAYERS");
                    (h.parse().expect("hidden"), l.parse().expect("layers"))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(320, 1), (640, 1), (640, 2), (1024, 2), (1536, 2), (1024, 4)]);

    let multi = MultiDataset::load_dir(&dir, "shard_").expect("labels");
    let data = multi.view_depth(depth).unwrap();
    println!(
        "{} positions, depths {:?}, sweeping depth {depth} at batch {batch}, max {max_epochs} epochs, patience {patience}, min_delta {min_delta}",
        data.len(),
        multi.depths
    );
    println!("grid: {grid:?}");

    let device = Default::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    std::fs::create_dir_all(&out).unwrap();
    let mut log = std::fs::File::create(out.join("sweep.jsonl")).unwrap();
    let mut rows = Vec::new();

    for (hidden, layers) in grid {
        let params = policy_params(hidden, layers);
        println!("--- hidden={hidden} layers={layers} params={params} ---");
        let ppo = PPOMoveSelector::<B>::new(
            PPOConfig::new(
                PolicyConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
                ValueConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
            ),
            &device,
        );
        let t0 = std::time::Instant::now();
        let (trained, s) = behaviour_clone(
            ppo,
            data,
            CloneStop { max_epochs, patience, min_delta },
            batch,
            0.001,
            &device,
        );
        let secs = t0.elapsed().as_secs_f32();

        // Keep every checkpoint: the sweep ranks on validation agreement, but
        // the winner still has to be measured by playing games.
        let cdir = out.join(format!("h{hidden}x{layers}"));
        std::fs::create_dir_all(&cdir).unwrap();
        trained.save(&cdir, "best").unwrap();

        let gap = 100.0 * (s.best_train - s.best_val);
        writeln!(
            log,
            "{{\"hidden\":{hidden},\"layers\":{layers},\"params\":{params},\"batch\":{batch},\"train\":{:.2},\"val\":{:.2},\"gap\":{:.2},\"best_epoch\":{},\"epochs_run\":{},\"stopped_early\":{},\"secs\":{:.0}}}",
            100.0 * s.best_train,
            100.0 * s.best_val,
            gap,
            s.best_epoch,
            s.epochs_run,
            s.stopped_early,
            secs
        )
        .unwrap();
        log.flush().unwrap();
        println!(
            "SWEEP hidden={hidden} layers={layers} params={params} train={:.2} val={:.2} gap={:.2} best_epoch={} epochs={} early={} {:.0}s",
            100.0 * s.best_train,
            100.0 * s.best_val,
            gap,
            s.best_epoch,
            s.epochs_run,
            s.stopped_early,
            secs
        );
        rows.push((hidden, layers, params, 100.0 * s.best_train, 100.0 * s.best_val, gap, s.best_epoch, s.epochs_run, s.stopped_early, secs));
    }

    // A cell that stopped on the epoch cap is a lower bound, not a plateau, so
    // say which is which rather than ranking them as if they were the same.
    println!("\n  hidden  layers    params   train     val     gap  best_ep  epochs  stop        secs");
    for (h, l, p, tr, v, g, be, er, early, secs) in &rows {
        println!(
            "  {h:>6}  {l:>6}  {p:>8}  {tr:>6.2}  {v:>6.2}  {g:>6.2}  {be:>7}  {er:>6}  {:<10}  {secs:>6.0}",
            if *early { "plateau" } else { "CAP" }
        );
    }
}
