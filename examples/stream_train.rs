#![recursion_limit = "512"]
//! Clone a teacher from a label directory that stays on disk.
//!
//! The in-memory route (`pipeline`) decodes every shard up front, which costs
//! about 2 KiB of RAM per position and stops being possible somewhere past a
//! few million. This reads the same shards a window at a time.
//!
//! Args: <labels_dir> <depth> <out_dir> [epochs] [batch] [buffer_shards]
//!
//! `depth` is 255 for the full-depth generator's labels.
use azul_tiles_rs::players::ppo::pretrain::CloneStop;
use azul_tiles_rs::players::ppo::stream::{behaviour_clone_streaming, ShardSet};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector};

#[cfg(feature = "gpu")]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

/// Peak resident set, which is the number this example exists to keep small.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn main() {
    env_logger::init();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: stream_train <labels_dir> <depth> <out_dir> [epochs] [batch] [buffer_shards]");
        std::process::exit(2);
    }
    let dir = std::path::PathBuf::from(&a[1]);
    let depth: u8 = a[2].parse().unwrap();
    let out = std::path::PathBuf::from(&a[3]);
    let epochs: usize = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(40);
    let batch: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(256);
    let buffer: usize = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(64);

    let t0 = std::time::Instant::now();
    let all = ShardSet::open(&dir, "shard_", depth).expect("labels");
    let (train, val) = all.split(0.1);
    println!(
        "{} positions in {} shards ({:.0}s to index)\n  train {} / val {}, buffer {buffer} shards, batch {batch}",
        all.len(),
        all.shards(),
        t0.elapsed().as_secs_f32(),
        train.len(),
        val.len(),
    );
    println!("  RSS after indexing: {:.0} MiB", peak_rss_mib());

    let device = Default::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
    let t1 = std::time::Instant::now();
    let (ppo, summary) = behaviour_clone_streaming(
        ppo,
        &train,
        &val,
        CloneStop { max_epochs: epochs, ..Default::default() },
        batch,
        buffer,
        0.001,
        &device,
    )
    .expect("train");

    let secs = t1.elapsed().as_secs_f32();
    println!(
        "clone: {summary:?}\n  {:.0}s total, {:.1}s/epoch, peak RSS {:.0} MiB",
        secs,
        secs / summary.epochs_run.max(1) as f32,
        peak_rss_mib(),
    );
    std::fs::create_dir_all(&out).unwrap();
    ppo.save(&out, "best").unwrap();
    println!("saved to {}", out.display());
}
