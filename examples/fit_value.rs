#![recursion_limit = "512"]
//! Fit the critic's value head to a search's root values, and nothing else.
//! Args: <labels_dir> <depth> <out_dir> [max_epochs] [batch] [lr]
//!
//! `pipeline` also clones the policy, which is most of its cost and none of
//! what a search evaluator needs -- the evaluator reads only the value head.
//! This loads only the states and one depth's values, too: the action masks
//! are 1.5 GB on the 2.1M position set and a value fit never looks at them,
//! which on a box with no swap is the difference between fitting and not.
//!
//! Distilling depth-d root values caps the evaluation's quality near depth d
//! plus whatever search amplification adds on top. That ceiling is the point
//! of doing it first: it is the cheapest possible learned evaluator, built
//! from labels already on disk, and what it is for is to find out whether the
//! thing that sinks a learned evaluation here is its quality or its speed.
//!
//! The `gpu` feature builds the wgpu backend, but **it does not currently
//! work for the value fit**: burn's WGSL codegen emits `powf_primitive` twice
//! into one fused shader module and the device rejects it ("redefinition of
//! `powf_primitive`"), then panics again in a destructor during cleanup. The
//! same trap already cost this file's loss function its `powi_scalar`; this is
//! a second instance of it that the workaround does not reach, and it is not
//! attributed here. Cloning the policy on the GPU is unaffected, so this is
//! specific to the value path. Run on the CPU:
//!     cargo run --release --example fit_value -- /tmp/labels_v3 3 /tmp/value_d3
//!
//! That costs little: the measured GPU/CPU comparison on a net this small is a
//! tie, the GPU's only real advantage being that it frees the cores.
use azul_tiles_rs::players::ppo::pretrain::{pretrain_value, CloneStop, MultiDataset};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector};

#[cfg(feature = "gpu")]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let depth: u8 = a[2].parse().unwrap();
    let out = std::path::PathBuf::from(&a[3]);
    let max_epochs: usize = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(60);
    let batch: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(256);
    let lr: f64 = a.get(6).map(|v| v.parse().unwrap()).unwrap_or(0.001);

    let t0 = std::time::Instant::now();
    let (states, values) =
        MultiDataset::load_dir_values(&dir, "shard_", depth).expect("labels");
    println!(
        "{} positions at depth {depth}, loaded in {:.0}s",
        values.len(),
        t0.elapsed().as_secs_f32()
    );

    let device = Default::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    std::fs::create_dir_all(&out).unwrap();
    let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
    let (ppo, summary) = pretrain_value(
        ppo,
        &states,
        &values,
        CloneStop { max_epochs, ..Default::default() },
        batch,
        lr,
        &device,
        // Write every improvement rather than only the final model. An epoch
        // over this set is minutes, so the checkpoint on disk is what makes
        // the run survive a kill -- and makes it raceable before it finishes.
        |best, epoch, ev| {
            best.save(&out, "best").expect("checkpoint");
            println!("  saved epoch {epoch} (ev {ev:+.3}) to {}", out.display());
        },
    );
    println!("value fit: {summary:?}");

    ppo.save(&out, "best").unwrap();
    println!("saved to {} in {:.0}s", out.display(), t0.elapsed().as_secs_f32());
}
