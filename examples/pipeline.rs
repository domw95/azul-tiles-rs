#![recursion_limit = "512"]
//! Clone a teacher from a multi-depth label set and initialise the critic.
//! Args: <labels_dir> <depth> <out_dir> [policy_epochs] [value_epochs]
use azul_tiles_rs::players::ppo::pretrain::{
    behaviour_clone, pretrain_value, CloneStop, MultiDataset,
};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector};
// Cloning is pure batched supervised training over a fixed dataset -- large
// matmuls, no sequential dependency, no per-sample device syncs -- so unlike
// the RL rollout it is worth putting on a GPU.
//     cargo run --release --features gpu --example pipeline
#[cfg(feature = "gpu")]
type B = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(not(feature = "gpu"))]
type B = burn::backend::Autodiff<burn::backend::NdArray>;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let depth: u8 = a[2].parse().unwrap();
    let out = std::path::PathBuf::from(&a[3]);
    let max_epochs: usize = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(200);
    let ve: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(6);

    let multi = MultiDataset::load_dir(&dir, "shard_").expect("labels");
    let di = multi.depths.iter().position(|&x| x == depth).expect("depth present");
    let data = multi.view_depth(depth).unwrap();
    println!("{} positions, depths {:?}, training on depth {depth}", data.len(), multi.depths);

    let device = Default::default();
    #[cfg(feature = "gpu")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    let t0 = std::time::Instant::now();
    let ppo = PPOMoveSelector::<B>::new(PPOConfig::default(), &device);
    let (ppo, summary) = behaviour_clone(
        ppo,
        data,
        CloneStop { max_epochs, ..Default::default() },
        256,
        0.001,
        &device,
    );
    println!("clone: {summary:?}");
    let (ppo, value_summary) = pretrain_value(
        ppo,
        data.states,
        &multi.values[di],
        CloneStop { max_epochs: ve, ..Default::default() },
        256,
        0.001,
        &device,
        |_, _, _| {},
    );
    println!("value: {value_summary:?}");

    std::fs::create_dir_all(&out).unwrap();
    ppo.save(&out, "best").unwrap();
    println!("saved to {} in {:.0}s", out.display(), t0.elapsed().as_secs_f32());
}
