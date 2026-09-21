//! Does the depth-2 policy need a bigger network? Args: <dataset> <epochs>
use azul_tiles_rs::players::ppo::pretrain::{behaviour_clone, Dataset};
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector, PolicyConfig, ValueConfig, STATE_SIZE};
use burn::backend::{Autodiff, NdArray};

type B = Autodiff<NdArray>;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = Dataset::load(std::path::Path::new(&a[1])).expect("dataset");
    let epochs: usize = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(12);
    let device = Default::default();
    println!("{} positions", data.len());

    for (hidden, layers) in [(320usize, 1usize), (768, 2), (1536, 2), (768, 4)] {
        println!("--- hidden={hidden} layers={layers} ---");
        let ppo = PPOMoveSelector::<B>::new(
            PPOConfig::new(
                PolicyConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
                ValueConfig::new(STATE_SIZE, hidden).with_hidden_layers(layers),
            ),
            &device,
        );
        let t0 = std::time::Instant::now();
        let trained = behaviour_clone(ppo, data.view(), epochs, 256, 0.001, &device);
        let dir = std::path::PathBuf::from(format!("/tmp/bccap_{hidden}_{layers}"));
        std::fs::create_dir_all(&dir).unwrap();
        trained.save(&dir, "best").unwrap();
        println!("CAP hidden={hidden} layers={layers} took {:.0}s", t0.elapsed().as_secs_f32());
    }
}
