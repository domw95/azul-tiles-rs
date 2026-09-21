//! Sanity-check a label directory before relying on it. Args: <dir>
use azul_tiles_rs::players::ppo::pretrain::MultiDataset;
use azul_tiles_rs::players::ppo::{ACTION_SIZE, STATE_SIZE};

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let d = MultiDataset::load_dir(std::path::Path::new(&dir), "shard_").expect("load");
    let n = d.len();
    println!("positions {n}, depths {:?}", d.depths);
    assert_eq!(d.states.len(), n * STATE_SIZE, "state array length");
    assert_eq!(d.masks.len(), n * ACTION_SIZE, "mask array length");

    for (i, &depth) in d.depths.iter().enumerate() {
        let t = &d.targets[i];
        let v = &d.values[i];
        assert_eq!(t.len(), n);
        assert_eq!(v.len(), n);
        // Every label must be a legal move at its own position.
        let illegal = (0..n)
            .filter(|&p| d.masks[p * ACTION_SIZE + t[p] as usize] != 0.0)
            .count();
        let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
        let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        println!(
            "  depth {depth}: illegal labels {illegal}, non-finite values {nonfinite}, mean value {mean:+.2}"
        );
    }
    // How often do the depths disagree? If they never do, the deeper search is
    // buying nothing and there is no point paying for it.
    if d.depths.len() >= 2 {
        for i in 1..d.depths.len() {
            let diff = (0..n).filter(|&p| d.targets[0][p] != d.targets[i][p]).count();
            println!(
                "  depth {} vs depth {}: differ on {:.1}% of positions",
                d.depths[0],
                d.depths[i],
                100.0 * diff as f32 / n as f32
            );
        }
    }
}
