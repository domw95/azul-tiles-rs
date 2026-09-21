//! Sanity-check a label directory before relying on it. Args: <dir>
use azul_tiles_rs::players::ppo::pretrain::MultiDataset;
use azul_tiles_rs::players::ppo::{ACTION_SIZE, STATE_SIZE};

/// Check each shard's byte size against the count in its own header, before
/// spending minutes loading gigabytes.
///
/// A shard whose generator was killed part way through still has a plausible
/// header and still loads -- `load_shard` slices only what the header asks
/// for -- so a truncated or over-long file is silent until the numbers come
/// out strange. The layout is: n:u64, nd:u8, depths:[u8; nd], then n states,
/// n masks, then per depth n targets and n values.
fn check_shard_sizes(dir: &std::path::Path) -> std::io::Result<()> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard_")))
        .collect();
    paths.sort();
    let mut bad = 0;
    for p in &paths {
        let mut head = [0u8; 9];
        let f = std::fs::File::open(p)?;
        std::io::Read::read_exact(&mut &f, &mut head)?;
        let n = u64::from_le_bytes(head[0..8].try_into().unwrap()) as usize;
        let nd = head[8] as usize;
        let expect = 9 + nd + n * (STATE_SIZE * 4 + ACTION_SIZE * 4 + nd * 8);
        let actual = f.metadata()?.len() as usize;
        if actual != expect {
            println!(
                "  {}: header says {n} positions at {nd} depths, so {expect} bytes, but the file is {actual}",
                p.file_name().unwrap().to_string_lossy()
            );
            bad += 1;
        }
    }
    println!("{} shards, {bad} with a size that disagrees with its header", paths.len());
    Ok(())
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    check_shard_sizes(std::path::Path::new(&dir)).expect("shard sizes");
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
