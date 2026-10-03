//! Where in a game do the expensive positions live?
//!
//! Stratifying the teacher's error by exact-solve cost showed the sharpest
//! available lever: positions whose round takes a large tree to solve carry
//! ~58 points of gap between what a clone could learn and what depth-2 targets
//! deliver, against ~33 points on cheap positions. Upweighting or relabelling
//! by solve cost therefore concentrates effort where it is worth most.
//!
//! But upweighting is only safe if solve cost is *not* a proxy for a narrow
//! slice of the game. If the expensive positions are all the opening ply of
//! round 1, then training on them shifts the distribution away from what a
//! policy actually meets in play -- the same error as covering only the
//! principal variation, in the other direction. This answers that by joining
//! recorded node counts to where each position sits in its game.
//!
//! Costs no search: the replay holds the moves played, so the game is rebuilt
//! by replaying them, and the node count comes from `meta_*.bin`.
//!
//! Args: <labels_dir> [max_shards]
use azul_tiles_rs::gamestate::{Gamestate, State};

/// Node counts in position order, as `gen_full` wrote them.
fn meta_nodes(path: &std::path::Path) -> std::io::Result<Vec<u32>> {
    let b = std::fs::read(path)?;
    let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    // Layout: n:u64, depth:[u8; n], exhaustive:[u8; n], nodes:[u32; n], secs.
    let at = 8 + 2 * n;
    Ok(b[at..at + 4 * n]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// `(seed, plies, played)` per game.
fn replay(path: &std::path::Path) -> std::io::Result<Vec<(u64, Vec<i32>)>> {
    let b = std::fs::read(path)?;
    let ns = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let mut at = 16;
    let seeds: Vec<u64> = b[at..at + 8 * ns]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    at += 8 * ns;
    let plies: Vec<u32> = b[at..at + 4 * ns]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    at += 4 * ns;
    let played: Vec<i32> = b[at..]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut out = Vec::with_capacity(ns);
    let mut off = 0usize;
    for (i, &p) in plies.iter().enumerate() {
        let p = p as usize;
        out.push((seeds[i], played[off..off + p].to_vec()));
        off += p;
    }
    Ok(out)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).map(|s| s.as_str()).unwrap_or("labels_full_heur"));
    let max_shards: usize = a.get(2).map(|v| v.parse().unwrap()).unwrap_or(400);

    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("meta_"))
        })
        .collect();
    shards.sort();
    shards.truncate(max_shards);

    // (round index, ply within round) -> counts, split by whether the exact
    // solve was cheap or dear. 1M is the fleet's node budget, so "dear" here
    // means "at or beyond what the generator was willing to spend".
    const DEAR: u32 = 1_000_000;
    let mut by_round = [[0u64; 2]; 8];
    let mut by_ply = [[0u64; 2]; 16];
    let (mut total, mut dear_total, mut unmatched) = (0u64, 0u64, 0u64);

    for meta_path in &shards {
        let name = meta_path.file_name().unwrap().to_str().unwrap();
        let rp = dir.join(name.replacen("meta_", "replay_", 1));
        let (Ok(nodes), Ok(games)) = (meta_nodes(meta_path), replay(&rp)) else {
            continue;
        };
        let mut idx = 0usize;
        for (seed, played) in games {
            let mut gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
            let (mut round, mut ply_in_round) = (0usize, 0usize);
            for &code in &played {
                if idx >= nodes.len() {
                    break;
                }
                if gs.is_round_over() {
                    if gs.end_round() == State::GameEnd {
                        break;
                    }
                    round += 1;
                    ply_in_round = 0;
                }
                // Replay indices are in real action space; recover the move by
                // matching, since `Move` has no from_index.
                let moves = gs.get_moves();
                let Some(mv) = moves.iter().copied().find(|m| m.to_index() as i32 == code) else {
                    unmatched += 1;
                    break;
                };
                let dear = (nodes[idx] >= DEAR) as usize;
                by_round[round.min(7)][dear] += 1;
                by_ply[ply_in_round.min(15)][dear] += 1;
                total += 1;
                dear_total += dear as u64;
                idx += 1;
                ply_in_round += 1;
                gs.play_move(mv);
            }
        }
    }

    println!(
        "{} shards, {total} positions rebuilt, {:.1}% dear (>= {DEAR} nodes), {unmatched} unmatched\n",
        shards.len(),
        100.0 * dear_total as f64 / total.max(1) as f64
    );
    let show = |label: &str, rows: &[[u64; 2]], name: &str| {
        println!("{label}");
        println!("  {name:>5} {:>9} {:>9} {:>9}", "n", "dear", "dear %");
        for (i, r) in rows.iter().enumerate() {
            let n = r[0] + r[1];
            if n == 0 {
                continue;
            }
            println!("  {i:>5} {n:>9} {:>9} {:>8.1}%", r[1], 100.0 * r[1] as f64 / n as f64);
        }
        println!();
    };
    show("by round index (0 = first round of the game):", &by_round, "round");
    show("by ply within round (0 = first to move after the deal):", &by_ply, "ply");
}
