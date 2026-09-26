//! Is the opening position seat-symmetric under the policy's encoding?
//!
//! Both generators hardcode `first_player = 0`, so seat 0 is the first mover of
//! round 1 in every game and seat 1 never is. That looks like a coverage hole
//! for a trainer switching to alternating seats -- the agent would play seat 1
//! in round 1, a position class the data only holds from the other side.
//!
//! But the encoding is taken from the mover's point of view, and at the first
//! ply of a game both boards are empty, no wall is filled, no score is on the
//! board and the first-player tile is still in the centre. If the two views are
//! bit-identical then seat 0's coverage *is* seat 1's coverage and the hole
//! does not exist. Asserted rather than argued, because the whole question is
//! whether a specific pair of float arrays are equal.
use azul_tiles_rs::gamestate::Gamestate;
use azul_tiles_rs::players::nn::{gs_to_array_ordered, FactoryOrder};

fn main() {
    let mut differ = 0;
    let mut checked = 0;
    for seed in 0..200u64 {
        let gs = Gamestate::<2, 6>::new_2_player_with_seed(seed, 0);
        let order = FactoryOrder::canonical(&gs);
        let as_seat0 = gs_to_array_ordered(&gs, 0, &order);
        let as_seat1 = gs_to_array_ordered(&gs, 1, &order);
        checked += 1;
        let d: Vec<usize> = as_seat0
            .as_slice()
            .iter()
            .zip(as_seat1.as_slice().iter())
            .enumerate()
            .filter(|(_, (a, b))| (*a - *b).abs() > 1e-9)
            .map(|(i, _)| i)
            .collect();
        if !d.is_empty() {
            if differ == 0 {
                println!("seed {seed}: {} of {} features differ, at indices {:?}",
                    d.len(), as_seat0.len(), &d[..d.len().min(12)]);
            }
            differ += 1;
        }
    }
    println!("\n{differ} of {checked} opening positions differ between seat views");
    if differ == 0 {
        println!("=> the opening is seat-symmetric under this encoding, so hardcoding");
        println!("   first_player = 0 leaves no round-1 coverage hole for seat 1.");
    } else {
        println!("=> the opening is NOT seat-symmetric, so seat 1 genuinely lacks");
        println!("   first-mover-of-round-1 coverage and the two skews compound.");
    }
}
