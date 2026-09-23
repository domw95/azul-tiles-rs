//! Paired scoring: two checkpoints on identical deals. Args: <a_dir> <b_dir> [games]
//!
//! Unpaired win rates waste most of their samples. Azul's per-game margin has a
//! standard deviation of about 15.6 points against a fixed opponent, so at 300
//! games a win rate carries +/-2.8pp of noise per seat and cannot resolve
//! anything under roughly 8 points -- which has already overturned several
//! conclusions on this project after re-measurement.
//!
//! Playing both candidates on the same seeds makes the deal a shared factor
//! rather than a source of noise: the comparison is then the mean of per-deal
//! differences, whose variance is 2*sd^2*(1-rho) instead of 2*sd^2. The
//! correlation rho is reported so the size of that saving is visible rather
//! than assumed, and it is not guaranteed to be large -- a seed fixes the bag
//! order, but discards depend on play, so paired games drift apart in later
//! rounds.
//!
//! Reported per seat and pooled: each arm's win rate and mean margin, the mean
//! paired difference with its standard error, and a McNemar count over the
//! deals the two arms disagree on.
use azul_tiles_rs::gamestate::{Gamestate, State};
use azul_tiles_rs::players::minimax::{Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::ppo::PPOMoveSelector;
use azul_tiles_rs::players::Player;
use burn::backend::NdArray;
use minimaxer::negamax::SearchOptions;

type B = NdArray;

fn opponent_depth() -> u8 {
    std::env::var("AZUL_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn opponent() -> Minimaxer<ScoreEvaluator> {
    Minimaxer::new(
        SearchOptions {
            max_depth: Some(opponent_depth()),
            alpha_beta: true,
            sort_on_create: true,
            sort_on_create_min_depth: 1,
            tt_bits: 20,
            ..Default::default()
        },
        "opponent",
        ScoreEvaluator,
    )
}

/// One game: the policy in `seat`, minimax in the other. Returns its margin.
fn play(ppo: &PPOMoveSelector<B>, seat: usize, seed: u64) -> f32 {
    let mut opp = opponent();
    let mut gs = Gamestate::new_2_player_with_seed(seed, 0);
    loop {
        let moves = gs.get_moves();
        let state = if gs.current_player() as usize == seat {
            let m = ppo.pick_move_greedy(&gs, &moves);
            gs.play_move(m)
        } else {
            gs.play_move(opp.pick_move(&gs, moves))
        };
        if state == State::RoundEnd && gs.end_round() == State::GameEnd {
            break;
        }
    }
    let s = gs.scores();
    s[seat] as f32 - s[1 - seat] as f32
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64
}

fn sd(v: &[f32], m: f64) -> f64 {
    (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// Report one seat's worth of paired games, and return the raw differences so
/// the seats can be pooled.
fn report(label: &str, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let diffs: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let (ma, mb, md) = (mean(a), mean(b), mean(&diffs));
    let (sa, sb, sd_d) = (sd(a, ma), sd(b, mb), sd(&diffs, md));
    // Correlation implied by the variance identity var(d) = va + vb - 2*cov.
    let rho = if sa > 0.0 && sb > 0.0 {
        ((sa * sa + sb * sb - sd_d * sd_d) / (2.0 * sa * sb)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let se_paired = sd_d / (n as f64).sqrt();
    let se_unpaired = (sa * sa / n as f64 + sb * sb / n as f64).sqrt();
    let wins_a = a.iter().filter(|&&x| x > 0.0).count();
    let wins_b = b.iter().filter(|&&x| x > 0.0).count();
    // McNemar: only the deals where the two arms disagree carry information
    // about which is better on a win/loss basis.
    let a_only = a.iter().zip(b).filter(|(&x, &y)| x > 0.0 && y <= 0.0).count();
    let b_only = a.iter().zip(b).filter(|(&x, &y)| x <= 0.0 && y > 0.0).count();

    println!(
        "{label}: A {:.1}% margin {:+.2} | B {:.1}% margin {:+.2}",
        100.0 * wins_a as f64 / n as f64,
        ma,
        100.0 * wins_b as f64 / n as f64,
        mb
    );
    println!(
        "{label}: paired diff {:+.2} +/- {:.2} (z {:+.2}) | rho {:.3} | unpaired se would be {:.2} ({:.2}x wider)",
        md,
        se_paired,
        if se_paired > 0.0 { md / se_paired } else { 0.0 },
        rho,
        se_unpaired,
        if se_paired > 0.0 { se_unpaired / se_paired } else { 1.0 }
    );
    println!("{label}: McNemar A-only {a_only}, B-only {b_only}, agreed {}", n - a_only - b_only);
    diffs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir_a, dir_b) = (args[1].clone(), args[2].clone());
    let games: u64 = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(1000);
    let device = Default::default();

    let a = PPOMoveSelector::<B>::from_checkpoint(std::path::Path::new(&dir_a), "best", &device)
        .expect("load A");
    let b = PPOMoveSelector::<B>::from_checkpoint(std::path::Path::new(&dir_b), "best", &device)
        .expect("load B");
    println!("A = {dir_a}\nB = {dir_b}\n{games} paired games per seat, opponent depth {}", opponent_depth());

    let mut all = Vec::new();
    for seat in [0usize, 1] {
        let mut ma = Vec::with_capacity(games as usize);
        let mut mb = Vec::with_capacity(games as usize);
        for seed in 9_000_000..9_000_000 + games {
            ma.push(play(&a, seat, seed));
            mb.push(play(&b, seat, seed));
        }
        all.extend(report(&format!("seat{seat}"), &ma, &mb));
    }

    let md = mean(&all);
    let sdd = sd(&all, md);
    let se = sdd / (all.len() as f64).sqrt();
    println!(
        "\nPOOLED over {} paired games: diff {:+.2} +/- {:.2} (z {:+.2}), 95% CI [{:+.2}, {:+.2}]",
        all.len(),
        md,
        se,
        if se > 0.0 { md / se } else { 0.0 },
        md - 1.96 * se,
        md + 1.96 * se
    );
    println!("(positive favours A; a CI spanning zero means the comparison is unresolved)");
}
