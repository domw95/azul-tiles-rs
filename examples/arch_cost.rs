#![recursion_limit = "512"]
//! What a batch-1 forward costs as a function of network shape.
//! Args: none. Run with `MATMUL_NUM_THREADS=1`.
//!
//! Two questions at once, and they have one experiment between them: is burn
//! the limit, or is the network simply too big? Sweeping the shape separates
//! them, because the two answers have different signatures. A cost that is
//! flat across shapes is fixed per-call overhead and says the framework is the
//! problem. A cost proportional to the weights says the arithmetic is, and no
//! amount of hand-rolling fixes it -- only a smaller or incremental network
//! does. The intercept and the slope are the two answers.
//!
//! The last shape is the one that matters. NNUE's proposal is that the first
//! layer becomes an accumulator updated a few dozen adds at a time rather than
//! recomputed, so what actually runs per leaf is only the tail: 256 -> 32 -> 1.
//! Timing that directly says what the NNUE path would cost per leaf *before*
//! quantisation and SIMD, which is the honest floor for "is this worth
//! building".
use azul_tiles_rs::players::ppo::{PPOConfig, PPOMoveSelector, PolicyConfig, ValueConfig};
use burn::backend::NdArray;
use burn::tensor::cast::ToElement as _;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

type B = NdArray;

struct Shape {
    label: &'static str,
    input: usize,
    hidden: usize,
    layers: usize,
}

/// A dense f32 network with nothing underneath it.
///
/// Here to answer "is burn the limit" by subtraction rather than inference:
/// the same shape, the same arithmetic, no framework. Deliberately naive --
/// plain loops, f32, no SIMD intrinsics, no quantisation, no incremental
/// anything. That makes it a *ceiling* on what leaving burn costs and a floor
/// on what hand-rolling could achieve, which is the useful direction: if even
/// this is fast enough, the optimised version certainly is.
struct Dense {
    /// Row-major `[out * in]` per layer, with its bias.
    layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
}

impl Dense {
    fn new(sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for w in sizes.windows(2) {
            let (i, o) = (w[0], w[1]);
            layers.push((
                (0..i * o).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect(),
                (0..o).map(|k| (k % 5) as f32 * 0.01).collect(),
                i,
                o,
            ));
        }
        Self { layers }
    }

    fn forward(&self, x: &[f32], scratch: &mut Vec<f32>) -> f32 {
        let mut cur = x.to_vec();
        for (li, (w, b, i, o)) in self.layers.iter().enumerate() {
            scratch.clear();
            scratch.resize(*o, 0.0);
            for oi in 0..*o {
                let row = &w[oi * i..oi * i + i];
                let mut sum = b[oi];
                for (k, v) in row.iter().enumerate() {
                    sum += v * cur[k];
                }
                // ReLU on the hidden layers, raw on the output, matching the
                // burn `Value` head.
                scratch[oi] = if li + 1 == self.layers.len() {
                    sum
                } else {
                    sum.max(0.0)
                };
            }
            cur.clear();
            cur.extend_from_slice(scratch);
        }
        cur[0]
    }
}

fn main() {
    let shapes = [
        Shape { label: "321->320->320->1  (current)", input: 321, hidden: 320, layers: 1 },
        Shape { label: "321->256->256->1", input: 321, hidden: 256, layers: 1 },
        Shape { label: "321->256->1", input: 321, hidden: 256, layers: 0 },
        Shape { label: "321->128->1", input: 321, hidden: 128, layers: 0 },
        Shape { label: "321->64->1", input: 321, hidden: 64, layers: 0 },
        Shape { label: "321->32->1", input: 321, hidden: 32, layers: 0 },
        Shape { label: "321->8->1", input: 321, hidden: 8, layers: 0 },
        // The NNUE tail: what runs per leaf once the first layer is an
        // incremental accumulator rather than a matmul.
        Shape { label: "256->32->1  (NNUE tail)", input: 256, hidden: 32, layers: 0 },
        // Almost no arithmetic at all. Whatever this costs is burn's per-call
        // overhead with the network subtracted out, and it is the floor no
        // amount of shrinking can get under while burn is in the search loop.
        Shape { label: "1->1->1  (burn floor)", input: 1, hidden: 1, layers: 0 },
    ];

    let device = Default::default();
    let reps = 3000;
    println!(
        "{:<30} {:>9} {:>9} {:>11} {:>10}",
        "shape", "MACs", "wt KiB", "ns/eval", "ns/kMAC"
    );

    for s in &shapes {
        let net = PPOMoveSelector::<B>::new(
            PPOConfig::new(
                PolicyConfig::new(s.input, s.hidden),
                ValueConfig::new(s.input, s.hidden).with_hidden_layers(s.layers),
            ),
            &device,
        );
        let macs = s.input * s.hidden + s.layers * s.hidden * s.hidden + s.hidden;
        // Weights are f32 here. This column is the one to watch: a batch-1
        // forward has to read every weight exactly once and reuses none of
        // them, so it is a streaming read of this much memory, and once it
        // stops fitting in cache that is the floor whatever the code does.
        let kib = (macs * 4) as f64 / 1024.0;

        let input: Vec<f32> = (0..s.input).map(|i| (i % 7) as f32 * 0.1).collect();
        let make = |v: &Vec<f32>| {
            Tensor::<B, 1>::from_data(TensorData::new(v.clone(), [s.input]), &device)
        };
        for _ in 0..64 {
            std::hint::black_box(net.value(make(&input)).into_scalar().to_f32());
        }

        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(net.value(make(&input)).into_scalar().to_f32());
        }
        let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
        println!(
            "{:<30} {macs:>9} {kib:>9.1} {ns:>11.0} {:>10.1}",
            s.label,
            ns / (macs as f64 / 1000.0)
        );
    }

    // The same arithmetic with no framework under it. Same run, so the load
    // that inflates the numbers above inflates these too and the comparison
    // survives a contended box; absolute nanoseconds here do not.
    println!("\nhand-rolled f32, same run:");
    for (label, sizes) in [
        ("321->320->320->1", vec![321usize, 320, 320, 1]),
        ("321->256->1", vec![321, 256, 1]),
        ("256->32->1  (NNUE tail)", vec![256, 32, 1]),
    ] {
        let net = Dense::new(&sizes);
        let input: Vec<f32> = (0..sizes[0]).map(|i| (i % 7) as f32 * 0.1).collect();
        let mut scratch = Vec::new();
        for _ in 0..256 {
            std::hint::black_box(net.forward(&input, &mut scratch));
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(net.forward(std::hint::black_box(&input), &mut scratch));
        }
        let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
        let macs: usize = sizes.windows(2).map(|w| w[0] * w[1]).sum();
        println!(
            "{:<30} {macs:>9} {:>9.1} {ns:>11.0} {:>10.1}",
            label,
            (macs * 4) as f64 / 1024.0,
            ns / (macs as f64 / 1000.0)
        );
    }
}
