use azul_tiles_rs::{
    gamestate::Gamestate,
    players::{MoveRankPlayer2, MoveWeightPlayer},
    runner::Runner,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    // Create a load of walls
    let mut runner =
        Runner::new_2_player([Box::new(MoveRankPlayer2), Box::new(MoveRankPlayer2)], None);
    let walls = (0..100)
        .flat_map(|_| {
            let mut gamestate = Gamestate::new_2_player();
            while runner.play_round(&mut gamestate) {}
            gamestate.boards().map(|b| b.wall)
        })
        .collect::<Vec<_>>();

    c.bench_function("wall_score", |b| {
        b.iter(|| {
            for wall in black_box(&walls) {
                black_box(wall.score());
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
