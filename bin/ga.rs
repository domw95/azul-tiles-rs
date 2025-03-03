use azul_tiles_rs::{
    players::{nn::MoveSelectNN, MoveRankPlayer2, MoveWeightPlayer, SLNNPlayer},
    runner::Population,
};

fn main() {
    let players = (0..400).map(|_| MoveSelectNN::new_random()).collect();
    let opponent = Box::new(MoveRankPlayer2::new());
    let mut population = Population::new(players, opponent);

    let n_games = 50;
    let best = population.rank_players(n_games);
    dbg!(&best);
    for generation in 0..100000 {
        population.evolve();
        let best = population.rank_players(n_games);
        println!(
            "Gen: {}, Score: {}, Wins: {}",
            generation,
            best.2.score / best.2.games as f64,
            best.2.winner_count.player0
        );
        serde_json::to_writer_pretty(std::fs::File::create("move_select_nn.json").unwrap(), &best)
            .unwrap();
    }
    population.evolve();
    dbg!(&population.rank_players(n_games));
}
