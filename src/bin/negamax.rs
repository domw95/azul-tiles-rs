use azul_tiles_rs::players::minimax::Minimaxer;
use azul_tiles_rs::players::Player;
use azul_tiles_rs::runner::PlayerRanker;
use minimaxer::negamax::SearchOptions;

fn main() {
    env_logger::init();
    // Compare performance of a bunch of minimaxer based players
    let players: Vec<Box<dyn Player<2, 6>>> = vec![
        // Search to depth 1 for every move
        Box::new(Minimaxer::new(
            SearchOptions {
                max_depth: Some(1),
                ..Default::default()
            },
            "Depth 1",
        )),
        // Search to depth 2 for every move
        Box::new(Minimaxer::new(
            SearchOptions {
                max_depth: Some(2),
                ..Default::default()
            },
            "Depth 2",
        )),
        // Search for 100ms
        Box::new(Minimaxer::new(
            SearchOptions {
                iterative: true,
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(100)),
                ..Default::default()
            },
            "100ms",
        )),
        // Search for 200ms
        Box::new(Minimaxer::new(
            SearchOptions {
                iterative: true,
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(200)),
                ..Default::default()
            },
            "200ms",
        )),
    ];

    let mut ranker = PlayerRanker::new(players);
    ranker.rank_players(10);
}
