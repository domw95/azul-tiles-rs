use azul_tiles_rs::players::minimax::{HeuristicEvaluator, Minimaxer, ScoreEvaluator};
use azul_tiles_rs::players::Player;
use azul_tiles_rs::runner::PlayerRanker;
use minimaxer::negamax::SearchOptions;

fn main() {
    env_logger::init();
    // Compare performance of a bunch of minimaxer based players
    let players: Vec<Box<dyn Player<2, 6>>> = vec![
        // Search to depth 1 for every move
        // Box::new(Minimaxer::new(
        //     SearchOptions {
        //         max_depth: Some(1),
        //         ..Default::default()
        //     },
        //     "Depth 1",
        //     ScoreEvaluator,
        // )),
        // Search to depth 2 for every move
        Box::new(Minimaxer::new(
            SearchOptions {
                max_depth: Some(2),
                ..Default::default()
            },
            "Depth 2",
            ScoreEvaluator,
        )),
        // Search for 100ms
        Box::new(Minimaxer::new(
            SearchOptions {
                iterative: true,
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(10)),
                ..Default::default()
            },
            "10ms",
            ScoreEvaluator,
        )),
        Box::new(Minimaxer::new(
            SearchOptions {
                iterative: true,
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(10)),
                ..Default::default()
            },
            "Heuristic 10ms",
            HeuristicEvaluator::default(),
        )),
        Box::new(Minimaxer::new(
            SearchOptions {
                iterative: true,
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(10)),
                ..Default::default()
            },
            "Heuristic 10ms No Wall",
            HeuristicEvaluator::new_no_wall_weight(0.5),
        )),
        // // Search for 100ms parallel
        // Box::new(Minimaxer::new(
        //     SearchOptions {
        //         iterative: true,
        //         alpha_beta: true,
        //         max_time: Some(std::time::Duration::from_millis(100)),
        //         parallel: true,
        //         ..Default::default()
        //     },
        //     "100ms parallel",
        //     ScoreEvaluator,
        // )),
    ];

    let mut ranker = PlayerRanker::new(players);
    ranker.rank_players(20);
}
