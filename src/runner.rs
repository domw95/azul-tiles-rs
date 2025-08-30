//! Module for running (multiple) games from start to finish with players

use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

use log::{debug, info};
use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};
use rand_distr::Bernoulli;

use crate::{
    gamestate::{Gamestate, State},
    players::{EvolvingPlayer, Player},
};

/// Game runner
///
/// Runs head to head games between two players,
/// optionally playing the same game with each player
/// playing first
pub struct Runner<const P: usize, const F: usize> {
    players: [Box<dyn Player<P, F>>; P],
    rng: rand::prelude::SmallRng,
}

impl Runner<2, 6> {
    /// Create a new runner with 2 players and optional seed
    pub fn new_2_player(players: [Box<dyn Player<2, 6>>; 2], seed: Option<u64>) -> Self {
        Self {
            players,
            rng: SmallRng::seed_from_u64(seed.unwrap_or(rand::thread_rng().next_u64())),
        }
    }

    /// Run the matchup between the two players
    fn run_matchup(&mut self, games: u32) -> MatchUpResult {
        (0..games)
            .map(|_| {
                let seed = self.rng.next_u64();
                self.play_game_pair(seed)
            })
            .sum()
    }

    /// Play a pair of games with each player starting first
    fn play_game_pair(&mut self, seed: u64) -> GamePairResult {
        let g1 = self.play_game(seed, 0);
        let g2 = self.play_game(seed, 1);
        GamePairResult::new([g1, g2])
    }

    fn play_game(&mut self, seed: u64, first_player: u8) -> GameResult {
        let mut gs = Gamestate::new_2_player_with_seed(seed, first_player);
        while self.play_round(&mut gs) {}
        GameResult::new(&gs)
    }

    pub fn play_round(&mut self, gs: &mut Gamestate<2, 6>) -> bool {
        loop {
            let moves = gs.get_moves();
            let move_ = self.players[gs.current_player() as usize].pick_move(&gs, moves);
            if gs.play_move(move_) == State::RoundEnd {
                return gs.end_round() != State::GameEnd;
            }
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct GameResult {
    scores: [u8; 2],
    winner: Winner,
}

#[derive(Debug, Clone, Copy)]
enum Winner {
    Player0,
    Player1,
    Draw,
}

impl Winner {
    fn new(score: &[u8; 2]) -> Self {
        match score[0].cmp(&score[1]) {
            std::cmp::Ordering::Less => Self::Player1,
            std::cmp::Ordering::Greater => Self::Player0,
            std::cmp::Ordering::Equal => Self::Draw,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct WinnerCount {
    pub player0: u32,
    pub player1: u32,
    pub draw: u32,
}

impl WinnerCount {
    pub fn invert(&self) -> Self {
        Self {
            player0: self.player1,
            player1: self.player0,
            draw: self.draw,
        }
    }
}

impl AddAssign<Winner> for WinnerCount {
    fn add_assign(&mut self, rhs: Winner) {
        match rhs {
            Winner::Player0 => self.player0 += 1,
            Winner::Player1 => self.player1 += 1,
            Winner::Draw => self.draw += 1,
        }
    }
}

impl GameResult {
    fn new(gs: &Gamestate<2, 6>) -> Self {
        let scores = gs.scores();
        let winner = Winner::new(&scores);
        Self { scores, winner }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GamePairResult {
    results: [GameResult; 2],
    score: f64,
}

impl GamePairResult {
    fn new(results: [GameResult; 2]) -> Self {
        Self {
            results,
            score: results
                .iter()
                .map(|r| r.scores[0] as f64 - r.scores[1] as f64)
                .sum(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct MatchUpResult {
    pub games: u32,
    pub score: f64,
    pub winner_count: WinnerCount,
}

impl MatchUpResult {
    pub fn average_score(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.score / self.games as f64
        }
    }

    pub fn invert(&self) -> Self {
        Self {
            games: self.games,
            score: -self.score,
            winner_count: self.winner_count.invert(),
        }
    }
}

impl AddAssign<GamePairResult> for MatchUpResult {
    fn add_assign(&mut self, rhs: GamePairResult) {
        self.games += 2;
        self.score += rhs.score;
        self.winner_count += rhs.results[0].winner;
        self.winner_count += rhs.results[1].winner;
    }
}

impl Sum<GamePairResult> for MatchUpResult {
    fn sum<I: Iterator<Item = GamePairResult>>(iter: I) -> Self {
        let mut result = Self::default();
        for r in iter {
            result += r;
        }
        result
    }
}

/// Rank a list of players by running them all against each other
pub struct PlayerRanker {
    players: Vec<Box<dyn Player<2, 6>>>,
    results: Vec<Vec<MatchUpResult>>,
}

impl PlayerRanker {
    pub fn new(players: Vec<Box<dyn Player<2, 6>>>) -> Self {
        let mut results = vec![vec![]; players.len()];
        for v in &mut results {
            v.resize(players.len(), MatchUpResult::default());
        }
        Self { players, results }
    }

    /// Rank a vec of players by playing them against each other
    pub fn rank_players(&mut self, games: u32) {
        // create a vec of vec of empty match results

        let seed = rand::random();
        // Run each matchup
        for i in 0..self.players.len() {
            for j in (i + 1)..self.players.len() {
                let player1 = dyn_clone::clone_box(&*self.players[i]);
                let player2 = dyn_clone::clone_box(&*self.players[j]);
                let mut runner = Runner::new_2_player([player1, player2], Some(seed));
                let result = runner.run_matchup(games);
                self.results[i][j] = result.invert();
                self.results[j][i] = result;
                info!(
                    "Matchup {} vs {}: {:?}",
                    self.players[i].name(),
                    self.players[j].name(),
                    result
                );
            }
        }
        // Print the upper triangular matrix of results as csv
        for p in self.players.iter() {
            print!("{},", p.name());
        }
        println!();
        for result in self.results.iter() {
            for r in result {
                print!("{:?},", r.average_score());
            }
            println!();
        }
    }
}

pub struct Population<T> {
    players: Option<Vec<T>>,
    ranked_players: Option<Vec<(T, f64, MatchUpResult)>>,
    opponent: Box<dyn Player<2, 6>>,
}

impl<T: Clone + EvolvingPlayer + Player<2, 6> + 'static> Population<T> {
    pub fn new(players: Vec<T>, opponent: Box<dyn Player<2, 6>>) -> Self {
        Self {
            players: Some(players),
            ranked_players: None,
            opponent,
        }
    }

    /// Rank a vec of players by playing them against each other
    pub fn rank_players(&mut self, games: u32) -> (T, f64, MatchUpResult) {
        // Create vec of ranked players against the opponent
        let mut players = self
            .players
            .take()
            .unwrap()
            .into_iter()
            .map(|p| {
                // compare the player to opponent
                let mut runner = Runner::new_2_player(
                    [Box::new(p.clone()), dyn_clone::clone_box(&*self.opponent)],
                    Some(0),
                );
                let result = runner.run_matchup(games);
                (p, 0.0, result)
            })
            .collect::<Vec<_>>();

        // compare each player to each other
        // let seed = rand::random();
        // for i in 0..players.len() {
        //     for j in (i + 1)..players.len() {
        //         let player1 = players[i].0.clone();
        //         let player2 = players[j].0.clone();
        //         let mut runner =
        //             Runner::new_2_player([Box::new(player1), Box::new(player2)], Some(seed));
        //         let result = runner.run_matchup(games);
        //         players[i].1 += result.score;
        //         players[j].1 -= result.score;
        //     }
        // }
        // sort by score
        // players.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        players.sort_by(
            |a, b| match b.2.winner_count.player0.cmp(&a.2.winner_count.player0) {
                std::cmp::Ordering::Less => std::cmp::Ordering::Less,
                std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
                std::cmp::Ordering::Equal => b.2.score.partial_cmp(&a.2.score).unwrap(),
            },
        );
        let best = players.first().unwrap().clone();
        self.ranked_players = Some(players);
        best
    }

    pub fn evolve(&mut self) {
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let ranked_players = self.ranked_players.take().unwrap();
        let mut next_pop = Vec::with_capacity(ranked_players.len());
        // Keep the top 10% of players
        let top = ranked_players.len() / 10;
        for (player, _, _) in ranked_players.iter().take(top) {
            next_pop.push(player.clone());
        }
        let top = ranked_players.len() / 10;
        let prob = Bernoulli::new(0.1).unwrap();
        // Mutate the top 10% of players 6 times
        for (player, _, _) in ranked_players.iter().take(top) {
            for _ in 0..6 {
                next_pop.push(player.mutate(prob, &mut rng));
            }
        }

        // Add crossover players
        while next_pop.len() < ranked_players.len() {
            let i = rng.gen_range(0..top);
            let j = loop {
                let j = rng.gen_range(0..top);
                if i != j {
                    break j;
                }
            };
            let player1 = &ranked_players[i].0;
            let player2 = &ranked_players[j].0;
            next_pop.push(player1.crossover(player2, prob));
        }

        // Create last players randomly
        // while next_pop.len() < ranked_players.len() {
        //     next_pop.push(T::birth())
        // }
        self.players = Some(next_pop);
    }
}

#[cfg(test)]
mod test {

    use crate::players::{MoveRankPlayer2, MoveWeightPlayer, RandomPlayer};

    use super::{Population, Runner};

    #[test]
    fn test_compare_players() {
        let player1 = Box::new(crate::players::MoveRankPlayer);
        let player2 = Box::new(crate::players::MoveRankPlayer2);
        let mut runner = Runner::new_2_player([player1, player2], Some(rand::random()));
        let result = runner.run_matchup(10000);
        dbg!(result);
    }

    #[test]
    fn test_rank_players() {
        let players = (0..100).map(|_| MoveWeightPlayer::new_random()).collect();
        let opponent = Box::new(MoveRankPlayer2::new());
        let mut population = Population::new(players, opponent);
        let best = population.rank_players(10);
        dbg!(&best);
        for _ in 0..10 {
            population.evolve();
            let best = population.rank_players(10);
            dbg!(&best.2.winner_count.player0);
        }
    }
}
