#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use std::{fs::File, path::PathBuf};

use azul_tiles_rs::{
    gamestate::{Destination, Gamestate, Move, Source},
    playerboard::{wall::WALL_COLOURS, RowIndex},
    players::{
        self,
        minimax::Minimaxer,
        nn::MoveSelectNN,
        ppo::{PPOMoveSelector, PolicyConfig, ValueConfig},
    },
    runner::MatchUpResult,
    tiles::{Tile, TileGroup},
};
use burn::{
    backend::{NdArray, Wgpu},
    tensor::Device,
};
use eframe::egui;
use egui::{Color32, FontId, Key, PointerButton, Pos2, Rect, Stroke, Vec2};

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 1000.0]),

        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| {
            // This gives us image support:
            // egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::<MyApp>::default())
        }),
    )
}

enum Player {
    Ai(Box<dyn players::Player<2, 6>>),
    Human,
}

struct MyApp {
    gs: Gamestate<2, 6>,

    players: [Player; 2],

    /// UI config that changes with screen size
    config: UIConfig,
    /// Track selection of move for human player
    selection: Selection,
}

impl MyApp {
    fn new() -> Self {
        Self::default()
    }

    fn advance_gamestate(&mut self) {
        match self.gs.state() {
            azul_tiles_rs::gamestate::State::RoundActive => {
                let player = &mut self.players[self.gs.current_player() as usize];
                if let Player::Ai(player) = player {
                    let moves = self.gs.get_moves();

                    let m = player.pick_move(&self.gs, moves);
                    self.gs.play_move(m);
                }
            }
            azul_tiles_rs::gamestate::State::RoundEnd => {
                self.gs.end_round();
            }
            azul_tiles_rs::gamestate::State::GameEnd => (),
        }
    }
}

impl MyApp {}

fn key_to_number(key: &Key) -> Option<usize> {
    match key {
        Key::Num0 => Some(0),
        Key::Num1 => Some(1),
        Key::Num2 => Some(2),
        Key::Num3 => Some(3),
        Key::Num4 => Some(4),
        Key::Num5 => Some(5),
        _ => None,
    }
}

type Backend = NdArray;
impl Default for MyApp {
    fn default() -> Self {
        // let (player, _, _): (MoveSelectNN, f64, MatchUpResult) =
        //     serde_json::from_reader(File::open("move_select_nn.json").unwrap()).unwrap();
        let player = Minimaxer::new(
            minimaxer::negamax::SearchOptions {
                alpha_beta: true,
                max_time: Some(std::time::Duration::from_millis(10)),
                iterative: true,
                ..Default::default()
            },
            "Minimaxer",
            players::minimax::ScoreEvaluator,
        );
        let device = Device::<Backend>::default();
        let ppo = PPOMoveSelector::<Backend>::from_file(
            PolicyConfig::new(150, 240),
            ValueConfig::new(150, 240),
            &PathBuf::from("ppo/checkpoint_200"),
            &device,
        );
        Self {
            gs: Gamestate::new_2_player_with_seed(rand::random(), 0),
            config: UIConfig::default(),
            players: [
                Player::Human,
                // Player::Ai(Box::new(azul_tiles_rs::players::MoveRankPlayer)),
                // Player::Ai(Box::new(azul_tiles_rs::players::MoveRankPlayer2)),
                Player::Ai(Box::new(player)),
                // Player::Ai(Box::new(ppo)),
            ],
            selection: Selection::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let window_size = ui.available_size();
            self.config.update(&window_size);

            let key = ctx.input(|input| {
                for event in &input.events {
                    if let egui::Event::Key {
                        key,
                        physical_key: _,
                        pressed: true,
                        repeat: _,
                        modifiers: _,
                    } = event
                    {
                        return Some(*key);
                    }
                }
                None
            });

            let click = ctx.input(|input| {
                for event in &input.events {
                    if let egui::Event::PointerButton {
                        pos,
                        button: PointerButton::Primary,
                        pressed: true,
                        modifiers: _,
                    } = event
                    {
                        return Some(*pos);
                    }
                }
                None
            });

            // Perform actions from space button
            if let Some(Key::Space) = key {
                self.advance_gamestate();
            } else if key == Some(Key::Escape) {
                self.selection = Selection::default();
            } else if let Some(key) = key {
                // If current player is human
                if let Player::Human = self.players[self.gs.current_player() as usize] {
                    // get list of available moves
                    let moves = self.gs.get_moves();
                    // Check if factory selected
                    if let Some(factory) = self.selection.factory {
                        // Check if tile selected
                        if let Some(tile) = self.selection.tile {
                            // Select row
                            if let Some(row) = key_to_number(&key) {
                                let m = if row == 0 {
                                    // Floor
                                    moves.iter().find(|m| {
                                        m.source == Source(factory as u8)
                                            && m.tile == tile
                                            && m.destination == Destination::Floor
                                    })
                                } else {
                                    // Row move
                                    let row = RowIndex::from(row as u8 - 1);
                                    moves.iter().find(|m| {
                                        m.source == Source(factory as u8)
                                            && m.tile == tile
                                            && m.destination == Destination::Row(row)
                                    })
                                };
                                if let Some(m) = m {
                                    self.gs.play_move(*m);
                                    self.selection = Selection::default();
                                } else {
                                    self.selection.row = None;
                                }
                            }
                        } else {
                            // Select tile if valid move
                            if let Some(tile) = key_to_number(&key) {
                                if tile < 5 {
                                    if factory == 0 {
                                        // centre, select by colour
                                        let centre = self.gs.centre();
                                        let tile = Tile::from(tile);
                                        let count = centre.get_count(tile);
                                        if count > 0 {
                                            self.selection.tile = Some(tile);
                                        }
                                    } else {
                                        // factory, select by tile
                                        let tiles =
                                            self.gs.factories()[factory].unwrap().tile_vec();

                                        if tile > 0 && tile < 5 {
                                            let tile = tiles[tile - 1];
                                            if tiles.iter().any(|t| t == &tile) {
                                                self.selection.tile = Some(tile);
                                            }
                                        }
                                    }
                                }
                            }
                            // If a tile has been set, store list of valid moves for highlighting on board
                            if let Some(tile) = self.selection.tile {
                                self.selection.moves = moves
                                    .iter()
                                    .filter(|m| m.tile == tile && m.source == Source(factory as u8))
                                    .cloned()
                                    .collect();
                            }
                        }
                    } else {
                        // Select factory if valid move
                        if let Some(factory) = key_to_number(&key) {
                            if moves.iter().any(|m| m.source == Source(factory as u8)) {
                                self.selection.factory = Some(factory);
                            }
                        }
                    }
                }
            }

            let mut highlight = Highlight::default();
            if self.gs.state() == azul_tiles_rs::gamestate::State::RoundActive {
                highlight.board = Some(self.gs.current_player() as usize);
            }
            highlight.factory = self.selection.factory;
            highlight.tile = self.selection.tile;
            highlight.rows = self.selection.moves.iter().fold([false; 5], |mut acc, m| {
                if let Destination::Row(ind) = m.destination {
                    acc[ind as usize] = true;
                }
                acc
            });
            highlight.floor = self
                .selection
                .moves
                .iter()
                .any(|m| m.destination == Destination::Floor);

            if let Some(click) = draw_game(ui, &self.config, &self.gs, highlight, click) {
                // if human turn, update selection
                if let Player::Human = self.players[self.gs.current_player() as usize] {
                    let moves = self.gs.get_moves();
                    let m = match click {
                        Click::Factory(factory, tile) => {
                            self.selection.factory = Some(factory as usize);
                            self.selection.tile = Some(tile);
                            self.selection.moves = moves
                                .iter()
                                .filter(|m| m.tile == tile && m.source == Source(factory))
                                .cloned()
                                .collect();
                            None
                        }
                        Click::Row(row) => {
                            if let Some(factory) = self.selection.factory {
                                if let Some(tile) = self.selection.tile {
                                    // find move
                                    moves.iter().find(|m| {
                                        m.source == Source(factory as u8)
                                            && m.tile == tile
                                            && m.destination == Destination::Row(row)
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        Click::Floor => {
                            if let Some(factory) = self.selection.factory {
                                if let Some(tile) = self.selection.tile {
                                    // find move
                                    moves.iter().find(|m| {
                                        m.source == Source(factory as u8)
                                            && m.tile == tile
                                            && m.destination == Destination::Floor
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(m) = m {
                        self.gs.play_move(*m);
                        self.selection = Selection::default();
                    }
                }
            } else if let Some(click) = click {
                self.advance_gamestate();
            }
        });
    }
}

#[derive(Debug, Clone, Copy)]
enum Click {
    Factory(u8, Tile),
    Row(RowIndex),
    Floor,
}

#[derive(Debug, Default)]
struct UIConfig {
    window_size: Vec2,
    pub tile_size: f32,
    pub tile_spacing: f32,
    pub tile_rounding: f32,
    pub boards: [BoardUI; 2],
    pub factories: [FactoryUI; 5],
    pub centre: CentreUI,
    pub bag: BagUI,
}

impl UIConfig {
    fn new(window_size: &Vec2) -> Self {
        let mut conf = Self::default();
        conf.update(window_size);
        conf
    }

    fn update(&mut self, window_size: &Vec2) {
        if *window_size == self.window_size {
            return;
        }
        self.window_size = *window_size;
        let height = window_size.y;
        let width = window_size.x;
        self.tile_size = (0.04 * height).clamp(20.0, 50.0);
        self.tile_spacing = self.tile_size * 0.2;
        self.tile_rounding = 0.1 * self.tile_size;
        let board_y_0 = 0.8 * height;
        let board_y_1 = 0.2 * height;
        self.boards[0] = BoardUI::new(
            Pos2::new(0.5 * width, board_y_0),
            self.tile_size,
            self.tile_spacing,
        );
        self.boards[1] = BoardUI::new(
            Pos2::new(0.5 * width, board_y_1),
            self.tile_size,
            self.tile_spacing,
        );
        let factory_space = self.tile_size / 3.0;
        let factory_gap =
            2.0 * (self.tile_size + self.tile_spacing) + self.tile_spacing + factory_space;

        let factory_left = Pos2::new(0.5 * width - 2.5 * factory_gap, 0.5 * height);

        for i in 1..6 {
            self.factories[i - 1] = FactoryUI::new(
                factory_left + Vec2::new(i as f32 * factory_gap, 0.0),
                self.tile_size,
                self.tile_spacing,
            );
        }

        self.bag = BagUI::new(
            Pos2::new(
                0.5 * width - 7.0 * (self.tile_size + self.tile_spacing),
                board_y_1,
            ),
            self.tile_size,
            self.tile_spacing,
        );

        self.centre = CentreUI::new(factory_left, self.tile_size, self.tile_spacing);
    }
}

/// UI layout for a gameboard
#[derive(Debug, Default)]
struct BoardUI {
    centre: Pos2,
    border: Vec2,
    wall: [[Pos2; 5]; 5],
    rows: [[Pos2; 5]; 5],
    floor: [Pos2; 7],
    score: Pos2,
}

impl BoardUI {
    fn new(centre: Pos2, tile_size: f32, tile_space: f32) -> Self {
        let wall_tl = centre + Vec2::new(tile_size + tile_space, -(2.5 * (tile_size + tile_space)));
        let mut wall: [[Pos2; 5]; 5] = Default::default();
        for i in 0..5 {
            for j in 0..5 {
                wall[i][j] = wall_tl
                    + Vec2::new(
                        j as f32 * (tile_size + tile_space),
                        i as f32 * (tile_size + tile_space),
                    );
            }
        }
        let floor_first = centre
            + Vec2::new(
                -5.0 * (tile_size + tile_space),
                2.5 * (tile_size + tile_space) + tile_space,
            );
        let mut floor: [Pos2; 7] = Default::default();
        for i in 0..7 {
            floor[i] = floor_first + Vec2::new(i as f32 * (tile_size + tile_space), 0.0);
        }

        // rows
        let row_tr = wall_tl + Vec2::new(-2.0 * (tile_size + tile_space), 0.0);
        let mut rows: [[Pos2; 5]; 5] = Default::default();
        for i in 0..5 {
            for j in 0..5 {
                rows[i][j] = row_tr
                    + Vec2::new(
                        j as f32 * -(tile_size + tile_space),
                        i as f32 * (tile_size + tile_space),
                    );
            }
        }

        let score = rows[1][3];

        Self {
            centre,
            wall,
            floor,
            border: Vec2::new(
                11.0 * (tile_size + tile_space) + tile_space,
                6.0 * (tile_size + tile_space) + 3.0 * tile_space,
            ),
            rows,
            score,
        }
    }
}

#[derive(Debug, Default)]
struct FactoryUI {
    centre: Pos2,
    border: Vec2,
    tiles: [Pos2; 4],
}

impl FactoryUI {
    fn new(centre: Pos2, tile_size: f32, tile_space: f32) -> Self {
        let mut tiles: [Pos2; 4] = Default::default();
        for i in 0..4 {
            let (x, y) = match i {
                0 => (-1.0, -1.0),
                1 => (1.0, -1.0),
                2 => (-1.0, 1.0),
                3 => (1.0, 1.0),
                _ => unreachable!(),
            };
            tiles[i] = centre
                + Vec2::new(
                    x * 0.5 * (tile_size + tile_space),
                    y * 0.5 * (tile_size + tile_space),
                );
        }
        Self {
            centre,
            border: Vec2::new(
                2.0 * (tile_size + tile_space) + tile_space,
                2.0 * (tile_size + tile_space) + tile_space,
            ),
            tiles,
        }
    }
}

#[derive(Debug, Default)]
struct CentreUI {
    centre: Pos2,
    border: Vec2,
    tiles: [Pos2; 6],
}

impl CentreUI {
    fn new(centre: Pos2, tile_size: f32, tile_space: f32) -> Self {
        let mut c = CentreUI::default();
        c.centre = centre;
        for i in 0..6 {
            let (y, x) = match i {
                0 => (-1.0, -1.0),
                1 => (0.0, -1.0),
                2 => (1.0, -1.0),
                3 => (-1.0, 1.0),
                4 => (0.0, 1.0),
                5 => (1.0, 1.0),
                _ => unreachable!(),
            };
            c.tiles[i] = centre
                + Vec2::new(
                    x * 0.5 * (tile_size + tile_space),
                    y * 1.0 * (tile_size + tile_space),
                )
        }
        c.border = Vec2::new(
            2.0 * (tile_size + tile_space) + tile_space,
            3.0 * (tile_size + tile_space) + tile_space,
        );
        c
    }
}

#[derive(Debug, Default)]
struct BagUI {
    centre: Pos2,
    border: Vec2,
    tiles: [Pos2; 5],
}

impl BagUI {
    fn new(centre: Pos2, tile_size: f32, tile_space: f32) -> Self {
        let mut b = BagUI::default();
        b.centre = centre;
        let top = centre - Vec2::new(0.0, 2.0 * (tile_size + tile_space));
        for i in 0..5 {
            b.tiles[i] = top + Vec2::new(0.0, i as f32 * (tile_size + tile_space))
        }
        b.border = Vec2::new(
            1.0 * (tile_size + tile_space) + tile_space,
            5.0 * (tile_size + tile_space) + tile_space,
        );
        b
    }
}

/// Indicates which parts of the UI should be highlighted for selection
#[derive(Debug, Default, Clone)]
struct Highlight {
    // Which board is highlighted for active player
    board: Option<usize>,
    tile: Option<Tile>,
    factory: Option<usize>,
    rows: [bool; 5],
    floor: bool,
}

#[derive(Debug, Default, Clone)]
struct Selection {
    moves: Vec<Move>,
    factory: Option<usize>,
    tile: Option<Tile>,
    row: Option<RowIndex>,
}

fn draw_game(
    ui: &mut egui::Ui,
    config: &UIConfig,
    gs: &Gamestate<2, 6>,
    highlight: Highlight,
    click: Option<Pos2>,
) -> Option<Click> {
    let mut clicked = None;
    // Draw player boards
    for i in 0..2 {
        clicked = clicked.or(draw_board(ui, config, gs, i, &highlight, click));
    }

    // Draw centre and factories
    clicked = clicked.or(draw_centre(ui, config, gs, &highlight, click));

    for i in 0..5 {
        clicked = clicked.or(draw_factory(ui, config, gs, i, &highlight, click));
    }

    // Draw bag
    draw_bag(ui, config, gs.tilebag());
    clicked
}
// Draw bag of tiles
fn draw_bag(ui: &mut egui::Ui, config: &UIConfig, bag: &TileGroup) {
    for (i, (&count, tile)) in bag.into_iter().enumerate() {
        if count > 0 {
            draw_tile_with_text(
                ui,
                config,
                tile_to_colour(&tile),
                config.bag.tiles[i],
                &count.to_string(),
                tile_to_text_colour(&tile),
                None,
            );
        } else {
            draw_tile_border(
                ui,
                config,
                tile_to_colour(&tile),
                config.bag.tiles[i],
                1.0,
                None,
            );
        }
    }
}

fn draw_centre(
    ui: &mut egui::Ui,
    config: &UIConfig,
    gs: &Gamestate<2, 6>,
    highlight: &Highlight,
    click: Option<Pos2>,
) -> Option<Click> {
    let centre = gs.centre();
    let selected = highlight.factory == Some(0);
    ui.painter().rect_stroke(
        Rect::from_center_size(config.centre.centre, config.centre.border),
        config.tile_rounding,
        if selected {
            Stroke::new(3.0, Color32::PURPLE)
        } else {
            Stroke::new(1.0, Color32::WHITE)
        },
        egui::StrokeKind::Inside,
    );

    let mut clicked = None;

    for (i, (&count, tile)) in centre.into_iter().enumerate() {
        if count > 0 {
            // draw tile with digit
            if draw_tile_with_text(
                ui,
                config,
                tile_to_colour(&tile),
                config.centre.tiles[i],
                &count.to_string(),
                tile_to_text_colour(&tile),
                click,
            ) {
                clicked = Some(Click::Factory(0, tile));
            }
            if selected && highlight.tile == Some(tile) {
                draw_tile_border(
                    ui,
                    config,
                    Color32::PURPLE,
                    config.centre.tiles[i],
                    3.0,
                    None,
                );
            }
        } else {
            // draw outline
            draw_tile_border(
                ui,
                config,
                tile_to_colour(&tile),
                config.centre.tiles[i],
                1.0,
                None,
            );
        }
    }
    if gs.first_player_tile() {
        draw_tile(ui, config, Color32::PURPLE, config.centre.tiles[5], click);
    }
    clicked
}

/// Draw factory to screen
fn draw_factory(
    ui: &mut egui::Ui,
    config: &UIConfig,
    gs: &Gamestate<2, 6>,
    factory: usize,
    highlight: &Highlight,
    click: Option<Pos2>,
) -> Option<Click> {
    let selected = highlight.factory == Some(factory + 1);
    // Draw border
    ui.painter().rect_stroke(
        Rect::from_center_size(
            config.factories[factory].centre,
            config.factories[factory].border,
        ),
        config.tile_rounding,
        if selected {
            Stroke::new(3.0, Color32::PURPLE)
        } else {
            Stroke::new(1.0, Color32::WHITE)
        },
        egui::StrokeKind::Inside,
    );

    let conf = &config.factories[factory];

    let mut clicked = None;

    if let Some(factory_group) = gs.factories()[factory + 1] {
        for (i, tile) in factory_group.tile_vec().iter().enumerate() {
            if draw_tile(ui, config, tile_to_colour(tile), conf.tiles[i], click) {
                clicked = Some(Click::Factory(factory as u8 + 1, *tile));
            }
            if selected && highlight.tile == Some(*tile) {
                draw_tile_border(ui, config, Color32::PURPLE, conf.tiles[i], 3.0, None);
            }
        }
    }
    clicked
}

/// Draw player board to screen
fn draw_board(
    ui: &mut egui::Ui,
    config: &UIConfig,
    gs: &Gamestate<2, 6>,
    board: usize,
    highlight: &Highlight,
    click: Option<Pos2>,
) -> Option<Click> {
    let selected = highlight.board == Some(board);
    // Draw border
    ui.painter().rect_stroke(
        Rect::from_center_size(config.boards[board].centre, config.boards[board].border),
        config.tile_rounding,
        if selected {
            Stroke::new(3.0, Color32::PURPLE)
        } else {
            Stroke::new(1.0, Color32::WHITE)
        },
        egui::StrokeKind::Inside,
    );
    // Draw wall
    for i in 0usize..5 {
        for j in 0usize..5 {
            let tile = gs.boards()[board].wall[(i.into(), j.into())];
            if let Some(tile) = tile {
                draw_tile(
                    ui,
                    config,
                    tile_to_colour(&tile),
                    config.boards[board].wall[i][j],
                    None,
                );
            } else {
                draw_tile_border(
                    ui,
                    config,
                    tile_to_colour(&WALL_COLOURS[i][j]),
                    config.boards[board].wall[i][j],
                    1.0,
                    None,
                );
            }
        }
    }

    let mut clicked = None;

    // Draw rows
    for i in 0usize..5 {
        let colour = if selected && highlight.rows[i] {
            Color32::PURPLE
        } else {
            Color32::WHITE
        };
        for j in 0..(i + 1) {
            let tile = gs.boards()[board].rows[i].tile();

            if gs.boards()[board].rows[i].count() as usize > j {
                if let Some(tile) = tile {
                    if draw_tile(
                        ui,
                        config,
                        tile_to_colour(&tile),
                        config.boards[board].rows[i][j],
                        click,
                    ) {
                        clicked = Some(Click::Row(RowIndex::from(i as u8)));
                    }
                }
            } else if draw_tile_border(
                ui,
                config,
                colour,
                config.boards[board].rows[i][j],
                1.0,
                click,
            ) {
                clicked = Some(Click::Row(RowIndex::from(i as u8)));
            }
        }
    }

    let factory_colour = if selected && highlight.floor {
        Color32::PURPLE
    } else {
        Color32::WHITE
    };

    let scores = ["-1", "-1", "-2", "-2", "-2", "-3", "-3"];
    for (pos, score) in config.boards[board].floor.iter().zip(scores.iter()) {
        if draw_tile_border_with_text(
            ui,
            config,
            factory_colour,
            *pos,
            score,
            Color32::WHITE,
            click,
        ) {
            clicked = Some(Click::Floor);
        }
    }
    // Check if player has first player token
    let offset = if gs.boards()[board].first_player_tile {
        draw_tile(
            ui,
            config,
            Color32::PURPLE,
            config.boards[board].floor[0],
            click,
        );
        1
    } else {
        0
    };
    // Draw floor
    for (i, tile) in gs.boards()[board]
        .floor
        .tile_vec()
        .iter()
        .take(7 - offset)
        .enumerate()
    {
        draw_tile(
            ui,
            config,
            tile_to_colour(&tile),
            config.boards[board].floor[i + offset],
            click,
        );
    }

    // Score
    let mut font = FontId {
        size: config.tile_size,
        ..Default::default()
    };
    ui.painter().text(
        config.boards[board].score,
        egui::Align2::CENTER_CENTER,
        gs.boards()[board].score.to_string()
            + "|"
            + &gs.boards()[board].predicted_score.to_string(),
        font,
        Color32::WHITE,
    );
    clicked
}

/// Draw a tile to the screen
fn draw_tile(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    click: Option<Pos2>,
) -> bool {
    ui.painter().rect_filled(
        Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size)),
        config.tile_rounding,
        colour,
    );
    if let Some(click) = click {
        if Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size))
            .contains(click)
        {
            return true;
        }
    }
    false
}

/// Draw a tile to the screen
fn draw_tile_with_text(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    text: &str,
    text_colour: Color32,
    click: Option<Pos2>,
) -> bool {
    let b = draw_tile(ui, config, colour, pos, click);
    draw_text(ui, pos, text, text_colour);
    b
}

fn draw_text(ui: &mut egui::Ui, pos: Pos2, text: &str, text_colour: Color32) {
    ui.painter().text(
        pos,
        egui::Align2::CENTER_CENTER,
        text,
        FontId::default(),
        text_colour,
    );
}

/// Draw a tile border
fn draw_tile_border(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    width: f32,
    click: Option<Pos2>,
) -> bool {
    ui.painter().rect_stroke(
        Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size)),
        config.tile_rounding,
        Stroke::new(width, colour),
        egui::StrokeKind::Inside,
    );
    if let Some(click) = click {
        if Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size))
            .contains(click)
        {
            return true;
        }
    }
    false
}

fn draw_tile_border_with_text(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    text: &str,
    text_colour: Color32,
    click: Option<Pos2>,
) -> bool {
    let b = draw_tile_border(ui, config, colour, pos, 1.0, click);
    draw_text(ui, pos, text, text_colour);
    b
}

fn tile_to_colour(tile: &Tile) -> egui::Color32 {
    match tile {
        Tile::Blue => egui::Color32::BLUE,
        Tile::Yellow => egui::Color32::YELLOW,
        Tile::Red => egui::Color32::RED,
        Tile::Black => egui::Color32::GREEN,
        Tile::White => egui::Color32::WHITE,
    }
}

fn tile_to_text_colour(tile: &Tile) -> egui::Color32 {
    match tile {
        Tile::Blue => egui::Color32::WHITE,
        Tile::Yellow => egui::Color32::BLACK,
        Tile::Red => egui::Color32::BLACK,
        Tile::Black => egui::Color32::BLACK,
        Tile::White => egui::Color32::BLACK,
    }
}
