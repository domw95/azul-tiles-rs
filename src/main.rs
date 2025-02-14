#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use azul_tiles_rs::{
    gamestate::Gamestate,
    player::Player,
    playerboard::{wall::WALL_COLOURS, PlayerBoard},
    tiles::{Tile, TileGroup},
};
use eframe::{egui, egui_glow::painter};
use egui::{Color32, FontId, Key, Painter, Pos2, Rect, Sense, Stroke, Vec2};

fn main() -> eframe::Result {
    // env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
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

struct MyApp {
    name: String,
    age: u32,
    gs: Gamestate<2, 6>,

    players: [Box<dyn Player<2, 6>>; 2],

    /// UI config that changes with screen size
    config: UIConfig,
}

impl MyApp {
    fn new() -> Self {
        let mut app = Self::default();

        app
    }
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            gs: Gamestate::new_2_player(),
            config: UIConfig::default(),
            players: [
                Box::new(azul_tiles_rs::player::MoveRankPlayer),
                Box::new(azul_tiles_rs::player::MoveRankPlayer),
            ],
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let window_size = ui.available_size();
            self.config.update(&window_size);

            ctx.input(|input| {
                if input.key_pressed(Key::Space) {
                    match self.gs.state() {
                        azul_tiles_rs::gamestate::State::RoundActive => {
                            let moves = self.gs.get_moves();
                            let player = &mut self.players[self.gs.current_player() as usize];
                            let m = player.pick_move(&self.gs, moves);
                            self.gs.play_move(m);
                        }
                        azul_tiles_rs::gamestate::State::RoundEnd => {
                            self.gs.end_round();
                        }
                        azul_tiles_rs::gamestate::State::GameEnd => (),
                    }
                }
            });

            draw_game(ui, &self.config, &self.gs);
        });
    }
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
                0 => (-1.0, 1.0),
                1 => (1.0, 1.0),
                2 => (-1.0, -1.0),
                3 => (1.0, -1.0),
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

fn draw_game(ui: &mut egui::Ui, config: &UIConfig, gs: &Gamestate<2, 6>) {
    // Draw player boards
    for i in 0..2 {
        draw_board(ui, config, gs, i);
    }

    // Draw centre and factories
    draw_centre(ui, config, gs);

    for i in 0..5 {
        draw_factory(ui, config, gs, i);
    }

    // Draw bag
    draw_bag(ui, config, gs.tilebag());
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
            );
        } else {
            draw_tile_border(ui, config, tile_to_colour(&tile), config.bag.tiles[i]);
        }
    }
}

fn draw_centre(ui: &mut egui::Ui, config: &UIConfig, gs: &Gamestate<2, 6>) {
    let centre = gs.centre();
    ui.painter().rect_stroke(
        Rect::from_center_size(config.centre.centre, config.centre.border),
        config.tile_rounding,
        Stroke::new(1.0, Color32::WHITE),
        egui::StrokeKind::Inside,
    );
    for (i, (&count, tile)) in centre.into_iter().enumerate() {
        if count > 0 {
            // draw tile with digit
            draw_tile_with_text(
                ui,
                config,
                tile_to_colour(&tile),
                config.centre.tiles[i],
                &count.to_string(),
                tile_to_text_colour(&tile),
            );
        } else {
            // draw outline
            draw_tile_border(ui, config, tile_to_colour(&tile), config.centre.tiles[i]);
        }
    }
    if gs.first_player_tile() {
        draw_tile(ui, config, Color32::PURPLE, config.centre.tiles[5]);
    }
}

/// Draw factory to screen
fn draw_factory(ui: &mut egui::Ui, config: &UIConfig, gs: &Gamestate<2, 6>, factory: usize) {
    // Draw border
    ui.painter().rect_stroke(
        Rect::from_center_size(
            config.factories[factory].centre,
            config.factories[factory].border,
        ),
        config.tile_rounding,
        Stroke::new(1.0, Color32::WHITE),
        egui::StrokeKind::Inside,
    );

    let conf = &config.factories[factory];

    if let Some(factory) = gs.factories()[factory + 1] {
        for (i, tile) in factory.tile_vec().iter().enumerate() {
            draw_tile(ui, config, tile_to_colour(tile), conf.tiles[i]);
        }
    }
}

/// Draw player board to screen
fn draw_board(ui: &mut egui::Ui, config: &UIConfig, gs: &Gamestate<2, 6>, board: usize) {
    // Draw border
    ui.painter().rect_stroke(
        Rect::from_center_size(config.boards[board].centre, config.boards[board].border),
        config.tile_rounding,
        Stroke::new(1.0, Color32::WHITE),
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
                );
            } else {
                draw_tile_border(
                    ui,
                    config,
                    tile_to_colour(&WALL_COLOURS[i][j]),
                    config.boards[board].wall[i][j],
                );
            }
        }
    }

    // Draw rows
    for i in 0usize..5 {
        for j in 0..(i + 1) {
            let tile = gs.boards()[board].rows[i].tile();

            if gs.boards()[board].rows[i].count() as usize > j {
                if let Some(tile) = tile {
                    draw_tile(
                        ui,
                        config,
                        tile_to_colour(&tile),
                        config.boards[board].rows[i][j],
                    );
                }
            } else {
                draw_tile_border(ui, config, Color32::WHITE, config.boards[board].rows[i][j]);
            }
        }
    }

    let scores = ["-1", "-1", "-2", "-2", "-2", "-3", "-3"];
    for (pos, score) in config.boards[board].floor.iter().zip(scores.iter()) {
        draw_tile_border_with_text(ui, config, Color32::WHITE, *pos, score, Color32::WHITE);
    }
    // Check if player has first player token
    let offset = if gs.boards()[board].first_player_tile {
        draw_tile(ui, config, Color32::PURPLE, config.boards[board].floor[0]);
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
        gs.boards()[board].score.to_string(),
        font,
        Color32::WHITE,
    );
}

/// Draw a tile to the screen
fn draw_tile(ui: &mut egui::Ui, config: &UIConfig, colour: Color32, pos: Pos2) {
    ui.painter().rect_filled(
        Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size)),
        config.tile_rounding,
        colour,
    );
}

/// Draw a tile to the screen
fn draw_tile_with_text(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    text: &str,
    text_colour: Color32,
) {
    draw_tile(ui, config, colour, pos);
    draw_text(ui, pos, text, text_colour);
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
fn draw_tile_border(ui: &mut egui::Ui, config: &UIConfig, colour: Color32, pos: Pos2) {
    ui.painter().rect_stroke(
        Rect::from_center_size(pos, Vec2::new(config.tile_size, config.tile_size)),
        config.tile_rounding,
        Stroke::new(1.0, colour),
        egui::StrokeKind::Inside,
    );
}

fn draw_tile_border_with_text(
    ui: &mut egui::Ui,
    config: &UIConfig,
    colour: Color32,
    pos: Pos2,
    text: &str,
    text_colour: Color32,
) {
    draw_tile_border(ui, config, colour, pos);
    draw_text(ui, pos, text, text_colour);
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
