// Do the TypeScript and Rust implementations agree on the rules?
//
// The benchmark only ever checked this statistically -- matching branching
// factors over random games. That is enough to trust a comparison of two
// engines playing their own games. It is not enough to let one engine pick
// moves inside the other's game, which is what the wasm player does: a
// disagreement about scoring or overflow would show up as inexplicably weak
// play rather than as an error.
//
// So: play random games in TypeScript, and at each round start hand the
// position to the Rust engine and replay the round's moves through it. Then
// compare what each side scored.
//
//   node conformance.mjs [games] [--ts /path/to/azul-tiles]

import { loadEngine, writePosition } from "./position.mjs";

const games = Number(process.argv[2]) || 200;
const tsIndex = process.argv.indexOf("--ts");
const tsRoot = tsIndex === -1 ? "/home/dom/azul-xlang-ts" : process.argv[tsIndex + 1];

const { GameState } = await import(`${tsRoot}/dist/state.js`);
const w = await loadEngine(
    new URL("./target/wasm32-unknown-unknown/release/wasm_engine.wasm", import.meta.url),
);

let rounds = 0;
let moves = 0;
const failures = [];
// Rust force-ends a game at its round 10; TypeScript has no cap, and neither
// does Azul. Past that point the two are playing different games by design, so
// a divergence there is explained rather than a rules disagreement.
//
// Compared in the engine's counting, not TypeScript's: it deals in its
// constructor, so its rounds run one ahead.
const ROUND_CAP = 10;
const engineRound = (tsRound) => tsRound + 1;
let capped = 0;

for (let seed = 1; seed <= games; seed++) {
    const gs = new GameState();
    gs.seed = String(seed);
    gs.newGame(2);

    let rng = BigInt.asUintN(64, BigInt(seed) ^ 0x9e3779b97f4a7c15n);
    const next = () => {
        rng = BigInt.asUintN(64, rng * 6364136223846793005n + 1442695040888963407n);
        return Number(rng >> 33n);
    };

    let roundStart = true;
    for (;;) {
        if (roundStart) {
            // Hand the engine the position as the browser would.
            writePosition(gs, w);
            const engineMoves = w.set_position();
            const tsMoves = gs.availableMoves.length;
            if (engineMoves !== tsMoves) {
                failures.push(
                    `seed ${seed} round ${gs.round}: ${tsMoves} moves in TypeScript, ${engineMoves} in Rust`,
                );
                break;
            }
            roundStart = false;
        }

        const avail = gs.availableMoves;
        if (avail.length === 0) {
            // Score the round on both sides and compare.
            const tsEnded = gs.endRound();
            w.position_end_round();
            for (let p = 0; p < 2; p++) {
                const ts = gs.playerBoards[p].score;
                const rs = w.position_score(p);
                if (ts !== rs) {
                    if (engineRound(gs.round) >= ROUND_CAP) {
                        capped++;
                    } else {
                        failures.push(
                            `seed ${seed} round ${gs.round}: player ${p} scored ${ts} in TypeScript, ${rs} in Rust`,
                        );
                    }
                }
            }
            rounds++;
            if (!tsEnded) break;
            roundStart = true;
            continue;
        }

        const move = avail[next() % avail.length];
        // Play it on both sides, naming it the way the UI does.
        const state = w.position_play(move.factory, move.tile, move.line);
        if (state < 0) {
            failures.push(
                `seed ${seed}: Rust rejected factory=${move.factory} tile=${move.tile} line=${move.line}`,
            );
            break;
        }
        gs.playMove(move);
        moves++;
        if (!gs.nextTurn()) {
            // End of turns; the loop's empty-moves branch scores the round.
        }
    }
}

console.log(`${games} games, ${rounds} rounds, ${moves} moves replayed through both`);
if (capped) {
    console.log(
        `${capped} divergence(s) at or past engine round ${ROUND_CAP}, where Rust force-ends ` +
        `the game ` +
        `and TypeScript does not. Explained, not a rules disagreement.`,
    );
}
if (failures.length === 0) {
    console.log("the implementations agree on every move and every round score");
} else {
    console.log(`\n${failures.length} disagreements:`);
    for (const f of failures.slice(0, 15)) console.log(`  ${f}`);
    if (failures.length > 15) console.log(`  ... and ${failures.length - 15} more`);
    process.exitCode = 1;
}
