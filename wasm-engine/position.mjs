// Serialises a TypeScript `GameState` into the wasm position buffer.
//
// Shared by the conformance check and by the player, so that what is verified
// is what is played. The layout is documented on `set_position` in
// src/lib.rs; this is the other half of it.

export const FIRST_PLAYER_TILE = 5;
export const NULL_TILE = -1;

/**
 * Takes the module rather than a view on purpose. Any wasm allocation can
 * grow linear memory, which detaches every existing view of it, so a view
 * cached across calls works right up until the search allocates and then
 * throws. Making a fresh one per write costs nothing and removes the trap.
 *
 * @param gs    an azul-tiles GameState
 * @param w     the module's exports
 * @param forRound  round number, in TypeScript's counting. Converted: the
 *                  engine deals in its constructor, so its first playable
 *                  round is 1 where TypeScript's is 0. The evaluation crosses
 *                  every term with rounds remaining, so getting this wrong
 *                  skews the whole evaluation rather than one term.
 */
export function writePosition(gs, w, forRound = gs.round) {
    const buf = positionView(w);
    buf.fill(0);
    buf[0] = gs.activePlayer;
    buf[1] = gs.firstTile === FIRST_PLAYER_TILE ? 1 : 0;
    buf[2] = Math.min(255, forRound + 1);

    for (let f = 0; f < 6; f++) {
        const factory = gs.factory[f] ?? [];
        for (const tile of factory) {
            if (tile >= 0 && tile < 5) buf[3 + f * 5 + tile]++;
        }
    }

    for (let p = 0; p < 2; p++) {
        const base = 33 + p * 42;
        const pb = gs.playerBoards[p];

        for (let r = 0; r < 5; r++) {
            for (let c = 0; c < 5; c++) {
                buf[base + r * 5 + c] = pb.wall[r][c] !== NULL_TILE ? 1 : 0;
            }
        }

        for (let r = 0; r < 5; r++) {
            const line = pb.lines[r];
            // 255 is the empty marker; a line only ever holds one colour.
            buf[base + 25 + r * 2] = line.length ? line[0] : 255;
            buf[base + 25 + r * 2 + 1] = line.length;
        }

        for (const tile of pb.floor) {
            if (tile === FIRST_PLAYER_TILE) buf[base + 40] = 1;
            else if (tile >= 0 && tile < 5) buf[base + 35 + tile]++;
        }

        // The committed score only. The engine recomputes its own prediction
        // from the board rather than taking one on trust.
        buf[base + 41] = Math.max(0, Math.min(255, pb.score));
    }
}

/** Unpack what `search_move` returns into the fields the web UI names. */
export function unpackMove(packed) {
    if (packed < 0) return null;
    return { factory: (packed >> 8) & 0xff, tile: (packed >> 4) & 0xf, line: packed & 0xf };
}

/** Load a wasm module with the clock it needs and nothing else. */
export async function loadEngine(wasmPath) {
    const { readFile } = await import("node:fs/promises");
    const { performance } = await import("node:perf_hooks");
    const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), {
        env: { now_ms: () => performance.now() },
    });
    return instance.exports;
}

/** A fresh Uint8Array over the module's position buffer. Never cache it: see
 *  `writePosition`. */
export function positionView(w) {
    return new Uint8Array(w.memory.buffer, w.position_buffer(), w.position_buffer_len());
}
