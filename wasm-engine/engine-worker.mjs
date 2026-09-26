// The pondering worker.
//
// Runs in a browser Web Worker and in node's worker_threads, because the
// prototype has to be measurable here and droppable into tiles-web there.
//
// The shape that matters: pondering is a chain of short slices scheduled
// through the task queue, never a `while` loop. A loop would hold the thread
// and the worker would not see a message until it finished -- which for an
// unbounded "search until solved" is never. Between slices the queue drains,
// so a request is answered within one slice at worst.

const isNode = typeof globalThis.process?.versions?.node === "string";

let post, onMessage, loadWasm;
if (isNode) {
    const { parentPort } = await import("node:worker_threads");
    const { readFile } = await import("node:fs/promises");
    const { performance } = await import("node:perf_hooks");
    globalThis.performance ??= performance;
    post = (m) => parentPort.postMessage(m);
    onMessage = (fn) => parentPort.on("message", fn);
    loadWasm = async (url) => readFile(url);
} else {
    post = (m) => self.postMessage(m);
    onMessage = (fn) => { self.onmessage = (e) => fn(e.data); };
    loadWasm = async (url) => (await fetch(url)).arrayBuffer();
}

const wasmUrl = new URL("./target/wasm32-unknown-unknown/release/wasm_engine.wasm", import.meta.url);
const { instance } = await WebAssembly.instantiate(await loadWasm(wasmUrl), {
    env: { now_ms: () => performance.now() },
});
const w = instance.exports;

/** How long one uninterruptible chunk of search is. This is the worst-case
 *  response latency, so it is the one number to turn down if the UI feels
 *  sticky, and up if the slicing overhead shows. */
let sliceMs = 250;
let pondering = false;
let scheduled = false;

function status(extra = {}) {
    return {
        round: w.round(),
        player: w.current_player(),
        moveCount: w.move_count(),
        depth: w.depth(),
        solved: w.solved() === 1,
        nodes: w.nodes(),
        treeSize: w.tree_size(),
        value: w.best_value(),
        scores: [w.score(0), w.score(1)],
        ...extra,
    };
}

function schedule() {
    if (!pondering || scheduled) return;
    scheduled = true;
    // A macrotask, so messages queued behind it are delivered first.
    setTimeout(step, 0);
}

function step() {
    scheduled = false;
    if (!pondering) return;
    if (w.move_count() === 0) { pondering = false; post({ type: "idle", ...status() }); return; }
    const solvedNow = w.ponder(sliceMs) === 1;
    if (solvedNow) {
        pondering = false;
        post({ type: "solved", ...status() });
        return;
    }
    schedule();
}

onMessage((msg) => {
    switch (msg.type) {
        case "newGame":
            w.new_game(msg.seed);
            if (msg.sliceMs) sliceMs = msg.sliceMs;
            pondering = true;
            schedule();
            post({ type: "ready", ...status() });
            break;

        // Answered from the last completed deepening pass, so this never waits
        // for the search. The only delay is the slice already in flight.
        case "requestMove": {
            // Normally free: the last completed pass, or the ordering the
            // re-rooted tree carried over. `ensure_move` only does work on a
            // cold root, and is bounded so the reply still beats the deadline.
            let index = w.best_index();
            if (index < 0) index = w.ensure_move(msg.deadlineMs ?? 500);
            post({ type: "move", id: msg.id, index, ...status() });
            break;
        }

        case "play": {
            const state = w.play_index(msg.index);
            pondering = true;
            schedule();
            post({ type: "played", state, ...status() });
            break;
        }

        case "endRound": {
            const state = w.end_round();
            pondering = state !== 2;
            schedule();
            post({ type: "roundStarted", state, ...status() });
            break;
        }

        case "stop":
            pondering = false;
            post({ type: "stopped", ...status() });
            break;
    }
});

post({ type: "booted" });
