// Drives the pondering worker through a full game and measures the thing the
// requirement is about: how long a move request actually takes to answer.
//
// Player 1 is the engine. Player 0 plays at random, standing in for a human,
// and takes `--think` ms over each move -- time the worker spends pondering
// rather than idling, which is the whole point of the design.
//
//   node demo.mjs [--think 300] [--slice 250] [--seed 7]

import { Worker } from "node:worker_threads";
import { performance } from "node:perf_hooks";

const arg = (name, fallback) => {
    const i = process.argv.indexOf(`--${name}`);
    return i === -1 ? fallback : Number(process.argv[i + 1]);
};
const THINK_MS = arg("think", 300);
const SLICE_MS = arg("slice", 250);
const SEED = arg("seed", 7);
const DEADLINE_MS = 1000;

const worker = new Worker(new URL("./engine-worker.mjs", import.meta.url));
const waiters = new Map();
let nextId = 1;
let onEvent = () => {};

worker.on("message", (msg) => {
    if (msg.type === "move" && waiters.has(msg.id)) {
        waiters.get(msg.id)(msg);
        waiters.delete(msg.id);
        return;
    }
    onEvent(msg);
});

const next = (type) => new Promise((res) => {
    onEvent = (msg) => { if (msg.type === type) { onEvent = () => {}; res(msg); } };
});

function requestMove() {
    const id = nextId++;
    const t = performance.now();
    return new Promise((res) => {
        waiters.set(id, (msg) => res({ ...msg, latency: performance.now() - t }));
        worker.postMessage({ type: "requestMove", id });
    });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

await next("booted");
worker.postMessage({ type: "newGame", seed: SEED, sliceMs: SLICE_MS });
let st = await next("ready");

console.log(`slice=${SLICE_MS}ms  simulated human think time=${THINK_MS}ms  deadline=${DEADLINE_MS}ms\n`);

const latencies = [];
const perRound = new Map();
let moves = 0;

while (st.state !== 2 && moves < 400) {
    if (st.moveCount === 0) {
        const before = st;
        st = await next2("endRound", "roundStarted");
        const r = perRound.get(before.round);
        if (r) {
            console.log(
                `round ${before.round}: ${r.moves} engine moves, max latency ${r.max.toFixed(1)}ms, ` +
                `deepest ${r.depth}${r.solvedAt ? `, solved after ${r.solvedAt} engine moves` : ", not solved"}`,
            );
        }
        continue;
    }

    if (st.player === 1) {
        // The engine's turn. It has been pondering through the human's think
        // time, so this should come back with a deep answer immediately.
        const res = await requestMove();
        latencies.push(res.latency);
        const r = perRound.get(res.round) ?? { moves: 0, max: 0, depth: 0, solvedAt: null };
        r.moves++;
        r.max = Math.max(r.max, res.latency);
        r.depth = Math.max(r.depth, res.depth);
        if (res.solved && r.solvedAt === null) r.solvedAt = r.moves;
        perRound.set(res.round, r);
        if (res.index < 0) throw new Error("engine had no move to give");
        worker.postMessage({ type: "play", index: res.index });
        st = await next("played");
        moves++;
    } else {
        // The human's turn: think, then play something legal.
        await sleep(THINK_MS);
        worker.postMessage({ type: "play", index: Math.floor(Math.random() * st.moveCount) });
        st = await next("played");
        moves++;
    }
}

function next2(send, expect) {
    worker.postMessage({ type: send });
    return next(expect);
}

latencies.sort((a, b) => a - b);
const pct = (p) => latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * p))];
console.log(
    `\n${latencies.length} engine moves over ${st.round} rounds` +
    `\nlatency: median ${pct(0.5).toFixed(1)}ms  p95 ${pct(0.95).toFixed(1)}ms  max ${latencies.at(-1).toFixed(1)}ms` +
    `\nover ${DEADLINE_MS}ms deadline: ${latencies.filter((l) => l > DEADLINE_MS).length}` +
    `\nfinal scores: ${st.scores.join(" - ")}  tree ${st.treeSize} nodes  searched ${st.nodes.toLocaleString()}`,
);
await worker.terminate();
