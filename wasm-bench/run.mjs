// Drives the wasm benchmark under node, which is V8 -- the same engine the
// browser would run it on. The only import the module needs is a clock.
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const path = new URL("./target/wasm32-unknown-unknown/release/wasm_bench.wasm", import.meta.url);
const { instance } = await WebAssembly.instantiate(await readFile(path), {
    env: { now_ms: () => performance.now() },
});
const w = instance.exports;

w.warmup();

const ms = w.bench_playout(500);
console.log(`playout    games=500 moves=${w.last_nodes()} ms=${ms.toFixed(1)} moves_per_sec=${(w.last_nodes() / ms * 1000).toFixed(0)}`);

for (const [label, fn] of [["fixed", w.bench_fixed], ["deepen", w.bench_deepen]]) {
    for (const depth of [3, 4, 5]) {
        const ms = fn(depth);
        const nodes = w.last_nodes();
        console.log(
            `${label.padEnd(10)} depth=${depth} nodes=${String(nodes).padStart(10)}` +
            ` ms=${ms.toFixed(1).padStart(9)} knodes_per_sec=${(nodes / ms).toFixed(0).padStart(6)}`,
        );
    }
}

const tms = w.bench_timed(100);
console.log(`timed      budget=100ms mean_depth=${w.last_mean_depth().toFixed(2)} nodes=${w.last_nodes()} ms=${tms.toFixed(1)} knodes_per_sec=${(w.last_nodes() / tms).toFixed(0)}`);
