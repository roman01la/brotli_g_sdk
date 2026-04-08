// Standalone streaming smoke test. Run with `tsx scripts/stream-check.ts`.
// Exercises BrotligStreamDecoder.push() with varying chunk sizes and
// byte-compares against the golden fixture.
//
// Kept out of the default `npm run test` run because the Dawn native
// addon used by the Node `webgpu` package destabilizes after ~3-4
// dispatches per process, forcing us to split streaming cases into
// separate process runs.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { BrotligStreamDecoder } from "../src/decoder.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(__dirname, "..", "test", "fixtures");

const FIXTURE = process.argv[2] ?? "medium";
const CHUNK = Number(process.argv[3] ?? 4096);

const map: Record<string, [string, string]> = {
  tiny: ["tiny.brotlig", "tiny.bin.golden"],
  small: ["small.brotlig", "small.txt.golden"],
  medium: ["medium.brotlig", "medium.bin.golden"],
};
const entry = map[FIXTURE];
if (!entry) {
  console.error(`unknown fixture: ${FIXTURE}. Pick one of: tiny, small, medium.`);
  process.exit(2);
}
const [brotligName, goldenName] = entry;

const inp = new Uint8Array(await readFile(resolve(FIXTURES, brotligName)));
const golden = new Uint8Array(await readFile(resolve(FIXTURES, goldenName)));

const t0 = Date.now();
const dec = await BrotligStreamDecoder.create();
const parts: Uint8Array[] = [];
for (let i = 0; i < inp.length; i += CHUNK) {
  const slice = inp.subarray(i, Math.min(i + CHUNK, inp.length));
  const out = await dec.push(slice);
  if (out.length) parts.push(out);
}
const tail = await dec.end();
if (tail.length) parts.push(tail);

let total = 0;
for (const p of parts) total += p.length;
const joined = new Uint8Array(total);
let off = 0;
for (const p of parts) { joined.set(p, off); off += p.length; }

if (joined.length !== golden.length) {
  console.error(`size mismatch: got ${joined.length}, expected ${golden.length}`);
  process.exit(1);
}
for (let i = 0; i < joined.length; i++) {
  if (joined[i] !== golden[i]) {
    console.error(`byte mismatch @ ${i}: got ${joined[i]}, expected ${golden[i]}`);
    process.exit(1);
  }
}
console.log(`PASS  ${FIXTURE} chunk=${CHUNK}  ${joined.length}B in ${Date.now() - t0}ms`);
// See test/run.ts: bypass Dawn addon destructor which may SIGSEGV on exit.
process.exit(0);
