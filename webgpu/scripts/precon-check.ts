// Standalone verifier for preconditioned BC fixtures.
// Runs the WebGPU one-shot decode() against bc1..bc5 fixtures and
// byte-compares against the .raw.golden files. Exits 0 on pass, nonzero on any
// mismatch. Calls process.exit(0) at the end to dodge the Dawn addon SIGSEGV
// on process teardown.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { decode } from "../src/decoder.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(__dirname, "..", "test", "fixtures");

const cases: Array<{ name: string; brotlig: string; golden: string }> = [
  { name: "bc1", brotlig: "bc1.brotlig", golden: "bc1.raw.golden" },
  { name: "bc2", brotlig: "bc2.brotlig", golden: "bc2.raw.golden" },
  { name: "bc3", brotlig: "bc3.brotlig", golden: "bc3.raw.golden" },
  { name: "bc4", brotlig: "bc4.brotlig", golden: "bc4.raw.golden" },
  { name: "bc5", brotlig: "bc5.brotlig", golden: "bc5.raw.golden" },
];

// Worker sub-mode: decode one fixture and write raw bytes to stdout.
if (process.argv[2] === "--worker") {
  const fname = process.argv[3];
  const inp = new Uint8Array(await readFile(resolve(FIXTURES, fname)));
  const out = await decode(inp);
  process.stdout.write(Buffer.from(out));
  process.exit(0);
}

let failures = 0;

for (const c of cases) {
  const golden = new Uint8Array(await readFile(resolve(FIXTURES, c.golden)));
  const t0 = Date.now();
  // Spawn a fresh node process for each fixture: the Dawn-backed device is a
  // process-global resource and state from a prior decode() leaks into the
  // next in the same process.
  const r = spawnSync(process.execPath, [
    "--import", "tsx",
    fileURLToPath(import.meta.url),
    "--worker", c.brotlig,
  ], { encoding: "buffer", maxBuffer: 1 << 24 });
  if (r.status !== 0 && r.status !== 139) {
    console.error(`FAIL  ${c.name}  worker exited ${r.status}: ${r.stderr?.toString()}`);
    failures++;
    continue;
  }
  const out = new Uint8Array(r.stdout);
  if (out.length !== golden.length) {
    console.error(`FAIL  ${c.name}  size mismatch: got ${out.length}, expected ${golden.length}`);
    failures++;
  }
  let firstMismatch = -1;
  const n = Math.min(out.length, golden.length);
  for (let i = 0; i < n; i++) {
    if (out[i] !== golden[i]) { firstMismatch = i; break; }
  }
  if (firstMismatch >= 0) {
    const ctxLo = Math.max(0, firstMismatch - 8);
    const ctxHi = Math.min(n, firstMismatch + 16);
    const gotHex = Array.from(out.subarray(ctxLo, ctxHi)).map(b => b.toString(16).padStart(2,"0")).join(" ");
    const expHex = Array.from(golden.subarray(ctxLo, ctxHi)).map(b => b.toString(16).padStart(2,"0")).join(" ");
    console.error(`FAIL  ${c.name}  byte mismatch @ ${firstMismatch}: got ${out[firstMismatch]}, expected ${golden[firstMismatch]}`);
    console.error(`        got [${ctxLo}..${ctxHi}): ${gotHex}`);
    console.error(`        exp [${ctxLo}..${ctxHi}): ${expHex}`);
    failures++;
  } else if (out.length === golden.length) {
    console.log(`PASS  ${c.name}  ${out.length}B in ${Date.now() - t0}ms`);
  }
}

if (failures > 0) {
  console.error(`\n${failures} fixture(s) failed.`);
  process.exit(1);
}
console.log("\nAll preconditioned fixtures passed.");
process.exit(0);
