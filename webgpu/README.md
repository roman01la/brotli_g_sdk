# @brotlig/webgpu

WebGPU port of AMD's Brotli-G GPU decoder. TypeScript host + WGSL shader,
runs in browsers and Node, validated byte-for-byte against the CPU
reference decoder across the full format matrix and a 393 MB real-world
asset.

## Status

End-to-end working. The WGSL shader is a complete port of
`../src/decoder/BrotliGCompute.hlsl` (~1900 LOC, 32-thread subgroup
compute). 25/25 tests pass in Chrome WebGPU, including:

- One-shot decode for tiny/small/medium random + Lorem ipsum fixtures
- One-shot decode for BC1..BC5 preconditioned texture fixtures
- Pathological inputs (all zeros, all ones, repeating patterns)
- Sub-stream page-level chunked streaming (1 B / 16 B / 1 KB / 4 KB /
  64 KB chunk sizes)
- Multi-stream concatenation
- Jagged streaming with deterministic xorshift32 chunk sizes
- Malformed-header rejection

Validated against a real 393 MB `emscripten.pack` asset (1.76 GB/s
end-to-end via `decodeInto`, ~3.5 GB/s of pure GPU compute throughput).

BC7 is not supported because the upstream Brotli-G SDK has no BC7
preconditioning code (`BROTLIG_DATA_FORMAT` only defines BC1..BC5).

## Performance

On Apple Silicon Chrome WebGPU, decoding `emscripten.pack`
(393 MiB output, 89 MiB Brotli-G compressed):

| Method | Median | Throughput |
|---|---|---|
| `decode()` | 278 ms | 1349 MB/s |
| `decodeInto()` | 221 ms | 1698 MB/s |
| Pure GPU pipeline (steady state) | 113 ms | 3489 MB/s |
| Standard brotli WASM (`brotli-wasm`) | 991 ms | 379 MB/s |

The WebGPU port is **~4.6× faster than WASM brotli** on real-world
data, at the cost of ~65% larger compressed payloads (Brotli-G's
parallel page format trades compression density for GPU parallelism).

## Hard requirements

- WebGPU `subgroups` feature must be available on the adapter.
- The compute pipeline's subgroup size must be exactly 32. The decoder
  refuses to run otherwise (mirrors HLSL `numthreads(32, 1, 1)` and the
  shader's pervasive `WaveReadLaneAt(_, lane_constexpr)` patterns).
- `requiredSubgroupSize: 32` is requested at pipeline creation when the
  implementation supports it.
- Adapter-max `maxStorageBufferBindingSize` and `maxBufferSize` are
  requested to allow >128 MB inputs/outputs.
- Runs in Chrome stable (browser) and Node 22+ via the `webgpu` npm
  package (Dawn bindings). `requestBrotligDevice()` auto-detects.

## API

```ts
import {
  BrotligStreamDecoder,
  decode,
  decodeInto,
  requestBrotligDevice,
  parseStreamHeaderOnly,
  findCompleteStreams,
} from "@brotlig/webgpu";

// One-shot, allocates a fresh Uint8Array for the output.
const out = await decode(compressedBytes);

// Zero-allocation one-shot. Caller pre-allocates the destination
// (must be >= uncompressed size). Saves a 393 MB-class memcpy on
// large outputs; ~20% faster than decode() for non-trivial inputs.
const dest = new Uint8Array(uncompressedSize);
await decodeInto(compressedBytes, dest);

// Sub-stream page-level streaming. push() starts decoding as soon as
// the stream header + page offset table are resident, then dispatches
// each batch of newly-resident pages. Decoder state is recovered
// between dispatches via the per-stream meta record.
const dec = await BrotligStreamDecoder.create();
for await (const chunk of source) {
  const produced = await dec.push(chunk);
  if (produced.length) sink.write(produced);
}
sink.write(await dec.end());
dec.destroy();
```

`push()` accepts arbitrary byte slices, buffers them in a CPU ring,
parses the stream header as soon as enough bytes arrive, and uploads
each fully-resident page body to the GPU input buffer. Each dispatch
runs the compute shader with `pageLimit = pagesUploaded`; the inner
page loop breaks at `pageLimit` and the next dispatch picks up at the
right `pageCursor`.

## Format support

- Generic Brotli-G streams (any quality, any page size 32 KiB / 64 KiB /
  128 KiB)
- Preconditioned BC1, BC2, BC3, BC4, BC5 texture streams
- Multi-stream concatenated containers (decoded sequentially via the
  shader's atomic stream-index countdown)

## Running tests

The default test runner uses `tsx` (not vitest) because the Dawn native
addon used by the Node `webgpu` package destabilizes inside vitest's
worker pool:

```sh
cd webgpu
npm install
npm run typecheck
npm run test
```

The test runner force-exits to dodge a SIGSEGV in the Dawn addon's
process-exit destructor; the npm script tolerates exit code 139.

For chunked streaming and BC1..BC5 spot checks (each runs in a fresh
process to amortize the addon's per-process resource budget):

```sh
npx tsx scripts/stream-check.ts <fixture> <chunkSize>
npx tsx scripts/precon-check.ts
```

For full-matrix browser validation, build dist/ and serve the package:

```sh
npm run build
python3 -m http.server 8765 --bind 127.0.0.1
# then open http://127.0.0.1:8765/browser-test.html
# results land in window.__BROTLIG_RESULT__
```

## Test fixtures

Generated via the CPU-only CLI built from this repo
(`bin/brotlig_cpu`, see `../sample/CMakeLists.txt`'s
`brotlig_cpu_cli` target). The CPU CLI builds on macOS/Linux as well as
Windows; the original `brotlig_cli` target is Windows-only because it
links D3D12.

Generic fixtures: tiny (512 B random), small (4 KiB Lorem ipsum),
medium (1 MiB random), allzero/allone/repeat (64 KiB synthetic),
em.pack (393 MB emscripten cache, used for benchmarks; not committed).

Preconditioned fixtures: bc1..bc5 (64x64 synthetic gradient textures
compressed with `-precondition -data-format <N>`).

## Known limitations

- **Page size cap**: 128 KiB (the upstream `BROTLIG_MAX_PAGE_SIZE`).
  256 KiB pages would compress ~2% better but the shader's distance
  ring buffer truncates entries to 16 bits, which silently corrupts
  back-references with distances >= 65536. This is a pre-existing
  limitation in the original AMD HLSL shader, not a port artifact.
  Lifting it requires widening `gDistRingLo`/`gDistRingHi` and
  `ResolveDistances` to hold full 32-bit distances per slot.
- **Cross-lane workgroup memory races**: a handful of remaining sites
  in `ConditionerParams_Init` helpers, `ClearDictionary` -> `SetSymbol`,
  and `SpreadValue` rely on single-subgroup SIMT lockstep for memory
  visibility. WGSL does not formally guarantee this. Empirically
  correct on Chrome / Dawn / Tint today; may not be on every backend.
  Fixing requires either `subgroupBarrier` (not yet a resolved WGSL
  builtin) or restructuring the call sites for uniform control flow.
- **Node Dawn addon stability**: the `webgpu` npm package's native
  addon hits a mutex bug after ~10-12 dispatches per process. Affects
  Node test ergonomics only; browsers are unaffected.
