# @brotlig/webgpu

WebGPU port of the Brotli-G GPU decoder. This package is the in-progress
TypeScript host + WGSL shader replacement for the D3D12/HLSL decoder that
lives under `../src/decoder/BrotliGCompute.hlsl`.

## Status

Scaffolding only. The WGSL shader is a placeholder; no decoder logic is
ported yet. The public API shape (`BrotligStreamDecoder`, `decode`) is
final and will not churn. Preconditioned decode for BC1..BC5 is validated
end-to-end against CPU round-trip fixtures (`webgpu/scripts/precon-check.ts`).

## Hard requirements

- WebGPU `subgroups` feature must be available on the adapter.
- The compute pipeline's subgroup size must be exactly 32. The decoder
  refuses to run otherwise (it mirrors the HLSL `numthreads(32,1,1)`
  wavefront assumption).
- Runs in Chrome stable (browser) and Node 22+ via the `webgpu` npm
  package (Dawn bindings). `requestBrotligDevice()` auto-detects.

## API sketch

```ts
import { BrotligStreamDecoder, decode } from "@brotlig/webgpu";

// One-shot
const out = await decode(compressed);

// Streaming (sub-stream page-level, Option 2)
const dec = await BrotligStreamDecoder.create();
for await (const chunk of source) {
  const produced = await dec.push(chunk);
  sink.write(produced);
}
sink.write(await dec.end());
dec.destroy();
```

`push()` accepts partial byte ranges; it buffers until at least one
complete Brotli-G stream is resident, then dispatches compute work over
the pages that are fully available. Per-stream decoder state is
persisted across dispatches in a GPU-resident `state` buffer so later
pages can resume.
