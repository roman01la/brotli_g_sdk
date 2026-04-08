// Brotli-G streaming WebGPU decoder.
//
// Host-side dispatch loop for the WGSL port of the Brotli-G decoder.
// Supports page-level streaming: as compressed bytes arrive via push(),
// the host parses stream headers, determines how many pages are resident,
// and dispatches the shader with a pageLimit that bounds the inner page
// loop. State is kept in a persistent state buffer so decode can resume.

import { requestBrotligDevice, assertSubgroupSize32 } from "./device.js";
import { BROTLIG_WGSL } from "./shader.js";
import {
  findCompleteStreams,
  parseStreamHeaderOnly,
  type ParsedStreamHeader,
} from "./format.js";

export interface BrotligDecoderOptions {
  device?: GPUDevice;
  maxInputBytes?: number;
  maxOutputBytes?: number;
  maxActiveStreams?: number;
}

// Mirrors sample/BrotligGPUDecoder.cpp: dispatch = (2560, 1, 1),
// numthreads = (32, 1, 1). The reference D3D12 host uses this ceiling so
// there is always at least one workgroup per page the shader can pick up.
// TODO: make this a function of (active streams * pageLimit) once we have
// a tighter upper bound; 2560 is an over-dispatch that wastes groups.
const REFERENCE_DISPATCH_GROUPS = 2560;

// Per the WGSL port, the persistent state buffer is laid out as one 16 KB
// slot per active stream.
const STATE_STRIDE_BYTES = 16 * 1024;

// meta layout (u32 words):
//   [0] remaining stream count
//   [1] pageLimit
//   [2] resume flag bitmask
//   [3] reserved
//   [4 + (s-1)*4 .. +3] per-stream: rptr, wptr, pageCursor, savedStateOffset
const META_HEADER_WORDS = 4;
const META_PER_STREAM_WORDS = 4;
const META_BYTES_FOR_STREAMS = (n: number) =>
  (META_HEADER_WORDS + n * META_PER_STREAM_WORDS) * 4;

const DEFAULTS = {
  maxInputBytes: 1 * 1024 * 1024,
  maxOutputBytes: 4 * 1024 * 1024,
  maxActiveStreams: 1,
} as const;

function nextPow2(n: number): number {
  let v = 1;
  while (v < n) v <<= 1;
  return v;
}

// Cache shader module / pipeline / bind group layout per device. The Dawn
// native addon used in Node destabilizes when many pipelines are created
// against the same device; caching also speeds up repeated decoder
// instantiation in browsers.
interface PipelineCacheEntry {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}
const pipelineCache = new WeakMap<GPUDevice, Promise<PipelineCacheEntry>>();

function getPipelineCache(device: GPUDevice): Promise<PipelineCacheEntry> {
  const existing = pipelineCache.get(device);
  if (existing) return existing;
  const p = (async () => {
    const shaderModule = device.createShaderModule({ code: BROTLIG_WGSL });
    const info = await shaderModule.getCompilationInfo?.();
    if (info && info.messages.some((m) => m.type === "error")) {
      const errs = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
        .join("\n");
      throw new Error(`BROTLIG_WGSL compile errors:\n${errs}`);
    }
    await assertSubgroupSize32(device, shaderModule, "main");
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });
    return { pipeline, bindGroupLayout };
  })();
  pipelineCache.set(device, p);
  return p;
}

// Per-stream host-side bookkeeping for decodes that are already uploaded
// but not yet finished (pageCursor < numPages). With the current single-
// workgroup-per-stream model this list has at most one entry, but the
// structure is in place for future pipelining.
interface InFlightStream {
  stream: ParsedStreamHeader;
  // Absolute byte offset into the GPU output buffer where this stream's
  // uncompressed bytes start.
  outputBase: number;
  // Absolute byte offset into the GPU input buffer where this stream's
  // compressed bytes start.
  inputBase: number;
  // Slot in the state buffer (0..maxActiveStreams-1).
  stateSlot: number;
  // Number of pages already decoded by prior dispatches.
  pageCursor: number;
  // Bytes produced by prior dispatches (running wptr, relative to outputBase).
  producedBytes: number;
  // True once header + precon + page table have been uploaded to inputBase.
  headerUploaded: boolean;
  // How many pages' compressed body bytes have been uploaded so far.
  pagesUploaded: number;
  // Absolute input-buffer offset where the next page body bytes go.
  inputBodyWritten: number;
}

export class BrotligStreamDecoder {
  readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;

  private inputBuf: GPUBuffer;
  private inputCap: number;
  private metaBuf: GPUBuffer;
  private outputBuf: GPUBuffer;
  private outputCap: number;
  private stateBuf: GPUBuffer;
  private stateCap: number;
  private readbackBuf: GPUBuffer;
  private cachedBindGroup: GPUBindGroup | null = null;

  // Running append cursor into the GPU input buffer. Monotonic across the
  // life of the decoder; growBuffer preserves content.
  private inputWritten = 0;
  // Running append cursor into the GPU output buffer. Monotonic.
  private outputWritten = 0;

  private readonly maxActiveStreams: number;

  private ring: Uint8Array = new Uint8Array(0);
  private ringLen = 0;

  private inFlight: InFlightStream | null = null;
  private destroyed = false;

  private constructor(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    bindGroupLayout: GPUBindGroupLayout,
    inputBuf: GPUBuffer,
    inputCap: number,
    metaBuf: GPUBuffer,
    outputBuf: GPUBuffer,
    outputCap: number,
    stateBuf: GPUBuffer,
    stateCap: number,
    readbackBuf: GPUBuffer,
    maxActiveStreams: number,
  ) {
    this.device = device;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.inputBuf = inputBuf;
    this.inputCap = inputCap;
    this.metaBuf = metaBuf;
    this.outputBuf = outputBuf;
    this.outputCap = outputCap;
    this.stateBuf = stateBuf;
    this.stateCap = stateCap;
    this.readbackBuf = readbackBuf;
    this.maxActiveStreams = maxActiveStreams;
  }

  static async create(
    opts: BrotligDecoderOptions = {},
  ): Promise<BrotligStreamDecoder> {
    const device = opts.device ?? (await requestBrotligDevice());
    const maxInput = opts.maxInputBytes ?? DEFAULTS.maxInputBytes;
    const maxOutput = opts.maxOutputBytes ?? DEFAULTS.maxOutputBytes;
    const maxStreams = opts.maxActiveStreams ?? DEFAULTS.maxActiveStreams;

    const { pipeline, bindGroupLayout } = await getPipelineCache(device);

    const inputCap = nextPow2(Math.max(maxInput, 1024));
    const outputCap = nextPow2(Math.max(maxOutput, 1024));
    const stateCap = nextPow2(Math.max(maxStreams * STATE_STRIDE_BYTES, 1024));

    const inputBuf = device.createBuffer({
      size: inputCap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const metaBuf = device.createBuffer({
      size: 256,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const outputBuf = device.createBuffer({
      size: outputCap,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const stateBuf = device.createBuffer({
      size: stateCap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readbackBuf = device.createBuffer({
      size: outputCap,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Zero meta.
    device.queue.writeBuffer(metaBuf, 0, new Uint32Array(64));

    return new BrotligStreamDecoder(
      device,
      pipeline,
      bindGroupLayout,
      inputBuf,
      inputCap,
      metaBuf,
      outputBuf,
      outputCap,
      stateBuf,
      stateCap,
      readbackBuf,
      maxStreams,
    );
  }

  // Grow one of the persistent buffers, preserving existing content via a
  // GPU-side copy. For output, the readback buffer must grow in lockstep.
  private growBuffer(
    kind: "input" | "output" | "state",
    required: number,
  ): void {
    const newCap = nextPow2(required);
    // Cached bind group references the old buffers; invalidate it.
    this.cachedBindGroup = null;
    const encoder = this.device.createCommandEncoder();
    if (kind === "input") {
      const next = this.device.createBuffer({
        size: newCap,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      if (this.inputWritten > 0) {
        encoder.copyBufferToBuffer(this.inputBuf, 0, next, 0, this.inputWritten);
      }
      this.device.queue.submit([encoder.finish()]);
      this.inputBuf.destroy();
      this.inputBuf = next;
      this.inputCap = newCap;
    } else if (kind === "output") {
      const next = this.device.createBuffer({
        size: newCap,
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      if (this.outputWritten > 0) {
        encoder.copyBufferToBuffer(this.outputBuf, 0, next, 0, this.outputWritten);
      }
      this.device.queue.submit([encoder.finish()]);
      this.outputBuf.destroy();
      this.outputBuf = next;
      this.outputCap = newCap;
      // Readback must match output capacity.
      this.readbackBuf.destroy();
      this.readbackBuf = this.device.createBuffer({
        size: newCap,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    } else {
      // state: per-stream content persists across dispatches within a
      // stream's lifetime, so preserve it.
      const next = this.device.createBuffer({
        size: newCap,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      if (this.stateCap > 0) {
        encoder.copyBufferToBuffer(this.stateBuf, 0, next, 0, this.stateCap);
      }
      this.device.queue.submit([encoder.finish()]);
      this.stateBuf.destroy();
      this.stateBuf = next;
      this.stateCap = newCap;
    }
  }

  private appendToRing(chunk: Uint8Array): void {
    const needed = this.ringLen + chunk.length;
    if (needed > this.ring.length) {
      const grown = new Uint8Array(nextPow2(Math.max(needed, 1024)));
      grown.set(this.ring.subarray(0, this.ringLen));
      this.ring = grown;
    }
    this.ring.set(chunk, this.ringLen);
    this.ringLen += chunk.length;
  }

  // Compressed byte range (relative to dataOffset) for page `p`. Uses the
  // format.ts quirk: pageOffsets[0] holds the LAST page's compressed size;
  // pageOffsets[i>0] is the start offset of page i relative to dataOffset.
  private pageStart(stream: ParsedStreamHeader, p: number): number {
    return p === 0 ? 0 : stream.pageOffsets[p];
  }
  private pageEnd(stream: ParsedStreamHeader, p: number): number {
    const numPages = stream.header.numPages;
    if (p < numPages - 1) return stream.pageOffsets[p + 1];
    // last page: start + size (size stored in pageOffsets[0])
    const lastStart = numPages > 1 ? stream.pageOffsets[numPages - 1] : 0;
    return lastStart + stream.pageOffsets[0];
  }

  // Upload aligned to 4 bytes (writeBuffer requirement). Padding bytes
  // beyond the actual body are harmless as the shader only reads up to
  // each page's compressed size.
  private writeAligned(dstOffset: number, slice: Uint8Array): void {
    const padded = slice.byteLength % 4 === 0
      ? slice
      : (() => {
          const p = new Uint8Array(Math.ceil(slice.byteLength / 4) * 4);
          p.set(slice);
          return p;
        })();
    this.device.queue.writeBuffer(
      this.inputBuf,
      dstOffset,
      padded.buffer,
      padded.byteOffset,
      padded.byteLength,
    );
  }

  // Start a new in-flight stream as soon as the header + precon + page table
  // are resident in the ring. Uploads only the header bytes; page bodies are
  // uploaded lazily as they become resident. Returns false if the ring does
  // not yet contain enough bytes to parse the header.
  private tryStartStream(): boolean {
    if (this.ringLen === 0) return false;
    const view = this.ring.subarray(0, this.ringLen);
    const parsed = parseStreamHeaderOnly(view, 0);
    if (!parsed) return false;

    const headerBytes = parsed.totalHeaderBytes;
    const inputBase = this.inputWritten;
    // Reserve capacity for the header now; body will extend inputWritten
    // as pages are uploaded.
    if (inputBase + headerBytes > this.inputCap) {
      this.growBuffer("input", inputBase + headerBytes);
    }
    this.writeAligned(inputBase, view.subarray(0, headerBytes));
    this.inputWritten = inputBase + headerBytes;

    // Reserve an output region of uncompressedSize.
    const outputBase = this.outputWritten;
    const outputEnd = outputBase + parsed.header.uncompressedSize;
    if (outputEnd > this.outputCap) this.growBuffer("output", outputEnd);
    this.outputWritten = outputEnd;

    const stateSlot = 0;
    if ((stateSlot + 1) * STATE_STRIDE_BYTES > this.stateCap) {
      this.growBuffer("state", (stateSlot + 1) * STATE_STRIDE_BYTES);
    }

    this.inFlight = {
      stream: parsed,
      inputBase,
      outputBase,
      stateSlot,
      pageCursor: 0,
      producedBytes: 0,
      headerUploaded: true,
      pagesUploaded: 0,
      inputBodyWritten: inputBase + headerBytes,
    };

    // Drop the consumed header bytes from the ring.
    this.ring.copyWithin(0, headerBytes, this.ringLen);
    this.ringLen -= headerBytes;
    return true;
  }

  async push(chunk: Uint8Array): Promise<Uint8Array> {
    if (this.destroyed) throw new Error("BrotligStreamDecoder is destroyed");
    // Rewind persistent input/output cursors when no stream is mid-flight
    // and the CPU ring is empty. Without this, repeated decode() calls on
    // a cached decoder grow inputWritten/outputWritten monotonically and
    // eventually exceed any allocated buffer capacity.
    if (!this.inFlight && this.ringLen === 0) {
      this.inputWritten = 0;
      this.outputWritten = 0;
    }
    this.appendToRing(chunk);

    const produced: Uint8Array[] = [];

    // Incremental-upload / page-level-streaming loop:
    //   1. Start a stream as soon as its header+precon+page table arrive.
    //   2. Upload only the page bodies that are currently resident in the
    //      ring, advancing pagesUploaded.
    //   3. Dispatch with pageLimit = pagesUploaded. The shader breaks its
    //      inner page loop at pageLimit and spills state so the next
    //      dispatch resumes. Readback pulls only the newly produced bytes.
    //   4. Retire the stream when pageCursor >= numPages.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (!this.inFlight) {
        if (!this.tryStartStream()) break;
      }
      const cur = this.inFlight!;
      const stream = cur.stream;
      const numPages = stream.header.numPages;

      // Compute how many additional pages are fully resident in the ring.
      // pageStart(pagesUploaded) is the byte offset (relative to dataOffset)
      // from which the ring contents begin; an additional page p is resident
      // iff pageEnd(p) - pageStart(pagesUploaded) <= ringLen.
      const baseRel = this.pageStart(stream, cur.pagesUploaded);
      let newResidentPages = 0;
      while (cur.pagesUploaded + newResidentPages < numPages) {
        const p = cur.pagesUploaded + newResidentPages;
        const need = this.pageEnd(stream, p) - baseRel;
        if (need > this.ringLen) break;
        newResidentPages++;
      }

      if (newResidentPages > 0) {
        const firstNew = cur.pagesUploaded;
        const lastNew = firstNew + newResidentPages - 1;
        const uploadBytes =
          this.pageEnd(stream, lastNew) - this.pageStart(stream, firstNew);
        const dstOff = cur.inputBodyWritten;
        if (dstOff + uploadBytes > this.inputCap) {
          this.growBuffer("input", dstOff + uploadBytes);
        }
        this.writeAligned(dstOff, this.ring.subarray(0, uploadBytes));
        cur.inputBodyWritten = dstOff + uploadBytes;
        this.inputWritten = Math.max(this.inputWritten, cur.inputBodyWritten);
        cur.pagesUploaded += newResidentPages;

        // Drop the consumed body bytes from the ring.
        this.ring.copyWithin(0, uploadBytes, this.ringLen);
        this.ringLen -= uploadBytes;
      }

      // If no pages are available to dispatch beyond what has been decoded,
      // we cannot make progress with the current buffer state.
      if (cur.pagesUploaded <= cur.pageCursor) break;

      const pageLimit = cur.pagesUploaded;
      if (typeof process !== "undefined" && process.env?.BROTLIG_DEBUG) {
        // eslint-disable-next-line no-console
        console.log(
          `[brotlig] dispatch pageLimit=${pageLimit} pagesUploaded=${cur.pagesUploaded} numPages=${numPages} pageCursor=${cur.pageCursor}`,
        );
      }
      const newBytes = await this.dispatch(cur, pageLimit);
      if (newBytes.length > 0) produced.push(newBytes);

      if (cur.pageCursor >= numPages) {
        this.inFlight = null;
        continue;
      }
    }

    return concat(produced);
  }

  private async dispatch(
    cur: InFlightStream,
    pageLimit: number,
  ): Promise<Uint8Array> {
    // Write the meta buffer for this dispatch.
    // Layout: header(4) + 1 per-stream record(4) = 8 u32s.
    const meta = new Uint32Array(META_HEADER_WORDS + META_PER_STREAM_WORDS);
    meta[0] = 1; // remaining stream count (single stream per dispatch)
    meta[1] = pageLimit;
    meta[2] = cur.pageCursor > 0 ? 1 : 0; // resume flag
    meta[3] = 0;
    meta[4] = cur.inputBase; // rptr
    meta[5] = cur.outputBase + cur.producedBytes; // wptr
    meta[6] = cur.pageCursor; // pageCursor
    meta[7] = cur.stateSlot * STATE_STRIDE_BYTES; // savedStateOffset
    this.device.queue.writeBuffer(this.metaBuf, 0, meta.buffer, 0, META_BYTES_FOR_STREAMS(1));

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    // TODO: cache bind groups keyed by (inputBuf, metaBuf, outputBuf,
    // stateBuf) identity rather than rebuilding per dispatch. Rebuild is
    // only strictly required after growBuffer().
    if (!this.cachedBindGroup) {
      this.cachedBindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.inputBuf } },
          { binding: 1, resource: { buffer: this.metaBuf } },
          { binding: 2, resource: { buffer: this.outputBuf } },
          { binding: 3, resource: { buffer: this.stateBuf } },
        ],
      });
    }
    const bindGroup = this.cachedBindGroup;
    pass.setBindGroup(0, bindGroup);
    // TODO: REFERENCE_DISPATCH_GROUPS = 2560 is a D3D12-era over-dispatch
    // ceiling. Compute a tight count (~ pages * streams) once the shader's
    // work-claiming scheme is validated end-to-end on WebGPU.
    pass.dispatchWorkgroups(REFERENCE_DISPATCH_GROUPS, 1, 1);
    pass.end();

    // The shader writes each page at streamWptr + page_index * pageSize.
    // After a dispatch with pageLimit == pagesUploaded, the shader has
    // populated pages [pageCursor, pageLimit). We do not read meta[5]
    // back: the shader does not update the wptr slot during decode (wptr
    // is a function-local inside CSMain), so its value remains the
    // host-supplied streamWptr. Instead, produced bytes are deterministic
    // from pagesUploaded because page sizes are fixed (pageSize) except
    // the last page which uses lastPageSize.
    const header = cur.stream.header;
    const finishedAll = pageLimit >= header.numPages;
    const newEnd = finishedAll
      ? header.uncompressedSize
      : pageLimit * header.pageSize;
    const newStart = cur.producedBytes;
    const newLen = newEnd - newStart;

    // Copy the produced range to the readback buffer. We reuse the same
    // encoder so the copy is ordered after the compute pass.
    if (newLen > 0) {
      const srcOff = cur.outputBase + newStart;
      const copyLen = Math.ceil(newLen / 4) * 4;
      encoder.copyBufferToBuffer(
        this.outputBuf,
        srcOff,
        this.readbackBuf,
        0,
        copyLen,
      );
    }

    this.device.queue.submit([encoder.finish()]);

    let outBytes = new Uint8Array(0);
    if (newLen > 0) {
      const copyLen = Math.ceil(newLen / 4) * 4;
      await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, copyLen);
      const src = new Uint8Array(this.readbackBuf.getMappedRange(0, copyLen));
      // Slice to the exact unpadded length.
      outBytes = new Uint8Array(src.subarray(0, newLen));
      this.readbackBuf.unmap();
    }

    // Advance host-side cursors to reflect the pages the shader decoded.
    cur.producedBytes = newEnd;
    cur.pageCursor = pageLimit;

    return outBytes;
  }

  async end(): Promise<Uint8Array> {
    if (this.destroyed) throw new Error("BrotligStreamDecoder is destroyed");
    if (this.inFlight) {
      throw new Error(
        "BrotligStreamDecoder.end(): a stream is still in flight",
      );
    }
    if (this.ringLen !== 0) {
      throw new Error(
        `BrotligStreamDecoder.end(): ${this.ringLen} trailing bytes do not form a complete stream`,
      );
    }
    return new Uint8Array(0);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.inputBuf.destroy();
    this.metaBuf.destroy();
    this.outputBuf.destroy();
    this.stateBuf.destroy();
    this.readbackBuf.destroy();
  }
}

function concat(chunks: Uint8Array[]): Uint8Array {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

// One-shot convenience. Callers who need to decode many streams should
// create a BrotligStreamDecoder explicitly and reuse it via push()/end();
// decode() creates a fresh decoder per call.
export async function decode(
  input: Uint8Array,
  opts?: BrotligDecoderOptions,
): Promise<Uint8Array> {
  const dec = await BrotligStreamDecoder.create(opts);
  try {
    const a = await dec.push(input);
    const b = await dec.end();
    return concat([a, b]);
  } finally {
    dec.destroy();
  }
}

// Keep a reference to findCompleteStreams so callers that want whole-buffer
// parsing can still import it via the decoder module surface if desired.
// TODO: decide whether to re-export from index.ts instead.
export { findCompleteStreams };
