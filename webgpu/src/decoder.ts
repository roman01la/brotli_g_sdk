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
  parseStream,
  BROTLIG_STREAM_HEADER_SIZE,
  BROTLIG_PRECON_HEADER_SIZE,
  type ParsedStream,
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
  stream: ParsedStream;
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
  // Bytes produced by prior dispatches (running wptr).
  producedBytes: number;
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

  // Parse a stream header + (possibly partial) page table to determine how
  // many pages are fully resident in `buf` starting at `offset`. Returns
  // null if the header/page-table is not yet fully present.
  //
  // Uses the format.ts quirk: pageOffsets[0] is the compressed size of the
  // LAST page; pageOffsets[i>0] is the start offset of page i relative to
  // dataOffset. Therefore the end of page i (for i < numPages-1) is
  // pageOffsets[i+1], and the end of the last page is
  // pageOffsets[numPages-1] + pageOffsets[0].
  private residentPageCount(
    buf: Uint8Array,
    offset: number,
  ): { resident: number; parsed: ParsedStream } | null {
    // parseStream returns null while header/precon/page-table are partial,
    // but it also returns null if the *body* is partial. We want the latter
    // case to still succeed for partial-body streams, so we inline a
    // lighter-weight residency check using the same layout rules.
    if (buf.length - offset < BROTLIG_STREAM_HEADER_SIZE) return null;
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const id = view.getUint8(offset + 0);
    const magic = view.getUint8(offset + 1);
    if ((id ^ 0xff) !== magic) return null;
    const numPages = view.getUint16(offset + 2, true);
    const word = view.getUint32(offset + 4, true);
    const isPrecon = ((word >>> 20) & 1) === 1;
    let cursor = offset + BROTLIG_STREAM_HEADER_SIZE;
    if (isPrecon) cursor += BROTLIG_PRECON_HEADER_SIZE;
    const tableBytes = numPages * 4;
    if (buf.length - cursor < tableBytes) return null;

    // We have enough to read the page table; synthesize a ParsedStream via
    // parseStream with a virtual buffer that is "complete enough". Because
    // parseStream requires the full body, we temporarily relax by building
    // the ParsedStream manually from what we just read, but only when the
    // full stream is resident. Otherwise we return a minimal descriptor.
    const pageOffsets = new Uint32Array(numPages);
    for (let i = 0; i < numPages; i++) {
      pageOffsets[i] = view.getUint32(cursor + i * 4, true);
    }
    const dataOffset = cursor + tableBytes;
    const lastSize = numPages > 0 ? pageOffsets[0] : 0;
    const lastStart = numPages > 1 ? pageOffsets[numPages - 1] : 0;
    const pageDataBytes = lastStart + lastSize;
    const totalBytes = dataOffset - offset + pageDataBytes;

    // Count pages fully resident. Page i end offset (relative to dataOffset)
    // is pageOffsets[i+1] for i < numPages-1, else lastStart+lastSize.
    let resident = 0;
    for (let i = 0; i < numPages; i++) {
      const endRel =
        i < numPages - 1 ? pageOffsets[i + 1] : lastStart + lastSize;
      if (buf.length - dataOffset >= endRel) resident++;
      else break;
    }

    // If the whole stream is resident, prefer parseStream's canonical
    // ParsedStream (includes precondition).
    const fullyResident = resident === numPages;
    if (fullyResident) {
      const parsed = parseStream(buf, offset);
      if (!parsed) return null;
      return { resident, parsed };
    }

    // Construct a partial ParsedStream manually. We reuse the header but
    // mark totalBytes as what WOULD be present if fully resident. Callers
    // must not rely on buf containing all of [offset, offset+totalBytes).
    const pageSizeIdx = word & 0x3;
    const lastPageSize = (word >>> 2) & 0x3ffff;
    const pageSize = (32 * 1024) << pageSizeIdx;
    const uncompressedSize =
      numPages * pageSize - (lastPageSize === 0 ? 0 : pageSize - lastPageSize);
    const parsed: ParsedStream = {
      header: {
        id,
        numPages,
        pageSizeIdx,
        pageSize,
        lastPageSize,
        isPreconditioned: isPrecon,
        uncompressedSize,
      },
      precondition: null,
      pageOffsets,
      dataOffset,
      streamOffset: offset,
      totalBytes,
    };
    return { resident, parsed };
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

    // Loop: peel off streams as they make forward progress. Each iteration
    // makes at least one dispatch against the currently-pending stream (or
    // selects a new one if none is in-flight), and breaks when no stream
    // can make progress with the bytes currently buffered.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      // If no stream is in flight, try to start one from the ring head.
      if (!this.inFlight) {
        if (this.ringLen === 0) break;
        const view = this.ring.subarray(0, this.ringLen);
        const probe = this.residentPageCount(view, 0);
        if (!probe) break; // need more header bytes
        const { parsed } = probe;

        // Upload compressed bytes. For now we only start a stream when the
        // full stream is resident, because uploading a partial body would
        // require tracking how many bytes we've pushed into the GPU input
        // buffer for this stream across pushes. That enhancement is left
        // as a TODO; see residentPageCount() comment.
        // TODO: true incremental upload of compressed bytes as they arrive,
        // to unlock pageLimit < numPages on the very first dispatch.
        if (this.ringLen < parsed.totalBytes) break;

        // Ensure input buffer capacity, then upload [streamOffset, totalBytes).
        const inputBase = this.inputWritten;
        const inputEnd = inputBase + parsed.totalBytes;
        if (inputEnd > this.inputCap) this.growBuffer("input", inputEnd);
        const slice = view.subarray(0, parsed.totalBytes);
        // writeBuffer needs an aligned-size copy; pad if necessary.
        const padded = slice.byteLength % 4 === 0
          ? slice
          : (() => {
              const p = new Uint8Array(Math.ceil(slice.byteLength / 4) * 4);
              p.set(slice);
              return p;
            })();
        this.device.queue.writeBuffer(
          this.inputBuf,
          inputBase,
          padded.buffer,
          padded.byteOffset,
          padded.byteLength,
        );
        this.inputWritten = inputEnd;

        // Reserve an output region of uncompressedSize.
        const outputBase = this.outputWritten;
        const outputEnd = outputBase + parsed.header.uncompressedSize;
        if (outputEnd > this.outputCap) this.growBuffer("output", outputEnd);
        this.outputWritten = outputEnd;

        // Reserve a state slot. We currently only support one in-flight
        // stream at a time, so slot 0 is always used.
        // TODO: pool state slots to allow maxActiveStreams concurrent
        // streams and multi-stream dispatches.
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
        };

        // Drop the consumed ring bytes now that they're uploaded.
        this.ring.copyWithin(0, parsed.totalBytes, this.ringLen);
        this.ringLen -= parsed.totalBytes;
      }

      const cur = this.inFlight;
      // Determine pageLimit. Because we currently require the whole stream
      // to be resident before uploading (see above), pageLimit is always
      // numPages. The partial-residency code path in residentPageCount()
      // is in place so that when incremental upload is implemented this
      // becomes `residentPages` instead.
      // TODO: once incremental upload works, recompute residentPages per
      // push() and set pageLimit to the number of *newly* resident pages.
      const pageLimit = cur.stream.header.numPages;
      if (cur.pageCursor >= pageLimit) {
        // Already done; fall through to readback.
      } else {
        const newBytes = await this.dispatch(cur, pageLimit);
        if (newBytes.length > 0) produced.push(newBytes);
      }

      // If the stream is now fully decoded, retire it and loop to try the
      // next one.
      if (cur.pageCursor >= cur.stream.header.numPages) {
        this.inFlight = null;
        continue;
      }
      // Stream still pending but we made no forward progress possible
      // (e.g. waiting on more input). Break.
      break;
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

    // Assume the dispatch processes pageLimit - pageCursor pages and
    // produces the corresponding uncompressed bytes. Because we currently
    // only run a single whole-stream dispatch per stream, the new byte
    // range is [producedBytes, uncompressedSize).
    // TODO: to support partial dispatches we must read back meta[5] (wptr)
    // here to learn how many bytes the shader actually wrote. That adds
    // a map-round-trip per dispatch; defer until multi-dispatch streams
    // are exercised.
    const newEnd = cur.stream.header.uncompressedSize;
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

    // Advance host-side cursors to reflect what the shader (we assume)
    // produced. When partial-dispatch support lands, these must be set
    // from the read-back meta record instead.
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

// Cached decoder for the one-shot convenience path. Creating a fresh
// decoder per call leaks GPU buffers in the Dawn native addon used by
// the `webgpu` npm package. Reusing a single instance also makes repeat
// calls ~3x faster in browsers.
const decoderCache = new WeakMap<GPUDevice, Promise<BrotligStreamDecoder>>();

export async function decode(
  input: Uint8Array,
  opts?: BrotligDecoderOptions,
): Promise<Uint8Array> {
  const device = opts?.device ?? (await requestBrotligDevice());
  let decP = decoderCache.get(device);
  if (!decP) {
    decP = BrotligStreamDecoder.create({ ...opts, device });
    decoderCache.set(device, decP);
  }
  const dec = await decP;
  const a = await dec.push(input);
  const b = await dec.end();
  return concat([a, b]);
}

// Keep a reference to findCompleteStreams so callers that want whole-buffer
// parsing can still import it via the decoder module surface if desired.
// TODO: decide whether to re-export from index.ts instead.
export { findCompleteStreams };
