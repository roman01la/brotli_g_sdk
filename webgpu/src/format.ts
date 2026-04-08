// Brotli-G stream format helpers.
//
// Layout mirrors `inc/DataStream.h` (StreamHeader / PreconditionHeader) and
// `inc/common/BrotligConstants.h`. Wire format is little-endian.
//
// StreamHeader (8 bytes, BROTLIG_STREAM_HEADER_SIZE_BITS = 64):
//   u8  Id                         (== BROTLIG_STREAM_ID = 5)
//   u8  Magic                      (== Id ^ 0xff)
//   u16 NumPages
//   u32 bitfield:
//       [ 0: 1] PageSizeIdx        (2 bits)
//       [ 2:19] LastPageSize       (18 bits)
//       [20:20] Preconditioned     (1 bit)
//       [21:31] Reserved           (11 bits)
//
// PreconditionHeader (8 bytes, only present when Preconditioned == 1):
//   u32 word0:
//       [ 0: 0] Swizzled
//       [ 1: 1] PitchD3D12Aligned
//       [ 2:16] WidthInBlocks  (stored as value-1)
//       [17:31] HeightInBlocks (stored as value-1)
//   u32 word1:
//       [ 0: 7] Format
//       [ 8:12] NumMips        (stored as value-1)
//       [13:31] PitchInBytes   (stored as value-1)
//
// Page offset table: exactly NumPages u32 entries immediately following the
// (optional) precondition header. Per src/BrotligEncoder.cpp:201-221 and
// src/BrotligDecoder.cpp:138-151 the encoder stores:
//   pageTable[0]       = compressed size of the LAST page
//   pageTable[i > 0]   = start offset of page i relative to the byte after
//                        the page table
// Page 0 always starts at offset 0 after the table. Total compressed payload
// bytes = pageTable[numPages - 1] + pageTable[0] (last page start + size);
// for numPages == 1 that reduces to pageTable[0].

export const BROTLIG_STREAM_ID = 5;
export const BROTLIG_STREAM_HEADER_SIZE = 8;
export const BROTLIG_PRECON_HEADER_SIZE = 8;
export const BROTLIG_MIN_PAGE_SIZE = 32 * 1024;

export interface StreamHeader {
  id: number;
  numPages: number;
  pageSizeIdx: number;
  pageSize: number;
  lastPageSize: number;
  isPreconditioned: boolean;
  uncompressedSize: number;
}

export interface PreconditionHeader {
  swizzled: boolean;
  pitchD3D12Aligned: boolean;
  widthInBlocks: number;
  heightInBlocks: number;
  format: number;
  numMips: number;
  pitchInBytes: number;
}

export interface ParsedStream {
  header: StreamHeader;
  precondition: PreconditionHeader | null;
  /** Per-page table as stored on the wire (length == numPages). Entry 0
   *  holds the last page's compressed size; entries i>0 hold the start
   *  offset of page i relative to `dataOffset`. */
  pageOffsets: Uint32Array;
  /** Absolute byte offset (within `buf`) where page 0's compressed data
   *  begins, i.e. first byte after the page offset table. */
  dataOffset: number;
  /** Absolute byte offset (within `buf`) where this stream starts. */
  streamOffset: number;
  /** Total byte length of this stream (header + precon + table + pages). */
  totalBytes: number;
}

function bits(v: number, lo: number, hi: number): number {
  // Extract bits [lo..hi] inclusive from a 32-bit unsigned value.
  const width = hi - lo + 1;
  const mask = width === 32 ? 0xffffffff : (1 << width) - 1;
  return (v >>> lo) & mask;
}

export function parseStream(
  buf: Uint8Array,
  offset: number,
): ParsedStream | null {
  if (offset < 0 || offset > buf.length) return null;
  if (buf.length - offset < BROTLIG_STREAM_HEADER_SIZE) return null;

  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  const id = view.getUint8(offset + 0);
  const magic = view.getUint8(offset + 1);
  if ((id ^ 0xff) !== magic) return null;
  if (id !== BROTLIG_STREAM_ID) return null;

  const numPages = view.getUint16(offset + 2, true);
  const word = view.getUint32(offset + 4, true);
  const pageSizeIdx = bits(word, 0, 1);
  const lastPageSize = bits(word, 2, 19);
  const isPreconditioned = bits(word, 20, 20) === 1;

  const pageSize = BROTLIG_MIN_PAGE_SIZE << pageSizeIdx;
  const uncompressedSize =
    numPages * pageSize - (lastPageSize === 0 ? 0 : pageSize - lastPageSize);

  let cursor = offset + BROTLIG_STREAM_HEADER_SIZE;

  let precondition: PreconditionHeader | null = null;
  if (isPreconditioned) {
    if (buf.length - cursor < BROTLIG_PRECON_HEADER_SIZE) return null;
    const w0 = view.getUint32(cursor + 0, true);
    const w1 = view.getUint32(cursor + 4, true);
    precondition = {
      swizzled: bits(w0, 0, 0) === 1,
      pitchD3D12Aligned: bits(w0, 1, 1) === 1,
      widthInBlocks: bits(w0, 2, 16) + 1,
      heightInBlocks: bits(w0, 17, 31) + 1,
      format: bits(w1, 0, 7),
      numMips: bits(w1, 8, 12) + 1,
      pitchInBytes: bits(w1, 13, 31) + 1,
    };
    cursor += BROTLIG_PRECON_HEADER_SIZE;
  }

  const tableBytes = numPages * 4;
  if (buf.length - cursor < tableBytes) return null;

  const pageOffsets = new Uint32Array(numPages);
  for (let i = 0; i < numPages; i++) {
    pageOffsets[i] = view.getUint32(cursor + i * 4, true);
  }
  cursor += tableBytes;

  // dataOffset is the first byte after the page table. pageOffsets[0] holds
  // the last page's compressed size; all other entries are start offsets
  // relative to dataOffset. Total page data = lastStart + lastSize.
  const dataOffset = cursor;
  const lastPageStart = numPages > 1 ? pageOffsets[numPages - 1] : 0;
  const lastPageCompressedSize = numPages > 0 ? pageOffsets[0] : 0;
  const pageDataBytes = lastPageStart + lastPageCompressedSize;

  const totalBytes = dataOffset - offset + pageDataBytes;
  if (buf.length - offset < totalBytes) return null;

  const header: StreamHeader = {
    id,
    numPages,
    pageSizeIdx,
    pageSize,
    lastPageSize,
    isPreconditioned,
    uncompressedSize,
  };

  return {
    header,
    precondition,
    pageOffsets,
    dataOffset,
    streamOffset: offset,
    totalBytes,
  };
}

export function findCompleteStreams(buf: Uint8Array): {
  streams: ParsedStream[];
  consumed: number;
} {
  // The SDK buffer format (see src/BrotligDecoder.cpp:234-265) is a single
  // StreamHeader at offset 0; there is no top-level multi-stream container
  // header. We still loop so a caller can concatenate several complete
  // streams back-to-back in one buffer.
  const streams: ParsedStream[] = [];
  let consumed = 0;
  while (consumed < buf.length) {
    const parsed = parseStream(buf, consumed);
    if (!parsed) break;
    streams.push(parsed);
    consumed += parsed.totalBytes;
  }
  return { streams, consumed };
}
