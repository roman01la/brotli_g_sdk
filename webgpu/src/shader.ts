// WGSL port of BrotliGCompute.hlsl (reference: src/decoder/BrotliGCompute.hlsl,
// ~1881 LOC, entry CSMain, numthreads(32,1,1)).
//
// Translation rules: see task description. Highlights:
//   - HLSL ByteAddressBuffer -> array<u32> indexed by (byteAddr >> 2).
//     Unaligned byte loads synthesised via loadByte/load16/loadU32Unaligned.
//   - uint64_t -> vec2<u32> (lo, hi); helpers u64_*.
//   - min16uint -> u32.
//   - firstbithigh -> firstLeadingBit; firstbitlow -> firstTrailingBit.
//     Note: HLSL firstbithigh/firstbitlow of 0 returns 0xFFFFFFFF; WGSL
//     firstLeadingBit/firstTrailingBit of 0 returns 0xFFFFFFFF too, so the
//     semantics match.
//   - reversebits -> reverseBits32 helper (5-step swap).
//   - Wave intrinsics mapped to subgroup* ops; dynamic lane selections use
//     subgroupShuffle; constexpr lane selections use subgroupBroadcast.
//   - HLSL output.InterlockedOr/And -> atomicOr/And on array<atomic<u32>>.
//     Byte stores land on word-aligned addresses (addr & -4) so alignment
//     is preserved: OK.
//   - groupshared InterlockedOr -> atomicOr on workgroup-scoped atomic<u32>.
//
// Streaming additions (design):
//   meta layout:
//     meta[0] = remaining stream count
//     meta[1] = pageLimit          (host-set per dispatch; break inner loop
//                                   when next page.index would exceed this)
//     meta[2] = resumeFlag bitmask (bit i set => stream i+1 mid-decode,
//                                   restore ctx from state buffer)
//     meta[3] = reserved
//     meta[4 + (idx-1)*4 + 0] = rptr
//     meta[4 + (idx-1)*4 + 1] = wptr
//     meta[4 + (idx-1)*4 + 2] = pageCursor   (next page.index to attempt)
//     meta[4 + (idx-1)*4 + 3] = savedStateOffset (u32 index into state[])
//
//   state layout (fixed stride STATE_STRIDE_WORDS per stream; currently
//   4096 u32 = 16 KB. Per-stream block:
//     [0]  = hold.lo
//     [1]  = hold.hi
//     [2]  = validBits
//     [3]  = readPointer
//     [4]  = pageCursor
//     [5]  = npostfix
//     [6]  = n_direct
//     [7]  = isDeltaEncoded
//     [8..8+DICT_SZ)              gDictionary
//     [..+SYMLEN_DWORDS)          gSymbolLengths
//     [..+SEARCH_TABLE_LANES)     SearchTables[0] packed
//     [..+SEARCH_TABLE_LANES)     SearchTables[1] packed
//     remainder reserved
//
// Forbidden items (no HLSL emitter, no algorithm rewrites) honoured.

export const BROTLIG_WGSL_ENTRY_POINT = "main";

export const BROTLIG_WGSL: string = /* wgsl */ `
enable subgroups;
diagnostic(off, derivative_uniformity);
diagnostic(off, subgroup_uniformity);

//------------------------------------------------------------------------------
// Bindings
//------------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read>       input_buf  : array<u32>;
@group(0) @binding(1) var<storage, read_write> metaBuf    : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> output_buf : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> state_buf  : array<atomic<u32>>;

//------------------------------------------------------------------------------
// Constants (HLSL: top of file)
//------------------------------------------------------------------------------
const LOG_NUM_LANES : u32 = 5u;
const NUM_LANES : u32 = 32u;

const BROTLIG_WORK_PAGE_SIZE_UNIT : u32 = 32u * 1024u;
const BROTLIG_WORK_STREAM_HEADER_SIZE : u32 = 8u;
const BROTLIG_WORK_STREAM_PRECON_HEADER_SIZE : u32 = 8u;

const BROTLIG_NUM_CATEGORIES : u32 = 3u;

const NIBBLE_TOTAL_BITS : u32 = 4u;
const BYTE_TOTAL_BITS : u32 = 8u;
const DWORD_TOTAL_BITS : u32 = 32u;
const DWORD_TOTAL_BYTES : u32 = 4u;
const DWORD_TOTAL_NIBBLES : u32 = 8u;
const SHORT_TOTAL_BITS : u32 = 16u;

const BROTLIG_NUM_LITERAL_SYMBOLS : u32 = 256u;
const BROTLIG_EOS_COMMAND_SYMBOL : u32 = 704u;
const BROTLIG_NUM_COMMAND_SYMBOLS : u32 = 704u + 24u;
const BROTLIG_NUM_COMMAND_SYMBOLS_WITH_SENTINEL : u32 = 704u + 24u;
const BROTLIG_NUM_DISTANCE_SYMBOLS : u32 = 544u;
const BROTLIG_NUM_MAX_SYMBOLS : u32 = 704u + 24u;

const BROTLIG_REPEAT_PREVIOUS_CODE_LENGTH : u32 = 16u;
const BROTLIG_REPEAT_ZERO_CODE_LENGTH : u32 = 17u;
const BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG : u32 = 4u;
const BROTLIG_MAX_HUFFMAN_CODE_BITSIZE : u32 = 16u;
const BROTLIG_MAX_HUFFMAN_CODE_LENGTH : u32 = 15u;

const BROTLIG_LENGTH_ENCODER_SIZE : u32 = 18u;
const BROTLIG_LENGTH_ENCODER_MAX_KEY_LENGTH : u32 = 7u;
const BROTLIG_LENGTH_ENCODER_MAX_EXTRA_LENGTH : u32 = 3u;

const DECODER_LENGTH_SHIFT : u32 = 12u; // SHORT_TOTAL_BITS - BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG

// DIV_ROUND_UP(BROTLIG_NUM_COMMAND_SYMBOLS_WITH_SENTINEL, DWORD_TOTAL_NIBBLES)
// = (728 + 7) / 8 = 91
const SYMBOL_LENGTHS_DWORDS : u32 = 91u;
// DIV_ROUND_UP(256, 4) = 64
const DICTIONARY_COMPACT_LITERAL_SIZE : u32 = 64u;
// DIV_ROUND_UP(544, 3) = 182
const DICTIONARY_COMPACT_DISTANCE_SIZE : u32 = 182u;
// DIV_ROUND_UP(728, 3) = 243
const DICTIONARY_COMPACT_COMMAND_SIZE : u32 = 243u;
// 64 + 182 + 243 = 489
const DICTIONARY_COMPACT_TOTAL_SIZE : u32 = 489u;

const BROTLIG_MAX_NUM_SUB_BLOCK : u32 = 32u;
const BROTLIG_MAX_NUM_MIP_LEVELS : u32 = 32u;
const BROTLIG_PRECON_SWIZZLE_REGION_SIZE : u32 = 2u;

// Streaming state layout constants
const STATE_STRIDE_WORDS : u32 = 4096u; // 16 KB per stream
const STATE_OFF_HOLD_LO : u32 = 0u;
const STATE_OFF_HOLD_HI : u32 = 1u;
const STATE_OFF_VALID_BITS : u32 = 2u;
const STATE_OFF_READPTR : u32 = 3u;
const STATE_OFF_PAGECUR : u32 = 4u;
const STATE_OFF_NPOSTFIX : u32 = 5u;
const STATE_OFF_NDIRECT : u32 = 6u;
const STATE_OFF_ISDELTA : u32 = 7u;
const STATE_OFF_DICT : u32 = 8u;                   // length DICTIONARY_COMPACT_TOTAL_SIZE (489)
const STATE_OFF_SYMLEN : u32 = 8u + 489u;          // length SYMBOL_LENGTHS_DWORDS (91) = 588
const STATE_OFF_SEARCH0 : u32 = 8u + 489u + 91u;   // 588; 32 lanes
const STATE_OFF_SEARCH1 : u32 = 620u;              // 588 + 32; 32 lanes

//------------------------------------------------------------------------------
// Groupshared declarations
//------------------------------------------------------------------------------
var<workgroup> gDictionary       : array<atomic<u32>, 489>;  // DICTIONARY_COMPACT_TOTAL_SIZE
var<workgroup> gSymbolLengths    : array<atomic<u32>, 91>;   // SYMBOL_LENGTHS_DWORDS
var<workgroup> scoreBoard        : array<atomic<u32>, 5>;    // LOG_NUM_LANES

var<workgroup> gSizes_subblocks      : array<u32, 32>;
var<workgroup> gOffsets_subblocks    : array<u32, 32>;
var<workgroup> gOffsets_substreams   : array<u32, 32>;
var<workgroup> gColor_subblocks      : array<u32, 32>;
var<workgroup> gMip_widths           : array<u32, 32>;
var<workgroup> gMip_heights          : array<u32, 32>;
var<workgroup> gMip_pitches          : array<u32, 32>;
var<workgroup> gOffsets_mipbytes     : array<u32, 32>;
var<workgroup> gOffsets_mipblocks    : array<u32, 32>;

// SearchTables[2]: each holds a per-lane u32 "table" word. HLSL stores these
// as struct-scope statics replicated per-lane; in WGSL we keep a per-lane
// workgroup array and index with laneIx.
var<workgroup> gSearchTable0 : array<u32, 32>;
var<workgroup> gSearchTable1 : array<u32, 32>;

// Per-lane distance ring buffer (u64 as vec2<u32>). HLSL: static uint64_t distringbuffer.
var<workgroup> gDistRingLo : array<u32, 32>;
var<workgroup> gDistRingHi : array<u32, 32>;

// Literal buffer carry-over across decodecommand iterations.
var<workgroup> gLiteralBuffer : array<u32, 32>;
var<workgroup> gLiteralsKept  : atomic<u32>; // only lane 0 writes/reads as uniform

// WavePrefixSum / WaveActiveSum intermediate bookkeeping is done purely via
// subgroup ops; no groupshared staging needed.

// decoder-params / conditioner-params / insert-copy length tables live as
// private per-lane state. HLSL: static LengthCode24 insertLengthCode, copyLengthCode.
// These are rebuilt per page so they're fine as private variables.

//------------------------------------------------------------------------------
// u64 helpers (vec2<u32> = (lo, hi))
//------------------------------------------------------------------------------
fn u64_from_u32(x : u32) -> vec2<u32> { return vec2<u32>(x, 0u); }

fn u64_low(a : vec2<u32>) -> u32 { return a.x; }
fn u64_high(a : vec2<u32>) -> u32 { return a.y; }

fn u64_or_u32(a : vec2<u32>, b : u32) -> vec2<u32> { return vec2<u32>(a.x | b, a.y); }
fn u64_and_u32(a : vec2<u32>, b : u32) -> u32 { return a.x & b; }

fn u64_or(a : vec2<u32>, b : vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x | b.x, a.y | b.y);
}

fn u64_shl(a : vec2<u32>, s : u32) -> vec2<u32> {
    if (s == 0u) { return a; }
    if (s >= 64u) { return vec2<u32>(0u, 0u); }
    if (s >= 32u) {
        return vec2<u32>(0u, a.x << (s - 32u));
    }
    let hi = (a.y << s) | (a.x >> (32u - s));
    let lo = a.x << s;
    return vec2<u32>(lo, hi);
}

fn u64_shr(a : vec2<u32>, s : u32) -> vec2<u32> {
    if (s == 0u) { return a; }
    if (s >= 64u) { return vec2<u32>(0u, 0u); }
    if (s >= 32u) {
        return vec2<u32>(a.y >> (s - 32u), 0u);
    }
    let lo = (a.x >> s) | (a.y << (32u - s));
    let hi = a.y >> s;
    return vec2<u32>(lo, hi);
}

//------------------------------------------------------------------------------
// Bit / byte helpers
//------------------------------------------------------------------------------
fn MaskLsbs(n : u32) -> u32 {
    if (n >= 32u) { return 0xffffffffu; }
    return (1u << n) - 1u;
}

fn MaskLsbsA(a : u32, n : u32) -> u32 { return a & MaskLsbs(n); }

fn reverseBits32(xIn : u32) -> u32 {
    var x = xIn;
    x = ((x >> 1u) & 0x55555555u) | ((x & 0x55555555u) << 1u);
    x = ((x >> 2u) & 0x33333333u) | ((x & 0x33333333u) << 2u);
    x = ((x >> 4u) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4u);
    x = ((x >> 8u) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8u);
    x = (x >> 16u) | (x << 16u);
    return x;
}

fn loadU32Aligned(byteAddr : u32) -> u32 {
    return input_buf[byteAddr >> 2u];
}

fn loadU32Unaligned(byteAddr : u32) -> u32 {
    let w0 = input_buf[byteAddr >> 2u];
    let sh = (byteAddr & 3u) * 8u;
    if (sh == 0u) { return w0; }
    let w1 = input_buf[(byteAddr >> 2u) + 1u];
    return (w0 >> sh) | (w1 << (32u - sh));
}

fn loadByte(byteAddr : u32) -> u32 {
    let w = input_buf[byteAddr >> 2u];
    let sh = (byteAddr & 3u) * 8u;
    return (w >> sh) & 0xffu;
}

fn load16(byteAddr : u32) -> u32 {
    return loadU32Unaligned(byteAddr) & 0xffffu;
}

fn InputByteLoad(addr : u32) -> u32 {
    let shft = (addr & 3u) << 3u;
    let w = input_buf[addr >> 2u];
    return (w >> shft) & 0xffu;
}

// HLSL: output.Load byte from RWByteAddressBuffer via atomicOr pattern. Here
// we read atomically.
fn OutputByteLoad(addr : u32) -> u32 {
    let shft = (addr & 3u) << 3u;
    let w = atomicLoad(&output_buf[addr >> 2u]);
    return (w >> shft) & 0xffu;
}

// HLSL: output.InterlockedOr((data & 0xff) << shft). Word-aligned: OK.
fn ByteStore(addr : u32, data : u32) {
    let shft = (addr & 3u) << 3u;
    let wordIx = addr >> 2u;
    atomicOr(&output_buf[wordIx], (data & 0xffu) << shft);
}

fn ByteUpdate(addr : u32, data : u32) {
    let shft = (addr & 3u) << 3u;
    let wordIx = addr >> 2u;
    let clearMask = ~(0xffu << shft);
    let d = (data & 0xffu) << shft;
    atomicAnd(&output_buf[wordIx], clearMask);
    atomicOr(&output_buf[wordIx], d);
}

// HLSL: output.Store(addr, data) word-write. Emulate via And(0) + Or(data).
fn OutputStoreU32(byteAddr : u32, data : u32) {
    let wordIx = byteAddr >> 2u;
    atomicAnd(&output_buf[wordIx], 0u);
    atomicOr(&output_buf[wordIx], data);
}

fn MakeSizeTAddr(addr : u32, bitlength : u32) -> u32 {
    return addr / (32u / bitlength);
}
fn MakeAlignShift(addr : u32, bitlength : u32) -> u32 {
    return (addr & ((32u / bitlength) - 1u)) * bitlength;
}
fn MakeReadAlignData(addr : u32, data : u32, bitlength : u32) -> u32 {
    return (data >> MakeAlignShift(addr, bitlength)) & MaskLsbs(bitlength);
}
fn MakeWriteAlignData(addr : u32, data : u32, bitlength : u32) -> u32 {
    return (data & MaskLsbs(bitlength)) << MakeAlignShift(addr, bitlength);
}

fn BitFieldBit(d : u32, fromIn : u32, toIn : u32) -> u32 {
    let s = min(fromIn, 31u);
    let e = max(s, toIn);
    let m = 0xffffffffu >> (31u - e + s);
    return (d >> s) & m;
}

fn GetBitSize(num : u32) -> u32 {
    // HLSL: (firstbithigh(num - 1) + 1) % 32
    let fb = firstLeadingBit(num - 1u);
    // firstLeadingBit(0) = 0xffffffff; match HLSL where firstbithigh(0) also -1.
    return (fb + 1u) & 31u;
}

fn DivRoundUp(num : u32, divv : u32) -> u32 { return (num + divv - 1u) / divv; }

// HLSL: __div3correct helper. num += (num&3)==3; return (num>>2, num&3)
fn Div3Correct(nIn : u32) -> vec2<u32> {
    var n = nIn;
    if ((n & 3u) == 3u) { n = n + 1u; }
    return vec2<u32>(n >> 2u, n & 3u);
}
fn Div3bit8(num : u32) -> vec2<u32> {
    return Div3Correct((num * 0x55u + (num >> 2u)) >> 6u);
}
fn Div3bit14(num : u32) -> vec2<u32> {
    return Div3Correct((num * 0x15555u) >> 16u);
}

fn BitTable32(index : u32, fieldlength : u32, table : u32) -> u32 {
    return (table >> (index * fieldlength)) & MaskLsbs(fieldlength);
}
fn BitTable64(index : u32, fieldlength : u32, table : vec2<u32>) -> u32 {
    let shifted = u64_shr(table, index * fieldlength);
    return u64_and_u32(shifted, MaskLsbs(fieldlength));
}

//------------------------------------------------------------------------------
// Wave helpers (HLSL lines ~190..296)
//------------------------------------------------------------------------------
fn ActiveIndexPrevious(cond : bool, laneIx : u32) -> u32 {
    // HLSL returns int firstbithigh(mask & MaskLsbs(laneIx)); -1 if no set bit.
    let b = subgroupBallot(cond).x;
    let masked = b & MaskLsbs(laneIx);
    // firstLeadingBit(0) == 0xffffffff which matches HLSL -1.
    return firstLeadingBit(masked);
}

// HLSL WaveHistogram(tag, present) line 243
fn WaveHistogram1(tag : u32, present : bool, laneIx : u32) -> u32 {
    var result : u32 = 0u;
    var mask : u32 = subgroupBallot(present).x;
    loop {
        if (mask == 0u) { break; }
        let l = firstTrailingBit(mask);
        let t = subgroupShuffle(tag, l);
        let eqMask = subgroupBallot(t == tag).x;
        let n = countOneBits(eqMask & mask);
        if (laneIx == t) { result = n; }
        mask = mask & ~eqMask;
    }
    return result;
}

// HLSL WaveHistogram(tag, num, present) line 261
fn WaveHistogram2(tag : u32, num : u32, present : bool, laneIx : u32) -> u32 {
    var result : u32 = 0u;
    var mask : u32 = subgroupBallot((num != 0u) && present).x;
    loop {
        if (mask == 0u) { break; }
        let l = firstLeadingBit(mask);
        let t = subgroupShuffle(tag, l);
        let b = (t == tag);
        let nIn = select(0u, num, b);
        let n = subgroupAdd(nIn);
        if (laneIx == t) { result = n; }
        let bMask = subgroupBallot(b).x;
        mask = mask & ~bMask;
    }
    return result;
}

//------------------------------------------------------------------------------
// gDictionary helpers (HLSL 309..369)
//------------------------------------------------------------------------------
fn GetRange(t : u32) -> u32 {
    if (t == 0u) { return BROTLIG_NUM_LITERAL_SYMBOLS; }
    if (t == 1u) { return BROTLIG_NUM_COMMAND_SYMBOLS; }
    return BROTLIG_NUM_DISTANCE_SYMBOLS;
}

fn GetBase(t : u32) -> u32 {
    if (t == 0u) { return 0u; }
    if (t == 1u) { return DICTIONARY_COMPACT_LITERAL_SIZE; }
    return DICTIONARY_COMPACT_LITERAL_SIZE + DICTIONARY_COMPACT_COMMAND_SIZE;
}

fn ClearDictionary(t : u32, laneIx : u32) {
    var ptr : u32 = laneIx;
    let lim = GetRange(t) / 4u;
    loop {
        if (ptr >= lim) { break; }
        atomicStore(&gDictionary[GetBase(t) + ptr], 0u);
        ptr = ptr + NUM_LANES;
    }
}

struct DictAddr {
    ix : u32,
    sh : u32,
    dt : u32,
};

fn CheckDictionaryAddress(t : u32, addr : u32, data : u32) -> DictAddr {
    let field : u32 = select(10u, 8u, t == 0u);
    var a : vec2<u32>;
    if (t != 0u) {
        a = Div3bit14(addr);
    } else {
        a = vec2<u32>(addr >> 2u, addr & 3u);
    }
    let sh = a.y * field;
    let dt = (data & MaskLsbs(field)) << sh;
    return DictAddr(a.x, sh, dt);
}

fn SetSymbol(t : u32, addr : u32, data : u32) {
    let base = GetBase(t);
    let da = CheckDictionaryAddress(t, addr, data);
    atomicOr(&gDictionary[base + da.ix], da.dt);
    /* workgroupBarrier removed: single-subgroup WG */
}

fn GetSymbol(t : u32, addr : u32) -> u32 {
    let base = GetBase(t);
    let da = CheckDictionaryAddress(t, addr, 0u);
    let word = atomicLoad(&gDictionary[base + da.ix]);
    let field = select(10u, 8u, t == 0u);
    return (word >> da.sh) & MaskLsbs(field);
}

//------------------------------------------------------------------------------
// Symbol lengths helpers
//------------------------------------------------------------------------------
fn SymbolLengthsReset(laneIx : u32) {
    var i : u32 = laneIx;
    loop {
        if (i >= SYMBOL_LENGTHS_DWORDS) { break; }
        atomicStore(&gSymbolLengths[i], 0u);
        i = i + NUM_LANES;
    }
}

fn HuffmalLengthTableInsert(data : u32, rle : u32, addr : u32) {
    for (var i : u32 = 0u; i < rle; i = i + 1u) {
        let a = MakeSizeTAddr(addr + i, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
        let d = MakeWriteAlignData(addr + i, data, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
        atomicOr(&gSymbolLengths[a], d);
    }
}

// SymbolLengths private snapshot. HLSL: struct SymbolLengths { uint __data[3]; ... }
// DIV_ROUND_UP(91, 32) = 3
struct SymbolLengthsSnap {
    d0 : u32,
    d1 : u32,
    d2 : u32,
};

fn SymLenSnapCopy(laneIx : u32) -> SymbolLengthsSnap {
    var s : SymbolLengthsSnap;
    let a0 = 0u * NUM_LANES + laneIx;
    let a1 = 1u * NUM_LANES + laneIx;
    let a2 = 2u * NUM_LANES + laneIx;
    s.d0 = select(0u, atomicLoad(&gSymbolLengths[a0]), a0 < SYMBOL_LENGTHS_DWORDS);
    s.d1 = select(0u, atomicLoad(&gSymbolLengths[a1]), a1 < SYMBOL_LENGTHS_DWORDS);
    s.d2 = select(0u, atomicLoad(&gSymbolLengths[a2]), a2 < SYMBOL_LENGTHS_DWORDS);
    return s;
}

fn SymLenSnapLaneGet(s : SymbolLengthsSnap, addr : u32) -> u32 {
    let da = MakeSizeTAddr(addr, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
    let ra = da / NUM_LANES;
    var dd : u32;
    if (ra == 0u) { dd = s.d0; }
    else if (ra == 1u) { dd = s.d1; }
    else { dd = s.d2; }
    let pp = subgroupShuffle(dd, da & (NUM_LANES - 1u));
    return MakeReadAlignData(addr, pp, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
}

//------------------------------------------------------------------------------
// Huffman region helpers (HLSL 420..460)
//------------------------------------------------------------------------------
struct HuffmanRegionResult { offset : u32, keyReturn : u32 };

fn HuffmanDefineRegions(number : u32, bitlength : u32) -> HuffmanRegionResult {
    let GUARD_BITS : u32 = 1u;
    let offset = subgroupExclusiveAdd(number);
    let increment = number << (SHORT_TOTAL_BITS - bitlength - GUARD_BITS);
    let key = subgroupExclusiveAdd(increment);
    return HuffmanRegionResult(offset, key);
}

fn CompactHuffmanSearchEntry(c : u32, o : u32, l : u32) -> u32 {
    let mo = MaskLsbs(DECODER_LENGTH_SHIFT);
    let ml = MaskLsbs(SHORT_TOTAL_BITS - DECODER_LENGTH_SHIFT);
    let shiftedLength = (l & ml) << DECODER_LENGTH_SHIFT;
    return (c << SHORT_TOTAL_BITS) | (o & mo) | shiftedLength;
}

// HLSL: out uint table is a per-lane out parameter updated by broadcasting.
// We return the modified table word and the offset.
struct CompactedHuffmanResult { offset : u32, table : u32 };

fn CreateCompactedHuffmanSearchTable(number : u32, tableIn : u32, laneIx : u32) -> CompactedHuffmanResult {
    let r = HuffmanDefineRegions(number, laneIx);
    let offset = r.offset;
    var key = r.keyReturn;

    let isvalue = (number != 0u) && (laneIx < BROTLIG_MAX_HUFFMAN_CODE_BITSIZE);

    // HLSL: key = WaveReadLaneAt(key, laneIx + 1); dynamic lane -> subgroupShuffle.
    key = subgroupShuffle(key, (laneIx + 1u) & 31u);

    let val = CompactHuffmanSearchEntry(key, offset, laneIx);

    var table = tableIn;
    var mask = subgroupBallot(isvalue).x;
    var ix : u32 = 0u;
    loop {
        if (mask == 0u) { break; }
        let lane = firstTrailingBit(mask);
        let value = subgroupShuffle(val, lane);
        if (laneIx == ix) { table = value; }
        mask = mask & (mask - 1u);
        ix = ix + 1u;
    }

    return CompactedHuffmanResult(offset + number, table);
}

//------------------------------------------------------------------------------
// Decoder state (bit reader)    HLSL 195..241
//------------------------------------------------------------------------------
struct DecoderState {
    readPointer : u32,
    validBits : u32,
    holdLo : u32,
    holdHi : u32,
    lane : u32,
};

fn BS_FetchNextDword(bs : ptr<function, DecoderState>, en : bool) {
    if (en && (*bs).validBits <= DWORD_TOTAL_BITS) {
        let rb = loadU32Aligned((*bs).readPointer);
        let shifted = u64_shl(vec2<u32>(rb, 0u), (*bs).validBits);
        (*bs).holdLo = (*bs).holdLo | shifted.x;
        (*bs).holdHi = (*bs).holdHi | shifted.y;
        (*bs).validBits = (*bs).validBits + DWORD_TOTAL_BITS;
        (*bs).readPointer = (*bs).readPointer + DWORD_TOTAL_BYTES;
    }
}

fn BS_Init(bs : ptr<function, DecoderState>, i : u32, lane : u32) {
    (*bs).lane = lane;
    let lo = loadU32Aligned(i);
    let hi = loadU32Aligned(i + 4u);
    (*bs).holdLo = lo;
    (*bs).holdHi = hi;
    (*bs).validBits = DWORD_TOTAL_BITS * 2u;
    (*bs).readPointer = i + DWORD_TOTAL_BYTES * 2u;
}

fn BS_DropBitsNoFetch(bs : ptr<function, DecoderState>, sizeIn : u32, en : bool) {
    let sz = select(0u, sizeIn, en);
    let shifted = u64_shr(vec2<u32>((*bs).holdLo, (*bs).holdHi), sz);
    (*bs).holdLo = shifted.x;
    (*bs).holdHi = shifted.y;
    (*bs).validBits = (*bs).validBits - sz;
}

fn BS_DropBits(bs : ptr<function, DecoderState>, sizeIn : u32, en : bool) {
    BS_DropBitsNoFetch(bs, sizeIn, en);
    BS_FetchNextDword(bs, en);
}

fn BS_GetBits(bs : ptr<function, DecoderState>, size : u32) -> u32 {
    let lo = (*bs).holdLo;
    if (size >= 32u) { return lo; }
    return lo & MaskLsbs(size);
}

fn BS_GetAndDropBits(bs : ptr<function, DecoderState>, size : u32, en : bool) -> u32 {
    let r = select(0u, BS_GetBits(bs, size), en);
    BS_DropBits(bs, size, en);
    return r;
}

//------------------------------------------------------------------------------
// InlineDecoder  (HLSL 485..527)
// We keep the per-lane packed table in gSearchTable{0,1}.
//------------------------------------------------------------------------------
fn SearchTableGet(tbl : u32, laneIx : u32) -> u32 {
    if (tbl == 0u) { return gSearchTable0[laneIx]; }
    return gSearchTable1[laneIx];
}
fn SearchTableSet(tbl : u32, laneIx : u32, v : u32) {
    if (tbl == 0u) { gSearchTable0[laneIx] = v; return; }
    gSearchTable1[laneIx] = v;
}

fn SearchTableInit(tblIx : u32, t : u32, sel : u32, laneIx : u32) {
    var mine : u32;
    if (tblIx == 0u) { mine = gSearchTable0[laneIx]; } else { mine = gSearchTable1[laneIx]; }
    if (sel != 0u) {
        let partner = subgroupShuffle(t, laneIx ^ (NUM_LANES / 2u));
        if (laneIx >= NUM_LANES / 2u) { mine = partner; }
    } else {
        if (laneIx < NUM_LANES / 2u) { mine = t; }
    }
    if (tblIx == 0u) { gSearchTable0[laneIx] = mine; } else { gSearchTable1[laneIx] = mine; }
    /* workgroupBarrier removed: single-subgroup WG */
}

struct InlineDecodeResult { symbol : u32, length : u32 };

fn InlineDecode(tblIx : u32, selIn : u32, t : u32, key : u32, laneIx : u32) -> InlineDecodeResult {
    var sel : u32 = 0u;
    if (selIn != 0u) { sel = NUM_LANES / 2u; }
    let refv = (reverseBits32(key) >> 1u) >> 16u;
    var offset : u32 = 0u;
    var code : u32 = 0u;
    var length : u32 = BROTLIG_MAX_HUFFMAN_CODE_BITSIZE;
    var done : bool = false;

    for (var i : u32 = 0u; i < BROTLIG_MAX_HUFFMAN_CODE_BITSIZE; i = i + 1u) {
        // HLSL: d = WaveReadLaneAt(__table, sel + i). Dynamic -> subgroupShuffle.
        let mine = SearchTableGet(tblIx, laneIx);
        let d = subgroupShuffle(mine, (sel + i) & 31u);
        let newcode = (d >> SHORT_TOTAL_BITS) & 0xffffu;
        if (!done) {
            offset = d & 0xffffu;
        }
        done = refv < newcode;
        if (!done) { code = newcode; }
        if (subgroupAll(done)) { break; }
    }
    let l = (offset >> DECODER_LENGTH_SHIFT) & MaskLsbs(BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
    length = l;
    offset = (offset & MaskLsbs(DECODER_LENGTH_SHIFT)) +
             ((refv - code) >> (SHORT_TOTAL_BITS - 1u - l));
    let sym = GetSymbol(t, offset);
    return InlineDecodeResult(sym, length);
}

//------------------------------------------------------------------------------
// BaseSymbolTable operations (HLSL 531..595)
//------------------------------------------------------------------------------
fn HuffmanInflateTable(t : u32, symbol : u32, length : u32, offsetIn : u32, laneIx : u32) -> u32 {
    var toDo : u32 = subgroupBallot(length != 0u).x;
    var retv : u32 = 0u;
    var offset : u32 = offsetIn;
    loop {
        if (toDo == 0u) { break; }
        let lane = firstTrailingBit(toDo);
        let leng = subgroupShuffle(length, lane);
        let same = leng == length;
        let mask = subgroupBallot(same).x;
        let offs = countOneBits(mask & MaskLsbs(laneIx));
        if (same) { offset = offset + offs; }
        if (leng == laneIx) { retv = countOneBits(mask); }
        toDo = toDo & ~mask;
    }
    if (length != 0u) {
        SetSymbol(t, offset, symbol);
    }
    return retv;
}

fn BuildHuffmanTable(t : u32, offIn : u32, tSize : u32, laneIx : u32) {
    // off = WaveReadLaneAt(off, laneIx - 1). Dynamic shuffle; HLSL behavior for
    // lane 0 reads lane 0xffffffff which effectively wraps.  We replicate via
    // subgroupShuffle with wrap mask.
    var off : u32 = subgroupShuffle(offIn, (laneIx - 1u) & 31u);

    let lenSnap = SymLenSnapCopy(laneIx);
    ClearDictionary(t, laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    var symbol : u32 = laneIx;
    loop {
        if (!subgroupAny(symbol < tSize)) { break; }
        var length = SymLenSnapLaneGet(lenSnap, symbol);
        if (symbol >= tSize) { length = 0u; }
        let offset = subgroupShuffle(off, length & 31u);
        let add = HuffmanInflateTable(t, symbol, length, offset, laneIx);
        off = off + add;
        symbol = symbol + NUM_LANES;
    }
}

//------------------------------------------------------------------------------
// FixedCodeLenCounts (HLSL 599..610)
//------------------------------------------------------------------------------
fn FixedCodeLenCounts(laneIx : u32, nsym : u32, tree_select : u32) -> u32 {
    // length_counts[4][5]
    // {0,2,0,0,0}, {0,1,2,0,0}, {0,0,4,0,0}, {0,1,1,2,0}
    var table_index : u32;
    if (nsym < 4u) { table_index = nsym - 2u; }
    else if (tree_select != 0u) { table_index = 3u; }
    else { table_index = 2u; }
    if (laneIx >= nsym) { return 0u; }
    if (table_index == 0u) {
        if (laneIx == 1u) { return 2u; } else { return 0u; }
    } else if (table_index == 1u) {
        if (laneIx == 1u) { return 1u; }
        if (laneIx == 2u) { return 2u; }
        return 0u;
    } else if (table_index == 2u) {
        if (laneIx == 2u) { return 4u; }
        return 0u;
    } else {
        if (laneIx == 1u) { return 1u; }
        if (laneIx == 2u) { return 1u; }
        if (laneIx == 3u) { return 2u; }
        return 0u;
    }
}

//------------------------------------------------------------------------------
// ReadSymbolCodeLengths (HLSL 612..692)
//------------------------------------------------------------------------------
fn ReadSymbolCodeLengths(t : u32, bs : ptr<function, DecoderState>, numCodes : u32, tSize : u32) -> u32 {
    let laneIx = (*bs).lane;

    // "gLookup.dec.Reset(laneIx);" - no-op in HLSL.

    var lengths = BS_GetAndDropBits(bs, 5u, laneIx < numCodes);
    var swizzle : u32;
    if (laneIx == 0u) { swizzle = 4u; }
    else if (laneIx == 5u) { swizzle = 5u; }
    else if (laneIx == 6u) { swizzle = 7u; }
    else if (laneIx == 16u) { swizzle = 8u; }
    else if (laneIx == 17u) { swizzle = 6u; }
    else if (laneIx >= 7u) { swizzle = laneIx + 2u; }
    else { swizzle = laneIx - 1u; }
    lengths = subgroupShuffle(lengths, swizzle & 31u);
    lengths = select(0u, lengths & 0xfu, laneIx < BROTLIG_LENGTH_ENCODER_SIZE);

    let hist0 = WaveHistogram1(lengths, (lengths != 0u) && (laneIx < BROTLIG_LENGTH_ENCODER_SIZE), laneIx);

    // Local inline decoder using a scratch table. We stage it in gSearchTable0
    // temporarily is risky because we also need SearchTables[0]/[1] later in
    // loadcomplex. Instead hold it in a private per-lane variable and mimic
    // the shuffles directly.
    var dec_table : u32 = 0xfffff000u;
    let cr = CreateCompactedHuffmanSearchTable(hist0, dec_table, laneIx);
    var hist = cr.offset;
    dec_table = cr.table;
    hist = subgroupShuffle(hist, (lengths - 1u) & 31u);
    _ = HuffmanInflateTable(t, laneIx, lengths, hist, laneIx);

    SymbolLengthsReset(laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    var rle : u32 = 0u;
    var saved : u32 = 0xffffffffu;
    var offset : u32 = 0u;
    var ptr : u32 = 0u;

    loop {
        if (!subgroupAny(ptr < tSize)) { break; }
        var keylen : u32 = 0u;
        let key = BS_GetBits(bs, BROTLIG_LENGTH_ENCODER_MAX_KEY_LENGTH + BROTLIG_LENGTH_ENCODER_MAX_EXTRA_LENGTH);

        // Inline decode against dec_table (not SearchTables[]!). We reuse
        // InlineDecode logic specialised for this scratch table.
        let refv = (reverseBits32(key) >> 1u) >> 16u;
        var loffset : u32 = 0u;
        var lcode : u32 = 0u;
        var llength : u32 = BROTLIG_MAX_HUFFMAN_CODE_BITSIZE;
        var ldone : bool = false;
        for (var i : u32 = 0u; i < BROTLIG_MAX_HUFFMAN_CODE_BITSIZE; i = i + 1u) {
            let d = subgroupShuffle(dec_table, i & 31u);
            let newcode = (d >> SHORT_TOTAL_BITS) & 0xffffu;
            if (!ldone) { loffset = d & 0xffffu; }
            ldone = refv < newcode;
            if (!ldone) { lcode = newcode; }
            if (subgroupAll(ldone)) { break; }
        }
        let ll = (loffset >> DECODER_LENGTH_SHIFT) & MaskLsbs(BROTLIG_MAX_HUFFMAN_CODE_BITSIZE_LOG);
        llength = ll;
        let lsymOffset = (loffset & MaskLsbs(DECODER_LENGTH_SHIFT)) + ((refv - lcode) >> (SHORT_TOTAL_BITS - 1u - ll));
        var symbol = GetSymbol(t, lsymOffset);
        keylen = llength;

        let keyShifted = key >> keylen;

        let lastIx = ActiveIndexPrevious(symbol != BROTLIG_REPEAT_PREVIOUS_CODE_LENGTH, laneIx);
        var repeat : u32;
        if (lastIx == 0xffffffffu) { repeat = saved; }
        else { repeat = subgroupShuffle(symbol, lastIx & 31u); }

        if (symbol == BROTLIG_REPEAT_PREVIOUS_CODE_LENGTH) {
            symbol = repeat;
            rle = 3u + (keyShifted & 3u);
            keylen = keylen + 2u;
        } else if (symbol == BROTLIG_REPEAT_ZERO_CODE_LENGTH) {
            symbol = 0u;
            rle = 3u + (keyShifted & 7u);
            keylen = keylen + 3u;
        } else {
            rle = 1u;
        }
        if (symbol >= BROTLIG_REPEAT_PREVIOUS_CODE_LENGTH) { symbol = 0u; }

        ptr = ptr + subgroupExclusiveAdd(rle);
        let ptrValid = ptr < tSize;

        var _rle : u32 = 0u;
        if ((symbol > 0u) && ptrValid) {
            _rle = min(tSize - ptr, rle);
            HuffmalLengthTableInsert(symbol, _rle, ptr);
        }
        offset = offset + WaveHistogram2(symbol, _rle, true, laneIx);

        saved = subgroupShuffle(symbol, NUM_LANES - 1u);

        BS_DropBits(bs, keylen, ptr < tSize);

        // HLSL: ptr = WaveReadLaneAt(ptr + rle, NUM_LANES - 1);
        ptr = subgroupBroadcast(ptr + rle, 31u);
    }
    return offset;
}

//------------------------------------------------------------------------------
// SymbolTable load variants (HLSL 694..790)
//------------------------------------------------------------------------------
fn loadtrivial(t : u32, bs : ptr<function, DecoderState>, tSize : u32, nsym : u32) {
    let laneIx = (*bs).lane;
    let max_bits = firstLeadingBit(tSize - 1u) + 1u;
    let sym = BS_GetAndDropBits(bs, max_bits, laneIx == 0u);
    let code : u32 = select(1u << (BROTLIG_MAX_HUFFMAN_CODE_BITSIZE - 1u),
                            1u << (BROTLIG_MAX_HUFFMAN_CODE_BITSIZE - 2u),
                            laneIx == 0u);
    let _t = CompactHuffmanSearchEntry(code, 0u, 0u);
    if (t < 2u) {
        SearchTableInit(0u, _t, t & 1u, laneIx);
    } else {
        SearchTableInit(1u, _t, t & 1u, laneIx);
    }
    SetSymbol(t, laneIx, sym);
}

fn loadsimple(t : u32, bs : ptr<function, DecoderState>, tSize : u32, nsym : u32, tree_select : u32) {
    let laneIx = (*bs).lane;
    let count = FixedCodeLenCounts(laneIx, nsym, tree_select);
    var _t : u32 = 0xfffff000u;
    let cr = CreateCompactedHuffmanSearchTable(count, _t, laneIx);
    _t = cr.table;
    if (t < 2u) { SearchTableInit(0u, _t, t & 1u, laneIx); }
    else { SearchTableInit(1u, _t, t & 1u, laneIx); }
    let max_bits = firstLeadingBit(tSize - 1u) + 1u;
    let sym = BS_GetAndDropBits(bs, max_bits, laneIx < nsym);
    SetSymbol(t, laneIx, sym);
}

fn loadcomplex(t : u32, bs : ptr<function, DecoderState>, tSize : u32, numCodes : u32) {
    let laneIx = (*bs).lane;
    let count = ReadSymbolCodeLengths(t, bs, numCodes, tSize);
    var _t : u32 = 0xfffff000u;
    let cr = CreateCompactedHuffmanSearchTable(count, _t, laneIx);
    let off = cr.offset;
    _t = cr.table;
    if (t < 2u) { SearchTableInit(0u, _t, t & 1u, laneIx); }
    else { SearchTableInit(1u, _t, t & 1u, laneIx); }
    BuildHuffmanTable(t, off, tSize, laneIx);
}

fn ReadHuffmanCode_one(t : u32, bs : ptr<function, DecoderState>, tSize : u32) {
    let laneIx = (*bs).lane;
    let data = BS_GetBits(bs, 32u);
    let headerD = subgroupBroadcastFirst(data);
    BS_DropBits(bs, 6u, laneIx == 0u);
    let cSel = BitFieldBit(headerD, 0u, 1u);
    let nSym = BitFieldBit(headerD, 2u, 3u);
    let tSel = BitFieldBit(headerD, 4u, 4u);
    let nCom = BitFieldBit(headerD, 2u, 5u) + 4u;
    switch (cSel) {
        case 0u: { loadtrivial(t, bs, tSize, nSym); }
        case 1u: { loadsimple(t, bs, tSize, nSym + 1u, tSel); }
        case 2u: { loadcomplex(t, bs, tSize, nCom); }
        default: {}
    }
}

fn ReadHuffmanCode(bs : ptr<function, DecoderState>) {
    ReadHuffmanCode_one(1u, bs, BROTLIG_NUM_COMMAND_SYMBOLS_WITH_SENTINEL);
    ReadHuffmanCode_one(2u, bs, BROTLIG_NUM_DISTANCE_SYMBOLS);
    ReadHuffmanCode_one(0u, bs, BROTLIG_NUM_LITERAL_SYMBOLS);
}

//------------------------------------------------------------------------------
// Distance ring buffer + ResolveDistances (HLSL 1189..1259)
//------------------------------------------------------------------------------
fn InitDistRingBuffer(laneIx : u32) {
    // HLSL: distringbuffer = 0x0010000f000b0004ULL  (4, 11, 15, 16)
    gDistRingLo[laneIx] = 0x000b0004u;
    gDistRingHi[laneIx] = 0x0010000fu;
}

// Returns new dist.
struct DecoderParams {
    npostfix : u32,
    n_direct : u32,
    isDeltaEncoded : u32,
};

fn ResolveDistances(symIn : u32, bs : ptr<function, DecoderState>, dparams : DecoderParams, laneIx : u32, p : bool) -> u32 {
    let sym = symIn;
    let immed = p && (sym >= 16u);
    let stack = sym < 16u;
    let fetch = immed && ((dparams.n_direct == 0u) || (sym >= 16u + dparams.n_direct));
    let ix = BitTable32(sym, 2u, 0x555000e4u);
    let offsetTab = BitTable64(sym, 4u, vec2<u32>(0x71625344u, 0x71625371u)); // HLSL: 0x7162537162534444ULL
    // NOTE: HLSL literal 0x7162537162534444ULL = hi=0x71625371 lo=0x62534444 -- recheck
    // Actually 0x7162537162534444:
    //   hi32 = 0x71625371
    //   lo32 = 0x62534444
    // Correct below.
    let offsetTabFixed = BitTable64(sym, 4u, vec2<u32>(0x62534444u, 0x71625371u));
    let offsetFixed = offsetTabFixed;

    var dist : u32;
    if (immed && !fetch) {
        dist = sym - 15u;
    } else if (p) {
        dist = gDistRingLo[laneIx] & 0xffffu;
    } else {
        dist = 0u;
    }

    let update = sym != 0u;
    if (fetch) {
        let param = sym - dparams.n_direct - 16u;
        let hcode = param >> dparams.npostfix;
        let lcode = MaskLsbsA(param, dparams.npostfix);
        let ndistbits = 1u + (hcode >> 1u);
        let extra = BS_GetAndDropBits(bs, ndistbits, true);
        let off = ((2u + (hcode & 1u)) << ndistbits) - 4u;
        dist = ((off + extra) << dparams.npostfix) + lcode + dparams.n_direct + 1u;
    }

    let umask = subgroupBallot(update).x;
    let smask = subgroupBallot(stack).x;

    var mask = subgroupBallot(p && update).x;
    loop {
        if (mask == 0u) { break; }
        let lane = firstTrailingBit(mask);
        let lmsk = 1u << lane;
        mask = mask & ~lmsk;

        let idx = subgroupShuffle(ix, lane);
        let off2 = subgroupShuffle(offsetFixed, lane);
        let tmp = subgroupShuffle(dist, lane);

        let takefromstack = ((lane == laneIx) && stack) || ((lane < laneIx) && !update);

        // reg = min16uint (distringbuffer >> (idx * 16));
        let shiftAmt = idx * 16u;
        let dr = vec2<u32>(gDistRingLo[laneIx], gDistRingHi[laneIx]);
        let drShift = u64_shr(dr, shiftAmt);
        var reg = drShift.x & 0xffffu;
        reg = reg + off2 - 4u;
        if ((smask & lmsk) == 0u) { reg = tmp; }

        if (takefromstack) { dist = reg; }

        var sx : u32 = 0u;
        var sy : u32 = 0u;
        if ((umask & lmsk) != 0u) { sx = reg; sy = 16u; }
        let cur = vec2<u32>(gDistRingLo[laneIx], gDistRingHi[laneIx]);
        let shifted = u64_shl(cur, sy);
        let combined = u64_or_u32(shifted, sx);
        gDistRingLo[laneIx] = combined.x;
        gDistRingHi[laneIx] = combined.y;
    }
    return dist;
}

//------------------------------------------------------------------------------
// LengthCode24 (HLSL 1061..1075). Instead of a struct with per-lane extraHIbaseLO
// we use private per-lane variables written by Init.
//------------------------------------------------------------------------------
fn LengthCode24_Init(bias : u32, bitDelta : u32, extDelta : u32, laneIx : u32) -> u32 {
    let bitCnt = countOneBits(bitDelta & MaskLsbs(laneIx));
    let extCnt = countOneBits(extDelta & MaskLsbs(laneIx));
    let extra = select(24u, bitCnt + extCnt, laneIx < 24u - 1u);
    let rangev = 1u << extra;
    let base = bias + subgroupExclusiveAdd(rangev);
    return base | (extra << 16u);
}

//------------------------------------------------------------------------------
// Decoder (HLSL 1077..1187)
//------------------------------------------------------------------------------
struct BrotligCmd {
    icp_code : u32,
    insert_len : u32,
    copy_len : u32,
    copy_dist : u32,
};

struct CmdLut {
    insert_len_extra : u32,
    copy_len_extra : u32,
    dist_code : i32,
    ctx : u32,
    insert_len_offset : u32,
    copy_len_offset : u32,
};

// decodelit/icp/dis inline via InlineDecode on SearchTables[category>>1], sel=category&1.
struct DecodeSymLen { sym : u32, length : u32 };

fn decodeCategory(cat : u32, bits : u32, laneIx : u32) -> DecodeSymLen {
    let tblIx = cat >> 1u;
    let sel = cat & 1u;
    let r = InlineDecode(tblIx, sel, cat, bits, laneIx);
    return DecodeSymLen(r.symbol, r.length);
}

fn decodelut(icp_codeIn : u32, enIn : bool,
             copyLengthCodePerLane : u32, insertLengthCodePerLane : u32) -> CmdLut {
    var icp_code = icp_codeIn;
    var en = enIn && (icp_code != BROTLIG_EOS_COMMAND_SYMBOL);
    let ec = en && (icp_code < BROTLIG_EOS_COMMAND_SYMBOL);
    let ic = icp_code - BROTLIG_EOS_COMMAND_SYMBOL;

    var copyCode : u32 = 0u;
    var insertCode : u32 = 0u;
    if (en) {
        let copyLsbs = (icp_code >> 0u) & MaskLsbs(3u);
        let insertLsbs = (icp_code >> 3u) & MaskLsbs(3u);
        icp_code = icp_code >> 6u;
        copyCode = BitTable32(icp_code, 2u, 0x262444u);
        insertCode = BitTable32(icp_code, 2u, 0x298500u);
        copyCode = (copyCode << 3u) | copyLsbs;
        insertCode = (insertCode << 3u) | insertLsbs;
        if (!ec) { insertCode = ic; }
    }

    let copy = subgroupShuffle(copyLengthCodePerLane, copyCode & 31u);
    let insert = subgroupShuffle(insertLengthCodePerLane, insertCode & 31u);

    var r : CmdLut;
    r.copy_len_extra   = select(0u, (copy >> 16u) & 0xffffu, ec);
    r.copy_len_offset  = select(0u, copy & 0xffffu, ec);
    r.insert_len_extra  = select(0u, (insert >> 16u) & 0xffffu, en);
    r.insert_len_offset = select(0u, insert & 0xffffu, en);
    r.dist_code = 0;
    r.ctx = 0u;
    return r;
}

fn decodecommand(bs : ptr<function, DecoderState>, laneIx : u32, en : bool,
                 copyLengthCodePerLane : u32, insertLengthCodePerLane : u32) -> BrotligCmd {
    var cmd : BrotligCmd;
    var code = select(0u, BS_GetBits(bs, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE), en);
    let d = decodeCategory(1u, code, laneIx);
    var icp = d.sym;
    var len = d.length;
    if (!en) { icp = 0u; }

    let cutoff_mask = subgroupBallot(icp == BROTLIG_EOS_COMMAND_SYMBOL).x;
    let cutoff_ix = firstTrailingBit(cutoff_mask);
    let cutoff = laneIx > cutoff_ix;
    if (cutoff) { icp = BROTLIG_EOS_COMMAND_SYMBOL; len = 0u; }

    BS_DropBits(bs, len, en);

    cmd.icp_code = icp;
    let clut = decodelut(icp, en, copyLengthCodePerLane, insertLengthCodePerLane);
    cmd.insert_len = BS_GetAndDropBits(bs, clut.insert_len_extra, en);
    cmd.copy_len = BS_GetAndDropBits(bs, clut.copy_len_extra, en);
    cmd.insert_len = cmd.insert_len + clut.insert_len_offset;
    cmd.copy_len = cmd.copy_len + clut.copy_len_offset;
    cmd.copy_dist = 0u;
    return cmd;
}

fn decodeliteral(bs : ptr<function, DecoderState>, en : bool, laneIx : u32) -> u32 {
    var code = select(0u, BS_GetBits(bs, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE), en);
    let d = decodeCategory(0u, code, laneIx);
    BS_DropBits(bs, d.length, en);
    return d.sym;
}

fn decodedistance(bs : ptr<function, DecoderState>, laneIx : u32, p : bool) -> u32 {
    var code = select(0u, BS_GetBits(bs, BROTLIG_MAX_HUFFMAN_CODE_BITSIZE), p);
    let d = decodeCategory(2u, code, laneIx);
    BS_DropBits(bs, d.length, p);
    return select(0u, d.sym, p);
}

//------------------------------------------------------------------------------
// SpreadLiterals (HLSL 1270..1317)
//------------------------------------------------------------------------------
fn SpreadValue(v0 : u32, v1 : u32, v2 : u32, v3 : u32, v4 : u32,
               mask : u32, sum : u32, en : bool, laneIx : u32) -> u32 {
    if (laneIx < LOG_NUM_LANES) { atomicStore(&scoreBoard[laneIx], 0u); }
    /* workgroupBarrier removed: single-subgroup WG */
    if (en) {
        atomicOr(&scoreBoard[0], (v0 & mask) << sum);
        atomicOr(&scoreBoard[1], (v1 & mask) << sum);
        atomicOr(&scoreBoard[2], (v2 & mask) << sum);
        atomicOr(&scoreBoard[3], (v3 & mask) << sum);
        atomicOr(&scoreBoard[4], (v4 & mask) << sum);
    }
    /* workgroupBarrier removed: single-subgroup WG */
    var result : u32 = 0u;
    for (var i : i32 = i32(LOG_NUM_LANES) - 1; i >= 0; i = i - 1) {
        result = result | (((atomicLoad(&scoreBoard[i]) >> laneIx) & 1u) << u32(i));
    }
    return result;
}

struct SpreadOut {
    writeoff : u32,
    written : u32,
    sum : u32,
    numLit : u32,
    offset : u32,
};

fn SpreadLiterals(sumIn : u32, numLitIn : u32, offsetIn : u32, laneIx : u32) -> SpreadOut {
    var sum = sumIn;
    var numLit = numLitIn;
    var offset = offsetIn;

    let mask = select(MaskLsbs(numLit), 0xffffffffu, numLit >= DWORD_TOTAL_BITS);
    let cond = (numLit != 0u) && (sum < NUM_LANES);

    // lanes[i] = -((laneIx >> i) & 1)
    let l0 = 0u - ((laneIx >> 0u) & 1u);
    let l1 = 0u - ((laneIx >> 1u) & 1u);
    let l2 = 0u - ((laneIx >> 2u) & 1u);
    let l3 = 0u - ((laneIx >> 3u) & 1u);
    let l4 = 0u - ((laneIx >> 4u) & 1u);

    let srcIdx = SpreadValue(l0, l1, l2, l3, l4, mask, sum, cond, laneIx);
    var writeoff = subgroupShuffle(offset, srcIdx & 31u);

    let c0 : u32 = 0xaaaaaaaau;
    let c1 : u32 = 0xccccccccu;
    let c2 : u32 = 0xf0f0f0f0u;
    let c3 : u32 = 0xff00ff00u;
    let c4 : u32 = 0xffff0000u;
    let relIdx = SpreadValue(c0, c1, c2, c3, c4, mask, sum, cond, laneIx);

    let lastOff = min(NUM_LANES, numLit + sum);
    let consumed = select(0u, lastOff - sum, cond);
    let written = subgroupBroadcast(lastOff, 31u);

    numLit = numLit - consumed;
    offset = offset + consumed;
    sum = sum - min(sum, NUM_LANES);
    writeoff = writeoff + relIdx;

    return SpreadOut(writeoff, written, sum, numLit, offset);
}

//------------------------------------------------------------------------------
// WriteLiterals (HLSL 1324..1347)
//------------------------------------------------------------------------------
fn WriteLiterals(bs : ptr<function, DecoderState>, wptr : u32, numLitIn : u32, offsetIn : u32, laneIx : u32) {
    var numLit = numLitIn;
    var offset = offsetIn;

    var literalsKept : u32 = gLiteralBuffer[31];   // sentinel lane; see below
    // HLSL: static uint s_literalsKept is per-lane but synced via WaveReadLaneFirst.
    // We stage s_literalsKept in atomic<u32> gLiteralsKept shared across workgroup.
    literalsKept = atomicLoad(&gLiteralsKept);

    var endOffset : u32 = (NUM_LANES - literalsKept) % NUM_LANES;
    var startIx : u32 = subgroupExclusiveAdd(numLit) + endOffset;

    loop {
        if (!subgroupAny(numLit != 0u)) { break; }
        if (literalsKept == 0u) {
            let lit = decodeliteral(bs, true, laneIx);
            gLiteralBuffer[laneIx] = lit;
        }
        let sp = SpreadLiterals(startIx, numLit, offset, laneIx);
        numLit = sp.numLit;
        offset = sp.offset;
        startIx = sp.sum;
        let written = sp.written;
        let writeoff = sp.writeoff;

        let mask = (laneIx >= endOffset) && (laneIx < written);
        endOffset = written % NUM_LANES;
        literalsKept = subgroupBroadcastFirst((NUM_LANES - endOffset) % NUM_LANES);

        if (mask) {
            ByteStore(wptr + writeoff, gLiteralBuffer[laneIx]);
        }
    }
    if (laneIx == 0u) { atomicStore(&gLiteralsKept, literalsKept); }
}

//------------------------------------------------------------------------------
// Decompress (generic, non-preconditioned) HLSL 1349..1432
//------------------------------------------------------------------------------
fn Decompress(bs : ptr<function, DecoderState>, wptrIn : u32, dParams : DecoderParams,
              outlimit : u32, laneIx : u32,
              copyLC : u32, insertLC : u32) -> u32 {
    var wptr = wptrIn;

    InitDistRingBuffer(laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    var neos : bool = true;
    var cmd = decodecommand(bs, laneIx, neos, copyLC, insertLC);

    var length = cmd.copy_len + cmd.insert_len;
    var offset : u32 = 0u;

    neos = length > 0u;
    var literalsRead : u32 = 0u;

    loop {
        if (!subgroupAll(wptr < outlimit)) { break; }

        offset = subgroupExclusiveAdd(length);
        let numlits = cmd.insert_len;
        var writeptr = wptr + offset;

        let literalsTotal = subgroupAdd(numlits);
        // The HLSL literalsFetch bookkeeping is observational; actual reads
        // happen via decodeliteral inside WriteLiterals.
        literalsRead = (literalsRead - literalsTotal) & (NUM_LANES - 1u);

        let decode_en = (cmd.icp_code >= 128u) && neos && (cmd.icp_code < BROTLIG_EOS_COMMAND_SYMBOL);

        cmd.copy_dist = decodedistance(bs, laneIx, decode_en);
        cmd.copy_dist = ResolveDistances(cmd.copy_dist, bs, dParams, laneIx, neos);

        WriteLiterals(bs, wptr, numlits, offset, laneIx);

        offset = offset + numlits;
        writeptr = writeptr + numlits;

        wptr = wptr + subgroupBroadcast(offset + cmd.copy_len, 31u);

        if (writeptr < outlimit) {
            // no clamp
        } else {
            cmd.copy_len = 0u;
        }

        let mask = subgroupBallot(neos).x;
        for (var i : u32 = 0u; i < NUM_LANES; i = i + 1u) {
            if ((mask & (1u << i)) == 0u) { continue; }
            let cOffset = subgroupShuffle(cmd.copy_dist, i);
            let cLength = subgroupShuffle(cmd.copy_len, i);
            let cOutptr = subgroupShuffle(writeptr, i);
            if (cOffset == 0u || cLength == 0u) { continue; }
            let source = cOutptr - cOffset;
            let tgt = cOutptr;
            var j : u32 = laneIx;
            loop {
                if (j >= cLength) { break; }
                let data = OutputByteLoad(source + (j % cOffset));
                ByteStore(j + tgt, data);
                j = j + NUM_LANES;
            }
        }

        if (subgroupAny(length == 0u)) { break; }

        cmd = decodecommand(bs, laneIx, neos, copyLC, insertLC);
        length = cmd.copy_len + cmd.insert_len;
        neos = length > 0u;
    }
    return wptr;
}

//------------------------------------------------------------------------------
// DeserializeCompact (HLSL 1656..1688)
//------------------------------------------------------------------------------
fn DeserializeCompact(bs : ptr<function, DecoderState>, iSize : u32, laneIx : u32) {
    // Prefetch (unused explicitly but preserves semantics).
    let _tmp = BS_GetBits(bs, 32u) << 6u;

    let avgBsSize = DivRoundUp(iSize, NUM_LANES);
    let baseSizeBits = GetBitSize(avgBsSize + 1u);
    let realSizeBits = GetBitSize(iSize);
    let deltaLogBits = GetBitSize(realSizeBits + 1u);

    var baseSize = BS_GetAndDropBits(bs, baseSizeBits, true);
    var deltaSizeBits = BS_GetAndDropBits(bs, deltaLogBits, true);

    baseSize = subgroupBroadcastFirst(baseSize);
    deltaSizeBits = subgroupBroadcastFirst(deltaSizeBits);

    var delta : u32 = 0u;
    for (var i : u32 = 0u; i < NUM_LANES; i = i + 1u) {
        let bits = BS_GetAndDropBits(bs, deltaSizeBits, true);
        if (i == laneIx) { delta = bits; }
    }
    delta = delta + baseSize;

    let offset = subgroupExclusiveAdd(delta);

    let readPtr = (*bs).readPointer - ((*bs).validBits / NUM_LANES) * DWORD_TOTAL_BYTES;
    BS_Init(bs, readPtr + (offset / DWORD_TOTAL_BYTES) * DWORD_TOTAL_BYTES, laneIx);
    BS_DropBits(bs, (offset % DWORD_TOTAL_BYTES) * BYTE_TOTAL_BITS, true);
}

//------------------------------------------------------------------------------
// Uncompressed copy paths (HLSL 1590..1654)
//------------------------------------------------------------------------------
fn UncompressedMemCopy(source : u32, destination : u32, inputSize : u32, laneIx : u32) {
    var i : u32 = laneIx * DWORD_TOTAL_BYTES;
    loop {
        if (i >= inputSize) { break; }
        let data = loadU32Aligned(source + i);
        OutputStoreU32(destination + i, data);
        i = i + NUM_LANES * DWORD_TOTAL_BYTES;
    }
}

fn InitializePage(destination : u32, outputSize : u32, laneIx : u32) {
    var i : u32 = laneIx;
    let nw = (outputSize + 3u) / 4u;
    loop {
        if (i >= nw) { break; }
        OutputStoreU32(destination + i * 4u, 0u);
        i = i + NUM_LANES;
    }
}

//------------------------------------------------------------------------------
// Preconditioned decoder path (BC1/BC7/etc.). HLSL 792..1050, 1434..1588,
// 1690..1744, 1792..1868.
// ConditionerParams is held in a plain struct; the per-subblock / per-mip
// tables live in the existing groupshared arrays declared near the top of
// this file (gSizes_subblocks, gOffsets_subblocks, gOffsets_substreams,
// gColor_subblocks, gMip_widths/heights/pitches, gOffsets_mipbytes/
// mipblocks). ConditionerParams_Init must be called by all 32 lanes.
//------------------------------------------------------------------------------

struct ConditionerParams {
    isPreconditioned     : u32,
    isSwizzled           : u32,
    isPitch_D3D12_aligned: u32,
    format               : u32,
    width                : u32,
    height               : u32,
    pitch                : u32,
    num_mips             : u32,
    streamoff            : u32,
    blocksizebytes       : u32,
    num_subblocks        : u32,
    num_colorsubblocks   : u32,
    num_blocks           : u32,
};

// HLSL:832 ConditionerParams.Init
fn ConditionerParams_Init(dc : ptr<function, ConditionerParams>, laneIx : u32) {
    if ((*dc).isPreconditioned == 0u) { return; }

    // Zero all groupshared scratch tables. HLSL:837
    var j : u32 = laneIx;
    loop {
        if (j >= BROTLIG_MAX_NUM_MIP_LEVELS) { break; }
        gSizes_subblocks[j] = 0u;
        gOffsets_subblocks[j] = 0u;
        gOffsets_substreams[j] = 0u;
        gColor_subblocks[j] = 0u;
        gMip_widths[j] = 0u;
        gMip_heights[j] = 0u;
        gMip_pitches[j] = 0u;
        gOffsets_mipbytes[j] = 0u;
        gOffsets_mipblocks[j] = 0u;
        j = j + NUM_LANES;
    }

    // Format table. HLSL:851
    var bsz : u32 = 1u;
    var nsub : u32 = 1u;
    var ncol : u32 = 0u;
    var sz : u32 = 0u;
    var col : u32 = 0u;
    switch ((*dc).format) {
        case 1u: {
            bsz = 8u; nsub = 3u; ncol = 2u;
            if (laneIx == 0u) { sz = 2u; }
            else if (laneIx == 1u) { sz = 2u; }
            else if (laneIx == 2u) { sz = 4u; }
            if (laneIx == 0u) { col = 0u; }
            else if (laneIx == 1u) { col = 1u; }
        }
        case 2u: {
            bsz = 16u; nsub = 4u; ncol = 2u;
            if (laneIx == 0u) { sz = 8u; }
            else if (laneIx == 1u) { sz = 2u; }
            else if (laneIx == 2u) { sz = 2u; }
            else if (laneIx == 3u) { sz = 4u; }
            if (laneIx == 0u) { col = 1u; }
            else if (laneIx == 1u) { col = 2u; }
        }
        case 3u: {
            bsz = 16u; nsub = 6u; ncol = 2u;
            if (laneIx == 0u) { sz = 1u; }
            else if (laneIx == 1u) { sz = 1u; }
            else if (laneIx == 2u) { sz = 6u; }
            else if (laneIx == 3u) { sz = 2u; }
            else if (laneIx == 4u) { sz = 2u; }
            else if (laneIx == 5u) { sz = 4u; }
            if (laneIx == 0u) { col = 3u; }
            else if (laneIx == 1u) { col = 4u; }
        }
        case 4u: {
            bsz = 8u; nsub = 3u; ncol = 2u;
            if (laneIx == 0u) { sz = 1u; }
            else if (laneIx == 1u) { sz = 1u; }
            else if (laneIx == 2u) { sz = 6u; }
            if (laneIx == 0u) { col = 0u; }
            else if (laneIx == 1u) { col = 1u; }
        }
        case 5u: {
            bsz = 16u; nsub = 6u; ncol = 4u;
            if (laneIx == 0u) { sz = 1u; }
            else if (laneIx == 1u) { sz = 1u; }
            else if (laneIx == 2u) { sz = 6u; }
            else if (laneIx == 3u) { sz = 1u; }
            else if (laneIx == 4u) { sz = 1u; }
            else if (laneIx == 5u) { sz = 6u; }
            if (laneIx == 0u) { col = 0u; }
            else if (laneIx == 1u) { col = 1u; }
            else if (laneIx == 2u) { col = 3u; }
            else if (laneIx == 3u) { col = 4u; }
        }
        default: {
            bsz = 1u; nsub = 1u; ncol = 0u;
            if (laneIx == 0u) { sz = 1u; }
        }
    }
    (*dc).blocksizebytes = bsz;
    (*dc).num_subblocks = nsub;
    (*dc).num_colorsubblocks = ncol;
    gSizes_subblocks[laneIx] = sz;
    gColor_subblocks[laneIx] = col;
    // barrier avoided: intra-function cross-lane reads below use subgroupShuffle instead

    // HLSL:897 mip dimensions
    let nmips = (*dc).num_mips;
    let w = (*dc).width;
    let h = (*dc).height;
    let p = (*dc).pitch;
    var mw : u32 = 0u;
    var mh : u32 = 0u;
    if (laneIx == 0u) {
        mw = w;
        mh = h;
    } else if (laneIx < nmips) {
        mw = (((w * 4u) / (2u << (laneIx - 1u))) + 3u) / 4u;
        mh = (((h * 4u) / (2u << (laneIx - 1u))) + 3u) / 4u;
    }
    gMip_widths[laneIx] = mw;
    gMip_heights[laneIx] = mh;

    var mp : u32 = 0u;
    if (laneIx == 0u) {
        mp = p;
    } else if (laneIx < nmips) {
        if ((*dc).isPitch_D3D12_aligned != 0u) {
            mp = DivRoundUp(mw * bsz, 256u) * 256u;
        } else {
            mp = mw * bsz;
        }
    }
    gMip_pitches[laneIx] = mp;

    gOffsets_mipbytes[laneIx] = 0u;
    gOffsets_mipblocks[laneIx] = 0u;
    // barrier avoided: intra-function cross-lane reads below use subgroupShuffle instead

    // HLSL:904 mip offsets prefix-sum (serial in HLSL; keep structure).
    // Use subgroupShuffle to read per-lane mp,mh,mw instead of workgroup memory,
    // because cross-lane workgroup reads are not reliably visible even within a
    // single subgroup (BC3 regression).
    var mbytes : u32 = 0u;
    var mblocks : u32 = 0u;
    for (var m : u32 = 0u; m <= nmips; m = m + 1u) {
        let mp_m = subgroupShuffle(mp, m);
        let mh_m = subgroupShuffle(mh, m);
        let mw_m = subgroupShuffle(mw, m);
        if (m < laneIx) {
            mbytes = mbytes + (mp_m * mh_m);
            mblocks = mblocks + (mw_m * mh_m);
        }
    }
    gOffsets_mipbytes[laneIx] = mbytes;
    gOffsets_mipblocks[laneIx] = mblocks;

    (*dc).num_blocks = subgroupShuffle(mblocks, nmips);

    // HLSL:915 subblock offsets prefix-sum. Use subgroupShuffle for cross-lane
    // read of per-lane sz instead of gSizes_subblocks (see BC3 regression).
    var sboff : u32 = 0u;
    var ssoff : u32 = 0u;
    let nblk = (*dc).num_blocks;
    for (var s : u32 = 0u; s <= nsub; s = s + 1u) {
        let sbsize = subgroupShuffle(sz, s);
        if (s < laneIx) {
            sboff = sboff + sbsize;
            ssoff = ssoff + sbsize * nblk;
        }
    }
    gOffsets_subblocks[laneIx] = sboff;
    gOffsets_substreams[laneIx] = ssoff;
}

// HLSL:926 ConditionerParams.GetSub
fn CP_GetSub(ptr_ : u32, num_subblocks : u32) -> u32 {
    var sub : u32 = 0u;
    for (var i : u32 = 0u; i < num_subblocks; i = i + 1u) {
        let off = gOffsets_substreams[i];
        if (ptr_ >= off) { sub = i; }
    }
    return sub;
}

// HLSL:938 ConditionerParams.GetMip
fn CP_GetMip(ptr_ : u32, sbsize : u32, num_mips : u32) -> u32 {
    var mip : u32 = 0u;
    for (var i : u32 = 0u; i < num_mips; i = i + 1u) {
        let moff = gOffsets_mipblocks[i];
        if (ptr_ >= moff * sbsize) { mip = i; }
    }
    return mip;
}

// HLSL:978 DeconditionPtr
fn DeconditionPtr(addrIn : u32, dc : ConditionerParams) -> u32 {
    var addr : u32 = addrIn - dc.streamoff;
    let sub = CP_GetSub(addr, dc.num_subblocks);
    let soffset = gOffsets_substreams[sub];
    let sbsize  = gSizes_subblocks[sub];
    let sboffset= gOffsets_subblocks[sub];

    var offsetAddr : u32 = addr - soffset;

    let mip = CP_GetMip(offsetAddr, sbsize, dc.num_mips);
    let moffset_block = gOffsets_mipblocks[mip];
    let mip_pos       = gOffsets_mipbytes[mip];
    let mip_width     = gMip_widths[mip];
    let mip_height    = gMip_heights[mip];
    let mip_pitch     = gMip_pitches[mip];

    offsetAddr = offsetAddr - (moffset_block * sbsize);

    let swRegion = BROTLIG_PRECON_SWIZZLE_REGION_SIZE;
    let isMipSwizzled = (dc.isSwizzled != 0u) && (mip_width >= swRegion) && (mip_height >= swRegion);
    let rem_width  = mip_width % swRegion;
    let rem_height = mip_height % swRegion;
    let eff_width  = mip_width - rem_width;
    let eff_height = mip_height - rem_height;

    let block = offsetAddr / sbsize;
    var row : u32 = block / mip_width;
    var col : u32 = block % mip_width;

    if (isMipSwizzled && (row < eff_height) && (col < eff_width)) {
        let eff_block = block - (row * rem_width);
        let blockGrp_width = eff_width / swRegion;

        let blockGrp   = eff_block / (swRegion * swRegion);
        let blockInGrp = eff_block % (swRegion * swRegion);

        let oblockGrpRow = blockGrp / blockGrp_width;
        let oblockGrpCol = blockGrp % blockGrp_width;

        let oblockRowInGrp = blockInGrp / swRegion;
        let oblockColInGrp = blockInGrp % swRegion;

        row = swRegion * oblockGrpRow + oblockRowInGrp;
        col = swRegion * oblockGrpCol + oblockColInGrp;
    }

    let block_pos = (row * mip_pitch) + (col * dc.blocksizebytes);
    let byte_pos  = offsetAddr % sbsize;

    return dc.streamoff + mip_pos + block_pos + sboffset + byte_pos;
}

// HLSL:1033 DeconByteLoad / Store / Update
fn DeconByteLoad(addr : u32, dc : ConditionerParams) -> u32 {
    return OutputByteLoad(DeconditionPtr(addr, dc));
}
fn DeconByteStore(addr : u32, data : u32, dc : ConditionerParams) {
    ByteStore(DeconditionPtr(addr, dc), data);
}
fn DeconByteUpdate(addr : u32, data : u32, dc : ConditionerParams) {
    ByteUpdate(DeconditionPtr(addr, dc), data);
}

// HLSL:950 ConditionerParams.HasColor
struct HasColorOut {
    hasColor : bool,
    sub_start: u32,
    sub_end  : u32,
};
fn CP_HasColor(dc : ConditionerParams, start : u32, end : u32) -> HasColorOut {
    var out_ : HasColorOut;
    out_.hasColor = false;
    out_.sub_start = 0u;
    out_.sub_end = 0u;
    for (var i : u32 = 0u; i < dc.num_colorsubblocks; i = i + 1u) {
        let sub = gColor_subblocks[i];
        let color_start = gOffsets_substreams[sub];
        let color_end   = gOffsets_substreams[sub + 1u];
        // OVERLAP(a1,a2,b1,b2) = a1<b2 && b1<a2
        if ((color_start < end) && (start < color_end)) {
            var ss : u32 = 0u;
            if (color_start > start) { ss = color_start - start; }
            ss = ss + start;
            var se : u32;
            if (color_end < end) { se = color_end - start; } else { se = end - start; }
            se = se + start;
            out_.sub_start = ss;
            out_.sub_end = se;
            out_.hasColor = true;
            return out_;
        }
    }
    return out_;
}

// HLSL:1461 DeltaDecode
fn DeltaDecode(destination : u32, dc : ConditionerParams, outlimit : u32, laneIx : u32) {
    var page_start : u32 = destination - dc.streamoff;
    let page_end : u32 = outlimit - dc.streamoff;

    loop {
        if (page_start >= page_end) { break; }
        let hc = CP_HasColor(dc, page_start, page_end);
        if (!hc.hasColor) {
            page_start = page_end;
            continue;
        }
        let sub_start = hc.sub_start;
        let sub_end   = hc.sub_end;

        var ref_ : u32 = 0u;
        if (laneIx == 0u) {
            ref_ = DeconByteLoad(sub_start, dc);
        }
        ref_ = subgroupBroadcastFirst(ref_);

        var i : u32 = sub_start + 1u + laneIx;
        loop {
            if (i >= sub_end) { break; }
            let delta = DeconByteLoad(i + dc.streamoff, dc);
            let deltasum = subgroupExclusiveAdd(delta);
            var byte : u32 = delta + deltasum + ref_;
            byte = byte & MaskLsbs(BYTE_TOTAL_BITS);
            DeconByteUpdate(i + dc.streamoff, byte, dc);
            ref_ = subgroupBroadcast(byte, 31u);
            i = i + NUM_LANES;
        }
        page_start = sub_end;
    }
}

// HLSL:1434 DeconditionLiterals (mirrors WriteLiterals but writes through
// DeconditionPtr).
fn DeconditionLiterals(bs : ptr<function, DecoderState>, wptr : u32, numLitIn : u32,
                       offsetIn : u32, dc : ConditionerParams, laneIx : u32) {
    var numLit = numLitIn;
    var offset = offsetIn;

    var literalsKept : u32 = atomicLoad(&gLiteralsKept);
    var endOffset : u32 = (NUM_LANES - literalsKept) % NUM_LANES;
    var startIx : u32 = subgroupExclusiveAdd(numLit) + endOffset;

    loop {
        if (!subgroupAny(numLit != 0u)) { break; }
        if (literalsKept == 0u) {
            let lit = decodeliteral(bs, true, laneIx);
            gLiteralBuffer[laneIx] = lit;
        }
        let sp = SpreadLiterals(startIx, numLit, offset, laneIx);
        numLit = sp.numLit;
        offset = sp.offset;
        startIx = sp.sum;
        let written = sp.written;
        let writeoff = sp.writeoff;

        let mask = (laneIx >= endOffset) && (laneIx < written);
        endOffset = written % NUM_LANES;
        literalsKept = subgroupBroadcastFirst((NUM_LANES - endOffset) % NUM_LANES);

        if (mask) {
            DeconByteStore(wptr + writeoff, gLiteralBuffer[laneIx], dc);
        }
    }
    if (laneIx == 0u) { atomicStore(&gLiteralsKept, literalsKept); }
}

// HLSL:1498 DecompressAndDecondition
fn DecompressAndDecondition(bs : ptr<function, DecoderState>, wptrIn : u32,
                            dParams : DecoderParams, dc : ConditionerParams,
                            outlimit : u32, laneIx : u32,
                            copyLC : u32, insertLC : u32) -> u32 {
    var wptr = wptrIn;
    let oWptr = wptrIn;

    InitDistRingBuffer(laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    var neos : bool = true;
    var cmd = decodecommand(bs, laneIx, neos, copyLC, insertLC);
    var length = cmd.copy_len + cmd.insert_len;
    var offset : u32 = 0u;
    neos = length > 0u;

    var literalsRead : u32 = 0u;

    loop {
        if (!subgroupAll(wptr < outlimit)) { break; }

        offset = subgroupExclusiveAdd(length);
        let numlits = cmd.insert_len;
        var writeptr = wptr + offset;

        let literalsTotal = subgroupAdd(numlits);
        literalsRead = (literalsRead - literalsTotal) & (NUM_LANES - 1u);

        let decode_en = (cmd.icp_code >= 128u) && neos && (cmd.icp_code < BROTLIG_EOS_COMMAND_SYMBOL);

        cmd.copy_dist = decodedistance(bs, laneIx, decode_en);
        cmd.copy_dist = ResolveDistances(cmd.copy_dist, bs, dParams, laneIx, neos);

        DeconditionLiterals(bs, wptr, numlits, offset, dc, laneIx);

        offset = offset + numlits;
        writeptr = writeptr + numlits;

        wptr = wptr + subgroupBroadcast(offset + cmd.copy_len, 31u);

        if (writeptr >= outlimit) {
            cmd.copy_len = 0u;
        }

        let mask = subgroupBallot(neos).x;
        for (var i : u32 = 0u; i < NUM_LANES; i = i + 1u) {
            if ((mask & (1u << i)) == 0u) { continue; }
            let cOffset = subgroupShuffle(cmd.copy_dist, i);
            let cLength = subgroupShuffle(cmd.copy_len, i);
            let cOutptr = subgroupShuffle(writeptr, i);
            if (cOffset == 0u || cLength == 0u) { continue; }
            let source = cOutptr - cOffset;
            let tgt = cOutptr;
            var j : u32 = laneIx;
            loop {
                if (j >= cLength) { break; }
                let data = DeconByteLoad(source + (j % cOffset), dc);
                DeconByteStore(j + tgt, data, dc);
                j = j + NUM_LANES;
            }
        }

        if (subgroupAny(length == 0u)) { break; }

        cmd = decodecommand(bs, laneIx, neos, copyLC, insertLC);
        length = cmd.copy_len + cmd.insert_len;
        neos = length > 0u;
    }

    if (dParams.isDeltaEncoded != 0u) {
        DeltaDecode(oWptr, dc, outlimit, laneIx);
    }

    return wptr;
}

// HLSL:1630 UncompressedDecondition
fn UncompressedDecondition(source : u32, destination : u32, inputSize : u32,
                           dc : ConditionerParams, laneIx : u32) {
    var i : u32 = laneIx;
    loop {
        if (i >= inputSize) { break; }
        let data = InputByteLoad(source + i);
        DeconByteStore(destination + i, data, dc);
        i = i + NUM_LANES;
    }
}

// HLSL:1690 InitializePage (preconditioned variant, byte-at-a-time via
// DeconByteStore).
fn InitializePage_precon(destination : u32, outputSize : u32, dc : ConditionerParams, laneIx : u32) {
    var i : u32 = laneIx;
    loop {
        if (i >= outputSize) { break; }
        DeconByteStore(destination + i, 0u, dc);
        i = i + NUM_LANES;
    }
}

// HLSL:1705 Process (preconditioned branch).
fn Process_preconditioned(source : u32, destination : u32, inputSize : u32,
                          outputSize : u32, dc : ConditionerParams, laneIx : u32) {
    InitializePage_precon(destination, outputSize, dc, laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    if (inputSize == outputSize) {
        UncompressedDecondition(source, destination, inputSize, dc, laneIx);
        return;
    }

    var i : u32 = laneIx;
    loop {
        if (i >= DICTIONARY_COMPACT_TOTAL_SIZE) { break; }
        atomicStore(&gDictionary[i], 0u);
        i = i + NUM_LANES;
    }
    if (laneIx == 0u) { atomicStore(&gLiteralsKept, 0u); }
    /* workgroupBarrier removed: single-subgroup WG */

    let copyLC   = LengthCode24_Init(2u, 0x3eaa80u, 0x000000u, laneIx);
    let insertLC = LengthCode24_Init(0u, 0x3eaa80u, 0x310020u, laneIx);

    var bs : DecoderState;
    BS_Init(&bs, source, laneIx);

    var dparams : DecoderParams;
    dparams.npostfix = BS_GetAndDropBits(&bs, 2u, true);
    dparams.n_direct = BS_GetAndDropBits(&bs, 4u, true);
    dparams.isDeltaEncoded = BS_GetAndDropBits(&bs, 1u, true);
    BS_DropBits(&bs, 1u, true);

    dparams.n_direct = dparams.n_direct << dparams.npostfix;

    dparams.npostfix = subgroupBroadcastFirst(dparams.npostfix);
    dparams.n_direct = subgroupBroadcastFirst(dparams.n_direct);
    dparams.isDeltaEncoded = subgroupBroadcastFirst(dparams.isDeltaEncoded);

    DeserializeCompact(&bs, inputSize, laneIx);
    ReadHuffmanCode(&bs);

    _ = DecompressAndDecondition(&bs, destination, dparams, dc,
                                 destination + outputSize, laneIx, copyLC, insertLC);
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Process a single page (generic only). HLSL 1705..1745
//------------------------------------------------------------------------------
fn Process_generic(source : u32, destination : u32, inputSize : u32, outputSize : u32, laneIx : u32) {
    InitializePage(destination, outputSize, laneIx);
    /* workgroupBarrier removed: single-subgroup WG */

    // Header says uncompressed? HLSL: if (Uncompressed(...)) return;
    // We approximate by comparing sizes: HLSL Uncompressed() only runs when
    // inputSize == outputSize.
    if (inputSize == outputSize) {
        UncompressedMemCopy(source, destination, inputSize, laneIx);
        return;
    }

    var i : u32 = laneIx;
    loop {
        if (i >= DICTIONARY_COMPACT_TOTAL_SIZE) { break; }
        atomicStore(&gDictionary[i], 0u);
        i = i + NUM_LANES;
    }
    if (laneIx == 0u) { atomicStore(&gLiteralsKept, 0u); }
    /* workgroupBarrier removed: single-subgroup WG */

    let copyLC   = LengthCode24_Init(2u, 0x3eaa80u, 0x000000u, laneIx);
    let insertLC = LengthCode24_Init(0u, 0x3eaa80u, 0x310020u, laneIx);

    var bs : DecoderState;
    BS_Init(&bs, source, laneIx);

    var dparams : DecoderParams;
    dparams.npostfix = BS_GetAndDropBits(&bs, 2u, true);
    dparams.n_direct = BS_GetAndDropBits(&bs, 4u, true);
    dparams.isDeltaEncoded = BS_GetAndDropBits(&bs, 1u, true);
    BS_DropBits(&bs, 1u, true);

    dparams.n_direct = dparams.n_direct << dparams.npostfix;

    dparams.npostfix = subgroupBroadcastFirst(dparams.npostfix);
    dparams.n_direct = subgroupBroadcastFirst(dparams.n_direct);
    dparams.isDeltaEncoded = subgroupBroadcastFirst(dparams.isDeltaEncoded);

    DeserializeCompact(&bs, inputSize, laneIx);
    ReadHuffmanCode(&bs);

    var dest = destination;
    dest = Decompress(&bs, dest, dparams, destination + outputSize, laneIx, copyLC, insertLC);
}

//------------------------------------------------------------------------------
// Streaming: spill / restore groupshared tables + bit reader state
//------------------------------------------------------------------------------
fn stateWordIndex(streamIndex : u32, offset : u32) -> u32 {
    return (streamIndex - 1u) * STATE_STRIDE_WORDS + offset;
}

fn spillState(streamIndex : u32, bs : DecoderState, pageCursor : u32, dp : DecoderParams, laneIx : u32) {
    let base = (streamIndex - 1u) * STATE_STRIDE_WORDS;
    if (laneIx == 0u) {
        atomicStore(&state_buf[base + STATE_OFF_HOLD_LO], bs.holdLo);
        atomicStore(&state_buf[base + STATE_OFF_HOLD_HI], bs.holdHi);
        atomicStore(&state_buf[base + STATE_OFF_VALID_BITS], bs.validBits);
        atomicStore(&state_buf[base + STATE_OFF_READPTR], bs.readPointer);
        atomicStore(&state_buf[base + STATE_OFF_PAGECUR], pageCursor);
        atomicStore(&state_buf[base + STATE_OFF_NPOSTFIX], dp.npostfix);
        atomicStore(&state_buf[base + STATE_OFF_NDIRECT], dp.n_direct);
        atomicStore(&state_buf[base + STATE_OFF_ISDELTA], dp.isDeltaEncoded);
    }
    // gDictionary spill
    var i : u32 = laneIx;
    loop {
        if (i >= DICTIONARY_COMPACT_TOTAL_SIZE) { break; }
        atomicStore(&state_buf[base + STATE_OFF_DICT + i], atomicLoad(&gDictionary[i]));
        i = i + NUM_LANES;
    }
    // gSymbolLengths spill
    i = laneIx;
    loop {
        if (i >= SYMBOL_LENGTHS_DWORDS) { break; }
        atomicStore(&state_buf[base + STATE_OFF_SYMLEN + i], atomicLoad(&gSymbolLengths[i]));
        i = i + NUM_LANES;
    }
    // SearchTable spills: exactly NUM_LANES lanes.
    atomicStore(&state_buf[base + STATE_OFF_SEARCH0 + laneIx], gSearchTable0[laneIx]);
    atomicStore(&state_buf[base + STATE_OFF_SEARCH1 + laneIx], gSearchTable1[laneIx]);
    /* workgroupBarrier removed: single-subgroup WG */
}

fn restoreState(streamIndex : u32, bs : ptr<function, DecoderState>, dp : ptr<function, DecoderParams>, laneIx : u32) -> u32 {
    let base = (streamIndex - 1u) * STATE_STRIDE_WORDS;
    (*bs).holdLo = atomicLoad(&state_buf[base + STATE_OFF_HOLD_LO]);
    (*bs).holdHi = atomicLoad(&state_buf[base + STATE_OFF_HOLD_HI]);
    (*bs).validBits = atomicLoad(&state_buf[base + STATE_OFF_VALID_BITS]);
    (*bs).readPointer = atomicLoad(&state_buf[base + STATE_OFF_READPTR]);
    (*bs).lane = laneIx;
    let pageCursor = atomicLoad(&state_buf[base + STATE_OFF_PAGECUR]);
    (*dp).npostfix = atomicLoad(&state_buf[base + STATE_OFF_NPOSTFIX]);
    (*dp).n_direct = atomicLoad(&state_buf[base + STATE_OFF_NDIRECT]);
    (*dp).isDeltaEncoded = atomicLoad(&state_buf[base + STATE_OFF_ISDELTA]);

    var i : u32 = laneIx;
    loop {
        if (i >= DICTIONARY_COMPACT_TOTAL_SIZE) { break; }
        atomicStore(&gDictionary[i], atomicLoad(&state_buf[base + STATE_OFF_DICT + i]));
        i = i + NUM_LANES;
    }
    i = laneIx;
    loop {
        if (i >= SYMBOL_LENGTHS_DWORDS) { break; }
        atomicStore(&gSymbolLengths[i], atomicLoad(&state_buf[base + STATE_OFF_SYMLEN + i]));
        i = i + NUM_LANES;
    }
    gSearchTable0[laneIx] = atomicLoad(&state_buf[base + STATE_OFF_SEARCH0 + laneIx]);
    gSearchTable1[laneIx] = atomicLoad(&state_buf[base + STATE_OFF_SEARCH1 + laneIx]);
    /* workgroupBarrier removed: single-subgroup WG */
    return pageCursor;
}

//------------------------------------------------------------------------------
// CSMain  (HLSL 1753..1881)
//------------------------------------------------------------------------------
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane : u32) {
    let laneIx = sg_lane;

    // Initialise literals kept per workgroup.
    if (laneIx == 0u) { atomicStore(&gLiteralsKept, 0u); }
    /* workgroupBarrier removed: single-subgroup WG */

    var streamIndex : u32 = 0u;
    if (laneIx == 0u) {
        streamIndex = atomicLoad(\&metaBuf[0]);
    }
    streamIndex = subgroupBroadcastFirst(streamIndex);

    loop {
        if (streamIndex == 0u) { break; }

        // Per-stream base in meta table (layout: [idx, pageLimit, resumeFlag, rsv,
        // then per-stream quads [rptr, wptr, pageCursor, savedStateOff]]).
        let perStreamBase : u32 = 4u + (streamIndex - 1u) * 4u;

        // Load rptr, wptr via lanes 0,1.
        var streamRptr : u32 = 0u;
        var streamWptr : u32 = 0u;
        if (laneIx == 0u) {
            streamRptr = atomicLoad(\&metaBuf[perStreamBase + 0u]);
        }
        if (laneIx == 1u) {
            streamWptr = atomicLoad(\&metaBuf[perStreamBase + 1u]);
        }
        streamRptr = subgroupBroadcast(streamRptr, 0u);
        streamWptr = subgroupBroadcast(streamWptr, 1u);

        // HLSL parses readControlWord1 + readControlWord2 + lastPageRsize
        // from the first three input dwords at stream.rptr. Use separate lane
        // reads and broadcast.
        var ctrlLoad : u32 = 0u;
        if (laneIx == 0u) { ctrlLoad = loadU32Aligned(streamRptr + 0u); }
        else if (laneIx == 1u) { ctrlLoad = loadU32Aligned(streamRptr + 4u); }
        else if (laneIx == 2u) { ctrlLoad = loadU32Aligned(streamRptr + 8u); }
        let readControlWord1 = subgroupBroadcast(ctrlLoad, 0u);
        let readControlWord2 = subgroupBroadcast(ctrlLoad, 1u);
        var lastPageRsize    = subgroupBroadcast(ctrlLoad, 2u);

        let numPages         = BitFieldBit(readControlWord1, 16u, 31u);
        let isPreconditioned = BitFieldBit(readControlWord2, 20u, 20u);
        let pageSize         = BROTLIG_WORK_PAGE_SIZE_UNIT << BitFieldBit(readControlWord2, 0u, 1u);
        let lastPageSize     = BitFieldBit(readControlWord2, 2u, 19u);

        // HLSL:1792 precondition header parse. Three dwords immediately after
        // the stream header: eControlWord1, eControlWord2, lastPageRsize
        // (override). Only read when isPreconditioned.
        var dcparams : ConditionerParams;
        dcparams.isPreconditioned = 0u;
        dcparams.isSwizzled = 0u;
        dcparams.isPitch_D3D12_aligned = 0u;
        dcparams.format = 0u;
        dcparams.width = 0u;
        dcparams.height = 0u;
        dcparams.pitch = 0u;
        dcparams.num_mips = 0u;
        dcparams.streamoff = 0u;
        dcparams.blocksizebytes = 0u;
        dcparams.num_subblocks = 0u;
        dcparams.num_colorsubblocks = 0u;
        dcparams.num_blocks = 0u;

        if (isPreconditioned != 0u) {
            var pLoad : u32 = 0u;
            let pBase = streamRptr + BROTLIG_WORK_STREAM_HEADER_SIZE;
            if (laneIx == 0u) { pLoad = loadU32Aligned(pBase + 0u); }
            else if (laneIx == 1u) { pLoad = loadU32Aligned(pBase + 4u); }
            else if (laneIx == 2u) { pLoad = loadU32Aligned(pBase + 8u); }
            let eCW1 = subgroupBroadcast(pLoad, 0u);
            let eCW2 = subgroupBroadcast(pLoad, 1u);
            lastPageRsize = subgroupBroadcast(pLoad, 2u);

            dcparams.isPreconditioned       = 1u;
            dcparams.isSwizzled             = BitFieldBit(eCW1, 0u, 0u);
            dcparams.isPitch_D3D12_aligned  = BitFieldBit(eCW1, 1u, 1u);
            dcparams.width                  = BitFieldBit(eCW1, 2u, 16u) + 1u;
            dcparams.height                 = BitFieldBit(eCW1, 17u, 31u) + 1u;
            dcparams.format                 = BitFieldBit(eCW2, 0u, 7u);
            dcparams.num_mips               = BitFieldBit(eCW2, 8u, 12u) + 1u;
            dcparams.pitch                  = BitFieldBit(eCW2, 13u, 31u) + 1u;
            dcparams.streamoff              = streamWptr;

            // Initialise groupshared conditioner tables once per stream.
            ConditionerParams_Init(&dcparams, laneIx);
            /* workgroupBarrier removed: single-subgroup WG */
        }

        // Streaming: read pageLimit + resume flag
        var pageLimit : u32 = 0u;
        var resumeFlag : u32 = 0u;
        if (laneIx == 0u) {
            pageLimit = atomicLoad(\&metaBuf[1]);
            resumeFlag = atomicLoad(\&metaBuf[2]);
        }
        pageLimit = subgroupBroadcastFirst(pageLimit);
        resumeFlag = subgroupBroadcastFirst(resumeFlag);

        let resumeBit = 1u << ((streamIndex - 1u) & 31u);
        let isResume = (resumeFlag & resumeBit) != 0u;

        // If resuming, restore tables + bit reader. Otherwise parse header
        // normally in Process_generic on the first page encounter.
        var bsSaved : DecoderState;
        var dpSaved : DecoderParams;
        var startPage : u32 = 0u;
        if (isResume) {
            startPage = restoreState(streamIndex, &bsSaved, &dpSaved, laneIx);
            // Seed literal bookkeeping.
            if (laneIx == 0u) { atomicStore(&gLiteralsKept, 0u); }
            /* workgroupBarrier removed: single-subgroup WG */
        }

        let pageDesc = BROTLIG_WORK_STREAM_HEADER_SIZE + streamRptr
                     + select(0u, BROTLIG_WORK_STREAM_PRECON_HEADER_SIZE, isPreconditioned != 0u);

        var finishedAllPages : bool = false;
        var suspended : bool = false;
        var currentPage : u32 = startPage;

        loop {
            var page_index : u32 = 0u;
            if (laneIx == 0u) {
                // atomicAdd counter meta[perStreamBase + 2] == pageCursor.
                page_index = atomicAdd(\&metaBuf[perStreamBase + 2u], 1u);
            }
            page_index = subgroupBroadcastFirst(page_index);

            if (page_index >= numPages) {
                finishedAllPages = true;
                break;
            }
            // Streaming break: if we've hit the dispatch's page budget, stop
            // and spill state. Note: we keep the incremented pageCursor so
            // the next dispatch will pick up at page_index.
            if (page_index >= pageLimit) {
                suspended = true;
                // Rewind meta cursor so next dispatch re-reads this same index.
                if (laneIx == 0u) {
                    atomicStore(\&metaBuf[perStreamBase + 2u], page_index);
                }
                break;
            }

            // Decode one page.
            var readSizeIndex : u32 = 0u;
            if (laneIx == 0u) {
                readSizeIndex = loadU32Aligned(pageDesc + page_index * 4u + 0u);
            } else if (laneIx == 1u) {
                readSizeIndex = loadU32Aligned(pageDesc + page_index * 4u + 4u);
            }
            let rs0 = subgroupBroadcast(readSizeIndex, 0u);
            let rs1 = subgroupBroadcast(readSizeIndex, 1u);

            var pageRptr : u32 = select(0u, rs0, page_index > 0u);
            var pageRsize : u32 = select(rs1 - pageRptr, lastPageRsize, page_index == numPages - 1u);
            let pageWptr = streamWptr + page_index * pageSize;
            let pageWsize = select(pageSize, lastPageSize, (page_index >= numPages - 1u) && (lastPageSize != 0u));
            pageRptr = pageRptr + pageDesc + numPages * 4u;

            if (isPreconditioned != 0u) {
                Process_preconditioned(pageRptr, pageWptr, pageRsize, pageWsize, dcparams, laneIx);
            } else {
                Process_generic(pageRptr, pageWptr, pageRsize, pageWsize, laneIx);
            }
            currentPage = page_index + 1u;
        }

        if (suspended) {
            // Spill: we have no in-flight bit reader at this boundary (pages
            // always reset bs at start), so spilling is mostly a no-op beyond
            // persisting pageCursor. Nevertheless, we set the resume flag.
            var bsBlank : DecoderState;
            bsBlank.holdLo = 0u; bsBlank.holdHi = 0u; bsBlank.validBits = 0u;
            bsBlank.readPointer = 0u; bsBlank.lane = laneIx;
            var dpBlank : DecoderParams;
            dpBlank.npostfix = 0u; dpBlank.n_direct = 0u; dpBlank.isDeltaEncoded = 0u;
            spillState(streamIndex, bsBlank, currentPage, dpBlank, laneIx);
            if (laneIx == 0u) {
                atomicOr(\&metaBuf[2], resumeBit);
            }
            // Do NOT decrement streamIndex; next dispatch resumes it.
            break;
        }

        if (finishedAllPages) {
            // Clear resume flag bit and advance.
            if (laneIx == 0u) {
                atomicAnd(\&metaBuf[2], ~resumeBit);
                let prev = atomicCompareExchangeWeak(\&metaBuf[0], streamIndex, streamIndex - 1u);
                streamIndex = select(prev.old_value, streamIndex - 1u, prev.exchanged);
            }
            streamIndex = subgroupBroadcastFirst(streamIndex);
        } else {
            streamIndex = subgroupBroadcastFirst(streamIndex);
        }
    }

    _ = lid;
}
`;
