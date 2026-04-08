// Device acquisition for the Brotli-G WebGPU decoder.
//
// Works in both browsers (via `navigator.gpu`) and Node 22+ (via the
// `webgpu` npm package which exposes a Dawn-backed `navigator.gpu`).

const SUBGROUPS_FEATURE = "subgroups" as GPUFeatureName;

async function getGPU(): Promise<GPU> {
  const nav =
    typeof navigator !== "undefined"
      ? (navigator as Navigator & { gpu?: GPU })
      : undefined;
  if (nav?.gpu) return nav.gpu;

  // Node fallback: dynamic import so browser bundlers don't choke.
  try {
    // `webgpu` is an optional peer (Node only); use a computed specifier
    // so TypeScript doesn't require its type declarations at build time.
    const specifier = "webgpu";
    const mod = (await import(/* @vite-ignore */ specifier)) as {
      create?: (args?: string[]) => GPU;
      globals?: Record<string, unknown>;
      default?: { create?: (args?: string[]) => GPU; globals?: Record<string, unknown> };
    };
    const create = mod.create ?? mod.default?.create;
    const globals = mod.globals ?? mod.default?.globals;
    if (!create) {
      throw new Error("webgpu package loaded but no create() export found");
    }
    // Install GPU* constant classes (GPUShaderStage, GPUBufferUsage, GPUMapMode, ...)
    // on globalThis so code written against the browser API works unchanged.
    if (globals) {
      const g = globalThis as Record<string, unknown>;
      for (const [k, v] of Object.entries(globals)) {
        if (g[k] === undefined) g[k] = v;
      }
    }
    // The `webgpu` npm package's create() returns a GPU instance directly.
    return create([]);
  } catch (err) {
    throw new Error(
      `No WebGPU implementation available. In a browser ensure navigator.gpu exists; ` +
        `in Node install the \`webgpu\` package (Node 22+). Cause: ${String(err)}`,
    );
  }
}

// Cache the device. In Node with the `webgpu` Dawn addon, creating multiple
// GPU/adapter/device instances across test files destabilizes the native
// addon (worker exits). In browsers caching is also preferable.
let cachedDevice: Promise<GPUDevice> | null = null;

export function requestBrotligDevice(): Promise<GPUDevice> {
  if (cachedDevice) return cachedDevice;
  cachedDevice = (async () => {
    const gpu = await getGPU();
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("requestAdapter() returned null - no WebGPU adapter found");
    }
    if (!adapter.features.has(SUBGROUPS_FEATURE)) {
      throw new Error(
        "Adapter does not support the 'subgroups' feature, which is required " +
          "by the Brotli-G decoder (port of a 32-thread wavefront HLSL kernel).",
      );
    }
    return adapter.requestDevice({ requiredFeatures: [SUBGROUPS_FEATURE] });
  })().catch((e) => {
    cachedDevice = null;
    throw e;
  });
  return cachedDevice;
}

// Asserts the compute pipeline will run with subgroup size == 32.
//
// TODO: The WebGPU subgroup-size query surface is still in flux. Chrome
// exposes `GPUAdapterInfo.subgroupMinSize`/`subgroupMaxSize` behind the
// subgroups feature, and pipelines can be created with
// `requiredSubgroupSize`. Once the final API ships we should:
//   1. Prefer creating the pipeline with `requiredSubgroupSize: 32`
//      inside `compute: { ..., requiredSubgroupSize: 32 }`.
//   2. Fall back to reading `adapter.info.subgroupMinSize` /
//      `subgroupMaxSize` and refusing if 32 is out of range.
// For now we do a best-effort check against adapter info and otherwise
// accept and warn.
export async function assertSubgroupSize32(
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  entryPoint: string,
): Promise<void> {
  // Touch the arguments so the scaffold compiles cleanly even though the
  // real probing logic is not wired yet.
  void shaderModule;
  void entryPoint;

  const info = (device as GPUDevice & {
    adapterInfo?: { subgroupMinSize?: number; subgroupMaxSize?: number };
  }).adapterInfo;

  const min = info?.subgroupMinSize;
  const max = info?.subgroupMaxSize;
  if (typeof min === "number" && typeof max === "number") {
    if (min > 32 || max < 32) {
      throw new Error(
        `Brotli-G decoder requires subgroup size 32, adapter reports [${min}, ${max}]`,
      );
    }
    return;
  }

  // TODO: once requiredSubgroupSize is standardized, create a throwaway
  // pipeline here with requiredSubgroupSize:32 and rely on WebGPU to
  // reject mismatched adapters.
}
