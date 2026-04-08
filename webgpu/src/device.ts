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
// Finding 4: this is now load-bearing. The caller also requests
// `requiredSubgroupSize: 32` on the compute pipeline; this function
// additionally probes GPUAdapterInfo.subgroupMinSize/subgroupMaxSize where
// available and throws if 32 is outside the adapter's supported range.
// If neither field is queryable we trust the pipeline-side
// requiredSubgroupSize request and return.
export async function assertSubgroupSize32(
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  entryPoint: string,
): Promise<void> {
  void shaderModule;
  void entryPoint;

  const info = (device as GPUDevice & {
    adapterInfo?: { subgroupMinSize?: number; subgroupMaxSize?: number };
  }).adapterInfo;

  const min = info?.subgroupMinSize;
  const max = info?.subgroupMaxSize;
  if (typeof min === "number" && min > 32) {
    throw new Error(
      `Brotli-G decoder requires subgroup size 32, adapter reports subgroupMinSize=${min}`,
    );
  }
  if (typeof max === "number" && max < 32) {
    throw new Error(
      `Brotli-G decoder requires subgroup size 32, adapter reports subgroupMaxSize=${max}`,
    );
  }
}
