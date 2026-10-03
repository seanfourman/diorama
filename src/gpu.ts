export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<Gpu> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not available in this browser. Try a current Chrome or Edge.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Check that hardware acceleration is enabled.');
  }

  // Scenes run to millions of Gaussians, so ask for the largest buffers this GPU
  // supports instead of WebGPU's conservative defaults.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => {
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
  });
  device.onuncapturederror = (event) => {
    console.error('WebGPU error:', event.error.message);
  };

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Could not create a WebGPU canvas context.');
  }
  // The tile rasterizer writes pixels from a compute shader, so the canvas needs
  // storage usage. rgba8unorm allows that; the usual bgra8unorm needs an extra feature.
  const format: GPUTextureFormat = 'rgba8unorm';
  context.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    alphaMode: 'opaque',
  });

  return { adapter, device, context, format };
}

/** Matches the canvas's pixel size to its on-screen size so the image stays sharp. */
export function resizeToDisplay(canvas: HTMLCanvasElement, maxSize: number): void {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.min(maxSize, Math.max(1, Math.floor(canvas.clientWidth * dpr)));
  const height = Math.min(maxSize, Math.max(1, Math.floor(canvas.clientHeight * dpr)));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}
