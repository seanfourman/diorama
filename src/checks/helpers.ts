// Shared plumbing for the GPU checks.

export type RGB = [number, number, number];

/**
 * Creates an rgba8unorm target (size × size, or [width, height]), lets `record`
 * draw into it, and returns its pixels.
 */
export async function renderPixels(
  device: GPUDevice,
  size: number | [number, number],
  record: (encoder: GPUCommandEncoder, target: GPUTextureView) => void,
): Promise<Uint8Array> {
  const [width, height] = typeof size === 'number' ? [size, size] : size;
  // Texture-to-buffer copies need each row padded to a multiple of 256 bytes.
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const texture = device.createTexture({
    label: 'check target',
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: 'check pixels',
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  record(encoder, texture.createView());
  encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow }, [width, height]);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(readback.getMappedRange());
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    pixels.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
  }
  readback.unmap();
  readback.destroy();
  texture.destroy();
  return pixels;
}

/** Copies a GPU buffer (created with COPY_SRC) back to the CPU. */
export async function readBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<ArrayBuffer> {
  const readback = device.createBuffer({
    label: 'check readback',
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  // The mapped range is gone after unmap(), so copy it out first.
  const data = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();
  return data;
}

export function pixelAt(pixels: Uint8Array, width: number, [x, y]: [number, number]): RGB {
  const i = (y * width + x) * 4;
  return [pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255];
}

export function maxDiff(a: readonly number[], b: readonly number[]): number {
  // NaN propagates through Math.max, so a NaN anywhere fails the check.
  return Math.max(...a.map((value, i) => Math.abs(value - b[i])));
}

export function formatRgb([r, g, b]: RGB): string {
  return `(${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`;
}

export function verdict(failures: string[]): string {
  return failures.length ? 'FAIL' : 'PASS';
}
