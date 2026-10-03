// Copying GPU data back to the CPU.

/** Copies a GPU buffer (created with COPY_SRC) back to the CPU. */
export async function readBuffer(device: GPUDevice, buffer: GPUBuffer): Promise<ArrayBuffer> {
  const readback = device.createBuffer({
    label: 'readback',
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

/** Copies an rgba8unorm texture (created with COPY_SRC) back to the CPU, 4 bytes per pixel, row by row. */
export async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint8Array> {
  const { width, height } = texture;
  // Texture-to-buffer copies need each row padded to a multiple of 256 bytes.
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const readback = device.createBuffer({
    label: 'texture readback',
    size: bytesPerRow * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
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
  return pixels;
}
