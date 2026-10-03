import lossWgsl from './shaders/loss.wgsl?raw';

const LAMBDA = 0.2; // must match loss.wgsl

// The 11-tap Gaussian window (σ = 1.5) that SSIM blurs with, normalized; entry k
// is the weight at offsets ±k.
function windowWeights(): number[] {
  const raw = Array.from({ length: 6 }, (_, k) => Math.exp(-(k * k) / (2 * 1.5 * 1.5)));
  const total = raw[0] + 2 * raw.slice(1).reduce((sum, w) => sum + w, 0);
  return raw.map((w) => w / total);
}

// The training loss on the GPU (M3): 0.8 × L1 + 0.2 × (1 − SSIM) between a render
// and a photo, and its gradient with respect to every rendered pixel, ready for the
// renderer's backward pass. See loss.wgsl for the math.
export class TrainingLoss {
  readonly width: number;
  readonly height: number;
  /** dL/d(pixel rgb), a vec4 per pixel, row by row. */
  readonly pixelGrads: GPUBuffer;
  /** Each pixel's share of the loss, without the constant λ. */
  readonly pixelLoss: GPUBuffer;
  private readonly device: GPUDevice;
  private readonly pipelines: Record<'moments' | 'blur' | 'ssim' | 'gradient', GPUComputePipeline>;
  private readonly buffers: GPUBuffer[] = [];
  private readonly moments: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly blurred: GPUBuffer;
  private readonly partials: GPUBuffer;
  // One uniform buffer per blur configuration: (horizontal, map count).
  private readonly params: { h5: GPUBuffer; v5: GPUBuffer; h3: GPUBuffer; v3: GPUBuffer };

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.width = width;
    this.height = height;
    const module = device.createShaderModule({ label: 'loss', code: lossWgsl });
    const pipeline = (entryPoint: string) =>
      device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    this.pipelines = {
      moments: pipeline('moments_pass'),
      blur: pipeline('blur_pass'),
      ssim: pipeline('ssim_pass'),
      gradient: pipeline('gradient_pass'),
    };
    const pixels = width * height;
    const buffer = (label: string, size: number, usage = GPUBufferUsage.STORAGE) => {
      const created = device.createBuffer({ label, size, usage });
      this.buffers.push(created);
      return created;
    };
    this.moments = buffer('loss moments', 5 * pixels * 16);
    this.scratch = buffer('loss blur scratch', 5 * pixels * 16);
    this.blurred = buffer('loss blurred', 5 * pixels * 16);
    this.partials = buffer('loss partials', 4 * pixels * 16);
    this.pixelGrads = buffer('pixel gradients', pixels * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.pixelLoss = buffer('pixel loss', pixels * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const weights = windowWeights();
    const uniform = (horizontal: number, mapCount: number) => {
      const data = new ArrayBuffer(48);
      new Uint32Array(data, 0, 4).set([width, height, horizontal, mapCount]);
      new Float32Array(data, 16, 8).set(weights);
      const created = buffer('loss params', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(created, 0, data);
      return created;
    };
    this.params = { h5: uniform(1, 5), v5: uniform(0, 5), h3: uniform(1, 3), v3: uniform(0, 3) };
  }

  /**
   * Records the loss between `rendered` (the renderer's pixel buffer: a vec4 per
   * pixel, rgb used) and `target` (the photo, an rgba8unorm texture of the same size).
   */
  encode(encoder: GPUCommandEncoder, rendered: GPUBuffer, target: GPUTextureView): void {
    const { moments, scratch, blurred, partials, params } = this;
    const pass = encoder.beginComputePass({ label: 'loss' });
    const run = (pipeline: GPUComputePipeline, resources: Record<number, GPUBuffer | GPUTextureView>) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: Object.entries(resources).map(([binding, resource]) => ({
            binding: Number(binding),
            resource: resource instanceof GPUBuffer ? { buffer: resource } : resource,
          })),
        }),
      );
      pass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(this.height / 16));
    };
    run(this.pipelines.moments, { 0: params.h5, 1: rendered, 2: target, 3: moments });
    run(this.pipelines.blur, { 0: params.h5, 4: moments, 5: scratch });
    run(this.pipelines.blur, { 0: params.v5, 4: scratch, 5: blurred });
    run(this.pipelines.ssim, { 0: params.h5, 6: blurred, 7: partials });
    run(this.pipelines.blur, { 0: params.h3, 4: partials, 5: scratch });
    run(this.pipelines.blur, { 0: params.v3, 4: scratch, 5: blurred });
    run(this.pipelines.gradient, {
      0: params.h5,
      1: rendered,
      2: target,
      6: blurred,
      7: partials,
      9: this.pixelGrads,
      10: this.pixelLoss,
    });
    pass.end();
  }

  /** Reads back the loss from the last encode (which must have been submitted). */
  async read(): Promise<number> {
    const readback = this.device.createBuffer({
      size: this.pixelLoss.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.pixelLoss, 0, readback, 0, this.pixelLoss.size);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const shares = new Float32Array(readback.getMappedRange());
    let sum = 0;
    for (const share of shares) sum += share;
    readback.unmap();
    readback.destroy();
    return sum + LAMBDA;
  }

  destroy(): void {
    this.buffers.forEach((buffer) => buffer.destroy());
  }
}
