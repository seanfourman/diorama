import { turboGradient } from './colormap';
import type { CameraData, GaussianRenderer } from './gaussianRenderer';
import type { Mat4 } from './mat4';
import { FLOATS_PER_SPLAT } from './gaussians';
import blitWgsl from './shaders/blit.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import inspectWgsl from './shaders/inspect.wgsl?raw';
import type { RGB } from './tileRasterizer';
import type { Trainer, TrainingView } from './trainer';

// The Inside view (M4): what the renderer and the optimizer are doing, drawn in
// place of the normal image. Some modes recolor the Gaussians (depth, the pull
// on each one, when densification made it); some draw per-pixel heatmaps (work,
// error); and two compare a render with its photo. See inspect.wgsl.
// Notes: docs/steps/4-inside-view.md

export type InsideMode = 'color' | 'depth' | 'work' | 'gradient' | 'age' | 'compare' | 'error';

export interface ModeInfo {
  mode: InsideMode;
  /** The key that selects it. */
  key: string;
  name: string;
  /** Needs training's statistics. */
  training: boolean;
  /** Needs the camera at a photo's viewpoint. */
  photo: boolean;
  /** What the colors mean. */
  legend: string;
  /** A Turbo color bar, and its two ends; none for modes without a scale. */
  scale?: { reverse?: boolean; low: string; high: string };
}

export const MODES: ModeInfo[] = [
  { mode: 'color', key: 'Digit1', name: 'Color', training: false, photo: false, legend: 'The scene as the Gaussians draw it.' },
  {
    mode: 'depth',
    key: 'Digit2',
    name: 'Depth',
    training: false,
    photo: false,
    legend:
      'Distance from the camera, on a log scale fitted to what is in view. Each Gaussian is recolored and blended as usual.',
    scale: { reverse: true, low: 'near', high: 'far' },
  },
  {
    mode: 'work',
    key: 'Digit3',
    name: 'Work',
    training: false,
    photo: false,
    legend:
      'How many splats each pixel walked through before it was done, on a log scale. The backward pass ' +
      'walks the same lists, so the red areas are the expensive ones to render and to train.',
    scale: { low: '16', high: '4,096+' },
  },
  {
    mode: 'gradient',
    key: 'Digit4',
    name: 'Pull',
    training: true,
    photo: false,
    legend:
      "The average pull on each Gaussian's 2D center since the last densification: the signal that decides " +
      'densification. Orange and red are over the threshold; those get cloned or split next time. Gray: not seen yet.',
    scale: { low: '1% of threshold', high: '10×' },
  },
  {
    mode: 'age',
    key: 'Digit5',
    name: 'Densification',
    training: true,
    photo: false,
    legend:
      'Where densification added Gaussians: green ones were cloned, orange ones split from a larger one, gray ones ' +
      'came from the photos’ point cloud. New ones glow, then fade over about 1,000 steps.',
  },
  {
    mode: 'compare',
    key: 'Digit6',
    name: 'Compare',
    training: true,
    photo: true,
    legend: "Left: the render from the photo's viewpoint. Right: the photo.",
  },
  {
    mode: 'error',
    key: 'Digit7',
    name: 'Error',
    training: true,
    photo: true,
    legend: "|render − photo| per pixel, averaged over red, green and blue, from the photo's viewpoint.",
    scale: { low: '0', high: '0.2+' },
  },
];

/** The splat counts at the two ends of the work scale. */
export const WORK_RANGE: [number, number] = [16, 4096];
const ERROR_MAX = 0.2;
const TILE = 16; // must match the per-pixel passes' workgroup size in inspect.wgsl

type PipelineName = 'depth' | 'gradient' | 'age' | 'work' | 'error';

/**
 * The depths at the two ends of the depth scale for a camera: the 5th and 95th
 * percentiles of `points` (xyz triples, a sample of the scene) in front of it.
 */
export function visibleDepthRange(points: Float32Array, view: Mat4): [number, number] {
  const depths: number[] = [];
  for (let i = 0; i < points.length; i += 3) {
    // The view looks down −z, so depth is minus the view-space z.
    const depth = -(view[2] * points[i] + view[6] * points[i + 1] + view[10] * points[i + 2] + view[14]);
    if (depth > 1e-3) depths.push(depth);
  }
  if (depths.length < 2) return [0.1, 100];
  depths.sort((a, b) => a - b);
  const near = depths[Math.floor(depths.length * 0.05)];
  const far = depths[Math.floor(depths.length * 0.95)];
  return [near, Math.max(far, 2 * near)];
}

/** Up to `count` of `positions` (xyz triples), evenly spaced, for visibleDepthRange. */
export function samplePoints(positions: Float32Array, stride: number, count = 4096): Float32Array {
  const total = Math.floor(positions.length / stride);
  const step = Math.max(1, Math.floor(total / count));
  const sample = new Float32Array(Math.ceil(total / step) * 3);
  for (let i = 0, k = 0; i < total; i += step, k += 3) sample.set(positions.subarray(i * stride, i * stride + 3), k);
  return sample;
}

export class InsideView {
  mode: InsideMode = 'color';
  /** The depths at the two ends of the depth scale. Pages update it with visibleDepthRange. */
  depthRange: [number, number];
  private readonly device: GPUDevice;
  private readonly renderer: GaussianRenderer;
  private readonly trainer?: Trainer;
  private readonly params: GPUBuffer;
  private readonly pipelines: Record<PipelineName, GPUComputePipeline>;
  private readonly blitPipeline: GPURenderPipeline;
  private readonly blitParams: GPUBuffer;
  private readonly sampler: GPUSampler;
  // Photo-sized images for the photo modes: the render, and the error heatmap.
  private photoTargets?: { render: GPUTexture; heat: GPUTexture };
  // Per-tile sums of squared error, from error_heatmap.
  private partials?: GPUBuffer;

  /**
   * `depthRange` sets the depths at the two ends of the depth scale. Without a
   * trainer, only the color, depth and work modes are available.
   */
  constructor(device: GPUDevice, renderer: GaussianRenderer, depthRange: [number, number], trainer?: Trainer) {
    this.device = device;
    this.renderer = renderer;
    this.trainer = trainer;
    this.depthRange = depthRange;
    this.params = device.createBuffer({ label: 'inspect params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blitParams = device.createBuffer({ label: 'blit params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: 'inspect', code: `${commonWgsl}\n${inspectWgsl}` });
    const pipeline = (entryPoint: string) =>
      device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    this.pipelines = {
      depth: pipeline('colorize_depth'),
      gradient: pipeline('colorize_gradient'),
      age: pipeline('colorize_age'),
      work: pipeline('work_heatmap'),
      error: pipeline('error_heatmap'),
    };
    const blit = device.createShaderModule({ label: 'blit', code: blitWgsl });
    this.blitPipeline = device.createRenderPipeline({
      label: 'blit',
      layout: 'auto',
      vertex: { module: blit, entryPoint: 'fullscreen' },
      fragment: { module: blit, entryPoint: 'draw', targets: [{ format: 'rgba8unorm' }] },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  /** The modes that work here: all of them while training, otherwise only those that need no training data. */
  get modes(): ModeInfo[] {
    return MODES.filter((info) => this.trainer || !info.training);
  }

  get info(): ModeInfo {
    return MODES.find(({ mode }) => mode === this.mode)!;
  }

  /** Switches mode if `code` is a mode's key. Returns whether it was. */
  handleKey(code: string): boolean {
    const info = this.modes.find(({ key }) => key === code);
    if (info) this.mode = info.mode;
    return Boolean(info);
  }

  /** The legend's color bar, as a CSS background, or '' for modes without a scale. */
  get scaleBackground(): string {
    const scale = this.info.scale;
    return scale ? turboGradient(scale.reverse) : '';
  }

  /**
   * Records a frame of the current mode onto `target` (the canvas), from
   * `camera`. The photo modes need `photo`: the view to compare with, which is
   * drawn from the photo's own camera and letterboxed. Without one, they fall
   * back to the color view.
   */
  encode(encoder: GPUCommandEncoder, target: GPUTexture, camera: CameraData, background: RGB, photo?: TrainingView): void {
    const { renderer } = this;
    const mode = this.info.photo && !photo ? 'color' : this.mode;
    this.writeParams();
    if (mode === 'compare' || mode === 'error') {
      const targets = this.photoTargetsFor(photo!.image);
      renderer.encode(encoder, targets.render.createView(), photo!.camera, background);
      if (mode === 'error') this.encodeError(encoder, photo!.image, targets.heat, this.errorPartials(1));
      const left = mode === 'compare' ? targets.render : targets.heat;
      const right = mode === 'compare' ? photo!.image : targets.heat;
      this.encodeBlit(encoder, target, left, right, mode === 'compare');
      return;
    }
    const recolor = mode === 'depth' || mode === 'gradient' || mode === 'age' ? mode : undefined;
    renderer.encode(encoder, target.createView(), camera, background, (pass) => {
      if (recolor) this.encodeColorize(pass, recolor);
    });
    if (mode === 'work') {
      const pass = encoder.beginComputePass({ label: 'work heatmap' });
      pass.setPipeline(this.pipelines.work);
      pass.setBindGroup(0, this.bindGroup(this.pipelines.work, { 0: this.params, 4: renderer.pixelLastBuffer!, 7: target.createView() }));
      pass.dispatchWorkgroups(Math.ceil(target.width / TILE), Math.ceil(target.height / TILE));
      pass.end();
    }
  }

  /**
   * The average PSNR of renders of `views` against their photos, measured on the
   * GPU (each render rounded to 8 bits, like the reference measures them). Much
   * faster than Trainer.evaluate, for the training curves.
   */
  async measurePsnr(views: TrainingView[], background: RGB): Promise<number> {
    const { device } = this;
    const [width, height] = views[0].camera.viewport;
    const tiles = Math.ceil(width / TILE) * Math.ceil(height / TILE);
    const partials = this.errorPartials(views.length * tiles);
    for (const [k, view] of views.entries()) {
      // One submission per view, since the camera and where this view's sums go
      // are both written with queue.writeBuffer.
      this.writeParams(k * tiles);
      const targets = this.photoTargetsFor(view.image);
      const encoder = device.createCommandEncoder({ label: 'measure PSNR' });
      this.renderer.encode(encoder, targets.render.createView(), view.camera, background);
      this.encodeError(encoder, view.image, targets.heat, partials);
      device.queue.submit([encoder.finish()]);
    }
    const readback = device.createBuffer({ size: partials.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(partials, 0, readback, 0, partials.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const sums = new Float32Array(readback.getMappedRange());
    let total = 0;
    for (let k = 0; k < views.length; k++) {
      let squaredError = 0;
      for (let t = 0; t < tiles; t++) squaredError += sums[k * tiles + t];
      total += 10 * Math.log10((3 * width * height) / squaredError);
    }
    readback.destroy();
    return total / views.length;
  }

  destroy(): void {
    for (const buffer of [this.params, this.blitParams, this.partials]) buffer?.destroy();
    this.photoTargets?.render.destroy();
    this.photoTargets?.heat.destroy();
  }

  // A buffer for at least `floats` per-tile error sums.
  private errorPartials(floats: number): GPUBuffer {
    const size = Math.max(4096, floats) * 4;
    if (!this.partials || this.partials.size < size) {
      this.partials?.destroy();
      this.partials = this.device.createBuffer({ label: 'error sums', size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    }
    return this.partials;
  }

  // The uniform for every pass; `firstTile` is where error_heatmap writes its sums.
  private writeParams(firstTile = 0): void {
    const data = new ArrayBuffer(32);
    new Uint32Array(data, 0, 1)[0] = firstTile;
    new Float32Array(data, 4, 7).set([
      this.trainer?.iteration ?? 0,
      this.trainer?.settings.densifyGradThreshold ?? 1,
      WORK_RANGE[0],
      ...this.depthRange,
      ERROR_MAX,
      WORK_RANGE[1],
    ]);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  private encodeColorize(pass: GPUComputePassEncoder, mode: 'depth' | 'gradient' | 'age'): void {
    const splats = this.renderer.splatBuffer!;
    const resources: Record<number, GPUBuffer> = { 0: this.params, 1: splats };
    if (mode === 'gradient') resources[2] = this.trainer!.buffers.stats;
    if (mode === 'age') resources[3] = this.trainer!.buffers.params;
    pass.setPipeline(this.pipelines[mode]);
    pass.setBindGroup(0, this.bindGroup(this.pipelines[mode], resources));
    pass.dispatchWorkgroups(Math.ceil(splats.size / (FLOATS_PER_SPLAT * 4) / 256));
  }

  private encodeError(encoder: GPUCommandEncoder, photo: GPUTexture, heat: GPUTexture, partials: GPUBuffer): void {
    const pass = encoder.beginComputePass({ label: 'error heatmap' });
    pass.setPipeline(this.pipelines.error);
    pass.setBindGroup(
      0,
      this.bindGroup(this.pipelines.error, {
        0: this.params,
        5: this.renderer.pixelBuffer!,
        6: photo.createView(),
        7: heat.createView(),
        8: partials,
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(photo.width / TILE), Math.ceil(photo.height / TILE));
    pass.end();
  }

  // Letterboxes `left` (and `right`, past the middle when splitting) onto the target.
  private encodeBlit(encoder: GPUCommandEncoder, target: GPUTexture, left: GPUTexture, right: GPUTexture, split: boolean): void {
    const scale = Math.min(target.width / left.width, target.height / left.height);
    const [width, height] = [left.width * scale, left.height * scale];
    const x0 = (target.width - width) / 2;
    const y0 = (target.height - height) / 2;
    this.device.queue.writeBuffer(
      this.blitParams,
      0,
      new Float32Array([x0, y0, x0 + width, y0 + height, split ? x0 + width / 2 : -10, 0, 0, 0]),
    );
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.blitPipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.blitParams } },
          { binding: 1, resource: left.createView() },
          { binding: 2, resource: right.createView() },
          { binding: 3, resource: this.sampler },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
  }

  private photoTargetsFor(photo: GPUTexture): { render: GPUTexture; heat: GPUTexture } {
    const current = this.photoTargets;
    if (current && current.render.width === photo.width && current.render.height === photo.height) return current;
    current?.render.destroy();
    current?.heat.destroy();
    const texture = (label: string) =>
      this.device.createTexture({
        label,
        size: [photo.width, photo.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
    this.photoTargets = { render: texture('photo-view render'), heat: texture('error heatmap') };
    return this.photoTargets;
  }

  private bindGroup(pipeline: GPUComputePipeline, resources: Record<number, GPUBuffer | GPUTextureView>): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(resources).map(([binding, resource]) => ({
        binding: Number(binding),
        resource: resource instanceof GPUBuffer ? { buffer: resource } : resource,
      })),
    });
  }
}
