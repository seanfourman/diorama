import { GaussianRenderer, type CameraData } from './gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS } from './gaussians';
import { psnr, ssim } from './metrics';
import { PrefixSum } from './prefixSum';
import { mulberry32 } from './random';
import { readBuffer, readTexture } from './readback';
import adamWgsl from './shaders/adam.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import densifyWgsl from './shaders/densify.wgsl?raw';
import gaussianMathWgsl from './shaders/gaussianMath.wgsl?raw';
import type { RGB } from './tileRasterizer';
import { TrainingLoss } from './trainingLoss';

const SH_C0 = 0.28209479177387814;
const WORKGROUP_SIZE = 256; // must match adam.wgsl and densify.wgsl
/** Adam moments per Gaussian: one per raw parameter and spherical-harmonic coefficient. */
const MOMENT_SLOTS = FLOATS_PER_GAUSSIAN + SH_REST_FLOATS;

/** One photo to learn from: its camera, and the photo as an rgba8unorm texture of the camera's viewport size. */
export interface TrainingView {
  name: string;
  camera: CameraData;
  image: GPUTexture;
}

/**
 * Raw training parameters. `params` has FLOATS_PER_GAUSSIAN per Gaussian, laid out
 * like packGaussians() but raw: position, opacity logit, log scale, (pad),
 * rotation, degree-0 coefficient, (pad). `sh` has SH_REST_FLOATS per Gaussian.
 */
export interface RawGaussians {
  params: Float32Array<ArrayBuffer>;
  sh: Float32Array<ArrayBuffer>;
}

/** Learning rates and schedules. The defaults are the reference's (arguments/__init__.py). */
export interface TrainingSettings {
  /** The scene's size. The position learning rate and the densification sizes scale with it (the reference's cameras_extent). */
  extent: number;
  /** The position learning rate decays exponentially from the first to the second over this many steps. */
  positionLrSteps: number;
  positionLr: [number, number];
  opacityLr: number;
  scaleLr: number;
  rotationLr: number;
  colorLr: number;
  shLr: number;
  /** Raise the spherical-harmonic degree every this many steps, up to 3. */
  shDegreeInterval: number;
  /** Densify every `densifyInterval` steps after step `densifyFrom`, until step `densifyUntil`. */
  densifyFrom: number;
  densifyUntil: number;
  densifyInterval: number;
  /** The average 2D-center gradient, in NDC units, from which a Gaussian is cloned or split. */
  densifyGradThreshold: number;
  /** Gaussians up to this fraction of the extent in size are cloned; larger ones are split. */
  percentDense: number;
  /** Gaussians fainter than this are pruned. */
  minOpacity: number;
  /** Every this many steps while densifying, cap every opacity at 0.01. After the first, also prune Gaussians over 0.1 × extent. */
  opacityResetInterval: number;
  background: RGB;
}

export const DEFAULT_SETTINGS: Omit<TrainingSettings, 'extent'> = {
  positionLrSteps: 30_000,
  positionLr: [0.00016, 0.0000016],
  opacityLr: 0.025,
  scaleLr: 0.005,
  rotationLr: 0.001,
  colorLr: 0.0025,
  shLr: 0.0025 / 20,
  shDegreeInterval: 1000,
  densifyFrom: 500,
  densifyUntil: 15_000,
  densifyInterval: 100,
  densifyGradThreshold: 0.0002,
  percentDense: 0.01,
  minOpacity: 0.005,
  opacityResetInterval: 3000,
  background: [0, 0, 0],
};

type PipelineName = 'activate' | 'adam' | 'stats' | 'resetOpacity' | 'countOutputs' | 'scatter';

/** What one densification did. */
export interface Densification {
  step: number;
  before: number;
  after: number;
  kept: number;
  cloned: number;
  split: number;
  removed: number;
}

// Trains Gaussians to match a set of photos (M3), following the reference's loop.
// Each step renders one training view, compares it with its photo
// (0.8 × L1 + 0.2 × D-SSIM), runs the backward pass, and takes an Adam step.
// Every 100 steps until step 15,000 it densifies: clones and splits Gaussians
// that are struggling, and prunes faint ones (densify.wgsl). Every 3,000 steps
// it resets opacities. Notes: docs/steps/3-training.md
export class Trainer {
  iteration = 0;
  shDegree = 0;
  readonly renderer: GaussianRenderer;
  readonly settings: TrainingSettings;
  /** Every densification so far, for the Inside view. */
  readonly densifications: Densification[] = [];
  private gaussianCount: number;
  private adamSteps = 0;
  private readonly device: GPUDevice;
  private readonly views: TrainingView[];
  private readonly loss: TrainingLoss;
  private readonly target: GPUTexture;
  private params: GPUBuffer;
  private moments: GPUBuffer;
  private stats: GPUBuffer;
  private readonly adamParams: GPUBuffer;
  private readonly pipelines: Record<PipelineName, GPUComputePipeline>;
  private readonly random = mulberry32(1);
  private queue: TrainingView[] = [];
  private bindGroups?: { adam: GPUBindGroup; stats: GPUBindGroup };
  // What the last optimize() left for maintain().
  private due?: { densify: boolean; reset: boolean };

  constructor(device: GPUDevice, raw: RawGaussians, views: TrainingView[], settings: TrainingSettings) {
    this.device = device;
    this.views = views;
    this.settings = settings;
    this.gaussianCount = raw.params.length / FLOATS_PER_GAUSSIAN;
    const [width, height] = views[0].camera.viewport;
    if (views.some(({ camera }) => camera.viewport[0] !== width || camera.viewport[1] !== height)) {
      throw new Error('All training views must be the same size.');
    }

    this.renderer = new GaussianRenderer(device);
    // Upload once activated on the CPU; from then on, the Adam step keeps the
    // renderer's copy in step with the raw parameters.
    this.renderer.setGaussians(activate(raw.params), { rest: raw.sh, degree: 0 });
    this.loss = new TrainingLoss(device, width, height);
    this.target = device.createTexture({
      label: 'training render',
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING,
    });

    this.params = this.storage('raw parameters', raw.params.byteLength);
    device.queue.writeBuffer(this.params, 0, raw.params);
    // Zero-initialized: Adam starts with no history.
    this.moments = this.storage('Adam moments', this.gaussianCount * MOMENT_SLOTS * 8);
    this.stats = this.storage('densification stats', this.gaussianCount * 16);
    this.adamParams = device.createBuffer({ label: 'Adam params', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const adam = device.createShaderModule({ label: 'adam', code: `${commonWgsl}\n${adamWgsl}` });
    const densify = device.createShaderModule({ label: 'densify', code: `${gaussianMathWgsl}\n${densifyWgsl}` });
    const pipeline = (module: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    this.pipelines = {
      activate: pipeline(adam, 'activate'),
      adam: pipeline(adam, 'adam_step'),
      stats: pipeline(adam, 'accumulate_stats'),
      resetOpacity: pipeline(adam, 'reset_opacity'),
      countOutputs: pipeline(densify, 'count_outputs'),
      scatter: pipeline(densify, 'scatter'),
    };
  }

  /** How many Gaussians there are now. Densification changes it. */
  get count(): number {
    return this.gaussianCount;
  }

  /** The raw parameters, Adam's moments and the densification stats. For checks. */
  get buffers(): { params: GPUBuffer; moments: GPUBuffer; stats: GPUBuffer } {
    return { params: this.params, moments: this.moments, stats: this.stats };
  }

  /** Runs one training step: optimize(), then maintain(). */
  async step(): Promise<void> {
    this.optimize();
    await this.maintain();
  }

  /**
   * The first part of a step: render one photo's view, the loss, the backward
   * pass and the Adam update, recorded and submitted without waiting for the GPU.
   * Measure between this and maintain(): like the reference, before the step's
   * densification and opacity reset, which change the image abruptly.
   */
  optimize(): void {
    const t = ++this.iteration;
    const { settings, renderer, device } = this;
    if (t % settings.shDegreeInterval === 0 && this.shDegree < 3) {
      renderer.setShDegree(++this.shDegree);
    }
    const densifying = t < settings.densifyUntil;
    // Like the reference, a densification step takes no optimizer step: the
    // reference's new parameter tensors have no gradients yet.
    const densifyNow = densifying && t > settings.densifyFrom && t % settings.densifyInterval === 0;

    const view = this.nextView();
    const encoder = device.createCommandEncoder({ label: `training step ${t}` });
    renderer.encode(encoder, this.target.createView(), view.camera, settings.background);
    this.loss.encode(encoder, renderer.pixelBuffer!, view.image.createView());
    renderer.encodeBackward(encoder, this.loss.pixelGrads);

    if (!densifyNow) this.adamSteps++;
    const progress = Math.min(1, t / settings.positionLrSteps);
    const [lrStart, lrEnd] = settings.positionLr;
    const positionLr = Math.exp(Math.log(lrStart) * (1 - progress) + Math.log(lrEnd) * progress) * settings.extent;
    device.queue.writeBuffer(
      this.adamParams,
      0,
      new Float32Array([
        1 - 0.9 ** this.adamSteps,
        1 - 0.999 ** this.adamSteps,
        positionLr,
        settings.opacityLr,
        settings.scaleLr,
        settings.rotationLr,
        settings.colorLr,
        settings.shLr,
        ...view.camera.viewport,
      ]),
    );
    const bindGroups = (this.bindGroups ??= this.createBindGroups());
    const pass = encoder.beginComputePass({ label: 'optimize' });
    // The statistics only matter for densifying, but the Inside view shows them
    // afterwards too, and they cost one small dispatch.
    pass.setPipeline(this.pipelines.stats);
    pass.setBindGroup(0, bindGroups.stats);
    pass.dispatchWorkgroups(this.groups());
    if (!densifyNow) {
      pass.setPipeline(this.pipelines.adam);
      pass.setBindGroup(0, bindGroups.adam);
      pass.dispatchWorkgroups(this.groups());
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    void renderer.afterSubmit();
    this.due = { densify: densifyNow, reset: densifying && t % settings.opacityResetInterval === 0 };
  }

  /** The rest of a step: its densification and opacity reset, if due. Densifying reads back the new count. */
  async maintain(): Promise<void> {
    const due = this.due;
    this.due = undefined;
    if (due?.densify) await this.densify(this.iteration > this.settings.opacityResetInterval);
    if (due?.reset) this.resetOpacity();
  }

  /**
   * Clones, splits and prunes (densify.wgsl), then starts the statistics over.
   * `pruneLarge` also removes Gaussians larger than 0.1 × extent. Public for checks.
   */
  async densify(pruneLarge: boolean): Promise<void> {
    const { device, settings, renderer } = this;
    const count = this.gaussianCount;
    const outputs = device.createBuffer({ label: 'densify outputs', size: count * 4, usage: GPUBufferUsage.STORAGE });
    const scan = new PrefixSum(device, outputs, count);
    const uniform = device.createBuffer({ label: 'densify params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new ArrayBuffer(32);
    new Float32Array(data, 0, 4).set([
      settings.densifyGradThreshold,
      settings.percentDense * settings.extent,
      settings.minOpacity,
      pruneLarge ? 0.1 * settings.extent : 0,
    ]);
    new Uint32Array(data, 16, 1)[0] = this.iteration; // the random seed
    device.queue.writeBuffer(uniform, 0, data);

    // Count each Gaussian's outputs and prefix-sum them, then read back the total
    // and the tally of what happened.
    const counters = device.createBuffer({ label: 'densify counters', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    let encoder = device.createCommandEncoder({ label: 'densify count' });
    let pass = encoder.beginComputePass({ label: 'densify count' });
    pass.setPipeline(this.pipelines.countOutputs);
    pass.setBindGroup(
      0,
      this.bindGroup(this.pipelines.countOutputs, { 0: uniform, 1: this.params, 2: this.stats, 3: outputs, 9: counters }),
    );
    pass.dispatchWorkgroups(this.groups(count));
    scan.encode(pass);
    pass.end();
    const readback = device.createBuffer({ label: 'densify count', size: 20, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(scan.total, 0, readback, 0, 4);
    encoder.copyBufferToBuffer(counters, 0, readback, 4, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const [newCount, removed, kept, cloned, split] = new Uint32Array(readback.getMappedRange());
    readback.destroy();
    counters.destroy();
    if (newCount === 0) throw new Error('Densification removed every Gaussian.');
    this.densifications.push({ step: this.iteration, before: count, after: newCount, kept, cloned, split, removed });

    // Write the new set, and its activated copy for the renderer.
    const params = this.storage('raw parameters', newCount * FLOATS_PER_GAUSSIAN * 4);
    const sh = this.storage('spherical harmonics', newCount * SH_REST_FLOATS * 4);
    const moments = this.storage('Adam moments', newCount * MOMENT_SLOTS * 8);
    const stats = this.storage('densification stats', newCount * 16);
    const gaussians = this.storage('gaussians', newCount * FLOATS_PER_GAUSSIAN * 4);
    encoder = device.createCommandEncoder({ label: 'densify scatter' });
    pass = encoder.beginComputePass({ label: 'densify scatter' });
    pass.setPipeline(this.pipelines.scatter);
    pass.setBindGroup(
      0,
      this.bindGroup(this.pipelines.scatter, {
        0: uniform,
        1: this.params,
        2: this.stats,
        3: outputs,
        4: renderer.shBuffer!,
        5: this.moments,
        6: params,
        7: sh,
        8: moments,
      }),
    );
    pass.dispatchWorkgroups(this.groups(count));
    pass.setPipeline(this.pipelines.activate);
    pass.setBindGroup(0, this.bindGroup(this.pipelines.activate, { 1: params, 6: gaussians }));
    pass.dispatchWorkgroups(this.groups(newCount));
    pass.end();
    device.queue.submit([encoder.finish()]);

    // The old buffers can go once the work above is submitted; the GPU keeps them until it's done.
    renderer.adoptGaussians(newCount, gaussians, sh, this.shDegree);
    for (const buffer of [this.params, this.moments, this.stats, outputs, uniform]) buffer.destroy();
    scan.destroy();
    this.params = params;
    this.moments = moments;
    this.stats = stats;
    this.gaussianCount = newCount;
    this.bindGroups = undefined;
  }

  /** Caps every opacity at 0.01 and clears Adam's history for it. Public for checks. */
  resetOpacity(): void {
    const encoder = this.device.createCommandEncoder({ label: 'reset opacity' });
    const pass = encoder.beginComputePass({ label: 'reset opacity' });
    pass.setPipeline(this.pipelines.resetOpacity);
    pass.setBindGroup(
      0,
      this.bindGroup(this.pipelines.resetOpacity, { 1: this.params, 3: this.moments, 6: this.renderer.gaussianBuffer! }),
    );
    pass.dispatchWorkgroups(this.groups());
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Copies the raw parameters back to the CPU, for saving. */
  async exportRaw(): Promise<RawGaussians> {
    return {
      params: new Float32Array(await readBuffer(this.device, this.params)),
      sh: new Float32Array(await readBuffer(this.device, this.renderer.shBuffer!)),
    };
  }

  /** The loss of the most recent step. */
  lastLoss(): Promise<number> {
    return this.loss.read();
  }

  /**
   * The average PSNR (dB) and SSIM of renders of `views` against their photos,
   * measured as the reference does: on renders saved as 8-bit images. SSIM is
   * slow on the CPU, so it's optional.
   */
  async evaluate(views: TrainingView[], withSsim = true): Promise<{ psnr: number; ssim: number }> {
    let psnrTotal = 0;
    let ssimTotal = 0;
    for (const view of views) {
      const encoder = this.device.createCommandEncoder();
      this.renderer.encode(encoder, this.target.createView(), view.camera, this.settings.background);
      this.device.queue.submit([encoder.finish()]);
      const [width, height] = view.camera.viewport;
      const rendered = new Float32Array(await readBuffer(this.device, this.renderer.pixelBuffer!)).subarray(0, width * height * 4);
      const photo = Float32Array.from(await readTexture(this.device, view.image), (value) => value / 255);
      for (let i = 0; i < rendered.length; i++) rendered[i] = Math.round(Math.min(1, Math.max(0, rendered[i])) * 255) / 255;
      psnrTotal += psnr(rendered, photo);
      if (withSsim) ssimTotal += ssim(rendered, photo, width, height);
    }
    return { psnr: psnrTotal / views.length, ssim: withSsim ? ssimTotal / views.length : NaN };
  }

  destroy(): void {
    this.renderer.destroy();
    this.loss.destroy();
    for (const resource of [this.target, this.params, this.moments, this.stats, this.adamParams]) resource.destroy();
  }

  // Like the reference: every view once, in a random order, then reshuffle.
  private nextView(): TrainingView {
    if (this.queue.length === 0) {
      this.queue = [...this.views];
      for (let i = this.queue.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        [this.queue[i], this.queue[j]] = [this.queue[j], this.queue[i]];
      }
    }
    return this.queue.pop()!;
  }

  private groups(count = this.gaussianCount): number {
    return Math.ceil(count / WORKGROUP_SIZE);
  }

  private storage(label: string, size: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private bindGroup(pipeline: GPUComputePipeline, resources: Record<number, GPUBuffer>): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(resources).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
    });
  }

  private createBindGroups(): { adam: GPUBindGroup; stats: GPUBindGroup } {
    const { renderer } = this;
    const gradients = renderer.gradients!;
    return {
      adam: this.bindGroup(this.pipelines.adam, {
        0: this.adamParams,
        1: this.params,
        2: renderer.shBuffer!,
        3: this.moments,
        4: gradients.gaussians,
        5: gradients.sh,
        6: renderer.gaussianBuffer!,
      }),
      stats: this.bindGroup(this.pipelines.stats, {
        0: this.adamParams,
        7: renderer.splatBuffer!,
        8: gradients.splats,
        9: this.stats,
      }),
    };
  }
}

/** Raw parameters to the renderer's activated layout, as adam.wgsl's activate does. */
export function activate(params: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> {
  const out = new Float32Array(params.length);
  for (let at = 0; at < params.length; at += FLOATS_PER_GAUSSIAN) {
    for (let k = 0; k < 3; k++) {
      out[at + k] = params[at + k];
      out[at + 4 + k] = Math.exp(params[at + 4 + k]);
      out[at + 12 + k] = 0.5 + SH_C0 * params[at + 12 + k];
    }
    out[at + 3] = 1 / (1 + Math.exp(-params[at + 3]));
    for (let k = 8; k < 12; k++) out[at + k] = params[at + k];
  }
  return out;
}
