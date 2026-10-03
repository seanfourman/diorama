import { FLOATS_PER_GAUSSIAN, FLOATS_PER_SPLAT, SH_REST_FLOATS } from './gaussians';
import { IDENTITY, type Mat4 } from './mat4';
import commonWgsl from './shaders/common.wgsl?raw';
import gaussianMathWgsl from './shaders/gaussianMath.wgsl?raw';
import preprocessWgsl from './shaders/preprocess.wgsl?raw';
import preprocessBackwardWgsl from './shaders/preprocessBackward.wgsl?raw';
import { TileRasterizer, type RGB } from './tileRasterizer';

const WORKGROUP_SIZE = 256; // must match @workgroup_size in preprocess.wgsl

export interface CameraData {
  view: Mat4;
  proj: Mat4;
  viewport: [number, number];
}

/** A scene's higher-order spherical harmonics: SH_REST_FLOATS per Gaussian, and the degree to use (1 to 3). */
export interface SphericalHarmonics {
  rest: Float32Array<ArrayBuffer>;
  degree: number;
}

export interface RendererOptions {
  /** How many (splat, tile) pairs to make room for at first. The buffers grow when a frame needs more. */
  initialPairCapacity?: number;
}

/** Gradients from the backward pass, laid out like the renderer's inputs. */
export interface Gradients {
  /** 16 floats per Gaussian, laid out like packGaussians(): position, opacity, scale, rotation, color. */
  gaussians: GPUBuffer;
  /** SH_REST_FLOATS per Gaussian. */
  sh: GPUBuffer;
  /** 9 floats per Gaussian: d/d(2D center x, y), d/d(conic a, b, c), d/d(opacity), d/d(color r, g, b). */
  splats: GPUBuffer;
}

interface Scene {
  count: number;
  gaussians: GPUBuffer;
  splats: GPUBuffer;
  sh: GPUBuffer;
  sceneParams: GPUBuffer;
  preprocessBindGroup: GPUBindGroup;
  rasterizer: TileRasterizer;
  // Made on the first backward pass.
  gradients?: Gradients & { bindGroup: GPUBindGroup };
}

// Renders 3D Gaussians. Each frame, a compute pass projects every Gaussian to a 2D
// splat (M1.3). Then the tile rasterizer (M1.5) sorts the splats into 16×16-pixel
// tiles and blends every pixel front to back. It draws into an rgba8unorm texture
// that allows storage writes. For training, the backward pass (M2) turns
// dL/d(pixel color) into gradients for every Gaussian parameter.
export class GaussianRenderer {
  private readonly device: GPUDevice;
  private readonly options: RendererOptions;
  private readonly preprocessPipeline: GPUComputePipeline;
  private readonly backwardPipeline: GPUComputePipeline;
  private readonly cameraBuffer: GPUBuffer;
  private scene?: Scene;
  private lastViewport: [number, number] = [0, 0];
  // For encodeDraw: a rasterizer over 2D splats that a check made.
  private drawRasterizer?: TileRasterizer;

  constructor(device: GPUDevice, options: RendererOptions = {}) {
    this.device = device;
    this.options = options;
    // Both preprocess shaders get the shared structs and the shared math prepended.
    const pipeline = (entryPoint: string, code: string) =>
      device.createComputePipeline({
        label: entryPoint,
        layout: 'auto',
        compute: {
          module: device.createShaderModule({ label: entryPoint, code: `${commonWgsl}\n${gaussianMathWgsl}\n${code}` }),
          entryPoint,
        },
      });
    this.preprocessPipeline = pipeline('preprocess', preprocessWgsl);
    this.backwardPipeline = pipeline('preprocess_backward', preprocessBackwardWgsl);
    // Camera in common.wgsl: two mat4x4 and a vec2, padded to 144 bytes.
    this.cameraBuffer = device.createBuffer({
      label: 'camera',
      size: 144,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Uploads Gaussians packed by packGaussians(). Without spherical harmonics, each
   * Gaussian's color is the same from every direction.
   */
  setGaussians(data: Float32Array<ArrayBuffer>, sh?: SphericalHarmonics): void {
    const count = data.length / FLOATS_PER_GAUSSIAN;
    if (count === 0) {
      this.destroyScene();
      return;
    }
    if (sh && sh.rest.length !== count * SH_REST_FLOATS) {
      throw new Error(`Expected ${count * SH_REST_FLOATS} spherical-harmonic floats, got ${sh.rest.length}.`);
    }
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const gaussians = this.upload('gaussians', data, usage);
    // Without spherical harmonics the shader never reads this buffer, but the
    // binding needs one. Given coefficients are kept even at degree 0, because
    // training raises the degree as it goes.
    const shBuffer = this.upload('spherical harmonics', sh ? sh.rest : new Float32Array(4), usage);
    this.destroyScene();
    this.createScene(count, gaussians, shBuffer, sh ? sh.degree : 0, this.options.initialPairCapacity);
  }

  /**
   * Draws `count` Gaussians already on the GPU, taking over their buffers: packed
   * like packGaussians(), and SH_REST_FLOATS coefficients each. Training's
   * densification makes these. The tile buffers keep their size, since the next
   * frame will likely need as many pairs as the last.
   */
  adoptGaussians(count: number, gaussians: GPUBuffer, sh: GPUBuffer, degree: number): void {
    const pairCapacity = this.scene?.rasterizer.pairCapacity ?? this.options.initialPairCapacity;
    this.destroyScene();
    this.createScene(count, gaussians, sh, degree, pairCapacity);
  }

  /** The 2D splats from the last preprocess, FLOATS_PER_SPLAT floats each. For checks and training. */
  get splatBuffer(): GPUBuffer | undefined {
    return this.scene?.splats;
  }

  /** The Gaussians being drawn, packed like packGaussians(). Training writes them in place. */
  get gaussianBuffer(): GPUBuffer | undefined {
    return this.scene?.gaussians;
  }

  /** The spherical-harmonic coefficients being drawn, SH_REST_FLOATS per Gaussian. Training writes them in place. */
  get shBuffer(): GPUBuffer | undefined {
    return this.scene?.sh;
  }

  /** Changes how many spherical-harmonic degrees the renderer uses, 0 to 3. */
  setShDegree(degree: number): void {
    if (this.scene) this.device.queue.writeBuffer(this.scene.sceneParams, 0, new Uint32Array([degree, 0, 0, 0]));
  }

  /** The scene's tile rasterizer. For checks. */
  get tiles(): TileRasterizer | undefined {
    return this.scene?.rasterizer;
  }

  /** The last frame's pixels as unrounded floats: rgb and final transmittance, a vec4 per pixel. */
  get pixelBuffer(): GPUBuffer | undefined {
    return this.scene?.rasterizer.pixelColor;
  }

  /** The last frame's per-pixel splat counts (see TileRasterizer.pixelLast). For the Inside view. */
  get pixelLastBuffer(): GPUBuffer | undefined {
    return this.scene?.rasterizer.pixelLast;
  }

  /** The gradients from the last backward pass. */
  get gradients(): Gradients | undefined {
    return this.scene?.gradients;
  }

  /**
   * Records one frame: project the Gaussians, then rasterize them by tile over
   * `background`. `afterPreprocess` can record into the same pass in between; the
   * Inside view recolors the splats there.
   */
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    camera: CameraData,
    background: RGB,
    afterPreprocess?: (pass: GPUComputePassEncoder) => void,
  ): void {
    this.writeCamera(camera);
    this.lastViewport = camera.viewport;
    const scene = this.scene;
    if (!scene) {
      const [r, g, b] = background;
      encoder
        .beginRenderPass({ colorAttachments: [{ view: target, clearValue: { r, g, b, a: 1 }, loadOp: 'clear', storeOp: 'store' }] })
        .end();
      return;
    }
    scene.rasterizer.encode(encoder, target, camera.viewport, background, (pass) => {
      pass.setPipeline(this.preprocessPipeline);
      pass.setBindGroup(0, scene.preprocessBindGroup);
      pass.dispatchWorkgroups(Math.ceil(scene.count / WORKGROUP_SIZE));
      afterPreprocess?.(pass);
    });
  }

  /**
   * Draws ready-made 2D splats (packed by packSplats()) front to back by their
   * depth, skipping the preprocess. For checks.
   */
  encodeDraw(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    splats: GPUBuffer,
    count: number,
    viewport: [number, number],
    background: RGB,
  ): void {
    // The tile stage only reads the viewport from the camera.
    this.writeCamera({ view: IDENTITY, proj: IDENTITY, viewport });
    this.drawRasterizer?.destroy();
    this.drawRasterizer = new TileRasterizer(this.device, this.cameraBuffer, splats, count, this.options.initialPairCapacity);
    this.drawRasterizer.encode(encoder, target, viewport, background);
  }

  /**
   * Records the backward pass of the frame just encoded (M2). `pixelGrads` holds
   * dL/d(pixel rgb), a vec4 per pixel, row by row. The results land in `gradients`.
   */
  encodeBackward(encoder: GPUCommandEncoder, pixelGrads: GPUBuffer): void {
    const scene = this.scene;
    if (!scene) return;
    scene.gradients ??= this.createGradients(scene);
    scene.rasterizer.encodeBackward(encoder, this.lastViewport, pixelGrads, scene.gradients.splats);
    const pass = encoder.beginComputePass({ label: 'preprocess backward' });
    pass.setPipeline(this.backwardPipeline);
    pass.setBindGroup(0, scene.gradients.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(scene.count / WORKGROUP_SIZE));
    pass.end();
  }

  /** Call after submitting a frame: grows the tile buffers if the frame needed more room. */
  afterSubmit(): Promise<void> {
    return this.scene?.rasterizer.afterSubmit() ?? Promise.resolve();
  }

  destroy(): void {
    this.destroyScene();
    this.drawRasterizer?.destroy();
    this.cameraBuffer.destroy();
  }

  private upload(label: string, contents: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, usage: number): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size: contents.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, contents);
    return buffer;
  }

  private createScene(count: number, gaussians: GPUBuffer, shBuffer: GPUBuffer, degree: number, pairCapacity?: number): void {
    const sceneParams = this.upload('scene params', new Uint32Array([degree, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    // Written by the preprocess, read by the tile stage (and copied out by the checks).
    const splats = this.device.createBuffer({
      label: 'splats',
      size: count * FLOATS_PER_SPLAT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.scene = {
      count,
      gaussians,
      splats,
      sh: shBuffer,
      sceneParams,
      preprocessBindGroup: this.device.createBindGroup({
        label: 'preprocess',
        layout: this.preprocessPipeline.getBindGroupLayout(0),
        entries: [this.cameraBuffer, gaussians, splats, shBuffer, sceneParams].map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      }),
      rasterizer: new TileRasterizer(this.device, this.cameraBuffer, splats, count, pairCapacity),
    };
  }

  private writeCamera({ view, proj, viewport }: CameraData): void {
    const data = new Float32Array(36);
    data.set(view, 0);
    data.set(proj, 16);
    data.set(viewport, 32);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  private createGradients(scene: Scene): Gradients & { bindGroup: GPUBindGroup } {
    const buffer = (label: string, floats: number) =>
      this.device.createBuffer({
        label,
        size: scene.count * floats * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    const splats = buffer('splat gradients', 9);
    const gaussians = buffer('gaussian gradients', FLOATS_PER_GAUSSIAN);
    const sh = buffer('spherical-harmonic gradients', SH_REST_FLOATS);
    return {
      splats,
      gaussians,
      sh,
      bindGroup: this.device.createBindGroup({
        label: 'preprocess backward',
        layout: this.backwardPipeline.getBindGroupLayout(0),
        entries: [this.cameraBuffer, scene.gaussians, scene.sh, scene.sceneParams, scene.splats, splats, gaussians, sh].map(
          (resource, binding) => ({ binding, resource: { buffer: resource } }),
        ),
      }),
    };
  }

  private destroyScene(): void {
    const scene = this.scene;
    for (const buffer of [scene?.gaussians, scene?.splats, scene?.sh, scene?.sceneParams]) buffer?.destroy();
    for (const buffer of [scene?.gradients?.splats, scene?.gradients?.gaussians, scene?.gradients?.sh]) buffer?.destroy();
    scene?.rasterizer.destroy();
    this.scene = undefined;
  }
}
