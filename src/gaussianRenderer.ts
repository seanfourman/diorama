import { FLOATS_PER_GAUSSIAN, FLOATS_PER_SPLAT, SH_REST_FLOATS } from './gaussians';
import { IDENTITY, type Mat4 } from './mat4';
import commonWgsl from './shaders/common.wgsl?raw';
import preprocessWgsl from './shaders/preprocess.wgsl?raw';
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

interface Scene {
  count: number;
  gaussians: GPUBuffer;
  splats: GPUBuffer;
  sh: GPUBuffer;
  sceneParams: GPUBuffer;
  preprocessBindGroup: GPUBindGroup;
  rasterizer: TileRasterizer;
}

// Renders 3D Gaussians. Each frame, a compute pass projects every Gaussian to a 2D
// splat (M1.3). Then the tile rasterizer (M1.5) sorts the splats into 16×16-pixel
// tiles and blends every pixel front to back. It draws into an rgba8unorm texture
// that allows storage writes.
export class GaussianRenderer {
  private readonly device: GPUDevice;
  private readonly options: RendererOptions;
  private readonly preprocessPipeline: GPUComputePipeline;
  private readonly cameraBuffer: GPUBuffer;
  private scene?: Scene;
  // For encodeDraw: a rasterizer over 2D splats that a check made.
  private drawRasterizer?: TileRasterizer;

  constructor(device: GPUDevice, options: RendererOptions = {}) {
    this.device = device;
    this.options = options;
    this.preprocessPipeline = device.createComputePipeline({
      label: 'preprocess',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'preprocess', code: `${commonWgsl}\n${preprocessWgsl}` }),
        entryPoint: 'preprocess',
      },
    });
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
    this.destroyScene();
    const count = data.length / FLOATS_PER_GAUSSIAN;
    if (count === 0) return;
    if (sh && sh.rest.length !== count * SH_REST_FLOATS) {
      throw new Error(`Expected ${count * SH_REST_FLOATS} spherical-harmonic floats, got ${sh.rest.length}.`);
    }
    const upload = (label: string, contents: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, usage: number) => {
      const buffer = this.device.createBuffer({ label, size: contents.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, contents);
      return buffer;
    };
    const gaussians = upload('gaussians', data, GPUBufferUsage.STORAGE);
    const degree = sh ? sh.degree : 0;
    // Without spherical harmonics the shader never reads this buffer, but the binding needs one.
    const shBuffer = upload('spherical harmonics', degree > 0 && sh ? sh.rest : new Float32Array(4), GPUBufferUsage.STORAGE);
    const sceneParams = upload('scene params', new Uint32Array([degree, 0, 0, 0]), GPUBufferUsage.UNIFORM);
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
      rasterizer: new TileRasterizer(this.device, this.cameraBuffer, splats, count, this.options.initialPairCapacity),
    };
  }

  /** The 2D splats from the last preprocess, FLOATS_PER_SPLAT floats each. For checks. */
  get splatBuffer(): GPUBuffer | undefined {
    return this.scene?.splats;
  }

  /** The scene's tile rasterizer. For checks. */
  get tiles(): TileRasterizer | undefined {
    return this.scene?.rasterizer;
  }

  /** Records one frame: project the Gaussians, then rasterize them by tile over `background`. */
  encode(encoder: GPUCommandEncoder, target: GPUTextureView, camera: CameraData, background: RGB): void {
    this.writeCamera(camera);
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

  /** Call after submitting a frame: grows the tile buffers if the frame needed more room. */
  afterSubmit(): Promise<void> {
    return this.scene?.rasterizer.afterSubmit() ?? Promise.resolve();
  }

  destroy(): void {
    this.destroyScene();
    this.drawRasterizer?.destroy();
    this.cameraBuffer.destroy();
  }

  private writeCamera({ view, proj, viewport }: CameraData): void {
    const data = new Float32Array(36);
    data.set(view, 0);
    data.set(proj, 16);
    data.set(viewport, 32);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  private destroyScene(): void {
    this.scene?.gaussians.destroy();
    this.scene?.splats.destroy();
    this.scene?.sh.destroy();
    this.scene?.sceneParams.destroy();
    this.scene?.rasterizer.destroy();
    this.scene = undefined;
  }
}
