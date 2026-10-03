import { PrefixSum } from './prefixSum';
import { RadixSort } from './radixSort';
import commonWgsl from './shaders/common.wgsl?raw';
import rasterizeWgsl from './shaders/rasterize.wgsl?raw';
import rasterizeBackwardWgsl from './shaders/rasterizeBackward.wgsl?raw';
import tilesWgsl from './shaders/tiles.wgsl?raw';

const TILE_SIZE = 16; // must match TILE_SIZE in common.wgsl
const WORKGROUP_SIZE = 256; // must match @workgroup_size in tiles.wgsl
const TILE_INDEX_BITS = 16; // how many bits of each pair's tile index the sort looks at

export type RGB = [number, number, number];

interface Pairs {
  capacity: number;
  sort: RadixSort;
  duplicateBindGroup: GPUBindGroup;
}

// The tile-based rasterizer (M1.5), following the reference:
// 1. count the 16×16-pixel tiles each 2D splat touches, and prefix-sum the counts
//    into slots;
// 2. write one (depth, tile) sort key per (splat, tile) pair into those slots,
//    and sort them;
// 3. find each tile's run of the sorted pairs;
// 4. blend each tile's pixels front to back, one workgroup per tile.
// It all runs on the GPU. The only readback is the pair count, which is used to
// grow the buffers when a frame needs more room.
export class TileRasterizer {
  /** How many (splat, tile) pairs the most recently read-back frame needed. */
  lastPairCount = 0;
  private readonly device: GPUDevice;
  private readonly camera: GPUBuffer;
  private readonly splats: GPUBuffer;
  private readonly count: number;
  private readonly countTilesPipeline: GPUComputePipeline;
  private readonly duplicatePipeline: GPUComputePipeline;
  private readonly findRangesPipeline: GPUComputePipeline;
  private readonly rasterizePipeline: GPUComputePipeline;
  private readonly backwardPipeline: GPUComputePipeline;
  private readonly tileOffsets: GPUBuffer;
  private readonly tileScan: PrefixSum;
  private readonly tileParams: GPUBuffer;
  private readonly rasterParams: GPUBuffer;
  private readonly readback: GPUBuffer;
  private readonly countTilesBindGroup: GPUBindGroup;
  private pairs: Pairs;
  private ranges?: GPUBuffer;
  private rangesTiles = 0;
  // Per pixel, for the loss and the backward pass: color and final transmittance
  // (vec4), and how far down its tile's list the last blended splat was (u32).
  private pixels?: { color: GPUBuffer; last: GPUBuffer; count: number };
  private readbackState: 'idle' | 'queued' | 'mapping' = 'idle';

  /**
   * Rasterizes `count` 2D splats from `splats`, reading the viewport from
   * `camera` (the Camera uniform in common.wgsl).
   */
  constructor(device: GPUDevice, camera: GPUBuffer, splats: GPUBuffer, count: number, initialPairCapacity?: number) {
    this.device = device;
    this.camera = camera;
    this.splats = splats;
    this.count = count;
    const tilesModule = device.createShaderModule({ label: 'tiles', code: `${commonWgsl}\n${tilesWgsl}` });
    const pipeline = (module: GPUShaderModule, entryPoint: string) =>
      device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
    this.countTilesPipeline = pipeline(tilesModule, 'count_tiles');
    this.duplicatePipeline = pipeline(tilesModule, 'duplicate');
    this.findRangesPipeline = pipeline(tilesModule, 'find_ranges');
    this.rasterizePipeline = pipeline(
      device.createShaderModule({ label: 'rasterize', code: `${commonWgsl}\n${rasterizeWgsl}` }),
      'rasterize',
    );
    this.backwardPipeline = pipeline(
      device.createShaderModule({ label: 'rasterize backward', code: `${commonWgsl}\n${rasterizeBackwardWgsl}` }),
      'rasterize_backward',
    );

    this.tileOffsets = device.createBuffer({ label: 'tile offsets', size: count * 4, usage: GPUBufferUsage.STORAGE });
    this.tileScan = new PrefixSum(device, this.tileOffsets, count);
    this.tileParams = device.createBuffer({
      label: 'tile params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.rasterParams = device.createBuffer({
      label: 'raster params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({
      label: 'pair count readback',
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.countTilesBindGroup = this.bindGroup(this.countTilesPipeline, { 0: camera, 1: splats, 2: this.tileOffsets });
    this.pairs = this.createPairs(initialPairCapacity ?? Math.max(1 << 16, count * 8));
  }

  get pairCapacity(): number {
    return this.pairs.capacity;
  }

  /** For checks: each sorted pair's splat, and each tile's [start, end) run of them. */
  get debugBuffers(): { pairSplats: GPUBuffer; ranges: GPUBuffer | undefined } {
    return { pairSplats: this.pairs.sort.values, ranges: this.ranges };
  }

  /**
   * The last frame's pixels as floats, row by row: rgb plus the transmittance
   * left at the end, as a vec4. Unlike the texture, it isn't clamped or rounded.
   */
  get pixelColor(): GPUBuffer | undefined {
    return this.pixels?.color;
  }

  /** The last frame's per-pixel count: how far down its tile's list each pixel's last blended splat was. */
  get pixelLast(): GPUBuffer | undefined {
    return this.pixels?.last;
  }

  /**
   * Records a frame. `before` records into the same compute pass first (the
   * renderer's preprocess). Draws into `target`: an rgba8unorm texture that
   * allows storage writes, `viewport` pixels in size.
   */
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    viewport: [number, number],
    background: RGB,
    before?: (pass: GPUComputePassEncoder) => void,
  ): void {
    const tilesX = Math.ceil(viewport[0] / TILE_SIZE);
    const tilesY = Math.ceil(viewport[1] / TILE_SIZE);
    if (tilesX * tilesY > 2 ** TILE_INDEX_BITS) {
      throw new Error(`A ${viewport.join('×')} viewport has more tiles than the sort can tell apart.`);
    }
    const ranges = this.rangesFor(tilesX * tilesY);
    const pixels = this.pixelsFor(viewport[0] * viewport[1]);
    encoder.clearBuffer(ranges); // tiles no splat touches keep an empty run
    this.device.queue.writeBuffer(this.rasterParams, 0, new Float32Array([...background, 1]));

    const pass = encoder.beginComputePass({ label: 'tiles' });
    before?.(pass);
    const splatGroups = Math.ceil(this.count / WORKGROUP_SIZE);
    pass.setPipeline(this.countTilesPipeline);
    pass.setBindGroup(0, this.countTilesBindGroup);
    pass.dispatchWorkgroups(splatGroups);
    this.tileScan.encode(pass);
    pass.setPipeline(this.duplicatePipeline);
    pass.setBindGroup(0, this.pairs.duplicateBindGroup);
    pass.dispatchWorkgroups(splatGroups);
    this.pairs.sort.encode(pass);
    pass.setPipeline(this.findRangesPipeline);
    pass.setBindGroup(
      0,
      this.bindGroup(this.findRangesPipeline, {
        3: this.tileScan.total,
        4: this.pairs.sort.keys,
        6: ranges,
        7: this.tileParams,
      }),
    );
    // WebGPU allows at most 65,535 workgroups along each dimension.
    const rangeGroups = Math.ceil(this.pairs.capacity / WORKGROUP_SIZE);
    const rangeRow = Math.min(rangeGroups, 65535);
    pass.dispatchWorkgroups(rangeRow, Math.ceil(rangeGroups / rangeRow));
    pass.setPipeline(this.rasterizePipeline);
    pass.setBindGroup(
      0,
      this.bindGroup(this.rasterizePipeline, {
        0: this.camera,
        1: this.rasterParams,
        2: this.splats,
        3: this.pairs.sort.values,
        4: ranges,
        5: target,
        6: pixels.color,
        7: pixels.last,
      }),
    );
    pass.dispatchWorkgroups(tilesX, tilesY);
    pass.end();

    // Read back how many pairs this frame needed, unless an earlier readback is
    // still in flight.
    if (this.readbackState === 'idle') {
      encoder.copyBufferToBuffer(this.tileScan.total, 0, this.readback, 0, 4);
      this.readbackState = 'queued';
    }
  }

  /**
   * Records the backward pass of the frame most recently encoded (M2): from
   * dL/d(pixel rgb) in `pixelGrads` (a vec4 per pixel, row by row) to each splat's
   * 9 gradients in `splatGrads`. It relies on that frame's tile lists and pixel
   * records, so nothing may be encoded in between.
   */
  encodeBackward(encoder: GPUCommandEncoder, viewport: [number, number], pixelGrads: GPUBuffer, splatGrads: GPUBuffer): void {
    const pixels = this.pixels;
    if (!pixels || !this.ranges) throw new Error('Encode a frame before its backward pass.');
    encoder.clearBuffer(splatGrads); // the backward pass adds into it
    const pass = encoder.beginComputePass({ label: 'rasterize backward' });
    pass.setPipeline(this.backwardPipeline);
    pass.setBindGroup(
      0,
      this.bindGroup(this.backwardPipeline, {
        0: this.camera,
        1: this.rasterParams,
        2: this.splats,
        3: this.pairs.sort.values,
        4: this.ranges,
        5: pixels.color,
        6: pixels.last,
        7: pixelGrads,
        8: splatGrads,
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(viewport[0] / TILE_SIZE), Math.ceil(viewport[1] / TILE_SIZE));
    pass.end();
  }

  /**
   * Call after submitting a frame. It reads back how many (splat, tile) pairs the
   * frame needed, and grows the pair buffers if they were too small. The frame
   * loop doesn't wait for it; checks do.
   */
  async afterSubmit(): Promise<void> {
    if (this.readbackState !== 'queued') return;
    this.readbackState = 'mapping';
    try {
      await this.readback.mapAsync(GPUMapMode.READ);
    } catch {
      return; // destroyed while the frame was in flight
    }
    this.lastPairCount = new Uint32Array(this.readback.getMappedRange())[0];
    this.readback.unmap();
    this.readbackState = 'idle';
    if (this.lastPairCount > this.pairs.capacity) {
      this.pairs.sort.destroy();
      this.pairs = this.createPairs(2 ** Math.ceil(Math.log2(this.lastPairCount * 1.25)));
    }
  }

  destroy(): void {
    this.pairs.sort.destroy();
    this.tileScan.destroy();
    for (const buffer of [
      this.tileOffsets,
      this.tileParams,
      this.rasterParams,
      this.readback,
      this.ranges,
      this.pixels?.color,
      this.pixels?.last,
    ]) {
      buffer?.destroy();
    }
  }

  private pixelsFor(count: number): { color: GPUBuffer; last: GPUBuffer; count: number } {
    if (!this.pixels || this.pixels.count < count) {
      this.pixels?.color.destroy();
      this.pixels?.last.destroy();
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
      this.pixels = {
        color: this.device.createBuffer({ label: 'pixel colors', size: count * 16, usage }),
        last: this.device.createBuffer({ label: 'pixel last splats', size: count * 4, usage }),
        count,
      };
    }
    return this.pixels;
  }

  private createPairs(capacity: number): Pairs {
    // Keys are (depth bits, tile index). Sorting the tile index as the high word
    // groups the pairs by tile, front to back within each tile.
    const sort = new RadixSort(this.device, capacity, { highBits: TILE_INDEX_BITS, count: this.tileScan.total });
    this.device.queue.writeBuffer(this.tileParams, 0, new Uint32Array([capacity, 0, 0, 0]));
    return {
      capacity,
      sort,
      duplicateBindGroup: this.bindGroup(this.duplicatePipeline, {
        0: this.camera,
        1: this.splats,
        2: this.tileOffsets,
        4: sort.keys,
        5: sort.values,
        7: this.tileParams,
      }),
    };
  }

  private rangesFor(tiles: number): GPUBuffer {
    if (!this.ranges || this.rangesTiles < tiles) {
      this.ranges?.destroy();
      this.ranges = this.device.createBuffer({
        label: 'tile ranges',
        size: tiles * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.rangesTiles = tiles;
    }
    return this.ranges;
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
