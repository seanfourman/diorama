import prefixSumWgsl from './shaders/prefixSum.wgsl?raw';

const BLOCK = 512; // values per workgroup; must match SCAN_BLOCK in prefixSum.wgsl

// Exclusive prefix sum of u32s on the GPU, in place (M1.4). Each workgroup scans a
// block of 512 values and writes the block's total one level up. The totals are
// scanned the same way, recursively, and then added back down. Buffers and
// dispatches are sized for the capacity; the actual count can come from a GPU
// buffer (M1.5).
export class PrefixSum {
  /** Once the scan has run, element 0 holds the sum of all the values. */
  readonly total: GPUBuffer;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly addPipeline: GPUComputePipeline;
  // Level 0 scans `data`; level k + 1 scans level k's block totals.
  private readonly levels: { blocks: number; bindGroup: GPUBindGroup }[] = [];
  private readonly ownedBuffers: GPUBuffer[] = [];

  /**
   * Scans the first values of `data` in place. `count`, if given, holds in
   * element 0 how many (at most `capacity`); otherwise all `capacity` are scanned.
   */
  constructor(device: GPUDevice, data: GPUBuffer, capacity: number, count?: GPUBuffer) {
    const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
      const created = device.createBuffer({ label, size, usage });
      this.ownedBuffers.push(created);
      return created;
    };
    const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    const layout = device.createBindGroupLayout({
      label: 'prefix sum',
      entries: [entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'storage'), entry(3, 'storage')],
    });
    // One layout for both entry points, so each level's bind group works with either.
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = device.createShaderModule({ label: 'prefix sum', code: prefixSumWgsl });
    this.scanPipeline = device.createComputePipeline({
      label: 'scan blocks',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'scan_blocks' },
    });
    this.addPipeline = device.createComputePipeline({
      label: 'add block sums',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'add_block_sums' },
    });

    let scanCount = count;
    if (!scanCount) {
      scanCount = buffer('prefix sum count', 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(scanCount, 0, new Uint32Array([capacity]));
    }
    let levelData = data;
    let levelCapacity = capacity;
    for (;;) {
      const blocks = Math.max(1, Math.ceil(levelCapacity / BLOCK));
      const params = buffer('prefix sum params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0, new Uint32Array([this.levels.length, capacity, 0, 0]));
      const sums = buffer('prefix sum block totals', blocks * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      this.levels.push({
        blocks,
        bindGroup: device.createBindGroup({
          label: `prefix sum level ${this.levels.length}`,
          layout,
          entries: [params, scanCount, levelData, sums].map((resource, binding) => ({
            binding,
            resource: { buffer: resource },
          })),
        }),
      });
      if (blocks === 1) {
        this.total = sums;
        break;
      }
      levelData = sums;
      levelCapacity = blocks;
    }
  }

  /** Records the scan into an open compute pass. */
  encode(pass: GPUComputePassEncoder): void {
    // Up the levels: scan each level's blocks, collecting block totals for the next.
    for (const level of this.levels) {
      pass.setPipeline(this.scanPipeline);
      pass.setBindGroup(0, level.bindGroup);
      pass.dispatchWorkgroups(level.blocks);
    }
    // Back down: offset each block by the now-scanned total of the blocks before it.
    for (let k = this.levels.length - 2; k >= 0; k--) {
      pass.setPipeline(this.addPipeline);
      pass.setBindGroup(0, this.levels[k].bindGroup);
      pass.dispatchWorkgroups(this.levels[k].blocks);
    }
  }

  destroy(): void {
    this.ownedBuffers.forEach((buffer) => buffer.destroy());
  }
}
