import { PrefixSum } from './prefixSum';
import radixSortWgsl from './shaders/radixSort.wgsl?raw';

const BLOCK_SIZE = 1024; // keys per workgroup; must match BLOCK_SIZE in radixSort.wgsl
const RADIX = 256; // 8-bit digits

// The key type, prepended to radixSort.wgsl: 32-bit keys, or 64-bit ones as
// (low word, high word).
const KEY_32 = 'alias Key = u32;\nfn key_word(key: Key, word: u32) -> u32 { return key; }\n';
const KEY_64 = 'alias Key = vec2<u32>;\nfn key_word(key: Key, word: u32) -> u32 { return key[word]; }\n';

export interface RadixSortOptions {
  /**
   * Sort 64-bit keys, stored as (low, high) pairs of u32s, looking at this many
   * low bits of the high word (at most 32). Without it, keys are single u32s.
   */
  highBits?: number;
  /** Holds in element 0 how many keys to sort (at most the capacity). Without it, all of them are sorted. */
  count?: GPUBuffer;
}

// Stable GPU radix sort of 32- or 64-bit keys, each carrying a 32-bit value (M1.4,
// 64-bit keys since M1.5). Each pass sorts by one 8-bit digit, least significant
// first: count each digit per block of keys, prefix-sum the counts into output
// positions, then scatter. The passes only add up to a full sort because each one
// is stable.
export class RadixSort {
  readonly capacity: number;
  /** The keys to sort. They end up sorted in place. */
  readonly keys: GPUBuffer;
  /** One value per key, moved along with it: typically the key's original index. */
  readonly values: GPUBuffer;
  private readonly blocks: number;
  private readonly countPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly passBindGroups: GPUBindGroup[] = [];
  private readonly prefixSum: PrefixSum;
  private readonly ownedBuffers: GPUBuffer[] = [];

  constructor(device: GPUDevice, capacity: number, { highBits = 0, count }: RadixSortOptions = {}) {
    this.capacity = capacity;
    this.blocks = Math.ceil(capacity / BLOCK_SIZE);
    const wide = highBits > 0;
    const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
      const created = device.createBuffer({ label, size, usage });
      this.ownedBuffers.push(created);
      return created;
    };
    const keyBytes = wide ? 8 : 4;
    const ioUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.keys = buffer('sort keys', capacity * keyBytes, ioUsage);
    this.values = buffer('sort values', capacity * 4, ioUsage);
    // Each pass reads one pair of buffers and writes the other.
    const scratchKeys = buffer('sort scratch keys', capacity * keyBytes, GPUBufferUsage.STORAGE);
    const scratchValues = buffer('sort scratch values', capacity * 4, GPUBufferUsage.STORAGE);
    // counts[digit × blocks + block]. Digit-major, so that one prefix sum over the
    // whole array turns every count into an output position.
    const counts = buffer('digit counts', RADIX * this.blocks * 4, GPUBufferUsage.STORAGE);
    const countsLength = buffer('digit counts length', 4, GPUBufferUsage.STORAGE);
    let sortCount = count;
    if (!sortCount) {
      sortCount = buffer('sort count', 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(sortCount, 0, new Uint32Array([capacity]));
    }

    const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    const layout = device.createBindGroupLayout({
      label: 'radix sort',
      entries: [
        entry(0, 'uniform'),
        entry(1, 'read-only-storage'),
        entry(2, 'read-only-storage'),
        entry(3, 'read-only-storage'),
        entry(4, 'storage'),
        entry(5, 'storage'),
        entry(6, 'storage'),
        entry(7, 'storage'),
      ],
    });
    // One layout for both entry points, so each pass's bind group works with either.
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = device.createShaderModule({
      label: 'radix sort',
      code: (wide ? KEY_64 : KEY_32) + radixSortWgsl,
    });
    this.countPipeline = device.createComputePipeline({
      label: 'count digits',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'count_digits' },
    });
    this.scatterPipeline = device.createComputePipeline({
      label: 'scatter',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'scatter' },
    });

    // Four passes over the low word, then two per 16 bits of the high word, which
    // keeps the total even so the result lands back in `keys` and `values`.
    const passes: { word: number; shift: number }[] = [0, 8, 16, 24].map((shift) => ({ word: 0, shift }));
    for (let p = 0; p < 2 * Math.ceil(highBits / 16); p++) passes.push({ word: 1, shift: p * 8 });
    passes.forEach(({ word, shift }, passIndex) => {
      const params = buffer('sort params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(params, 0, new Uint32Array([shift, word, capacity, 0]));
      const [keysIn, valuesIn, keysOut, valuesOut] =
        passIndex % 2 === 0
          ? [this.keys, this.values, scratchKeys, scratchValues]
          : [scratchKeys, scratchValues, this.keys, this.values];
      this.passBindGroups.push(
        device.createBindGroup({
          label: `radix sort pass ${passIndex}`,
          layout,
          entries: [params, sortCount, keysIn, valuesIn, keysOut, valuesOut, counts, countsLength].map(
            (resource, binding) => ({ binding, resource: { buffer: resource } }),
          ),
        }),
      );
    });
    this.prefixSum = new PrefixSum(device, counts, RADIX * this.blocks, countsLength);
  }

  /** Records the sort into an open compute pass. The result lands back in `keys` and `values`. */
  encode(pass: GPUComputePassEncoder): void {
    for (const bindGroup of this.passBindGroups) {
      pass.setPipeline(this.countPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(this.blocks);
      this.prefixSum.encode(pass);
      pass.setPipeline(this.scatterPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(this.blocks);
    }
  }

  destroy(): void {
    this.prefixSum.destroy();
    this.ownedBuffers.forEach((buffer) => buffer.destroy());
  }
}
