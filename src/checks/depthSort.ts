import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { packGaussians, type Gaussian } from '../gaussians';
import { IDENTITY, perspective, type Vec3 } from '../mat4';
import { PrefixSum } from '../prefixSum';
import { RadixSort } from '../radixSort';
import { mulberry32 } from '../random';
import { formatRgb, maxDiff, pixelAt, readBuffer, renderPixels, verdict, type RGB } from './helpers';
import { preprocess, shadePixel } from './reference';

// M1.4: check the GPU prefix sum and radix sort against the CPU, check that two
// overlapping Gaussians listed near-first are drawn with the near one on top, and
// time a million-key sort. Notes: docs/steps/1.4-depth-sort.md

const SCAN_SIZE = 300_000; // enough for three levels of 512-value blocks
const SORT_SIZES = [1, 1000, 1024, 100_003]; // one key, a partial block, a full block, many blocks
const BENCHMARK_SIZE = 1_000_000;
const SIZE = 64;
const TOLERANCE = 2.5 / 255;
const BLACK: RGB = [0, 0, 0];
// The same camera as the M1.3 check: 90° field of view, so a focal length of 32 px.
const CAMERA: CameraData = { view: IDENTITY, proj: perspective(Math.PI / 2, 1, 0.1, 100), viewport: [SIZE, SIZE] };

const count = (n: number) => n.toLocaleString('en-US');

export async function runDepthSortCheck(device: GPUDevice): Promise<string> {
  const scanFailure = await checkPrefixSum(device);
  const sortFailures: string[] = [];
  for (const size of SORT_SIZES) {
    const failure = await checkSort(device, size);
    if (failure) sortFailures.push(`  ${count(size)} keys: ${failure}`);
  }
  const orderFailures = await checkDrawOrder(device);
  const milliseconds = await timeSort(device, BENCHMARK_SIZE);
  return [
    'M1.4: depth sort',
    `Prefix sum: ${scanFailure ? 'FAIL' : 'PASS'} (${count(SCAN_SIZE)} values)`,
    ...(scanFailure ? [`  ${scanFailure}`] : []),
    `Radix sort: ${verdict(sortFailures)} (${SORT_SIZES.map(count).join(', ')} keys, against a stable CPU sort)`,
    ...sortFailures,
    `Draw order: ${verdict(orderFailures)}`,
    ...orderFailures,
    `Speed: ${count(BENCHMARK_SIZE)} keys sorted in ${milliseconds.toFixed(2)} ms (median of 5, wall clock)`,
  ].join('\n');
}

async function checkPrefixSum(device: GPUDevice): Promise<string | null> {
  const random = mulberry32(3);
  const input = Uint32Array.from({ length: SCAN_SIZE }, () => Math.floor(random() * 10));
  const buffer = device.createBuffer({
    label: 'prefix sum check',
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, input);
  const scan = new PrefixSum(device, buffer, SCAN_SIZE);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  scan.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const output = new Uint32Array(await readBuffer(device, buffer));
  scan.destroy();
  buffer.destroy();

  let sum = 0;
  for (let i = 0; i < SCAN_SIZE; i++) {
    if (output[i] !== sum) return `position ${i}: GPU ${output[i]}, CPU ${sum}`;
    sum += input[i];
  }
  return null;
}

async function checkSort(device: GPUDevice, size: number): Promise<string | null> {
  const random = mulberry32(size);
  // Half the keys use all 32 bits. The other half repeat a few small values, so
  // there are plenty of equal keys to show whether the sort keeps them in order.
  const keys = Uint32Array.from({ length: size }, (_, i) =>
    i % 2 === 0 ? Math.floor(random() * 2 ** 32) : Math.floor(random() * 16),
  );
  const indices = Uint32Array.from({ length: size }, (_, i) => i);
  const sort = new RadixSort(device, size);
  device.queue.writeBuffer(sort.keys, 0, keys);
  device.queue.writeBuffer(sort.values, 0, indices);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sort.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const gpuKeys = new Uint32Array(await readBuffer(device, sort.keys));
  const gpuIndices = new Uint32Array(await readBuffer(device, sort.values));
  sort.destroy();

  // A stable CPU sort: by key, with ties kept in their original order.
  const expected = Array.from(indices).sort((a, b) => keys[a] - keys[b] || a - b);
  for (let i = 0; i < size; i++) {
    if (gpuIndices[i] !== expected[i] || gpuKeys[i] !== keys[expected[i]]) {
      return (
        `position ${i}: GPU has key ${gpuKeys[i]} from index ${gpuIndices[i]}, ` +
        `CPU has key ${keys[expected[i]]} from index ${expected[i]}`
      );
    }
  }
  return null;
}

async function checkDrawOrder(device: GPUDevice): Promise<string[]> {
  const round = (position: Vec3, scale: number, color: Vec3): Gaussian => ({
    position,
    scale: [scale, scale, scale],
    rotation: [1, 0, 0, 0],
    color,
    opacity: 0.8,
  });
  // Listed near-first, so drawing them in list order would put the far blue one
  // on top. Both are σ = 3.2 px on screen (0.3 at depth 3, 0.5 at depth 5).
  const gaussians = [
    round([0, 0, -3], 0.3, [1, 0, 0]), // near, red
    round([0, 0, 2], 0.3, [0, 1, 0]), // behind the camera, so culled
    round([0, 0, -5], 0.5, [0, 0, 1]), // far, blue
  ];
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(packGaussians(gaussians));
  const pixels = await renderPixels(device, SIZE, (encoder, target) => renderer.encode(encoder, target, CAMERA, BLACK));
  renderer.destroy();

  // shadePixel orders them by depth itself, like the GPU.
  const splats = gaussians.map((gaussian) => preprocess(gaussian, CAMERA)).filter((splat) => splat !== null);
  const failures: string[] = [];
  const probes: [number, number][] = [
    [32, 32],
    [34, 30],
    [29, 35],
  ];
  for (const pixel of probes) {
    const gpu = pixelAt(pixels, SIZE, pixel);
    const cpu = shadePixel(splats, pixel, [SIZE, SIZE]);
    if (maxDiff(gpu, cpu) > TOLERANCE) {
      failures.push(`  pixel (${pixel.join(', ')}): GPU ${formatRgb(gpu)}  CPU ${formatRgb(cpu)}`);
    }
  }
  const [red, , blue] = pixelAt(pixels, SIZE, [32, 32]);
  if (!(red > blue)) {
    failures.push(`  the far blue splat is on top (red ${red.toFixed(3)}, blue ${blue.toFixed(3)} at the center)`);
  }
  return failures;
}

async function timeSort(device: GPUDevice, size: number): Promise<number> {
  const random = mulberry32(7);
  const keys = Uint32Array.from({ length: size }, () => Math.floor(random() * 2 ** 32));
  const sort = new RadixSort(device, size);
  const times: number[] = [];
  for (let run = 0; run < 6; run++) {
    device.queue.writeBuffer(sort.keys, 0, keys);
    await device.queue.onSubmittedWorkDone(); // keep the upload out of the timing
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    sort.encode(pass);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (run > 0) times.push(performance.now() - start); // the first run is a warm-up
  }
  sort.destroy();
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}
