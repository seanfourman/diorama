import { turbo } from '../colormap';
import type { CameraData } from '../gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS, unpackSplats } from '../gaussians';
import { InsideView, WORK_RANGE, visibleDepthRange } from '../insideView';
import { IDENTITY, perspective } from '../mat4';
import { psnr } from '../metrics';
import { mulberry32 } from '../random';
import { readTexture } from '../readback';
import { DEFAULT_SETTINGS, Trainer, type TrainingView } from '../trainer';
import { readBuffer, verdict } from './helpers';

// M4: the Inside view, against the same math on the CPU.
// - The recolored splats: depth, the pull on each center, and densification's stamps.
// - The work heatmap, from the rasterizer's per-pixel counts.
// - The error heatmap and the compare view, letterboxed: a 64×32 photo on a 64×64 canvas.
// - PSNR measured on the GPU against metrics.ts.
// Notes: docs/steps/4-inside-view.md

const SIZE = 64;
const DEPTH_RANGE: [number, number] = [1, 20];
const STEP = 1500;
const TOLERANCE = 1.5 / 255; // 8-bit rounding, plus float differences

export async function runInsideCheck(device: GPUDevice): Promise<string> {
  const random = mulberry32(51);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  const count = 300;
  const params = new Float32Array(count * FLOATS_PER_GAUSSIAN);
  const stats = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    const depth = between(2, 15);
    params.set([between(-0.6, 0.6) * depth, between(-0.6, 0.6) * depth, -depth], at);
    params[at + 3] = between(-2, 3); // opacity logit
    for (let k = 0; k < 3; k++) {
      params[at + 4 + k] = Math.log(between(0.05, 0.4));
      params[at + 12 + k] = between(-1.5, 1.5);
    }
    params.set([between(-1, 1), between(-1, 1), between(-1, 1), between(-1, 1)], at + 8);
    // Densification's stamps: a third from the points, a third cloned, a third split, at various steps.
    params[at + 7] = i % 3;
    params[at + 15] = i % 3 ? between(0, STEP) : 0;
    if (i % 5) stats.set([between(0, 0.002) * 4, 4, 0, 0], i * 4); // every fifth never seen
  }

  // The photo: 64×32 random pixels, seen from a camera of the same shape.
  const photoCamera: CameraData = { view: IDENTITY, proj: perspective(Math.PI / 3, 2, 0.1, 100), viewport: [SIZE, SIZE / 2] };
  const photoPixels = Uint8Array.from({ length: SIZE * (SIZE / 2) * 4 }, () => Math.floor(random() * 256));
  const photo = device.createTexture({
    size: [SIZE, SIZE / 2],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  device.queue.writeTexture({ texture: photo }, photoPixels, { bytesPerRow: SIZE * 4 }, [SIZE, SIZE / 2]);
  const view: TrainingView = { name: 'photo', camera: photoCamera, image: photo };
  const trainer = new Trainer(device, { params, sh: new Float32Array(count * SH_REST_FLOATS) }, [view], {
    ...DEFAULT_SETTINGS,
    extent: 1,
  });
  device.queue.writeBuffer(trainer.buffers.stats, 0, stats);
  trainer.iteration = STEP;
  const inside = new InsideView(device, trainer.renderer, DEPTH_RANGE, trainer);
  const canvas = device.createTexture({
    size: [SIZE, SIZE],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const camera: CameraData = { view: IDENTITY, proj: perspective(Math.PI / 3, 1, 0.1, 100), viewport: [SIZE, SIZE] };
  const draw = async (mode: InsideView['mode'], withPhoto = false) => {
    inside.mode = mode;
    const encoder = device.createCommandEncoder();
    inside.encode(encoder, canvas, camera, [0, 0, 0], withPhoto ? view : undefined);
    device.queue.submit([encoder.finish()]);
    return readTexture(device, canvas);
  };

  // The recolored splats: the tile stage blends whatever colors the splats hold.
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const expectedColor = {
    depth: (depth: number) => turbo(1 - Math.log(depth / DEPTH_RANGE[0]) / Math.log(DEPTH_RANGE[1] / DEPTH_RANGE[0])),
    gradient: (i: number) => {
      const [sum, seen] = [stats[i * 4], stats[i * 4 + 1]];
      return seen > 0 ? turbo((Math.log10(Math.max(sum / seen / 0.0002, 1e-6)) + 2) / 3) : [0.25, 0.25, 0.25];
    },
    age: (i: number) => {
      const kind = params[i * FLOATS_PER_GAUSSIAN + 7];
      if (kind === 0) return [0.35, 0.35, 0.35];
      const base = kind === 1 ? [0.2, 0.95, 0.35] : [1, 0.5, 0.1];
      const glow = 1 / 3 + (2 / 3) * Math.exp(-(STEP - params[i * FLOATS_PER_GAUSSIAN + 15]) / 1000);
      return base.map((c) => c * glow);
    },
  };
  const colorFailures: string[] = [];
  let visible = 0;
  for (const mode of ['depth', 'gradient', 'age'] as const) {
    await draw(mode);
    const splats = unpackSplats(new Float32Array(await readBuffer(device, trainer.renderer.splatBuffer!)));
    visible = 0;
    splats.forEach((splat, i) => {
      if (splat.radius === 0) return;
      visible++;
      const want = mode === 'depth' ? expectedColor.depth(splat.depth) : expectedColor[mode](i);
      const opacity = sigmoid(params[i * FLOATS_PER_GAUSSIAN + 3]);
      const off = Math.max(...want.map((c, k) => Math.abs(c - splat.color[k])), Math.abs(splat.color[3] - opacity));
      if (!(off < 2e-4) && colorFailures.length < 5) colorFailures.push(`  ${mode}: splat ${i} is off by ${off.toExponential(1)}`);
    });
  }

  // Work: each pixel's count, through the log scale and the colormap.
  const workPixels = await draw('work');
  const counts = new Uint32Array(await readBuffer(device, trainer.renderer.pixelLastBuffer!));
  let workWorst = 0;
  let busiest = 0;
  for (let p = 0; p < SIZE * SIZE; p++) {
    busiest = Math.max(busiest, counts[p]);
    const want = turbo(Math.log2(Math.max(counts[p], 1) / WORK_RANGE[0]) / Math.log2(WORK_RANGE[1] / WORK_RANGE[0]));
    workWorst = Math.max(workWorst, ...want.map((c, k) => Math.abs(c - workPixels[p * 4 + k] / 255)));
  }
  const workFailures = workWorst <= TOLERANCE ? [] : [`  off by ${(workWorst * 255).toFixed(1)}/255`];

  // Error, letterboxed: the 64×32 photo fills rows 16 to 47.
  const errorPixels = await draw('error', true);
  const rendered = new Float32Array(await readBuffer(device, trainer.renderer.pixelBuffer!)).subarray(0, SIZE * (SIZE / 2) * 4);
  const quantized = rendered.map((v, k) => (k % 4 === 3 ? 0 : Math.round(Math.min(1, Math.max(0, v)) * 255) / 255));
  const photoFloats = Float32Array.from(photoPixels, (v, k) => (k % 4 === 3 ? 0 : v / 255));
  let errorWorst = 0;
  let backdropWrong = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const got = Array.from(errorPixels.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 3), (v) => v / 255);
      if (y < 16 || y >= 48) {
        // blit.wgsl's backdrop
        if (Math.abs(got[0] - 0.07) > TOLERANCE || Math.abs(got[2] - 0.12) > TOLERANCE) backdropWrong++;
        continue;
      }
      const p = (y - 16) * SIZE + x;
      const mean = [0, 1, 2].reduce((sum, k) => sum + Math.abs(quantized[p * 4 + k] - photoFloats[p * 4 + k]), 0) / 3;
      const want = turbo(mean / 0.2);
      errorWorst = Math.max(errorWorst, ...want.map((c, k) => Math.abs(c - got[k])));
    }
  }
  const errorFailures = [
    ...(errorWorst <= TOLERANCE ? [] : [`  heatmap off by ${(errorWorst * 255).toFixed(1)}/255`]),
    ...(backdropWrong ? [`  ${backdropWrong} letterbox pixels aren't the backdrop`] : []),
  ];

  // Compare: the render left of the middle, a white line, the photo right of it.
  const comparePixels = await draw('compare', true);
  const renderTexture = Uint8Array.from(quantized, (v) => Math.round(v * 255));
  let compareWrong = 0;
  for (let y = 16; y < 48; y++) {
    for (let x = 0; x < SIZE; x++) {
      const got = comparePixels.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 3);
      const source = x < 31 ? renderTexture : x > 32 ? photoPixels : undefined;
      const want = source ? source.subarray(((y - 16) * SIZE + x) * 4, ((y - 16) * SIZE + x) * 4 + 3) : [255, 255, 255];
      if (want.some((value, k) => Math.abs(value - got[k]) > 1)) compareWrong++;
    }
  }
  const compareFailures = compareWrong ? [`  ${compareWrong} pixels wrong`] : [];

  // PSNR: the GPU's per-tile sums against metrics.ts, for the same render and two
  // photos. The second is the render itself, off by up to 2 levels, so rounding
  // the render to 8 bits (as the reference measures) visibly matters.
  const secondPixels = renderTexture.map((v, k) => (k % 4 === 3 ? 255 : Math.min(255, Math.max(0, v + Math.floor(random() * 5) - 2))));
  const second = device.createTexture({ size: [SIZE, SIZE / 2], format: 'rgba8unorm', usage: photo.usage });
  device.queue.writeTexture({ texture: second }, secondPixels, { bytesPerRow: SIZE * 4 }, [SIZE, SIZE / 2]);
  const gpuPsnr = await inside.measurePsnr([view, { name: 'second', camera: photoCamera, image: second }], [0, 0, 0]);
  const cpuPsnr = (psnr(quantized, photoFloats) + psnr(quantized, Float32Array.from(secondPixels, (v, k) => (k % 4 === 3 ? 0 : v / 255)))) / 2;
  const psnrFailures = Math.abs(gpuPsnr - cpuPsnr) < 1e-3 ? [] : [`  GPU ${gpuPsnr.toFixed(4)} dB, CPU ${cpuPsnr.toFixed(4)} dB`];

  // The depth scale's fit: 100 points at depths 1 to 100 straight ahead, and some behind the camera.
  const points = new Float32Array(Array.from({ length: 110 }, (_, k) => [0, 0, k < 100 ? -(k + 1) : k]).flat());
  const range = visibleDepthRange(points, IDENTITY);
  const rangeFailures = range[0] === 6 && range[1] === 96 ? [] : [`  got [${range.join(', ')}], expected [6, 96]`];

  inside.destroy();
  trainer.destroy();
  for (const texture of [photo, second, canvas]) texture.destroy();
  return [
    'M4: the Inside view',
    `Recolored splats: ${verdict(colorFailures)} (depth, pull and densification for ${visible} visible splats, against the CPU)`,
    ...colorFailures,
    `Work heatmap: ${verdict(workFailures)} (${SIZE * SIZE} pixels; the busiest walked ${busiest} splats)`,
    ...workFailures,
    `Error heatmap: ${verdict(errorFailures)} (a 64×32 photo letterboxed on a 64×64 canvas; off by up to ${(errorWorst * 255).toFixed(2)}/255)`,
    ...errorFailures,
    `Compare: ${verdict(compareFailures)} (render left, photo right, a line between)`,
    ...compareFailures,
    `PSNR on the GPU: ${verdict(psnrFailures)} (${gpuPsnr.toFixed(4)} dB against metrics.ts's ${cpuPsnr.toFixed(4)} dB, ` +
      'averaged over a random photo and a near-copy of the render)',
    ...psnrFailures,
    `Depth scale: ${verdict(rangeFailures)} (the 5th and 95th percentile of the depths in front of the camera)`,
    ...rangeFailures,
  ].join('\n');
}
