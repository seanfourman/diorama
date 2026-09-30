import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { packGaussians, unpackSplats, type Gaussian } from '../gaussians';
import { IDENTITY, lookAt, perspective } from '../mat4';
import { mulberry32 } from '../random';
import { trefoilKnot } from '../testScene';
import { maxDiff, pixelAt, readBuffer, renderPixels, verdict, type RGB } from './helpers';
import { blendPixel, tileCount, tileLists } from './reference';

// M1.5: render a random scene with the tile rasterizer, then:
// - compare every pixel with the same tile and blending math on the CPU;
// - compare every tile's sorted list of splats;
// - check that the pair buffers grow when a frame needs more room than they have.
// Also times a frame of the knot. Notes: docs/steps/1.5-tiles.md

// Not a multiple of 16, so the last row and column of tiles are partial.
const VIEWPORT: [number, number] = [120, 90];
const CAMERA: CameraData = { view: IDENTITY, proj: perspective(Math.PI / 2, 120 / 90, 0.1, 100), viewport: VIEWPORT };
const BACKGROUND: RGB = [0.1, 0.2, 0.3]; // not black, so the background term gets checked too
const SPLAT_COUNT = 300;
const INITIAL_CAPACITY = 256; // deliberately too small for the first frame
const TOLERANCE = 2.5 / 255;

const count = (n: number) => n.toLocaleString('en-US');

export async function runTilesCheck(device: GPUDevice): Promise<string> {
  const renderer = new GaussianRenderer(device, { initialPairCapacity: INITIAL_CAPACITY });
  renderer.setGaussians(packGaussians(randomScene(SPLAT_COUNT)));
  const frame = () =>
    renderPixels(device, VIEWPORT, (encoder, target) => renderer.encode(encoder, target, CAMERA, BACKGROUND));
  // The first frame needs more pairs than there's room for, and afterSubmit grows
  // the buffers. The second frame should then fit.
  await frame();
  await renderer.afterSubmit();
  const pixels = await frame();
  await renderer.afterSubmit();

  const tiles = renderer.tiles!;
  const pairCount = tiles.lastPairCount;
  const capacity = tiles.pairCapacity;
  const splats = unpackSplats(new Float32Array(await readBuffer(device, renderer.splatBuffer!)));
  const pairSplats = new Uint32Array(await readBuffer(device, tiles.debugBuffers.pairSplats));
  const ranges = new Uint32Array(await readBuffer(device, tiles.debugBuffers.ranges!));
  renderer.destroy();

  const growthFailures =
    capacity > INITIAL_CAPACITY && capacity >= pairCount
      ? []
      : [`  ${count(pairCount)} pairs needed, but the buffers hold ${count(capacity)}`];

  // The CPU works from the GPU's own 2D splats, so only the tile stage is under test.
  const lists = tileLists(splats, VIEWPORT);
  const listFailures: string[] = [];
  lists.forEach((cpu, tile) => {
    const gpu = Array.from(pairSplats.subarray(ranges[2 * tile], ranges[2 * tile + 1]));
    if (gpu.length !== cpu.length || gpu.some((index, k) => index !== cpu[k])) {
      listFailures.push(`  tile ${tile}: GPU [${gpu.join(', ')}], CPU [${cpu.join(', ')}]`);
    }
  });

  const [width, height] = VIEWPORT;
  const [tilesX, tilesY] = tileCount(VIEWPORT);
  let worst = 0;
  let wrongPixels = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const list = lists[Math.floor(y / 16) * tilesX + Math.floor(x / 16)];
      const difference = maxDiff(pixelAt(pixels, width, [x, y]), blendPixel(splats, list, [x, y], BACKGROUND));
      worst = Math.max(worst, difference);
      if (!(difference <= TOLERANCE)) wrongPixels++;
    }
  }
  const imageFailures = wrongPixels ? [`  ${count(wrongPixels)} pixels differ by more than ${TOLERANCE * 255}/255`] : [];

  const milliseconds = await timeKnotFrame(device);
  return [
    'M1.5: tiles',
    `Buffer growth: ${verdict(growthFailures)} (${count(INITIAL_CAPACITY)} → ${count(capacity)} slots for ${count(pairCount)} pairs)`,
    ...growthFailures,
    `Tile lists: ${verdict(listFailures)} (${tilesX * tilesY} tiles, ${count(pairCount)} pairs from ${SPLAT_COUNT} splats)`,
    ...listFailures.slice(0, 3),
    `Image: ${verdict(imageFailures)} (${count(width * height)} pixels, largest difference ${(worst * 255).toFixed(2)}/255)`,
    ...imageFailures,
    `Speed: the 2,000-splat knot at 1280×720 in ${milliseconds.toFixed(2)} ms a frame (median of 5, wall clock)`,
  ].join('\n');
}

// Gaussians scattered through the view, with random sizes, shapes, turns, colors
// and opacities.
function randomScene(splatCount: number): Gaussian[] {
  const random = mulberry32(11);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  return Array.from({ length: splatCount }, (): Gaussian => {
    const depth = between(2, 8);
    return {
      position: [between(-1.2, 1.2) * depth, between(-0.9, 0.9) * depth, -depth],
      scale: [between(0.02, 0.4), between(0.02, 0.4), between(0.02, 0.4)],
      rotation: [between(-1, 1), between(-1, 1), between(-1, 1), between(-1, 1)],
      color: [random(), random(), random()],
      opacity: between(0.2, 0.95),
    };
  });
}

async function timeKnotFrame(device: GPUDevice): Promise<number> {
  const [width, height] = [1280, 720];
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(packGaussians(trefoilKnot(2000)));
  const target = device.createTexture({
    label: 'timing target',
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING,
  });
  // Roughly the app's opening view: 3 units out, a little above.
  const camera: CameraData = {
    view: lookAt([1.49, 1.44, 2.17], [0, 0, 0], [0, 1, 0]),
    proj: perspective(Math.PI / 4, width / height, 0.05, 100),
    viewport: [width, height],
  };
  const times: number[] = [];
  for (let run = 0; run < 6; run++) {
    const encoder = device.createCommandEncoder();
    renderer.encode(encoder, target.createView(), camera, [0, 0, 0]);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (run > 0) times.push(performance.now() - start); // the first run is a warm-up
    await renderer.afterSubmit();
  }
  renderer.destroy();
  target.destroy();
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}
