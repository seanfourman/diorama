import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { FLOATS_PER_SPLAT, packGaussians, packSplats, unpackSplats, type Gaussian, type Splat2D } from '../gaussians';
import { IDENTITY, lookAt, perspective } from '../mat4';
import { mulberry32 } from '../random';
import { trefoilKnot } from '../testScene';
import { maxDiff, pixelAt, readBuffer, renderPixels, verdict, type RGB } from './helpers';
import { blendPixel, tileCount, tileLists } from './reference';

// M1.5: render a random scene with the tile rasterizer, then:
// - compare every pixel with the same tile and blending math on the CPU;
// - compare every tile's sorted list of splats;
// - check that the pair buffers grow when a frame needs more room than they have.
// Then the same pixel comparison for a dense, opaque scene, where most pixels
// stop blending early, and for a tile built so that all but 32 of its pixels
// stop early: the tile may only stop loading splats once every pixel has.
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
  const image = compareWithCpu(pixels, splats, lists);
  const opaque = await checkOpaqueScene(device);
  const earlyExit = await checkEarlyExit(device);

  const milliseconds = await timeKnotFrame(device);
  return [
    'M1.5: tiles',
    `Buffer growth: ${verdict(growthFailures)} (${count(INITIAL_CAPACITY)} → ${count(capacity)} slots for ${count(pairCount)} pairs)`,
    ...growthFailures,
    `Tile lists: ${verdict(listFailures)} (${tilesX * tilesY} tiles, ${count(pairCount)} pairs from ${SPLAT_COUNT} splats)`,
    ...listFailures.slice(0, 3),
    `Image: ${verdict(image.failures)} (${count(width * height)} pixels, largest difference ${(image.worst * 255).toFixed(2)}/255)`,
    ...image.failures,
    `Opaque image: ${verdict(opaque.failures)} (${opaque.summary}; largest difference ${(opaque.worst * 255).toFixed(2)}/255)`,
    ...opaque.failures,
    `Tile early exit: ${verdict(earlyExit.failures)} (224 pixels done in the first batch, 32 blending through three; ` +
      `largest difference ${(earlyExit.worst * 255).toFixed(2)}/255)`,
    ...earlyExit.failures,
    `Speed: the 2,000-splat knot at 1280×720 in ${milliseconds.toFixed(2)} ms a frame (median of 5, wall clock)`,
  ].join('\n');
}

// Every pixel against the CPU's blend of its tile's list.
function compareWithCpu(
  pixels: Uint8Array,
  splats: Splat2D[],
  lists: number[][],
  viewport = VIEWPORT,
): { failures: string[]; worst: number } {
  const [width, height] = viewport;
  const [tilesX] = tileCount(viewport);
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
  return { failures: wrongPixels ? [`  ${count(wrongPixels)} pixels differ by more than ${TOLERANCE * 255}/255`] : [], worst };
}

// 5,000 large, nearly opaque Gaussians: tiles hold several batches of splats,
// and most pixels are done long before their list ends.
async function checkOpaqueScene(device: GPUDevice): Promise<{ failures: string[]; worst: number; summary: string }> {
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(packGaussians(randomScene(5000, { scale: [0.1, 0.6], opacity: [0.9, 0.99] })));
  await renderPixels(device, VIEWPORT, (encoder, target) => renderer.encode(encoder, target, CAMERA, BACKGROUND));
  await renderer.afterSubmit(); // grow the pair buffers, then render again
  const pixels = await renderPixels(device, VIEWPORT, (encoder, target) => renderer.encode(encoder, target, CAMERA, BACKGROUND));
  const splats = unpackSplats(new Float32Array(await readBuffer(device, renderer.splatBuffer!)));
  const transmittance = new Float32Array(await readBuffer(device, renderer.pixelBuffer!));
  renderer.destroy();
  const lists = tileLists(splats, VIEWPORT);
  const { failures, worst } = compareWithCpu(pixels, splats, lists);
  let opaque = 0;
  for (let p = 0; p < VIEWPORT[0] * VIEWPORT[1]; p++) if (transmittance[p * 4 + 3] < 0.01) opaque++;
  const longest = Math.max(...lists.map((list) => list.length));
  return { failures, worst, summary: `${count(opaque)} pixels nearly opaque; tile lists up to ${count(longest)} splats` };
}

// One tile, drawn from hand-made 2D splats. In front: 4 thin, opaque splats on
// each of columns 0 to 13, which finish those pixels within the first batch.
// Behind: 600 broad, faint ones that columns 14 and 15 keep blending through
// three batches without ever becoming opaque.
async function checkEarlyExit(device: GPUDevice): Promise<{ failures: string[]; worst: number }> {
  const random = mulberry32(13);
  const splats: Splat2D[] = [];
  for (let column = 0; column < 14; column++) {
    for (let k = 0; k < 4; k++) {
      splats.push({
        mean: [column + 0.5, 8],
        depth: 1 + splats.length * 0.001,
        radius: 24,
        conic: [1 / 0.25 ** 2, 0, 1 / 100 ** 2], // 0.25 pixels wide, 100 tall
        color: [random(), random(), random(), 0.99],
      });
    }
  }
  for (let k = 0; k < 600; k++) {
    splats.push({
      mean: [random() * 16, random() * 16],
      depth: 5 + k * 0.001,
      radius: 60,
      conic: [1 / 20 ** 2, 0, 1 / 20 ** 2],
      color: [random(), random(), random(), 0.01],
    });
  }
  const viewport: [number, number] = [16, 16];
  const buffer = device.createBuffer({ size: splats.length * FLOATS_PER_SPLAT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, packSplats(splats));
  const renderer = new GaussianRenderer(device, { initialPairCapacity: 1024 });
  const pixels = await renderPixels(device, viewport, (encoder, target) =>
    renderer.encodeDraw(encoder, target, buffer, splats.length, viewport, BACKGROUND),
  );
  renderer.destroy();
  buffer.destroy();
  return compareWithCpu(pixels, splats, tileLists(splats, viewport), viewport);
}

// Gaussians scattered through the view, with random sizes, shapes, turns, colors
// and opacities.
function randomScene(
  splatCount: number,
  ranges: { scale: [number, number]; opacity: [number, number] } = { scale: [0.02, 0.4], opacity: [0.2, 0.95] },
): Gaussian[] {
  const random = mulberry32(11);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  const [smallest, largest] = ranges.scale;
  return Array.from({ length: splatCount }, (): Gaussian => {
    const depth = between(2, 8);
    return {
      position: [between(-1.2, 1.2) * depth, between(-0.9, 0.9) * depth, -depth],
      scale: [between(smallest, largest), between(smallest, largest), between(smallest, largest)],
      rotation: [between(-1, 1), between(-1, 1), between(-1, 1), between(-1, 1)],
      color: [random(), random(), random()],
      opacity: between(...ranges.opacity),
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
