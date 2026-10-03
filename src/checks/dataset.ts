import { loadDataset, loadViews, meanNeighborDistance2, initialGaussians } from '../loadDataset';
import { mulberry32 } from '../random';
import { readTexture } from '../readback';
import { parseCameras, verticalFov, type TrainingCamera } from '../sceneCameras';
import { activate } from '../trainer';
import { verdict } from './helpers';

// M3.1: reading a photo set for training.
// - Nearest neighbors on the GPU against the CPU, on random points.
// - The starting Gaussians built from them.
// - If data/train/sparse has been downloaded: the COLMAP cameras against the
//   cameras.json the reference wrote from the same reconstruction, and a photo
//   loaded onto the GPU.
// Notes: docs/steps/3-training.md

export async function runDatasetCheck(device: GPUDevice): Promise<string> {
  const { knnFailures, initFailures } = await checkNeighbors(device);
  return [
    'M3.1: training data',
    `Nearest neighbors: ${verdict(knnFailures)} (3,001 random points, GPU against CPU)`,
    ...knnFailures,
    `Starting Gaussians: ${verdict(initFailures)} (size, color, opacity and rotation from the points)`,
    ...initFailures,
    ...(await checkTrainPhotos(device)),
  ].join('\n');
}

async function checkNeighbors(device: GPUDevice): Promise<{ knnFailures: string[]; initFailures: string[] }> {
  // Not a multiple of the 256-point batches, and with one exact duplicate.
  const random = mulberry32(31);
  const count = 3001;
  const positions = Float32Array.from({ length: count * 3 }, () => random() * 4 - 2);
  positions.copyWithin(3 * 1000, 3 * 7, 3 * 8);
  const gpu = await meanNeighborDistance2(device, positions);
  const cpu = cpuNeighborDistance2(positions, count);
  const knnFailures: string[] = [];
  for (let i = 0; i < count && knnFailures.length < 5; i++) {
    if (!(Math.abs(gpu[i] - cpu[i]) <= 1e-5 * cpu[i] + 1e-9)) knnFailures.push(`  point ${i}: GPU ${gpu[i]}, CPU ${cpu[i]}`);
  }

  const colors = Uint8Array.from({ length: count * 3 }, () => Math.floor(random() * 256));
  const raw = await initialGaussians(device, positions, colors);
  const activated = activate(raw.params);
  const initFailures: string[] = [];
  for (let i = 0; i < count && initFailures.length < 5; i++) {
    const at = i * 16;
    const scale = Math.sqrt(Math.max(cpu[i], 1e-7));
    const problems = [
      ...[0, 1, 2].map((k) => Math.abs(activated[at + k] - positions[i * 3 + k]) > 1e-6 && `position ${k}`),
      ...[0, 1, 2].map((k) => Math.abs(activated[at + 4 + k] - scale) > 1e-4 * scale && `scale ${k}`),
      ...[0, 1, 2].map((k) => Math.abs(activated[at + 12 + k] - colors[i * 3 + k] / 255) > 1e-5 && `color ${k}`),
      Math.abs(activated[at + 3] - 0.1) > 1e-6 && 'opacity',
      [1, 0, 0, 0].some((q, k) => activated[at + 8 + k] !== q) && 'rotation',
    ].filter(Boolean);
    if (problems.length) initFailures.push(`  Gaussian ${i}: wrong ${problems.join(', ')}`);
  }
  return { knnFailures, initFailures };
}

function cpuNeighborDistance2(positions: Float32Array, count: number): Float64Array {
  const result = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const best = [Infinity, Infinity, Infinity];
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const dx = positions[j * 3] - positions[i * 3];
      const dy = positions[j * 3 + 1] - positions[i * 3 + 1];
      const dz = positions[j * 3 + 2] - positions[i * 3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best[2]) {
        best[2] = d2;
        best.sort((a, b) => a - b);
      }
    }
    result[i] = (best[0] + best[1] + best[2]) / 3;
  }
  return result;
}

async function checkTrainPhotos(device: GPUDevice): Promise<string[]> {
  let dataset;
  try {
    dataset = await loadDataset('train');
  } catch {
    return ['Photos: SKIP (data/train/sparse is missing; npm run download-photos fetches it)'];
  }
  const lines: string[] = [];

  // The reference's cameras.json came from the same COLMAP reconstruction, so the
  // poses must match to rounding.
  const response = await fetch('data/train/cameras.json');
  if (response.ok) {
    const reference = new Map(parseCameras(await response.json()).map((camera) => [camera.name, camera]));
    const all = [...dataset.train, ...dataset.test];
    const failures: string[] = [];
    let compared = 0;
    for (const camera of all) {
      const expected = reference.get(camera.name.replace(/\.[^.]+$/, ''));
      if (!expected) continue;
      compared++;
      const worst = Math.max(
        ...(['position', 'right', 'down', 'forward'] as const).flatMap((key) =>
          camera[key].map((value, k) => Math.abs(value - expected[key][k])),
        ),
        Math.abs(verticalFov(camera) - verticalFov(expected)),
      );
      if (!(worst < 1e-6) && failures.length < 5) failures.push(`  ${camera.name}: off by ${worst.toExponential(1)}`);
    }
    if (compared !== all.length) failures.push(`  only ${compared} of ${all.length} photos are in cameras.json`);
    lines.push(
      `COLMAP cameras: ${verdict(failures)} (${all.length} photos, poses and fields of view against the reference's cameras.json)`,
      ...failures,
    );
  }

  // One photo, through to the GPU.
  const photo: TrainingCamera = dataset.train[0];
  const [view] = await loadViews(device, 'train', [photo]);
  const pixels = await readTexture(device, view.image);
  view.image.destroy();
  const [width, height] = view.camera.viewport;
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
  const mean = sum / (3 * width * height * 255);
  const aspectError = Math.abs(width / height - photo.width / photo.height) / (photo.width / photo.height);
  const photoFailures = [
    ...(aspectError < 0.005 ? [] : [`  the photo is ${width}×${height}, but COLMAP saw ${photo.width}×${photo.height}`]),
    ...(mean > 0.05 && mean < 0.95 ? [] : [`  the photo's mean brightness is ${mean.toFixed(3)}`]),
  ];
  const count = (n: number) => n.toLocaleString('en-US');
  lines.push(
    `Photos: ${verdict(photoFailures)} ("train": ${dataset.train.length} to train on, ${dataset.test.length} held out, ` +
      `extent ${dataset.extent.toFixed(2)}; ${photo.name} is ${width}×${height})`,
    ...photoFailures,
  );

  const start = performance.now();
  const distance2 = await meanNeighborDistance2(device, dataset.positions);
  const milliseconds = performance.now() - start;
  const sizes = Array.from(distance2, Math.sqrt).sort((a, b) => a - b);
  lines.push(
    `Speed: nearest neighbors of all ${count(sizes.length)} points in ${milliseconds.toFixed(0)} ms ` +
      `(median starting size ${sizes[sizes.length >> 1].toFixed(4)})`,
  );
  return lines;
}
