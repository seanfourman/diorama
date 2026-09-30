// Downloads a trained 3D Gaussian Splatting scene into data/<name>/: the final
// point cloud (point_cloud.ply) and the training cameras (cameras.json).
// Usage: npm run download-scene [-- <name>]
//
// The scenes come from the original 3DGS release (Inria), mirrored on Hugging
// Face, and are covered by that release's license: research and personal use.
// data/ is git-ignored, so they never end up in the repository.
import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { once } from 'node:events';
import path from 'node:path';

const SCENES = {
  // Tanks and Temples "train": about 1.07 million Gaussians, 266 MB.
  train: 'https://huggingface.co/camenduru/gaussian-splatting/resolve/main/train',
};
const FILES = {
  'point_cloud.ply': 'point_cloud/iteration_30000/point_cloud.ply',
  'cameras.json': 'cameras.json',
};

const name = process.argv[2] ?? 'train';
const base = SCENES[name];
if (!base) {
  console.error(`Unknown scene "${name}". Available: ${Object.keys(SCENES).join(', ')}.`);
  process.exit(1);
}
const directory = path.join(import.meta.dirname, '..', 'data', name);
await mkdir(directory, { recursive: true });

for (const [file, remote] of Object.entries(FILES)) {
  const response = await fetch(`${base}/${remote}`);
  if (!response.ok) throw new Error(`${remote}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  const out = createWriteStream(path.join(directory, file));
  let received = 0;
  let nextReport = 0.1;
  for await (const chunk of response.body) {
    if (!out.write(chunk)) await once(out, 'drain');
    received += chunk.length;
    if (total && received / total >= nextReport) {
      console.log(`${file}: ${Math.round((100 * received) / total)}%`);
      nextReport += 0.1;
    }
  }
  out.end();
  await once(out, 'finish');
  console.log(`${file}: done, ${(received / 2 ** 20).toFixed(1)} MiB`);
}
console.log(`Saved to ${directory}`);
