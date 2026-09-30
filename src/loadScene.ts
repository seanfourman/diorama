import { parseGaussianPly, type GaussianScene } from './plyLoader';
import { parseCameras, type TrainingCamera } from './sceneCameras';

/**
 * Loads data/<name>/point_cloud.ply and data/<name>/cameras.json (see
 * scripts/download-scene.mjs), reporting progress as it goes.
 */
export async function loadScene(
  name: string,
  onProgress: (message: string) => void,
): Promise<GaussianScene & { cameras: TrainingCamera[] }> {
  const base = `data/${encodeURIComponent(name)}`;
  const camerasResponse = await fetch(`${base}/cameras.json`);
  if (!camerasResponse.ok) {
    throw new Error(`Couldn't load ${base}/cameras.json. Run "npm run download-scene -- ${name}" first.`);
  }
  const cameras = parseCameras(await camerasResponse.json());
  const ply = await fetchWithProgress(`${base}/point_cloud.ply`, (fraction) =>
    onProgress(`Loading ${name}… ${Math.round(fraction * 100)}%`),
  );
  onProgress(`Reading ${name}…`);
  return { ...parseGaussianPly(ply), cameras };
}

async function fetchWithProgress(url: string, onProgress: (fraction: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Couldn't load ${url} (HTTP ${response.status}).`);
  const total = Number(response.headers.get('content-length'));
  // Without a length, or with a compressed transfer, progress can't be measured.
  if (!response.body || !total || response.headers.get('content-encoding')) return response.arrayBuffer();
  const chunks: Uint8Array[] = [];
  let received = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / total));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes.buffer;
}
