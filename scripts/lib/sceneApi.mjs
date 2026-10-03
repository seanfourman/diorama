// The local scene API (M5): what lets the browser turn photos into a scene. The
// browser can't run COLMAP, so the dev server does, through these routes (all
// under /api/scenes/, scene and file names checked strictly):
//   GET  /api/scenes                         what's in data/: each scene and what it has
//   PUT  /api/scenes/<name>/input/<file>     one photo, as the request body
//   DELETE /api/scenes/<name>/input          clear the uploaded photos, to start over
//   POST /api/scenes/<name>/reconstruct      run COLMAP on the photos (?matcher=sequential)
//   GET  /api/scenes/<name>/status           how the reconstruction is going
//   PUT  /api/scenes/<name>/result/<file>    a trained scene's point_cloud.ply or cameras.json
// vite.config.ts mounts it on the dev server, and scripts/check.mjs on its own.
import { existsSync } from 'node:fs';
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { cleanLogLine, reconstruct } from './reconstruct.mjs';

const DATA = path.join(import.meta.dirname, '..', '..', 'data');
const NAME = /^[\w-]{1,64}$/;
const PHOTO = /^[\w-]{1,100}\.(jpe?g|png)$/i;
const RESULTS = new Set(['point_cloud.ply', 'cameras.json']);
const MAX_UPLOAD = 1 << 30; // a trained .ply can be a few hundred MB

// One reconstruction at a time; COLMAP already uses every core.
const jobs = new Map();
let running = false;

/** Handles the request if it's an API route; returns false otherwise. */
export async function handleSceneApi(req, res) {
  const url = new URL(req.url, 'http://localhost');
  if (!url.pathname.startsWith('/api/scenes')) return false;
  const [, , , name, action, file] = url.pathname.split('/').map(decodeURIComponent);
  try {
    if (req.method === 'GET' && !name) return send(res, 200, await listScenes());
    if (!NAME.test(name ?? '')) return send(res, 400, { error: 'Scene names are letters, digits, - and _.' });
    const sceneDir = path.join(DATA, name);
    if (req.method === 'PUT' && action === 'input' && PHOTO.test(file ?? '')) {
      await mkdir(path.join(sceneDir, 'input'), { recursive: true });
      await writeFile(path.join(sceneDir, 'input', file), await readBody(req));
      return send(res, 200, { saved: file });
    }
    if (req.method === 'DELETE' && action === 'input' && !file) {
      if (running) return send(res, 409, { error: 'A reconstruction is running.' });
      await rm(path.join(sceneDir, 'input'), { recursive: true, force: true });
      return send(res, 200, { cleared: name });
    }
    if (req.method === 'PUT' && action === 'result' && RESULTS.has(file)) {
      await mkdir(sceneDir, { recursive: true });
      await writeFile(path.join(sceneDir, file), await readBody(req));
      return send(res, 200, { saved: file });
    }
    if (req.method === 'POST' && action === 'reconstruct') {
      if (running) return send(res, 409, { error: 'A reconstruction is already running.' });
      startReconstruction(name, sceneDir, url.searchParams.get('matcher') === 'sequential' ? 'sequential' : 'exhaustive');
      return send(res, 202, jobs.get(name));
    }
    if (req.method === 'GET' && action === 'status') {
      return send(res, 200, jobs.get(name) ?? { state: 'idle' });
    }
    return send(res, 404, { error: 'No such route.' });
  } catch (error) {
    return send(res, 500, { error: error.message });
  }
}

function startReconstruction(name, sceneDir, matcher) {
  const job = { state: 'running', matcher, stage: 'Starting', lines: [], started: Date.now() };
  jobs.set(name, job);
  running = true;
  reconstruct(sceneDir, {
    matcher,
    onStage: (stage) => (job.stage = stage),
    onLog: (line) => {
      job.lines.push(cleanLogLine(line));
      if (job.lines.length > 12) job.lines.shift();
    },
  })
    .then(({ placed, total, quality, warnings }) => Object.assign(job, { state: 'done', stage: 'Done', placed, total, quality, warnings }))
    .catch((error) => Object.assign(job, { state: 'error', error: error.message }))
    .finally(() => {
      running = false;
      job.seconds = Math.round((Date.now() - job.started) / 1000);
    });
}

async function listScenes() {
  if (!existsSync(DATA)) return [];
  const scenes = [];
  for (const entry of await readdir(DATA, { withFileTypes: true })) {
    if (!entry.isDirectory() || !NAME.test(entry.name)) continue;
    const has = (file) => existsSync(path.join(DATA, entry.name, file));
    scenes.push({
      name: entry.name,
      photos: has('input') ? (await readdir(path.join(DATA, entry.name, 'input'))).length : 0,
      trainable: has('sparse/0/images.bin') && has('images'),
      trained: has('point_cloud.ply') && has('cameras.json'),
    });
  }
  return scenes;
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_UPLOAD) throw new Error('Upload too large.');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function send(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
  return true;
}
