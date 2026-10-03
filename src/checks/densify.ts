import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS } from '../gaussians';
import { IDENTITY } from '../mat4';
import { mulberry32 } from '../random';
import { DEFAULT_SETTINGS, Trainer, activate } from '../trainer';
import { readBuffer, verdict } from './helpers';

// M3.5: densification and pruning, on the GPU, against the reference's rules on
// the CPU. 2,000 Gaussians cover every case: seen or never seen, a gradient
// above or below the threshold, small enough to clone or big enough to split,
// faint, or too big. Also the opacity reset, and what densification records for
// the Inside view (M4): its tally, and each new Gaussian's kind and step.
// Notes: docs/steps/3-training.md

const COUNT = 2000;
const SLOTS = FLOATS_PER_GAUSSIAN + SH_REST_FLOATS;
const EXTENT = 1; // so clones are at most 0.01 in size, and anything over 0.1 is too big
const STEP = 700; // the step densification runs at, which it stamps on new Gaussians

export async function runDensifyCheck(device: GPUDevice): Promise<string> {
  const lines = ['M3.5: densification'];
  for (const pruneLarge of [false, true]) {
    const { failures, summary } = await checkDensify(device, pruneLarge);
    lines.push(`${pruneLarge ? 'With' : 'Without'} the size limit: ${verdict(failures)} (${summary})`, ...failures);
  }
  return lines.join('\n');
}

async function checkDensify(device: GPUDevice, pruneLarge: boolean): Promise<{ failures: string[]; summary: string }> {
  const random = mulberry32(pruneLarge ? 41 : 43);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  const params = new Float32Array(COUNT * FLOATS_PER_GAUSSIAN);
  const sh = Float32Array.from({ length: COUNT * SH_REST_FLOATS }, () => between(-1, 1));
  const moments = Float32Array.from({ length: COUNT * SLOTS * 2 }, () => between(-1, 1));
  const stats = new Float32Array(COUNT * 4);
  // Sizes and opacities stay clear of the thresholds, where float32 rounding could go either way.
  const sizeRanges = [
    [0.002, 0.009], // cloned
    [0.012, 0.08], // split
    [0.11, 0.15], // split into children under 0.1; too big to keep whole
    [0.17, 0.3], // too big even once split
  ];
  for (let i = 0; i < COUNT; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    const [lo, hi] = sizeRanges[Math.floor(random() * 4)];
    const size = between(lo, hi);
    const largest = Math.floor(random() * 3);
    for (let k = 0; k < 3; k++) {
      params[at + k] = between(-1, 1);
      params[at + 4 + k] = Math.log(k === largest ? size : size * between(0.2, 0.9));
      params[at + 12 + k] = between(-1, 1);
    }
    const opacity = random() < 0.15 ? between(0.001, 0.004) : between(0.006, 0.99);
    params[at + 3] = Math.log(opacity / (1 - opacity));
    for (let k = 8; k < 12; k++) params[at + k] = between(-1, 1);
    // A tenth were never seen; the rest have an average gradient on either side of the threshold.
    if (random() >= 0.1) {
      const seen = 1 + Math.floor(random() * 10);
      const average = random() < 0.5 ? between(0.00005, 0.00019) : between(0.00021, 0.001);
      stats.set([average * seen, seen, between(1, 30), 0], i * 4);
    }
  }

  // The view only matters for training steps, which this doesn't take.
  const image = device.createTexture({ size: [16, 16], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  const camera = { view: IDENTITY, proj: IDENTITY, viewport: [16, 16] as [number, number] };
  const trainer = new Trainer(device, { params: params.slice(), sh }, [{ name: 'unused', camera, image }], {
    ...DEFAULT_SETTINGS,
    extent: EXTENT,
  });
  device.queue.writeBuffer(trainer.buffers.moments, 0, moments);
  device.queue.writeBuffer(trainer.buffers.stats, 0, stats);
  trainer.iteration = STEP;
  await trainer.densify(pruneLarge);
  const read = async (buffer: GPUBuffer) => new Float32Array(await readBuffer(device, buffer));
  const got = {
    count: trainer.count,
    params: await read(trainer.buffers.params),
    sh: await read(trainer.renderer.shBuffer!),
    moments: await read(trainer.buffers.moments),
    stats: await read(trainer.buffers.stats),
    gaussians: await read(trainer.renderer.gaussianBuffer!),
  };

  // What the reference would do with each Gaussian.
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const removed = (opacity: number, size: number) => opacity < 0.005 || (pruneLarge && size > 0.1 * EXTENT);
  const expected: { source: number; kind: 'kept' | 'clone' | 'child' }[] = [];
  const tally = { kept: 0, cloned: 0, split: 0, removed: 0 };
  for (let i = 0; i < COUNT; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    const grad = stats[i * 4 + 1] > 0 ? stats[i * 4] / stats[i * 4 + 1] : 0;
    const size = Math.exp(Math.max(params[at + 4], params[at + 5], params[at + 6]));
    const opacity = sigmoid(params[at + 3]);
    const clone = grad >= 0.0002 && size <= 0.01 * EXTENT;
    const split = grad >= 0.0002 && !clone;
    if (removed(opacity, split ? size / 1.6 : size)) {
      tally.removed++;
    } else if (split) {
      tally.split++;
      expected.push({ source: i, kind: 'child' }, { source: i, kind: 'child' });
    } else {
      expected.push({ source: i, kind: 'kept' });
      if (clone) {
        tally.cloned++;
        expected.push({ source: i, kind: 'clone' });
      } else {
        tally.kept++;
      }
    }
  }

  const failures: string[] = [];
  const fail = (message: string) => failures.length < 5 && failures.push(`  ${message}`);
  if (got.count !== expected.length) fail(`${got.count} Gaussians afterwards, expected ${expected.length}`);
  const record = trainer.densifications[0];
  const tallied = { kept: record?.kept, cloned: record?.cloned, split: record?.split, removed: record?.removed };
  if (JSON.stringify(tallied) !== JSON.stringify(tally)) fail(`the GPU tallied ${JSON.stringify(tallied)}, expected ${JSON.stringify(tally)}`);
  // Each split child's offset from its parent, in the parent's own axes and units of its scale: should be N(0, 1).
  const offsets: number[] = [];
  expected.slice(0, got.count).forEach(({ source, kind }, o) => {
    const at = source * FLOATS_PER_GAUSSIAN;
    const out = o * FLOATS_PER_GAUSSIAN;
    for (let k = 0; k < FLOATS_PER_GAUSSIAN; k++) {
      const isPosition = k < 3;
      const isScale = k >= 4 && k < 7;
      if (kind === 'child' && isPosition) continue;
      // New Gaussians carry their kind (1 cloned, 2 split) in float 7 and the step in float 15.
      let want = kind === 'child' && isScale ? params[at + k] - Math.log(1.6) : params[at + k];
      if (kind !== 'kept' && k === 7) want = kind === 'clone' ? 1 : 2;
      if (kind !== 'kept' && k === 15) want = STEP;
      if (Math.abs(got.params[out + k] - want) > 1e-5) fail(`output ${o} (${kind} of ${source}): parameter ${k} is ${got.params[out + k]}, expected ${want}`);
    }
    for (let k = 0; k < SH_REST_FLOATS; k++) {
      if (got.sh[o * SH_REST_FLOATS + k] !== sh[source * SH_REST_FLOATS + k]) fail(`output ${o}: coefficient ${k} wasn't copied`);
    }
    for (let k = 0; k < SLOTS * 2; k++) {
      const want = kind === 'kept' ? moments[source * SLOTS * 2 + k] : 0;
      if (got.moments[o * SLOTS * 2 + k] !== want) fail(`output ${o} (${kind}): Adam moment ${k} is ${got.moments[o * SLOTS * 2 + k]}, expected ${want}`);
    }
    if (kind === 'child') {
      const axes = rotationColumns(params.subarray(at + 8, at + 12));
      const d = [0, 1, 2].map((k) => got.params[out + k] - params[at + k]);
      axes.forEach((axis, k) => offsets.push((axis[0] * d[0] + axis[1] * d[1] + axis[2] * d[2]) / Math.exp(params[at + 4 + k])));
    }
  });
  const mean = offsets.reduce((sum, x) => sum + x, 0) / offsets.length;
  const variance = offsets.reduce((sum, x) => sum + (x - mean) ** 2, 0) / offsets.length;
  if (!(Math.abs(mean) < 0.1 && Math.abs(variance - 1) < 0.15)) {
    fail(`split positions: mean ${mean.toFixed(3)} and variance ${variance.toFixed(3)} in the parent's units; expected 0 and 1`);
  }
  const activated = activate(got.params);
  if (activated.some((value, k) => !(Math.abs(value - got.gaussians[k]) <= 1e-5 * Math.max(1, Math.abs(value))))) {
    fail("the renderer's Gaussians don't match the new parameters");
  }
  if (got.stats.some((value) => value !== 0)) fail("the statistics didn't start over");

  // The opacity reset: every opacity capped at 0.01, and only its Adam history cleared.
  trainer.resetOpacity();
  const reset = {
    params: await read(trainer.buffers.params),
    moments: await read(trainer.buffers.moments),
    gaussians: await read(trainer.renderer.gaussianBuffer!),
  };
  const capped = Math.log(0.01 / 0.99);
  for (let o = 0; o < got.count; o++) {
    const at = o * FLOATS_PER_GAUSSIAN;
    const want = Math.min(got.params[at + 3], capped);
    if (Math.abs(reset.params[at + 3] - want) > 1e-5 || Math.abs(reset.gaussians[at + 3] - sigmoid(want)) > 1e-6) {
      fail(`opacity reset: Gaussian ${o} has opacity logit ${reset.params[at + 3]}, expected ${want}`);
    }
    for (let k = 0; k < SLOTS * 2; k++) {
      const s = o * SLOTS * 2 + k;
      const want = k === 6 || k === 7 ? 0 : got.moments[s];
      if (reset.moments[s] !== want) fail(`opacity reset: Gaussian ${o}'s Adam moment ${k} is ${reset.moments[s]}, expected ${want}`);
    }
  }
  trainer.destroy();
  image.destroy();
  return {
    failures,
    summary:
      `${COUNT} → ${got.count}: ${tally.kept} kept, ${tally.cloned} cloned, ${tally.split} split, ${tally.removed} removed; ` +
      `split positions mean ${mean.toFixed(3)}, variance ${variance.toFixed(3)}`,
  };
}

// The columns of the rotation matrix of quaternion (w, x, y, z), normalized as in
// gaussianMath.wgsl: the Gaussian's axes in world space.
function rotationColumns(raw: Float32Array): [number, number, number][] {
  const length = Math.hypot(...raw);
  const [w, x, y, z] = Array.from(raw, (v) => v / length);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
}
