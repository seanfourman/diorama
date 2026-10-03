import { loadDataset } from '../loadDataset';
import type { Vec3 } from '../mat4';
import { mulberry32 } from '../random';
import { camerasToJson, frameGaussians, parseCameras, type TrainingCamera } from '../sceneCameras';
import { verdict } from './helpers';

// M5: your own scenes.
// - cameras.json, which training writes next to a saved scene, read back.
// - Framing a scene that has no cameras.
// - If scripts/reconstruct.mjs has been run on the "train" photos (as
//   train-colmap): our COLMAP run against the reconstruction the dataset ships
//   with. Two reconstructions only agree up to scale, rotation and position, so
//   the check first finds the best such transform between the camera centers
//   (Horn's method), then measures what's left.
// Notes: docs/steps/5-your-photos.md

export async function runScenesCheck(): Promise<string> {
  const lines = ['M5: your scenes'];

  const random = mulberry32(61);
  const cameras: TrainingCamera[] = Array.from({ length: 5 }, (_, k) => {
    const [right, down, forward] = rotationColumns(randomQuaternion(random));
    return { name: `photo_${k}.jpg`, width: 800, height: 600, position: [random(), random(), random()], right, down, forward, fx: 700 + k, fy: 710 + k };
  });
  const back = parseCameras(JSON.parse(JSON.stringify(camerasToJson(cameras))));
  const jsonFailures = back.flatMap((camera, k) => {
    const original = cameras[k];
    const worst = Math.max(
      ...(['position', 'right', 'down', 'forward'] as const).flatMap((key) => camera[key].map((v, i) => Math.abs(v - original[key][i]))),
      Math.abs(camera.fx - original.fx),
      Math.abs(camera.fy - original.fy),
    );
    const sameName = camera.name === original.name.replace(/\.jpg$/, '');
    return worst < 1e-12 && sameName && camera.width === original.width ? [] : [`  camera ${k} came back different`];
  });
  lines.push(`cameras.json: ${verdict(jsonFailures)} (5 cameras written and read back)`, ...jsonFailures);

  // A ball of points around (1, 2, 3), with a median distance of 0.5.
  const positions = new Float32Array(16 * 2001);
  for (let i = 0; i < 2001; i++) {
    const direction = normalize([random() - 0.5, random() - 0.5, random() - 0.5]);
    const distance = i / 2000; // uniform in [0, 1], so the median is 0.5
    positions.set([1 + direction[0] * distance, 2 + direction[1] * distance, 3 + direction[2] * distance], i * 16);
  }
  const frame = frameGaussians(positions, 16);
  const frameError = Math.max(Math.abs(frame.center[0] - 1), Math.abs(frame.center[1] - 2), Math.abs(frame.center[2] - 3), Math.abs(frame.radius - 0.5));
  const frameFailures = frameError < 0.05 ? [] : [`  center (${frame.center.map((v) => v.toFixed(3)).join(', ')}), radius ${frame.radius.toFixed(3)}`];
  lines.push(`Framing without cameras: ${verdict(frameFailures)} (2,001 points around a known center)`, ...frameFailures);

  // The alignment itself, on points moved by a known scale, rotation and translation.
  const known = { scale: 2.5, columns: rotationColumns(randomQuaternion(random)), translation: [3, -1, 2] as Vec3 };
  const knownMatrix = [0, 1, 2].map((r) => known.columns.map((column) => column[r]));
  const from = Array.from({ length: 20 }, (): Vec3 => [random(), random(), random()]);
  const to = from.map((p) => add(scaled(multiply(knownMatrix, p), known.scale), known.translation));
  const found = similarity(from, to);
  const alignError = Math.max(
    Math.abs(found.scale - known.scale),
    ...found.rotation.flatMap((row, r) => row.map((v, c) => Math.abs(v - knownMatrix[r][c]))),
    ...found.translation.map((v, k) => Math.abs(v - known.translation[k])),
  );
  const alignFailures = alignError < 1e-9 ? [] : [`  off by ${alignError.toExponential(1)}`];
  lines.push(`Aligning two reconstructions: ${verdict(alignFailures)} (a known scale, rotation and translation recovered)`, ...alignFailures);

  lines.push(...(await compareReconstructions()));
  return lines.join('\n');
}

async function compareReconstructions(): Promise<string[]> {
  let ours, theirs;
  try {
    [ours, theirs] = await Promise.all([loadDataset('train-colmap'), loadDataset('train')]);
  } catch {
    return ['Our COLMAP run: SKIP (run "npm run reconstruct -- train-colmap --from=data/train/images --sequential" first)'];
  }
  const reference = new Map([...theirs.train, ...theirs.test].map((camera) => [camera.name, camera]));
  const pairs = [...ours.train, ...ours.test].flatMap((camera) => {
    const match = reference.get(camera.name);
    return match ? [[camera, match] as const] : [];
  });
  const { scale, rotation, translation } = similarity(pairs.map(([a]) => a.position), pairs.map(([, b]) => b.position));
  const apply = (p: Vec3) => add(scaled(multiply(rotation, p), scale), translation);
  const centerErrors = pairs.map(([a, b]) => Math.hypot(...subtract(apply(a.position), b.position)));
  const rms = Math.sqrt(centerErrors.reduce((sum, e) => sum + e * e, 0) / centerErrors.length);
  // The camera's own rotation, carried into the reference's frame, against the reference's.
  const angles = pairs.map(([a, b]) => {
    const axes = [a.right, a.down, a.forward].map((axis) => multiply(rotation, axis));
    const trace = dot(axes[0], b.right) + dot(axes[1], b.down) + dot(axes[2], b.forward);
    return (Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2))) * 180) / Math.PI;
  });
  const medianAngle = [...angles].sort((x, y) => x - y)[Math.floor(angles.length / 2)];
  const total = reference.size;
  const relative = rms / theirs.extent;
  const failures = [
    ...(pairs.length >= 0.95 * total ? [] : [`  only ${pairs.length} of ${total} photos placed`]),
    ...(relative < 0.01 ? [] : [`  camera centers off by ${(relative * 100).toFixed(2)}% of the scene`]),
    ...(medianAngle < 0.5 ? [] : [`  camera rotations off by ${medianAngle.toFixed(2)}° (median)`]),
  ];
  return [
    `Our COLMAP run: ${verdict(failures)} (${pairs.length} of ${total} photos placed; after aligning, camera centers within ` +
      `${(relative * 100).toFixed(2)}% of the scene (RMS), rotations within ${medianAngle.toFixed(2)}° (median))`,
    ...failures,
  ];
}

/**
 * The scale, rotation and translation that best map points `a` onto points `b`
 * (least squares), by Horn's method: the rotation is the quaternion that
 * maximizes Σ b'·R a', which is the top eigenvector of a 4×4 matrix built from
 * the centered points' cross-covariance.
 */
export function similarity(a: Vec3[], b: Vec3[]): { scale: number; rotation: number[][]; translation: Vec3 } {
  const mean = (points: Vec3[]): Vec3 => scaled(points.reduce((sum, p) => add(sum, p), [0, 0, 0]), 1 / points.length);
  const [ma, mb] = [mean(a), mean(b)];
  const ca = a.map((p) => subtract(p, ma));
  const cb = b.map((p) => subtract(p, mb));
  const s = [0, 1, 2].map((i) => [0, 1, 2].map((j) => ca.reduce((sum, p, k) => sum + p[i] * cb[k][j], 0)));
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
  const n = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  const columns = rotationColumns(topEigenvector(n));
  const matrix = [0, 1, 2].map((r) => columns.map((column) => column[r]));
  const numerator = ca.reduce((sum, p, k) => sum + dot(cb[k], multiply(matrix, p)), 0);
  const denominator = ca.reduce((sum, p) => sum + dot(p, p), 0);
  const scale = numerator / denominator;
  return { scale, rotation: matrix, translation: subtract(mb, scaled(multiply(matrix, ma), scale)) };
}

// The eigenvector of the largest eigenvalue of a symmetric matrix, by Jacobi rotations.
function topEigenvector(matrix: number[][]): [number, number, number, number] {
  const a = matrix.map((row) => [...row]);
  const v: number[][] = a.map((_, i) => a.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 50; sweep++) {
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-15) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) {
          const [akp, akq] = [a[k][p], a[k][q]];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const [apk, aqk] = [a[p][k], a[q][k]];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const [vkp, vkq] = [v[k][p], v[k][q]];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

// The columns of the rotation matrix of quaternion (w, x, y, z), normalized.
function rotationColumns(raw: number[]): [Vec3, Vec3, Vec3] {
  const length = Math.hypot(...raw);
  const [w, x, y, z] = raw.map((v) => v / length);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
}

function randomQuaternion(random: () => number): number[] {
  return [random() - 0.5, random() - 0.5, random() - 0.5, random() - 0.5];
}

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const subtract = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scaled = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a: Vec3): Vec3 => scaled(a, 1 / Math.hypot(...a));
const multiply = (m: number[][], v: Vec3): Vec3 => [dot(m[0] as Vec3, v), dot(m[1] as Vec3, v), dot(m[2] as Vec3, v)];
