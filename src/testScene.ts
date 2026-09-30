import type { Gaussian, Quat } from './gaussians';
import { normalize, subtract, type Vec3 } from './mat4';
import { mulberry32 } from './random';

// A stand-in scene until real ones load in M1.6: a trefoil knot drawn with
// stretched Gaussians that follow the curve, colored like a rainbow.
export function trefoilKnot(count: number, seed = 1): Gaussian[] {
  const random = mulberry32(seed);
  const gaussians: Gaussian[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i / count) * 2 * Math.PI;
    const [x, y, z] = knotPoint(t);
    const [jx, jy, jz] = randomInBall(random, 0.035);
    const along = 0.05 + 0.03 * random();
    const across = 0.012 + 0.01 * random();
    gaussians.push({
      position: [x + jx, y + jy, z + jz],
      scale: [along, across, across],
      // Turn each Gaussian's long (x) axis to follow the curve.
      rotation: rotationFromX(normalize(subtract(knotPoint(t + 1e-3), knotPoint(t - 1e-3)))),
      color: hsvToRgb(i / count, 0.7, 1),
      opacity: 0.9,
    });
  }
  return gaussians;
}

// A (2, 3) torus knot lying roughly flat, scaled to fit inside a unit sphere.
function knotPoint(t: number): Vec3 {
  const r = (2 + Math.cos(3 * t)) / 3;
  return [r * Math.cos(2 * t), Math.sin(3 * t) / 3, r * Math.sin(2 * t)];
}

// The quaternion that turns the x axis toward `direction` (a unit vector): the
// half-way quaternion (1 + x·d, x × d), normalized.
function rotationFromX([dx, dy, dz]: Vec3): Quat {
  const w = 1 + dx;
  if (w < 1e-6) return [0, 0, 1, 0]; // direction is −x: half a turn about y
  const length = Math.hypot(w, dz, dy);
  return [w / length, 0, -dz / length, dy / length];
}

function randomInBall(random: () => number, radius: number): Vec3 {
  for (;;) {
    const [x, y, z] = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
    if (x * x + y * y + z * z <= 1) return [x * radius, y * radius, z * radius];
  }
}

function hsvToRgb(h: number, s: number, v: number): Vec3 {
  const channel = (n: number) => {
    const k = (n + h * 6) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [channel(5), channel(3), channel(1)];
}
