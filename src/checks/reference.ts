import type { CameraData } from '../gaussianRenderer';
import type { Gaussian, Quat, Splat2D } from '../gaussians';
import { normalize, subtract, transform, type Mat4, type Vec3 } from '../mat4';
import type { RGB } from './helpers';

// The renderer's math in plain TypeScript, in double precision, for the checks
// to compare the GPU against. Mirrors shaders/preprocess.wgsl, shaders/tiles.wgsl
// and shaders/rasterize.wgsl.

const TILE_SIZE = 16;
const SH_C1 = 0.4886025119029199;
const SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152789498];
const SH_C3 = [
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
  1.445305721320277, -0.5900435899266435,
];

type Matrix = number[][]; // a list of rows

/** One Gaussian's spherical-harmonic coefficients 1 to 15 (45 floats, coefficient-major), and the degree to use. */
export interface GaussianSh {
  rest: ArrayLike<number>;
  degree: number;
}

/** Projects one Gaussian the way preprocess.wgsl does; null means culled. */
export function preprocess(g: Gaussian, camera: CameraData, sh?: GaussianSh): Splat2D | null {
  const {
    view,
    proj,
    viewport: [width, height],
  } = camera;
  const [vx, vy, vz] = transform(view, g.position);
  const depth = -vz;
  if (depth <= 0.2) return null;

  const r = rotationMatrix(g.rotation);
  const m = r.map((row) => row.map((value, col) => value * g.scale[col])); // R S
  const cov3d = multiply(m, transpose(m));

  const fx = (proj[0] * width) / 2;
  const fy = (proj[5] * height) / 2;
  const limitX = (1.3 * width) / 2 / fx;
  const limitY = (1.3 * height) / 2 / fy;
  const tx = clamp(vx / depth, -limitX, limitX) * depth;
  const ty = clamp(vy / depth, -limitY, limitY) * depth;
  const j = [
    [fx / depth, 0, (fx * tx) / depth ** 2],
    [0, -fy / depth, (-fy * ty) / depth ** 2],
  ];
  const w = [0, 1, 2].map((row) => [0, 1, 2].map((col) => view[col * 4 + row]));
  const jw = multiply(j, w);
  const cov = multiply(multiply(jw, cov3d), transpose(jw));
  const a = cov[0][0] + 0.3;
  const b = cov[0][1];
  const c = cov[1][1] + 0.3;

  const det = a * c - b * b;
  if (det <= 0) return null;
  const mid = (a + c) / 2;
  const radius = Math.ceil(3 * Math.sqrt(mid + Math.sqrt(Math.max(0.1, mid * mid - det))));

  const [cx, cy, , cw] = transform(proj, [vx, vy, vz]);
  const mean: [number, number] = [((cx / cw) * 0.5 + 0.5) * width, ((cy / cw) * -0.5 + 0.5) * height];
  if (mean[0] + radius < 0 || mean[1] + radius < 0 || mean[0] - radius > width || mean[1] - radius > height) {
    return null;
  }
  let color = g.color;
  if (sh && sh.degree > 0) {
    const extra = shColor(sh.rest, normalize(subtract(g.position, cameraCenter(view))), sh.degree);
    color = [color[0] + extra[0], color[1] + extra[1], color[2] + extra[2]];
  }
  return {
    mean,
    depth,
    radius,
    conic: [c / det, -b / det, a / det],
    color: [Math.max(0, color[0]), Math.max(0, color[1]), Math.max(0, color[2]), g.opacity],
  };
}

/** Where a view matrix [R | t] puts the camera: −Rᵀ t. */
function cameraCenter(view: Mat4): Vec3 {
  const center = (col: number) =>
    -(view[col * 4] * view[12] + view[col * 4 + 1] * view[13] + view[col * 4 + 2] * view[14]);
  return [center(0), center(1), center(2)];
}

/** The view-dependent part of a color: spherical harmonics of degrees 1 to `degree` in direction (x, y, z). */
function shColor(rest: ArrayLike<number>, [x, y, z]: Vec3, degree: number): Vec3 {
  const channel = (ch: number) => {
    const c = (k: number) => rest[(k - 1) * 3 + ch];
    let value = SH_C1 * (-y * c(1) + z * c(2) - x * c(3));
    if (degree > 1) {
      const [xx, yy, zz] = [x * x, y * y, z * z];
      value +=
        SH_C2[0] * x * y * c(4) +
        SH_C2[1] * y * z * c(5) +
        SH_C2[2] * (2 * zz - xx - yy) * c(6) +
        SH_C2[3] * x * z * c(7) +
        SH_C2[4] * (xx - yy) * c(8);
      if (degree > 2) {
        value +=
          SH_C3[0] * y * (3 * xx - yy) * c(9) +
          SH_C3[1] * x * y * z * c(10) +
          SH_C3[2] * y * (4 * zz - xx - yy) * c(11) +
          SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c(12) +
          SH_C3[4] * x * (4 * zz - xx - yy) * c(13) +
          SH_C3[5] * z * (xx - yy) * c(14) +
          SH_C3[6] * x * (xx - 3 * yy) * c(15);
      }
    }
    return value;
  };
  return [channel(0), channel(1), channel(2)];
}

export function tileCount([width, height]: [number, number]): [number, number] {
  return [Math.ceil(width / TILE_SIZE), Math.ceil(height / TILE_SIZE)];
}

/**
 * The tiles a splat's bounding square touches, as [min x, min y, end x, end y]
 * with the ends exclusive, like tile_rect in common.wgsl. The edges are rounded
 * to 32-bit floats first, as on the GPU, so a square ending right on a tile
 * boundary lands on the same side.
 */
export function tileRect({ mean, radius }: Splat2D, [tilesX, tilesY]: [number, number]): [number, number, number, number] {
  const tile = (edge: number) => Math.floor(Math.fround(edge) / TILE_SIZE);
  return [
    clamp(tile(mean[0] - radius), 0, tilesX),
    clamp(tile(mean[1] - radius), 0, tilesY),
    clamp(tile(mean[0] + radius) + 1, 0, tilesX),
    clamp(tile(mean[1] + radius) + 1, 0, tilesY),
  ];
}

/** Each tile's splats (as indices), front to back, in the order the GPU's sorted pairs list them. */
export function tileLists(splats: Splat2D[], viewport: [number, number]): number[][] {
  const tiles = tileCount(viewport);
  const [tilesX, tilesY] = tiles;
  const lists: number[][] = Array.from({ length: tilesX * tilesY }, () => []);
  splats.forEach((splat, index) => {
    if (splat.radius === 0) return;
    const [x0, y0, x1, y1] = tileRect(splat, tiles);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) lists[y * tilesX + x].push(index);
    }
  });
  // Array.sort is stable, so equal depths keep their list order, as on the GPU.
  for (const list of lists) list.sort((a, b) => splats[a].depth - splats[b].depth);
  return lists;
}

/** Blends one pixel's list of splats front to back over `background`, the way rasterize.wgsl does. */
export function blendPixel(splats: Splat2D[], list: number[], [x, y]: [number, number], background: RGB): RGB {
  let transmittance = 1;
  let rgb: RGB = [0, 0, 0];
  for (const index of list) {
    const { mean, conic: [a, b, c], color } = splats[index];
    // Pixels are sampled at their centers, hence the 0.5.
    const dx = x + 0.5 - mean[0];
    const dy = y + 0.5 - mean[1];
    const power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy;
    if (power > 0) continue;
    const alpha = Math.min(0.99, color[3] * Math.exp(power));
    if (alpha < 1 / 255) continue;
    const remaining = transmittance * (1 - alpha);
    if (remaining < 0.0001) break; // nearly opaque: stop without adding this splat
    const add = (i: number) => rgb[i] + color[i] * alpha * transmittance;
    rgb = [add(0), add(1), add(2)];
    transmittance = remaining;
  }
  const over = (i: number) => rgb[i] + transmittance * background[i];
  return [over(0), over(1), over(2)];
}

/** One pixel's color from the tile rasterizer, for spot checks. */
export function shadePixel(
  splats: Splat2D[],
  pixel: [number, number],
  viewport: [number, number],
  background: RGB = [0, 0, 0],
): RGB {
  const [tilesX] = tileCount(viewport);
  const tile = Math.floor(pixel[1] / TILE_SIZE) * tilesX + Math.floor(pixel[0] / TILE_SIZE);
  return blendPixel(splats, tileLists(splats, viewport)[tile], pixel, background);
}

function rotationMatrix([qw, qx, qy, qz]: Quat): Matrix {
  const n = Math.hypot(qw, qx, qy, qz);
  const [w, x, y, z] = [qw / n, qx / n, qy / n, qz / n];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

function multiply(a: Matrix, b: Matrix): Matrix {
  return a.map((row) => b[0].map((_, col) => row.reduce((sum, value, k) => sum + value * b[k][col], 0)));
}

function transpose(m: Matrix): Matrix {
  return m[0].map((_, col) => m.map((row) => row[col]));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
