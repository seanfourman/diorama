import type { Vec3 } from './mat4';

// CPU-side mirrors of the GPU structs in shaders/common.wgsl, and packers that
// lay them out the way the shaders read them.

/** A quaternion (w, x, y, z): the order the reference and its .ply files use. */
export type Quat = [number, number, number, number];

export interface Gaussian {
  position: Vec3;
  /** Standard deviations along the Gaussian's own x, y and z axes. */
  scale: Vec3;
  rotation: Quat;
  color: Vec3;
  opacity: number;
}

/** A Gaussian projected to the screen: what the draw stage needs. */
export interface Splat2D {
  /** Center in pixels, y down. */
  mean: [number, number];
  /** Distance in front of the camera. */
  depth: number;
  /** Half-size of the bounding square in pixels; 0 means culled. */
  radius: number;
  /** Inverse 2D covariance [[a, b], [b, c]] as (a, b, c). */
  conic: Vec3;
  /** rgb + opacity. */
  color: [number, number, number, number];
}

export const FLOATS_PER_GAUSSIAN = 16; // struct Gaussian: 64 bytes
export const FLOATS_PER_SPLAT = 12; // struct Splat2D: 48 bytes
// Spherical-harmonic coefficients 1 to 15 per Gaussian, rgb each, coefficient-major:
// (c1 r, c1 g, c1 b, c2 r, …). Coefficient 0 is already in the Gaussian's color.
export const SH_REST_FLOATS = 45;

export function packGaussians(gaussians: Gaussian[]): Float32Array<ArrayBuffer> {
  const data = new Float32Array(gaussians.length * FLOATS_PER_GAUSSIAN);
  gaussians.forEach(({ position, scale, rotation, color, opacity }, i) => {
    // Float 7 pads scale out to 16 bytes before the vec4 rotation; float 15 pads
    // the struct to 64 bytes.
    data.set([...position, opacity, ...scale, 0, ...rotation, ...color, 0], i * FLOATS_PER_GAUSSIAN);
  });
  return data;
}

export function packSplats(splats: Splat2D[]): Float32Array<ArrayBuffer> {
  const data = new Float32Array(splats.length * FLOATS_PER_SPLAT);
  splats.forEach(({ mean, depth, radius, conic, color }, i) => {
    // Float 7 pads conic out to 16 bytes before the vec4 color.
    data.set([...mean, depth, radius, ...conic, 0, ...color], i * FLOATS_PER_SPLAT);
  });
  return data;
}

export function unpackSplats(data: Float32Array): Splat2D[] {
  const splats: Splat2D[] = [];
  for (let offset = 0; offset < data.length; offset += FLOATS_PER_SPLAT) {
    const f = (k: number) => data[offset + k];
    splats.push({
      mean: [f(0), f(1)],
      depth: f(2),
      radius: f(3),
      conic: [f(4), f(5), f(6)],
      color: [f(8), f(9), f(10), f(11)],
    });
  }
  return splats;
}
