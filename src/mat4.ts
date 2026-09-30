// 4×4 matrices as 16 numbers in column-major order (m[column * 4 + row]), the
// layout WGSL reads a mat4x4<f32> in. Each row of the literals below is one
// *column* of the matrix.

export type Mat4 = number[];
export type Vec3 = [number, number, number];

export const IDENTITY: Mat4 = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

/** a × b, which applies b first and then a. */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out: Mat4 = [];
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

/** m × (x, y, z, 1), returned as clip-space (x, y, z, w). */
export function transform(m: Mat4, [x, y, z]: Vec3): [number, number, number, number] {
  const row = (r: number) => m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
  return [row(0), row(1), row(2), row(3)];
}

/**
 * Perspective projection for WebGPU, whose clip-space depth runs from 0 at the
 * near plane to 1 at the far plane. OpenGL's runs from −1 to 1, so OpenGL
 * formulas are subtly wrong here.
 */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const depth = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * depth, -1,
    0, 0, near * far * depth, 0,
  ];
}

/** View matrix for a camera at `eye` looking at `target`. The camera looks down its own −z axis. */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(subtract(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

/** a + b × s */
export function addScaled(a: Vec3, b: Vec3, s: number): Vec3 {
  return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / length, v[1] / length, v[2] / length];
}
