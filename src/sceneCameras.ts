import { dot, normalize, subtract, type Mat4, type Vec3 } from './mat4';

// The training cameras in a 3DGS model's cameras.json: where each photo was taken
// from, and how (M1.6).

export interface TrainingCamera {
  name: string;
  width: number;
  height: number;
  /** The camera's center in world space. */
  position: Vec3;
  /** The camera's axes in world space, OpenCV-style: x right, y down, z forward. */
  right: Vec3;
  down: Vec3;
  forward: Vec3;
  /** Focal lengths in pixels. */
  fx: number;
  fy: number;
}

interface CameraJson {
  img_name: string;
  width: number;
  height: number;
  position: Vec3;
  rotation: number[][];
  fx: number;
  fy: number;
}

export function parseCameras(json: unknown): TrainingCamera[] {
  return (json as CameraJson[]).map(({ img_name, width, height, position, rotation, fx, fy }) => {
    // `rotation` is camera-to-world, so its columns are the camera's axes in world space.
    const column = (j: number): Vec3 => [rotation[0][j], rotation[1][j], rotation[2][j]];
    return { name: img_name, width, height, position, right: column(0), down: column(1), forward: column(2), fx, fy };
  });
}

export function verticalFov(camera: TrainingCamera): number {
  return 2 * Math.atan(camera.height / (2 * camera.fy));
}

/** A training camera's view matrix, in this renderer's convention: x right, y up, looking down −z. */
export function cameraView({ position, right, down, forward }: TrainingCamera): Mat4 {
  const up: Vec3 = [-down[0], -down[1], -down[2]];
  const back: Vec3 = [-forward[0], -forward[1], -forward[2]];
  return [
    right[0], up[0], back[0], 0,
    right[1], up[1], back[1], 0,
    right[2], up[2], back[2], 0,
    -dot(right, position), -dot(up, position), -dot(back, position), 1,
  ];
}

/**
 * Where the training cameras look, and which way is up.
 * - `center` is the point closest to every camera's line of sight, by least squares.
 * - `up` is the average of the cameras' up directions.
 * - `radius` is the median distance from a camera to the center.
 */
export function frameScene(cameras: TrainingCamera[]): { center: Vec3; up: Vec3; radius: number } {
  // Minimize Σ |(I − f fᵀ)(c − p)|² over c, for each camera's forward axis f and
  // position p. That gives the linear system Σ (I − f fᵀ) c = Σ (I − f fᵀ) p.
  const a = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const b = [0, 0, 0];
  const up: Vec3 = [0, 0, 0];
  for (const { position, forward, down } of cameras) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const m = (r === c ? 1 : 0) - forward[r] * forward[c];
        a[r][c] += m;
        b[r] += m * position[c];
      }
      up[r] -= down[r];
    }
  }
  // Cameras that all look the same way (a forward-facing capture) don't pin down
  // a center, so fall back to their average position.
  const center = solve3(a, b, cameras.length) ?? average(cameras.map(({ position }) => position));
  const distances = cameras.map(({ position }) => Math.hypot(...subtract(position, center))).sort((x, y) => x - y);
  return { center, up: normalize(up), radius: distances[Math.floor(distances.length / 2)] };
}

// Cramer's rule, or null when the system is close to singular.
function solve3(a: number[][], b: number[], scale: number): Vec3 | null {
  const det = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(a);
  if (Math.abs(d) < 1e-6 * scale ** 3) return null;
  const replaced = (k: number) => a.map((row, r) => row.map((value, c) => (c === k ? b[r] : value)));
  return [det(replaced(0)) / d, det(replaced(1)) / d, det(replaced(2)) / d];
}

function average(points: Vec3[]): Vec3 {
  const sum = points.reduce<Vec3>((total, p) => [total[0] + p[0], total[1] + p[1], total[2] + p[2]], [0, 0, 0]);
  return [sum[0] / points.length, sum[1] / points.length, sum[2] / points.length];
}
