import type { Vec3 } from './mat4';
import type { TrainingCamera } from './sceneCameras';

// Reads a COLMAP sparse reconstruction, the binary cameras.bin, images.bin and
// points3D.bin (M3). That's what structure-from-motion produces from a set of
// photos: each photo's camera, and a cloud of 3D points seen in them. Training
// starts from those points.

export interface Reconstruction {
  /** One per registered photo, sorted by file name like the reference. */
  cameras: TrainingCamera[];
  /** xyz per point. */
  positions: Float32Array<ArrayBuffer>;
  /** rgb per point, 0 to 255. */
  colors: Uint8Array<ArrayBuffer>;
}

export function parseColmap(camerasBin: ArrayBuffer, imagesBin: ArrayBuffer, pointsBin: ArrayBuffer): Reconstruction {
  const intrinsics = readCameras(new Reader(camerasBin));
  const cameras = readImages(new Reader(imagesBin)).map(({ name, cameraId, q, t }): TrainingCamera => {
    const intrinsic = intrinsics.get(cameraId);
    if (!intrinsic) throw new Error(`Photo ${name} uses camera ${cameraId}, which isn't in cameras.bin.`);
    // COLMAP stores world → camera: x_camera = R x_world + t. The camera sits at
    // −Rᵀ t, and R's rows are its axes in world space (x right, y down, z forward).
    const r = rotationMatrix(q);
    const position: Vec3 = [0, 1, 2].map((c) => -(r[0][c] * t[0] + r[1][c] * t[1] + r[2][c] * t[2])) as Vec3;
    return { name, ...intrinsic, position, right: r[0], down: r[1], forward: r[2] };
  });
  cameras.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return { cameras, ...readPoints(new Reader(pointsBin)) };
}

class Reader {
  private readonly view: DataView;
  private offset = 0;
  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }
  u8(): number {
    return this.view.getUint8(this.offset++);
  }
  i32(): number {
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }
  u64(): number {
    const value = Number(this.view.getBigUint64(this.offset, true));
    this.offset += 8;
    return value;
  }
  f64(): number {
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }
  skip(bytes: number): void {
    this.offset += bytes;
  }
  string(): string {
    const start = this.offset;
    while (this.view.getUint8(this.offset) !== 0) this.offset++;
    const text = new TextDecoder().decode(new Uint8Array(this.view.buffer, this.view.byteOffset + start, this.offset - start));
    this.offset++; // the terminating zero
    return text;
  }
}

type Intrinsics = Pick<TrainingCamera, 'width' | 'height' | 'fx' | 'fy'>;

function readCameras(reader: Reader): Map<number, Intrinsics> {
  const cameras = new Map<number, Intrinsics>();
  for (let count = reader.u64(); count > 0; count--) {
    const id = reader.i32();
    const model = reader.i32();
    const width = reader.u64();
    const height = reader.u64();
    // Training needs undistorted photos: SIMPLE_PINHOLE (f, cx, cy) or PINHOLE (fx, fy, cx, cy).
    if (model === 0) {
      const f = reader.f64();
      reader.skip(16);
      cameras.set(id, { width, height, fx: f, fy: f });
    } else if (model === 1) {
      const fx = reader.f64();
      const fy = reader.f64();
      reader.skip(16);
      cameras.set(id, { width, height, fx, fy });
    } else {
      throw new Error(`Camera ${id} uses COLMAP model ${model}; only undistorted pinhole cameras are supported.`);
    }
  }
  return cameras;
}

function readImages(reader: Reader): { name: string; cameraId: number; q: number[]; t: Vec3 }[] {
  const images = [];
  for (let count = reader.u64(); count > 0; count--) {
    reader.i32(); // image id
    const q = [reader.f64(), reader.f64(), reader.f64(), reader.f64()]; // (w, x, y, z)
    const t: Vec3 = [reader.f64(), reader.f64(), reader.f64()];
    const cameraId = reader.i32();
    const name = reader.string();
    reader.skip(reader.u64() * 24); // the photo's 2D points: x, y and a point id each
    images.push({ name, cameraId, q, t });
  }
  return images;
}

function readPoints(reader: Reader): { positions: Float32Array<ArrayBuffer>; colors: Uint8Array<ArrayBuffer> } {
  const count = reader.u64();
  const positions = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 3);
  for (let i = 0; i < count; i++) {
    reader.u64(); // point id
    for (let k = 0; k < 3; k++) positions[i * 3 + k] = reader.f64();
    for (let k = 0; k < 3; k++) colors[i * 3 + k] = reader.u8();
    reader.f64(); // reprojection error
    reader.skip(reader.u64() * 8); // which photos saw it: an image id and point index each
  }
  return { positions, colors };
}

function rotationMatrix([w, x, y, z]: number[]): [Vec3, Vec3, Vec3] {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
