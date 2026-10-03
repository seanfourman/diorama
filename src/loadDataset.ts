import { parseColmap } from './colmap';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS } from './gaussians';
import { perspective } from './mat4';
import { readBuffer } from './readback';
import { cameraView, verticalFov, type TrainingCamera } from './sceneCameras';
import nearestNeighborsWgsl from './shaders/nearestNeighbors.wgsl?raw';
import type { RawGaussians, TrainingView } from './trainer';

const SH_C0 = 0.28209479177387814;
/** Like the reference's --eval: every 8th photo, by name, is held out for testing. */
const TEST_EVERY = 8;

// Loading a photo set for training (M3): data/<name>/images/*.jpg and
// data/<name>/sparse/0/*.bin (see scripts/download-photos.mjs).

export interface Dataset {
  name: string;
  /** The photos to learn from, and the held-out ones to measure on. */
  train: TrainingCamera[];
  test: TrainingCamera[];
  /** The structure-from-motion points: xyz, and rgb from 0 to 255. */
  positions: Float32Array<ArrayBuffer>;
  colors: Uint8Array<ArrayBuffer>;
  /** The scene's size, as the reference's cameras_extent: 1.1 × the farthest training camera from their average. */
  extent: number;
}

export async function loadDataset(name: string): Promise<Dataset> {
  const base = `data/${encodeURIComponent(name)}/sparse/0`;
  const fetchBinary = async (file: string) => {
    const response = await fetch(`${base}/${file}`);
    if (!response.ok) throw new Error(`Couldn't load ${base}/${file}. Run "npm run download-photos -- ${name}" first.`);
    return response.arrayBuffer();
  };
  const [cameras, images, points] = await Promise.all(['cameras.bin', 'images.bin', 'points3D.bin'].map(fetchBinary));
  const { cameras: all, positions, colors } = parseColmap(cameras, images, points);
  const train = all.filter((_, i) => i % TEST_EVERY !== 0);
  const test = all.filter((_, i) => i % TEST_EVERY === 0);
  const centers = train.map(({ position }) => position);
  const mean = [0, 1, 2].map((k) => centers.reduce((sum, c) => sum + c[k], 0) / centers.length);
  const farthest = Math.max(...centers.map((c) => Math.hypot(c[0] - mean[0], c[1] - mean[1], c[2] - mean[2])));
  return { name, train, test, positions, colors, extent: 1.1 * farthest };
}

/**
 * Loads each camera's photo onto the GPU and pairs it with the camera, at the
 * photo's own size (which can be smaller than the size COLMAP saw; the field of
 * view stays the same). Like the reference, the principal point is taken to be
 * the center.
 */
export async function loadViews(
  device: GPUDevice,
  name: string,
  cameras: TrainingCamera[],
  onProgress: (loaded: number) => void = () => {},
): Promise<TrainingView[]> {
  let loaded = 0;
  return Promise.all(
    cameras.map(async (camera) => {
      const response = await fetch(`data/${encodeURIComponent(name)}/images/${encodeURIComponent(camera.name)}`);
      if (!response.ok) throw new Error(`Couldn't load the photo ${camera.name}.`);
      // The raw sRGB values, as the reference reads them.
      const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' });
      const image = device.createTexture({
        label: camera.name,
        size: [bitmap.width, bitmap.height],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: image }, [bitmap.width, bitmap.height]);
      bitmap.close();
      onProgress(++loaded);
      return { name: camera.name, camera: trainingCameraData(camera, [image.width, image.height]), image };
    }),
  );
}

/** The renderer's camera for a photo's pose, rendered at `viewport`. */
export function trainingCameraData(camera: TrainingCamera, viewport: [number, number]): TrainingView['camera'] {
  return {
    view: cameraView(camera),
    proj: perspective(verticalFov(camera), viewport[0] / viewport[1], 0.01, 1000),
    viewport,
  };
}

/**
 * The reference's starting Gaussians: one per structure-from-motion point, with
 * the point's color, opacity 0.1, no rotation, and a round size: the square root
 * of the mean squared distance to its 3 nearest neighbors.
 */
export async function initialGaussians(device: GPUDevice, positions: Float32Array, colors: Uint8Array): Promise<RawGaussians> {
  const count = positions.length / 3;
  const distance2 = await meanNeighborDistance2(device, positions);
  const params = new Float32Array(count * FLOATS_PER_GAUSSIAN);
  const opacityLogit = Math.log(0.1 / 0.9);
  for (let i = 0; i < count; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    const logScale = 0.5 * Math.log(Math.max(distance2[i], 1e-7));
    for (let k = 0; k < 3; k++) {
      params[at + k] = positions[i * 3 + k];
      params[at + 4 + k] = logScale;
      params[at + 12 + k] = (colors[i * 3 + k] / 255 - 0.5) / SH_C0;
    }
    params[at + 3] = opacityLogit;
    params[at + 8] = 1; // the quaternion (1, 0, 0, 0)
  }
  return { params, sh: new Float32Array(count * SH_REST_FLOATS) };
}

/** For each point, the mean squared distance to its 3 nearest neighbors, on the GPU. */
export async function meanNeighborDistance2(device: GPUDevice, positions: Float32Array): Promise<Float32Array> {
  const count = positions.length / 3;
  const points = device.createBuffer({ label: 'points', size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const padded = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) padded.set(positions.subarray(i * 3, i * 3 + 3), i * 4);
  device.queue.writeBuffer(points, 0, padded);
  const result = device.createBuffer({
    label: 'neighbor distances',
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const pipeline = device.createComputePipeline({
    label: 'nearest neighbors',
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: nearestNeighborsWgsl }), entryPoint: 'nearest' },
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: points } },
        { binding: 1, resource: { buffer: result } },
      ],
    }),
  );
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const distance2 = new Float32Array(await readBuffer(device, result));
  points.destroy();
  result.destroy();
  return distance2;
}
