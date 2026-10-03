import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS, packGaussians, type Gaussian } from '../gaussians';
import { perspective } from '../mat4';
import { mulberry32 } from '../random';
import { cameraView, verticalFov } from '../sceneCameras';
import fixture from './fixtures/backward.json';
import { loadSceneOnce, readBuffer, renderPixels, verdict, type RGB } from './helpers';

// M2: the backward pass against PyTorch. reference/backward_reference.py renders a
// small random scene with the same math in float64, gets exact gradients from
// autograd (checked there against finite differences), and saves everything to
// fixtures/backward.json. This check renders the same scene on the GPU, runs the
// backward pass with the same dL/d(pixel), and compares every gradient. It also
// times forward and backward on the real scene. Notes: docs/steps/2-backward-pass.md

interface BackwardFixture {
  viewport: [number, number];
  view: number[];
  proj: number[];
  background: RGB;
  shDegree: number;
  gaussians: Gaussian[];
  shRest: number[][];
  pixelGrads: number[];
  image: number[];
  gradients: {
    position: number[];
    scale: number[];
    rotation: number[];
    opacity: number;
    color: number[];
    sh: number[];
  }[];
  finiteDifferenceError: Record<string, number>;
}

const data = fixture as unknown as BackwardFixture;
const TOLERANCE = 2e-3; // f32 on the GPU against float64, relative to the group's largest gradient

// Where each group sits in the 16-float gradient layout of a Gaussian.
const GROUPS = [
  { name: 'position', offset: 0, size: 3 },
  { name: 'scale', offset: 4, size: 3 },
  { name: 'rotation', offset: 8, size: 4 },
  { name: 'opacity', offset: 3, size: 1 },
  { name: 'color', offset: 12, size: 3 },
] as const;

export async function runBackwardCheck(device: GPUDevice): Promise<string> {
  const [width, height] = data.viewport;
  const camera: CameraData = { view: data.view, proj: data.proj, viewport: [width, height] };
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(packGaussians(data.gaussians), {
    rest: Float32Array.from(data.shRest.flat()),
    degree: data.shDegree,
  });
  await renderPixels(device, [width, height], (encoder, target) => renderer.encode(encoder, target, camera, data.background));
  const pixels = new Float32Array(await readBuffer(device, renderer.pixelBuffer!));
  let imageError = 0;
  for (let p = 0; p < width * height; p++) {
    for (let channel = 0; channel < 3; channel++) {
      imageError = Math.max(imageError, Math.abs(pixels[p * 4 + channel] - data.image[p * 3 + channel]));
    }
  }

  const pixelGrads = uploadPixelGrads(device, width * height, (p, channel) => data.pixelGrads[p * 3 + channel]);
  const encoder = device.createCommandEncoder();
  renderer.encodeBackward(encoder, pixelGrads);
  device.queue.submit([encoder.finish()]);
  const gaussianGrads = new Float32Array(await readBuffer(device, renderer.gradients!.gaussians));
  const shGrads = new Float32Array(await readBuffer(device, renderer.gradients!.sh));
  renderer.destroy();
  pixelGrads.destroy();

  // Per group: the largest disagreement, relative to the largest reference gradient.
  const errors: [string, number][] = GROUPS.map(({ name, offset, size }) => {
    let worst = 0;
    let largest = 0;
    data.gradients.forEach((expected, i) => {
      const reference = name === 'opacity' ? [expected.opacity] : expected[name];
      for (let k = 0; k < size; k++) {
        largest = Math.max(largest, Math.abs(reference[k]));
        worst = Math.max(worst, Math.abs(gaussianGrads[i * FLOATS_PER_GAUSSIAN + offset + k] - reference[k]));
      }
    });
    return [name, worst / largest];
  });
  let shWorst = 0;
  let shLargest = 0;
  data.gradients.forEach(({ sh }, i) => {
    sh.forEach((reference, k) => {
      shLargest = Math.max(shLargest, Math.abs(reference));
      shWorst = Math.max(shWorst, Math.abs(shGrads[i * SH_REST_FLOATS + k] - reference));
    });
  });
  errors.push(['spherical harmonics', shWorst / shLargest]);

  const imageFailures = imageError <= 1e-4 ? [] : [`  largest difference ${imageError.toExponential(1)}`];
  const gradientFailures = errors
    .filter(([, error]) => !(error <= TOLERANCE))
    .map(([name, error]) => `  ${name}: off by ${error.toExponential(1)} of the largest gradient`);
  const parameterCount = data.gaussians.length * (3 + 3 + 4 + 1 + 3 + SH_REST_FLOATS);
  const worstFiniteDifference = Math.max(...Object.values(data.finiteDifferenceError));
  return [
    'M2: backward pass',
    `Forward vs PyTorch: ${verdict(imageFailures)} (largest difference ${imageError.toExponential(1)})`,
    ...imageFailures,
    `Gradients vs PyTorch autograd: ${verdict(gradientFailures)} (${data.gaussians.length} Gaussians, ${parameterCount} numbers)`,
    `  largest error ÷ largest gradient: ${errors.map(([name, error]) => `${name} ${error.toExponential(1)}`).join(', ')}`,
    ...gradientFailures,
    `Autograd vs finite differences, in float64 when the fixture was made: ${worstFiniteDifference.toExponential(1)}`,
    ...(await timeRealScene(device)),
  ].join('\n');
}

/** A buffer of dL/d(pixel rgb), a vec4 per pixel with the fourth float unused. */
function uploadPixelGrads(device: GPUDevice, pixelCount: number, value: (pixel: number, channel: number) => number): GPUBuffer {
  const grads = new Float32Array(pixelCount * 4);
  for (let p = 0; p < pixelCount; p++) {
    for (let channel = 0; channel < 3; channel++) grads[p * 4 + channel] = value(p, channel);
  }
  const buffer = device.createBuffer({
    label: 'pixel gradients',
    size: grads.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, grads);
  return buffer;
}

// Forward and backward on the real scene from its first photo, at half the
// photo's size, which is about what training uses.
async function timeRealScene(device: GPUDevice): Promise<string[]> {
  let scene;
  try {
    scene = await loadSceneOnce('train');
  } catch {
    return ['Speed: SKIP (data/train is missing)'];
  }
  const photo = scene.cameras[0];
  const [width, height] = [Math.round(photo.width / 2), Math.round(photo.height / 2)];
  const camera: CameraData = {
    view: cameraView(photo),
    proj: perspective(verticalFov(photo), width / height, 0.01, 1000),
    viewport: [width, height],
  };
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(scene.gaussians, { rest: scene.shRest, degree: scene.shDegree });
  const target = device.createTexture({
    label: 'backward timing target',
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING,
  });
  const random = mulberry32(9);
  const pixelGrads = uploadPixelGrads(device, width * height, () => (random() - 0.5) / (width * height));
  const timed = async (record: (encoder: GPUCommandEncoder) => void) => {
    const encoder = device.createCommandEncoder();
    record(encoder);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };
  const forward: number[] = [];
  const backward: number[] = [];
  for (let run = 0; run < 8; run++) {
    const forwardTime = await timed((encoder) => renderer.encode(encoder, target.createView(), camera, [0, 0, 0]));
    await renderer.afterSubmit(); // lets the tile buffers grow during the warm-up runs
    const backwardTime = await timed((encoder) => renderer.encodeBackward(encoder, pixelGrads));
    if (run >= 3) {
      forward.push(forwardTime);
      backward.push(backwardTime);
    }
  }
  renderer.destroy();
  target.destroy();
  pixelGrads.destroy();
  const median = (times: number[]) => times.sort((a, b) => a - b)[Math.floor(times.length / 2)];
  return [
    `Speed: "train" at ${width}×${height}: forward ${median(forward).toFixed(1)} ms, ` +
      `backward ${median(backward).toFixed(1)} ms (median of 5, wall clock)`,
  ];
}
