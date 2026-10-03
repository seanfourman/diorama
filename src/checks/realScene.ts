import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS, packGaussians, unpackSplats, type Gaussian } from '../gaussians';
import { IDENTITY, lookAt, perspective } from '../mat4';
import { SH_C0, parseGaussianPly } from '../plyLoader';
import { mulberry32 } from '../random';
import { cameraView, verticalFov } from '../sceneCameras';
import { loadSceneOnce, readBuffer, renderPixels, verdict } from './helpers';
import { preprocess } from './reference';

// M1.6:
// - read a hand-made .ply and check every converted value;
// - compare the GPU's view-dependent colors (spherical harmonics) with the CPU's;
// - if data/train has been downloaded, render the real scene from its first
//   training photo and time it.
// Notes: docs/steps/1.6-real-scene.md

const PLY_PROPERTIES = [
  'x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
  ...Array.from({ length: 45 }, (_, k) => `f_rest_${k}`),
  'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
];
const SH_C1 = 0.4886025119029199;
const SIZE = 64;

export async function runRealSceneCheck(device: GPUDevice): Promise<string> {
  const plyFailures = checkPlyReading();
  const shFailures = await checkSphericalHarmonics(device);
  return [
    'M1.6: real scenes',
    `.ply reading: ${verdict(plyFailures)} (2 Gaussians, every property)`,
    ...plyFailures,
    `Spherical harmonics: ${verdict(shFailures)} (2 cases by hand, 20 random ones against the CPU)`,
    ...shFailures,
    ...(await checkRealScene(device)),
  ].join('\n');
}

// Two Gaussians in which every raw value is different: property k of Gaussian v is (62v + k) / 100.
function checkPlyReading(): string[] {
  const raw = (v: number, k: number) => (62 * v + k) / 100;
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    'element vertex 2',
    ...PLY_PROPERTIES.map((name) => `property float ${name}`),
    'end_header',
    '',
  ].join('\n');
  const headerBytes = new TextEncoder().encode(header);
  const body = Float32Array.from({ length: 2 * PLY_PROPERTIES.length }, (_, i) =>
    raw(Math.floor(i / PLY_PROPERTIES.length), i % PLY_PROPERTIES.length),
  );
  const file = new Uint8Array(headerBytes.length + body.byteLength);
  file.set(headerBytes);
  file.set(new Uint8Array(body.buffer), headerBytes.length);
  const scene = parseGaussianPly(file.buffer);

  const failures: string[] = [];
  if (scene.count !== 2 || scene.shDegree !== 3) {
    failures.push(`  read ${scene.count} Gaussians of degree ${scene.shDegree}; expected 2 of degree 3`);
  }
  const expect = (label: string, got: number, want: number) => {
    if (!(Math.abs(got - want) <= 1e-5 * Math.max(1, Math.abs(want)))) failures.push(`  ${label}: ${got}, expected ${want}`);
  };
  for (let v = 0; v < 2; v++) {
    const value = (name: string) => raw(v, PLY_PROPERTIES.indexOf(name));
    const read = (k: number) => scene.gaussians[v * FLOATS_PER_GAUSSIAN + k];
    for (let k = 0; k < 3; k++) {
      expect(`Gaussian ${v} position ${k}`, read(k), value('xyz'[k]));
      expect(`Gaussian ${v} scale ${k}`, read(4 + k), Math.exp(value(`scale_${k}`)));
      expect(`Gaussian ${v} color ${k}`, read(12 + k), 0.5 + SH_C0 * value(`f_dc_${k}`));
    }
    expect(`Gaussian ${v} opacity`, read(3), 1 / (1 + Math.exp(-value('opacity'))));
    for (let k = 0; k < 4; k++) expect(`Gaussian ${v} rotation ${k}`, read(8 + k), value(`rot_${k}`));
    // The file stores the coefficients channel by channel; shRest has them coefficient by coefficient.
    for (let c = 0; c < 15; c++) {
      for (let channel = 0; channel < 3; channel++) {
        expect(
          `Gaussian ${v} coefficient ${c + 1} channel ${channel}`,
          scene.shRest[v * SH_REST_FLOATS + c * 3 + channel],
          value(`f_rest_${channel * 15 + c}`),
        );
      }
    }
  }
  return failures.slice(0, 5);
}

async function checkSphericalHarmonics(device: GPUDevice): Promise<string[]> {
  const renderer = new GaussianRenderer(device);
  const colorsFrom = async (gaussians: Gaussian[], rest: Float32Array<ArrayBuffer>, camera: CameraData) => {
    renderer.setGaussians(packGaussians(gaussians), { rest, degree: 3 });
    await renderPixels(device, SIZE, (encoder, target) => renderer.encode(encoder, target, camera, [0, 0, 0]));
    return unpackSplats(new Float32Array(await readBuffer(device, renderer.splatBuffer!))).map(({ color }) => color);
  };
  const round = (position: [number, number, number]): Gaussian => ({
    position,
    scale: [0.2, 0.2, 0.2],
    rotation: [1, 0, 0, 0],
    color: [0.5, 0.5, 0.5],
    opacity: 0.9,
  });
  const failures: string[] = [];

  // By hand: two Gaussians with only coefficient 1 set, the one that multiplies −y.
  // From a camera at the origin, one is seen looking down at it (y = −2/√20) and
  // the other looking up (y = +2/√20), so one is brighter by SH_C1 × 2/√20 and the
  // other darker by the same amount.
  const byHandRest = new Float32Array(2 * SH_REST_FLOATS);
  byHandRest.fill(1, 0, 3);
  byHandRest.fill(1, SH_REST_FLOATS, SH_REST_FLOATS + 3);
  const identity: CameraData = { view: IDENTITY, proj: perspective(Math.PI / 2, 1, 0.1, 100), viewport: [SIZE, SIZE] };
  const [below, above] = await colorsFrom([round([0, -2, -4]), round([0, 2, -4])], byHandRest, identity);
  const shift = (SH_C1 * 2) / Math.sqrt(20);
  for (const [name, color, expected] of [
    ['below the view axis', below, 0.5 + shift],
    ['above the view axis', above, 0.5 - shift],
  ] as const) {
    if (color.slice(0, 3).some((channel) => Math.abs(channel - expected) > 1e-4)) {
      failures.push(`  ${name}: GPU (${color.slice(0, 3).map((c) => c.toFixed(4)).join(', ')}), expected ${expected.toFixed(4)}`);
    }
  }

  // Against the CPU: random Gaussians and coefficients, from a camera off to one side.
  const random = mulberry32(21);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  const gaussians = Array.from({ length: 20 }, () => round([between(-1.5, 1.5), between(-1.5, 1.5), between(-1.5, 1.5)]));
  const rest = Float32Array.from({ length: gaussians.length * SH_REST_FLOATS }, () => between(-0.5, 0.5));
  const camera: CameraData = {
    view: lookAt([2.5, 1.5, 4], [0, 0, 0], [0, 1, 0]),
    proj: perspective(Math.PI / 3, 1, 0.1, 100),
    viewport: [SIZE, SIZE],
  };
  const gpu = await colorsFrom(gaussians, rest, camera);
  renderer.destroy();
  gaussians.forEach((gaussian, i) => {
    const cpu = preprocess(gaussian, camera, { rest: rest.subarray(i * SH_REST_FLOATS, (i + 1) * SH_REST_FLOATS), degree: 3 });
    if (!cpu) {
      failures.push(`  random Gaussian ${i} was culled on the CPU`);
    } else if (cpu.color.slice(0, 3).some((channel, k) => Math.abs(channel - gpu[i][k]) > 1e-4)) {
      failures.push(`  random Gaussian ${i}: GPU (${gpu[i].slice(0, 3).join(', ')}), CPU (${cpu.color.slice(0, 3).join(', ')})`);
    }
  });
  return failures.slice(0, 5);
}

// Renders the real scene from its first training photo, at that photo's size.
async function checkRealScene(device: GPUDevice): Promise<string[]> {
  let scene;
  try {
    scene = await loadSceneOnce('train');
  } catch {
    return ['Real scene: SKIP (data/train is missing; npm run download-scene fetches it)'];
  }
  const photo = scene.cameras[0];
  const [width, height] = [photo.width, photo.height];
  const camera: CameraData = {
    view: cameraView(photo),
    proj: perspective(verticalFov(photo), width / height, 0.01, 1000),
    viewport: [width, height],
  };
  const renderer = new GaussianRenderer(device);
  renderer.setGaussians(scene.gaussians, { rest: scene.shRest, degree: scene.shDegree });
  const target = device.createTexture({
    label: 'real scene timing target',
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING,
  });
  // The first frames warm up and let the tile buffers grow to fit.
  const times: number[] = [];
  for (let run = 0; run < 8; run++) {
    const encoder = device.createCommandEncoder();
    renderer.encode(encoder, target.createView(), camera, [0, 0, 0]);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (run >= 3) times.push(performance.now() - start);
    await renderer.afterSubmit();
  }
  target.destroy();
  times.sort((a, b) => a - b);
  const milliseconds = times[Math.floor(times.length / 2)];
  const pairs = renderer.tiles!.lastPairCount;

  // A trained scene seen from one of its own photos should fill the frame.
  const pixels = await renderPixels(device, [width, height], (encoder, view) =>
    renderer.encode(encoder, view, camera, [0, 0, 0]),
  );
  renderer.destroy();
  let covered = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] || pixels[i + 1] || pixels[i + 2]) covered++;
  }
  const coverage = covered / (width * height);
  const count = (n: number) => n.toLocaleString('en-US');
  return [
    `Real scene: ${coverage > 0.5 ? 'PASS' : 'FAIL'} ("train": ${count(scene.count)} Gaussians, degree ${scene.shDegree}, ` +
      `${scene.cameras.length} photos; ${Math.round(coverage * 100)}% of photo ${photo.name}'s view covered)`,
    `Speed: "train" at ${width}×${height} in ${milliseconds.toFixed(1)} ms a frame, ` +
      `${count(pairs)} (splat, tile) pairs (median of 5, wall clock)`,
  ];
}
