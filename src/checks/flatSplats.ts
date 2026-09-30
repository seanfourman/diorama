import { packSplats, type Splat2D } from '../gaussians';
import { GaussianRenderer } from '../gaussianRenderer';
import { lookAt, multiply, perspective, transform, type Vec3 } from '../mat4';
import { formatRgb, maxDiff, pixelAt, renderPixels, verdict, type RGB } from './helpers';
import { shadePixel } from './reference';

// M1.2: draw two overlapping round splats into a small offscreen image, read the
// pixels back, and compare them with the same Gaussian and blending math on the
// CPU. Since M1.5 the splats go straight into the tile rasterizer, which blends
// them front to back by depth. Also sanity-checks the camera math.
// Notes: docs/steps/1.2-flat-splats.md

const SIZE = 64;
const TOLERANCE = 2.5 / 255; // 8-bit color, plus float differences between GPU and CPU

// σ = 4 px (a conic of 1/16), with the square reaching 3σ. The red splat is
// nearer, so it blends over the white one.
const SPLATS: Splat2D[] = [
  { mean: [20.5, 12.5], depth: 2, radius: 12, conic: [1 / 16, 0, 1 / 16], color: [1, 1, 1, 0.8] },
  { mean: [26.5, 12.5], depth: 1, radius: 12, conic: [1 / 16, 0, 1 / 16], color: [1, 0, 0, 0.5] },
];

// Expected colors worked out by hand from the Gaussian and blending formulas.
const PROBES: { name: string; pixel: [number, number]; expected: RGB }[] = [
  { name: 'white center', pixel: [20, 12], expected: [0.832, 0.67, 0.67] },
  { name: 'red center', pixel: [26, 12], expected: [0.63, 0.13, 0.13] },
  { name: '2σ below the white center', pixel: [20, 20], expected: [0.128, 0.106, 0.106] },
  { name: 'outside both quads', pixel: [39, 12], expected: [0, 0, 0] },
  { name: 'white center mirrored in y', pixel: [20, 51], expected: [0, 0, 0] },
];

export async function runFlatSplatsCheck(device: GPUDevice): Promise<string> {
  const renderer = new GaussianRenderer(device);
  const data = packSplats(SPLATS);
  const splatBuffer = device.createBuffer({
    label: 'flat splats',
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(splatBuffer, 0, data);
  const pixels = await renderPixels(device, SIZE, (encoder, target) =>
    renderer.encodeDraw(encoder, target, splatBuffer, SPLATS.length, [SIZE, SIZE], [0, 0, 0]),
  );
  splatBuffer.destroy();
  renderer.destroy();

  const failures: string[] = [];
  for (const { name, pixel, expected } of PROBES) {
    const gpu = pixelAt(pixels, SIZE, pixel);
    const cpu = shadePixel(SPLATS, pixel, [SIZE, SIZE]);
    // Checking the CPU version against hand-worked values catches the case where
    // the GPU and CPU are wrong in the same way.
    if (maxDiff(gpu, cpu) > TOLERANCE || maxDiff(cpu, expected) > TOLERANCE) {
      failures.push(`  ${name}: GPU ${formatRgb(gpu)}  CPU ${formatRgb(cpu)}  expected ${formatRgb(expected)}`);
    }
  }
  const cameraProblems = checkCameraMath();
  return [
    'M1.2: flat splats',
    `Pixel probes: ${verdict(failures)} (${PROBES.length - failures.length} of ${PROBES.length})`,
    ...failures,
    `Camera math: ${verdict(cameraProblems)}`,
    ...cameraProblems.map((problem) => `  ${problem}`),
  ].join('\n');
}

// A camera 5 units back on +z, looking at the origin.
function checkCameraMath(): string[] {
  const near = 0.1;
  const far = 100;
  const viewProj = multiply(perspective(Math.PI / 2, 1, near, far), lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]));
  const ndc = (point: Vec3): Vec3 => {
    const [x, y, z, w] = transform(viewProj, point);
    return [x / w, y / w, z / w];
  };
  const problems: string[] = [];
  const [cx, cy] = ndc([0, 0, 0]);
  if (Math.abs(cx) > 1e-9 || Math.abs(cy) > 1e-9) problems.push('the target is not at the screen center');
  if (!(ndc([1, 0, 0])[0] > 0)) problems.push('+x does not appear to the right');
  if (!(ndc([0, 1, 0])[1] > 0)) problems.push('+y does not appear above the center');
  // WebGPU's depth range: 0 at the near plane, 1 at the far plane.
  if (Math.abs(ndc([0, 0, 5 - near])[2]) > 1e-6) problems.push('the near plane does not map to depth 0');
  if (Math.abs(ndc([0, 0, 5 - far])[2] - 1) > 1e-6) problems.push('the far plane does not map to depth 1');
  return problems;
}
