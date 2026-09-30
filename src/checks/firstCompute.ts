import projectWgsl from '../shaders/project.wgsl?raw';

// M1.1: run the projection kernel on a few points, read the results back, and
// check them against the same math on the CPU. Brief: docs/steps/1.1-first-compute-pass.md

type Vec2 = [number, number];
type Vec3 = [number, number, number];

interface TestCase {
  name: string;
  /** 4×4 matrix in column-major order, the order WGSL reads a mat4x4<f32> in. */
  viewProj: readonly number[];
  expected: Vec2[];
}

const VIEWPORT: Vec2 = [800, 600];
const WORKGROUP_SIZE = 256; // must match @workgroup_size in project.wgsl
const TOLERANCE_PX = 1e-3;

const POINTS: Vec3[] = [
  [0, 0, 0],
  [0.5, 0.5, 0],
  [-1, -1, 0],
  [1, 1, 0],
  [0.25, -0.5, 0],
];

// Each row of these matrix literals is one *column* of the matrix.
const TESTS: TestCase[] = [
  {
    name: 'Test A (identity)',
    viewProj: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ],
    expected: [[400, 300], [600, 150], [0, 600], [800, 0], [500, 450]],
  },
  {
    name: 'Test B (translate x by 0.5)',
    viewProj: [
      1,   0, 0, 0,
      0,   1, 0, 0,
      0,   0, 1, 0,
      0.5, 0, 0, 1,
    ],
    expected: [[600, 300], [800, 150], [200, 600], [1000, 0], [700, 450]],
  },
];

export async function runFirstCompute(device: GPUDevice): Promise<string> {
  const module = device.createShaderModule({ label: 'project', code: projectWgsl });
  const pipeline = device.createComputePipeline({
    label: 'project',
    layout: 'auto',
    compute: { module, entryPoint: 'project' },
  });

  const lines = ['M1.1: first compute pass'];
  for (const test of TESTS) {
    const gpu = await projectOnGpu(device, pipeline, test.viewProj, POINTS);
    const cpu = POINTS.map((point) => projectOnCpu(test.viewProj, point));
    // Checking the CPU version against hand-computed values catches the case
    // where the GPU and CPU are wrong in the same way.
    const pass =
      maxError(gpu, cpu) <= TOLERANCE_PX && maxError(cpu, test.expected) <= TOLERANCE_PX;
    lines.push(`${test.name}: ${pass ? 'PASS' : 'FAIL'}`, `  GPU:      ${format(gpu)}`);
    if (!pass) {
      lines.push(`  CPU:      ${format(cpu)}`, `  expected: ${format(test.expected)}`);
    }
  }
  return lines.join('\n');
}

async function projectOnGpu(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  viewProj: readonly number[],
  points: Vec3[],
): Promise<Vec2[]> {
  // The Camera struct is 16 floats of matrix, 2 of viewport, then 2 of padding:
  // WGSL rounds a struct up to its largest alignment (16 bytes for mat4x4), so
  // it's 80 bytes, not 72.
  const cameraData = new Float32Array(20);
  cameraData.set(viewProj, 0);
  cameraData.set(VIEWPORT, 16);

  // One vec4 (16 bytes) per point. The shader only reads xyz; the 4th slot is padding.
  const positionData = new Float32Array(points.length * 4);
  points.forEach((point, i) => positionData.set(point, i * 4));

  const outputSize = points.length * 2 * Float32Array.BYTES_PER_ELEMENT;
  const camera = device.createBuffer({
    label: 'camera',
    size: cameraData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const positions = device.createBuffer({
    label: 'positions',
    size: positionData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const screen = device.createBuffer({
    label: 'screen',
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  // The CPU can't read a storage buffer, so the results get copied into one it can map.
  const readback = device.createBuffer({
    label: 'readback',
    size: outputSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(camera, 0, cameraData);
  device.queue.writeBuffer(positions, 0, positionData);

  const bindGroup = device.createBindGroup({
    label: 'project',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: screen } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: 'project' });
  const pass = encoder.beginComputePass({ label: 'project' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(points.length / WORKGROUP_SIZE));
  pass.end();
  encoder.copyBufferToBuffer(screen, 0, readback, 0, outputSize);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  // The mapped range is gone after unmap(), so copy it out first.
  const flat = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  for (const buffer of [camera, positions, screen, readback]) buffer.destroy();

  return points.map((_, i): Vec2 => [flat[2 * i], flat[2 * i + 1]]);
}

// The shader's math, on the CPU. m is column-major: m[column * 4 + row].
function projectOnCpu(m: readonly number[], [x, y, z]: Vec3): Vec2 {
  const clip = [0, 1, 2, 3].map((row) => m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]);
  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];
  // Clip-space y points up and screen y points down, hence the flip.
  return [(ndcX * 0.5 + 0.5) * VIEWPORT[0], (-ndcY * 0.5 + 0.5) * VIEWPORT[1]];
}

function maxError(a: Vec2[], b: Vec2[]): number {
  // NaN propagates through Math.max, so a NaN anywhere fails the check.
  return Math.max(...a.flatMap(([x, y], i) => [Math.abs(x - b[i][0]), Math.abs(y - b[i][1])]));
}

function format(points: Vec2[]): string {
  return points.map(([x, y]) => `(${+x.toFixed(3)}, ${+y.toFixed(3)})`).join(' ');
}
