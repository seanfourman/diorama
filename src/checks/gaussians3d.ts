import { GaussianRenderer, type CameraData } from "../gaussianRenderer";
import {
  packGaussians,
  unpackSplats,
  type Gaussian,
  type Quat,
  type Splat2D,
} from "../gaussians";
import { IDENTITY, perspective } from "../mat4";
import {
  formatRgb,
  maxDiff,
  pixelAt,
  readBuffer,
  renderPixels,
  verdict,
  type RGB,
} from "./helpers";
import { preprocess, shadePixel } from "./reference";

// M1.3: project a handful of 3D Gaussians to 2D splats on the GPU and compare
// every field with the same math on the CPU, and with values worked out by hand
// for the simple cases. Then render one stretched Gaussian and check its pixels
// and which way it points. Notes: docs/steps/1.3-3d-gaussians.md

const SIZE = 64;
const TOLERANCE = 2.5 / 255;
const BLACK: RGB = [0, 0, 0];
// A camera at the origin looking down −z with a 90° field of view: in a 64-pixel
// viewport, that's a focal length of 32 pixels.
const CAMERA: CameraData = {
  view: IDENTITY,
  proj: perspective(Math.PI / 2, 1, 0.1, 100),
  viewport: [SIZE, SIZE],
};

const white = (shape: Omit<Gaussian, "color" | "opacity">): Gaussian => ({
  ...shape,
  color: [1, 1, 1],
  opacity: 0.9,
});
// A 45° turn about the view axis (z): (cos 22.5°, 0, 0, sin 22.5°).
const TURN_45: Quat = [Math.cos(Math.PI / 8), 0, 0, Math.sin(Math.PI / 8)];
const STRETCHED = white({
  position: [0, 0, -4],
  scale: [1, 0.25, 0.25],
  rotation: TURN_45,
});

const CASES: {
  name: string;
  gaussian: Gaussian;
  expected?: Splat2D | "culled";
}[] = [
  {
    name: "round, straight ahead",
    gaussian: white({
      position: [0, 0, -4],
      scale: [0.5, 0.5, 0.5],
      rotation: [1, 0, 0, 0],
    }),
    // σ = 32 px × 0.5 / 4 = 4 px: a covariance of 16, plus the 0.3 blur, on each axis.
    expected: {
      mean: [32, 32],
      depth: 4,
      radius: 13,
      conic: [1 / 16.3, 0, 1 / 16.3],
      color: [1, 1, 1, 0.9],
    },
  },
  {
    name: "stretched along the up-right diagonal",
    gaussian: STRETCHED,
    // A 2D covariance of [[34.3, −30], [−30, 34.3]] with determinant 276.49. The
    // off-diagonal is negative because pixel y points down.
    expected: {
      mean: [32, 32],
      depth: 4,
      radius: 25,
      conic: [34.3 / 276.49, 30 / 276.49, 34.3 / 276.49],
      color: [1, 1, 1, 0.9],
    },
  },
  {
    name: "behind the camera",
    gaussian: white({
      position: [0, 0, 2],
      scale: [0.5, 0.5, 0.5],
      rotation: [1, 0, 0, 0],
    }),
    expected: "culled",
  },
  {
    name: "off to the side and tilted",
    gaussian: white({
      position: [1.2, 0.7, -5],
      scale: [0.3, 0.15, 0.6],
      rotation: [0.8, 0.3, -0.4, 0.2],
    }),
  },
  {
    name: "far outside the view",
    gaussian: white({
      position: [40, 0, -4],
      scale: [0.2, 0.2, 0.2],
      rotation: [1, 0, 0, 0],
    }),
    expected: "culled",
  },
];

// Pixels around the stretched Gaussian, whose long axis should point up and to the right.
const PROBES: { name: string; pixel: [number, number] }[] = [
  { name: "center", pixel: [32, 32] },
  { name: "on the long axis", pixel: [39, 24] },
  { name: "on the short axis", pixel: [39, 39] },
  { name: "near the far end", pixel: [44, 20] },
  { name: "outside its square", pixel: [60, 4] },
];

export async function runGaussiansCheck(device: GPUDevice): Promise<string> {
  const renderer = new GaussianRenderer(device);
  try {
    const projectionFailures = await checkProjection(device, renderer);
    const renderFailures = await checkRender(device, renderer);
    return [
      "M1.3: 3D Gaussians",
      `Projection: ${verdict(projectionFailures)} (${CASES.length} cases)`,
      ...projectionFailures,
      `Render: ${verdict(renderFailures)} (${PROBES.length} probes and the orientation)`,
      ...renderFailures,
    ].join("\n");
  } finally {
    renderer.destroy();
  }
}

async function checkProjection(
  device: GPUDevice,
  renderer: GaussianRenderer,
): Promise<string[]> {
  renderer.setGaussians(packGaussians(CASES.map(({ gaussian }) => gaussian)));
  // Only the splats the preprocess writes matter here; the drawn image is thrown away.
  await renderPixels(device, SIZE, (encoder, target) =>
    renderer.encode(encoder, target, CAMERA, BLACK),
  );
  const gpu = unpackSplats(
    new Float32Array(await readBuffer(device, renderer.splatBuffer!)),
  );

  const failures: string[] = [];
  CASES.forEach(({ name, gaussian, expected }, i) => {
    const cpu = preprocess(gaussian, CAMERA);
    const problems = compareSplats(
      "GPU",
      gpu[i].radius === 0 ? null : gpu[i],
      "CPU",
      cpu,
    );
    if (expected !== undefined) {
      problems.push(
        ...compareSplats(
          "CPU",
          cpu,
          "expected",
          expected === "culled" ? null : expected,
        ),
      );
    }
    if (problems.length) failures.push(`  ${name}: ${problems.join("; ")}`);
  });
  return failures;
}

async function checkRender(
  device: GPUDevice,
  renderer: GaussianRenderer,
): Promise<string[]> {
  renderer.setGaussians(packGaussians([STRETCHED]));
  const pixels = await renderPixels(device, SIZE, (encoder, target) =>
    renderer.encode(encoder, target, CAMERA, BLACK),
  );
  const cpuSplats = [preprocess(STRETCHED, CAMERA)].filter(
    (splat) => splat !== null,
  );

  const failures: string[] = [];
  for (const { name, pixel } of PROBES) {
    const gpu = pixelAt(pixels, SIZE, pixel);
    const cpu = shadePixel(cpuSplats, pixel, [SIZE, SIZE]);
    if (maxDiff(gpu, cpu) > TOLERANCE)
      failures.push(`  ${name}: GPU ${formatRgb(gpu)}  CPU ${formatRgb(cpu)}`);
  }
  // The orientation, stated directly: bright along the long axis, dark along the short one.
  const [long] = pixelAt(pixels, SIZE, [39, 24]);
  const [short] = pixelAt(pixels, SIZE, [39, 39]);
  if (!(long > 0.2 && short < 0.05)) {
    failures.push(
      `  orientation: ${long.toFixed(3)} on the long axis and ${short.toFixed(3)} on the short one`,
    );
  }
  return failures;
}

function compareSplats(
  nameA: string,
  a: Splat2D | null,
  nameB: string,
  b: Splat2D | null,
): string[] {
  if (!a || !b) {
    return a === b
      ? []
      : [
          `${nameA} ${a ? "kept" : "culled"} it but ${nameB} ${b ? "kept" : "culled"} it`,
        ];
  }
  // Relative to the value's size, with a floor of 1 for values near zero.
  const close = (x: number, y: number) =>
    Math.abs(x - y) <= 1e-4 * Math.max(1, Math.abs(y));
  const format = (values: readonly number[]) =>
    `(${values.map((v) => +v.toFixed(5)).join(", ")})`;
  const problems: string[] = [];
  if (!close(a.mean[0], b.mean[0]) || !close(a.mean[1], b.mean[1])) {
    problems.push(`mean ${format(a.mean)} vs ${format(b.mean)}`);
  }
  if (!close(a.depth, b.depth)) problems.push(`depth ${a.depth} vs ${b.depth}`);
  if (a.radius !== b.radius) problems.push(`radius ${a.radius} vs ${b.radius}`);
  if (!a.conic.every((value, k) => close(value, b.conic[k]))) {
    problems.push(`conic ${format(a.conic)} vs ${format(b.conic)}`);
  }
  return problems.map((problem) => `${nameA} vs ${nameB} ${problem}`);
}
