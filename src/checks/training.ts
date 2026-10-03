import { GaussianRenderer, type CameraData } from '../gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS, packGaussians, type Gaussian } from '../gaussians';
import { lookAt, perspective, type Vec3 } from '../mat4';
import { parseGaussianPly } from '../plyLoader';
import { writeGaussianPly } from '../plyWriter';
import { mulberry32 } from '../random';
import { DEFAULT_SETTINGS, Trainer, activate, type RawGaussians, type TrainingSettings, type TrainingView } from '../trainer';
import { verdict } from './helpers';

// M3.4: training works, end to end. Render 8 views of a random "true" scene as the
// photos, start from a scrambled copy of it, train, and check the renders got much
// closer to the photos. Then M3.5: start from a few Gaussians in the wrong places
// and check that densification grows them into a better fit than training alone.
// Notes: docs/steps/3-training.md

const SIZE = 96;
const STEPS = 400;
const SH_C0 = 0.28209479177387814;

export async function runTrainingCheck(device: GPUDevice): Promise<string> {
  const random = mulberry32(17);
  const between = (lo: number, hi: number) => lo + (hi - lo) * random();
  const truth: Gaussian[] = Array.from({ length: 60 }, () => ({
    position: [between(-1, 1), between(-1, 1), between(-1, 1)],
    scale: [between(0.08, 0.3), between(0.08, 0.3), between(0.08, 0.3)],
    rotation: [between(-1, 1), between(-1, 1), between(-1, 1), between(-1, 1)],
    color: [random(), random(), random()],
    opacity: between(0.6, 0.95),
  }));

  // Eight cameras on a ring around the scene, looking at its center.
  const cameras: CameraData[] = Array.from({ length: 8 }, (_, k) => {
    const angle = (k / 8) * 2 * Math.PI;
    const eye: Vec3 = [3.5 * Math.sin(angle), 1.2, 3.5 * Math.cos(angle)];
    return { view: lookAt(eye, [0, 0, 0], [0, 1, 0]), proj: perspective(Math.PI / 3.5, 1, 0.1, 100), viewport: [SIZE, SIZE] };
  });
  const photographer = new GaussianRenderer(device);
  photographer.setGaussians(packGaussians(truth));
  const views: TrainingView[] = cameras.map((camera, k) => {
    const image = device.createTexture({
      label: `photo ${k}`,
      size: [SIZE, SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const encoder = device.createCommandEncoder();
    photographer.encode(encoder, image.createView(), camera, [0, 0, 0]);
    device.queue.submit([encoder.finish()]);
    return { name: `view ${k}`, camera, image };
  });
  photographer.destroy();

  const train = async (params: Float32Array<ArrayBuffer>, overrides: Partial<TrainingSettings>) => {
    const count = params.length / FLOATS_PER_GAUSSIAN;
    const trainer = new Trainer(
      device,
      { params, sh: new Float32Array(count * SH_REST_FLOATS) },
      views,
      { ...DEFAULT_SETTINGS, extent: 3.85, ...overrides },
    );
    const before = (await trainer.evaluate(views, false)).psnr;
    const start = performance.now();
    for (let k = 0; k < STEPS; k++) await trainer.step();
    const loss = await trainer.lastLoss();
    const seconds = (performance.now() - start) / 1000;
    const after = (await trainer.evaluate(views, false)).psnr;
    const result = { before, after, loss, stepsPerSecond: STEPS / seconds, count: trainer.count, raw: await trainer.exportRaw() };
    trainer.destroy();
    return result;
  };

  // The starting point: every Gaussian moved, resized, recolored and half transparent.
  const scrambled = new Float32Array(truth.length * FLOATS_PER_GAUSSIAN);
  truth.forEach(({ position, scale }, i) => {
    const at = i * FLOATS_PER_GAUSSIAN;
    for (let k = 0; k < 3; k++) {
      scrambled[at + k] = position[k] + between(-0.15, 0.15);
      scrambled[at + 4 + k] = Math.log(scale[k] * between(0.6, 1.6));
      scrambled[at + 12 + k] = (random() - 0.5) / SH_C0; // a random color
    }
    scrambled[at + 3] = 0; // logit of 0.5
    scrambled.set([1, 0, 0, 0], at + 8);
  });
  const fit = await train(scrambled, {});
  const fitFailures = fit.after - fit.before >= 6 ? [] : [`  PSNR only went from ${fit.before.toFixed(1)} to ${fit.after.toFixed(1)} dB`];

  // Too few Gaussians: 12 gray ones scattered through the scene. Densify every 50
  // steps from step 100, instead of every 100 from 500, to fit the short run.
  const sparse = new Float32Array(12 * FLOATS_PER_GAUSSIAN);
  for (let i = 0; i < 12; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    for (let k = 0; k < 3; k++) {
      sparse[at + k] = between(-0.8, 0.8);
      sparse[at + 4 + k] = Math.log(0.25);
    }
    sparse[at + 3] = Math.log(0.1 / 0.9);
    sparse.set([1, 0, 0, 0], at + 8);
  }
  const plain = await train(sparse.slice(), { densifyUntil: 0 });
  const densified = await train(sparse.slice(), { densifyFrom: 100, densifyInterval: 50, densifyUntil: 351 });
  views.forEach(({ image }) => image.destroy());
  const densifyFailures = [
    ...(densified.count > 12 ? [] : [`  still ${densified.count} Gaussians`]),
    ...(densified.after > plain.after + 1 ? [] : ['  densifying didn\'t beat training alone by 1 dB']),
  ];

  const plyFailures = await checkPlyRoundTrip(fit.raw);

  return [
    'M3.4: training',
    `Learning a scene from 8 views: ${verdict(fitFailures)} (PSNR ${fit.before.toFixed(1)} → ${fit.after.toFixed(1)} dB ` +
      `in ${STEPS} steps, ${fit.stepsPerSecond.toFixed(0)} steps a second; last loss ${fit.loss.toFixed(4)})`,
    ...fitFailures,
    `Growing from 12 Gaussians: ${verdict(densifyFailures)} (with densification ${densified.after.toFixed(1)} dB ` +
      `and ${densified.count} Gaussians; without, ${plain.after.toFixed(1)} dB)`,
    ...densifyFailures,
    `Saving as .ply: ${verdict(plyFailures)} (the trained Gaussians, written and read back by the scene loader)`,
    ...plyFailures,
  ].join('\n');
}

async function checkPlyRoundTrip(raw: RawGaussians): Promise<string[]> {
  const scene = parseGaussianPly(await writeGaussianPly(raw).arrayBuffer());
  const expected = activate(raw.params);
  const failures: string[] = [];
  if (scene.count !== expected.length / FLOATS_PER_GAUSSIAN || scene.shDegree !== 3) {
    failures.push(`  read back ${scene.count} Gaussians of degree ${scene.shDegree}`);
  }
  const worst = (a: Float32Array, b: Float32Array) => a.reduce((most, value, k) => Math.max(most, Math.abs(value - b[k])), 0);
  const paramsError = worst(scene.gaussians, expected);
  const shError = worst(scene.shRest, raw.sh);
  if (!(paramsError < 1e-6 && shError === 0)) {
    failures.push(`  read back off by ${paramsError} (parameters) and ${shError} (spherical harmonics)`);
  }
  return failures;
}
