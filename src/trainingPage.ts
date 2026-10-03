import { TrainingCharts } from './charts';
import { resizeToDisplay } from './gpu';
import { InsidePanel } from './insidePanel';
import { InsideView, samplePoints, visibleDepthRange } from './insideView';
import { initialGaussians, loadDataset, loadViews } from './loadDataset';
import { writeGaussianPly } from './plyWriter';
import { frameScene, verticalFov } from './sceneCameras';
import { DEFAULT_SETTINGS, Trainer } from './trainer';
import { ViewerCamera } from './viewerCamera';

// The training page (?train=<name>, M3 and M4): trains on data/<name>'s photos
// while showing the Gaussians as they learn, from a camera you can move. The
// Inside view (keys 1 to 7) shows what the optimizer is doing, with live curves.
// Measures the held-out photos at step 7,000 and at the end, as the paper reports.

export interface PageElements {
  canvas: HTMLCanvasElement;
  summary: HTMLElement;
  log: HTMLPreElement;
}

export interface TrainingPageOptions {
  totalSteps: number;
  /** The Inside view mode to start in, by name (as in insideView.ts's MODES). */
  mode?: string;
  /** A photo to start at, numbered from 1: the training photos first, then the held-out ones. */
  photo?: number;
}

export const TRAINING_CONTROLS = [
  'Space pauses and resumes training. P saves the Gaussians as a .ply.',
  '1 to 7 switch the Inside view, H hides the panels. [ and ] step through the photos, the held-out ones last.',
].join('\n');
/** Roughly how long each frame spends training, in milliseconds. Longer trains faster; shorter keeps the view smooth. */
const FRAME_BUDGET = 100;
/** How often the PSNR curve measures the held-out photos, in steps. */
const PSNR_EVERY = 1000;

export async function runTrainingPage(
  device: GPUDevice,
  context: GPUCanvasContext,
  { canvas, summary, log }: PageElements,
  name: string,
  options: TrainingPageOptions,
  checkStatus: string,
): Promise<void> {
  const { totalSteps } = options;
  summary.textContent = `Reading ${name}…`;
  const dataset = await loadDataset(name);
  const cameras = [...dataset.train, ...dataset.test];
  const views = await loadViews(device, name, cameras, (loaded) => {
    summary.textContent = `Loading photos… ${loaded} of ${cameras.length}`;
  });
  const trainViews = views.slice(0, dataset.train.length);
  const testViews = views.slice(dataset.train.length);
  summary.textContent = 'Placing the first Gaussians…';
  const trainer = new Trainer(device, await initialGaussians(device, dataset.positions, dataset.colors), trainViews, {
    ...DEFAULT_SETTINGS,
    extent: dataset.extent,
  });
  const { settings } = trainer;
  const count = (n: number) => n.toLocaleString('en-US');
  const [width, height] = trainViews[0].camera.viewport;
  log.textContent +=
    `\n\nTraining "${name}" for ${count(totalSteps)} steps on ${trainViews.length} photos at ${width}×${height}, ` +
    `with ${testViews.length} held out. Starting from ${count(trainer.count)} points (scene extent ${dataset.extent.toFixed(2)}).` +
    `\n${TRAINING_CONTROLS}`;

  const { center, up, radius } = frameScene(dataset.train);
  const camera = new ViewerCamera(canvas, {
    target: center,
    up,
    radius,
    eye: dataset.train[0].position,
    poses: cameras.map((photo) => ({ position: photo.position, forward: photo.forward, fovY: verticalFov(photo) })),
  });
  if (options.photo) camera.goToPhoto(Math.min(cameras.length, Math.max(1, options.photo)) - 1);

  const inside = new InsideView(device, trainer.renderer, [0.1 * radius, 10 * radius], trainer);
  // The structure-from-motion points stand in for the scene when fitting the depth scale.
  const depthSample = samplePoints(dataset.positions, 3);
  if (options.mode && inside.modes.some(({ mode }) => mode === options.mode)) inside.mode = options.mode as typeof inside.mode;
  const panel = new InsidePanel(true);
  const resets = [];
  for (let step = settings.opacityResetInterval; step < settings.densifyUntil; step += settings.opacityResetInterval) resets.push(step);
  const charts = new TrainingCharts(panel.charts, { totalSteps, opacityResets: resets, densifyUntil: settings.densifyUntil });

  let paused = false;
  let saving = false;
  window.addEventListener('keydown', (event) => {
    if (inside.handleKey(event.code)) return;
    if (event.code === 'Space') {
      paused = !paused;
      event.preventDefault();
    } else if (event.code === 'KeyP' && !saving) {
      saving = true;
      void saveGaussians(trainer, `${name}-${trainer.iteration}.ply`).finally(() => (saving = false));
    }
  });

  const photoLabel = (index: number) =>
    `photo ${index + 1} of ${cameras.length}${index >= trainViews.length ? ' (held out)' : ''}`;
  // What the panel adds to the current mode's legend.
  const panelNote = (): string => {
    const index = camera.photoIndex;
    if (inside.info.photo) {
      return index < 0 ? 'Press [ or ] to jump to a photo; the held-out ones come last.' : `Showing ${photoLabel(index)}.`;
    }
    if (inside.mode === 'gradient' && trainer.iteration >= settings.densifyUntil) {
      return `Densification ended at step ${count(settings.densifyUntil)}; this is the average since then.`;
    }
    const last = trainer.densifications.at(-1);
    if (inside.mode === 'age' && last) {
      return (
        `Last densification, step ${count(last.step)}: ${count(last.cloned)} cloned, ${count(last.split)} split, ` +
        `${count(last.removed)} removed (${count(last.before)} → ${count(last.after)}).`
      );
    }
    return '';
  };

  const evaluations = new Set([7000, totalSteps]);
  let stepsPerFrame = 1;
  let trainingMs = 0;
  let loss = NaN;
  let lossReadAt = 0;
  let lastTime = performance.now();
  // Steps a second, measured over about a second: from `steps` done at `start`.
  let rate = { start: lastTime, steps: 0, perSecond: 0 };
  const showStatus = () => {
    const training = trainer.iteration < totalSteps;
    summary.textContent = [
      checkStatus,
      `${name}: step ${count(trainer.iteration)} of ${count(totalSteps)}`,
      `${count(trainer.count)} Gaussians`,
      `loss ${loss.toFixed(4)}`,
      training ? `${rate.perSecond.toFixed(0)} steps/s` : '',
      formatDuration(trainingMs),
      paused && training ? 'paused' : training ? '' : 'done',
      camera.photoIndex >= 0 ? photoLabel(camera.photoIndex) : camera.label,
    ]
      .filter(Boolean)
      .join(' · ');
  };
  charts.addPsnr(0, await inside.measurePsnr(testViews, settings.background));

  for (;;) {
    const frameStart = performance.now();
    const training = !paused && trainer.iteration < totalSteps;
    let evaluateNow = false;
    let measureNow = false;
    if (training) {
      for (let k = 0; k < stepsPerFrame && trainer.iteration < totalSteps && !evaluateNow && !measureNow; k++) {
        trainer.optimize();
        evaluateNow = evaluations.has(trainer.iteration);
        measureNow = trainer.iteration % PSNR_EVERY === 0;
        // A step to measure keeps its densification and opacity reset until after measuring.
        if (!evaluateNow && !measureNow) await trainer.maintain();
      }
    }

    // The live view, through the trainer's own renderer.
    camera.update(Math.max(0, frameStart - lastTime) / 1000);
    lastTime = frameStart;
    resizeToDisplay(canvas, device.limits.maxTextureDimension2D);
    const encoder = device.createCommandEncoder({ label: 'live view' });
    const photo = camera.photoIndex >= 0 ? views[camera.photoIndex] : undefined;
    const matrices = camera.matrices(canvas.width / canvas.height);
    if (inside.mode === 'depth') inside.depthRange = visibleDepthRange(depthSample, matrices.view);
    inside.encode(encoder, context.getCurrentTexture(), { ...matrices, viewport: [canvas.width, canvas.height] }, settings.background, photo);
    device.queue.submit([encoder.finish()]);
    void trainer.renderer.afterSubmit();
    await device.queue.onSubmittedWorkDone();

    const now = performance.now();
    if (training) {
      const frameMs = now - frameStart;
      trainingMs += frameMs;
      stepsPerFrame = Math.max(1, Math.min(500, Math.round(stepsPerFrame * Math.min(2, FRAME_BUDGET / frameMs))));
    }
    if (now - lossReadAt > 500 && trainer.iteration > 0 && training) {
      lossReadAt = now;
      loss = await trainer.lastLoss();
      charts.addLoss(trainer.iteration, loss);
    }
    charts.addCount(trainer.iteration, trainer.count);
    if (now - rate.start > 1000) {
      rate = { start: now, steps: trainer.iteration, perSecond: ((trainer.iteration - rate.steps) * 1000) / (now - rate.start) };
    }
    // Measuring isn't counted as training time.
    if (evaluateNow) {
      summary.textContent = `Measuring the ${testViews.length} held-out photos at step ${count(trainer.iteration)}…`;
      const { psnr, ssim } = await trainer.evaluate(testViews);
      charts.addPsnr(trainer.iteration, psnr);
      log.textContent +=
        `\nStep ${count(trainer.iteration)}: test PSNR ${psnr.toFixed(2)} dB, SSIM ${ssim.toFixed(3)} ` +
        `(${testViews.length} held-out photos); ${count(trainer.count)} Gaussians; ${formatDuration(trainingMs)} of training`;
    } else if (measureNow) {
      charts.addPsnr(trainer.iteration, await inside.measurePsnr(testViews, settings.background));
    }
    if (evaluateNow || measureNow) rate = { start: performance.now(), steps: trainer.iteration, perSecond: rate.perSecond };
    // The last step's densification or opacity reset would only spoil the result.
    if ((evaluateNow || measureNow) && trainer.iteration < totalSteps) await trainer.maintain();
    // Tells scripts/check.mjs the run has finished.
    if (evaluateNow && trainer.iteration >= totalSteps) document.body.dataset.status = 'done';

    showStatus();
    panel.update(inside, panelNote());
    charts.draw(trainer.densifications);
    await new Promise(requestAnimationFrame);
  }
}

async function saveGaussians(trainer: Trainer, fileName: string): Promise<void> {
  const url = URL.createObjectURL(writeGaussianPly(await trainer.exportRaw()));
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${Math.floor(seconds / 60)}:${pad(seconds % 60)}`;
}
