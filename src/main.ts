import { runBackwardCheck } from './checks/backward';
import { runDatasetCheck } from './checks/dataset';
import { runDensifyCheck } from './checks/densify';
import { runDepthSortCheck } from './checks/depthSort';
import { runFirstCompute } from './checks/firstCompute';
import { runFlatSplatsCheck } from './checks/flatSplats';
import { runGaussiansCheck } from './checks/gaussians3d';
import { runInsideCheck } from './checks/inside';
import { runLossCheck } from './checks/loss';
import { runRealSceneCheck } from './checks/realScene';
import { runTilesCheck } from './checks/tiles';
import { runTrainingCheck } from './checks/training';
import { GaussianRenderer } from './gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, packGaussians } from './gaussians';
import { initGpu, resizeToDisplay } from './gpu';
import { InsidePanel } from './insidePanel';
import { InsideView, samplePoints, visibleDepthRange, type InsideMode } from './insideView';
import { loadScene } from './loadScene';
import { frameScene, verticalFov } from './sceneCameras';
import { trefoilKnot } from './testScene';
import { runTrainingPage } from './trainingPage';
import type { RGB } from './tileRasterizer';
import { ViewerCamera } from './viewerCamera';

const KNOT_BACKGROUND: RGB = [0.07, 0.08, 0.12];
const CONTROLS = [
  'Drag to turn, scroll to zoom.',
  'WASD or arrows to fly, Q/E down and up, Shift for speed.',
  '[ and ] step through the training photos, O orbits, R resets.',
  '1 to 3 switch the Inside view: color, depth, and work per pixel. H hides the panels.',
].join('\n');

const params = new URLSearchParams(location.search);
const canvas = document.querySelector<HTMLCanvasElement>('#gpu-canvas')!;
const panel = document.querySelector<HTMLDetailsElement>('#panel')!;
const summary = document.querySelector<HTMLElement>('#summary')!;
const log = document.querySelector<HTMLPreElement>('#log')!;

async function main(): Promise<void> {
  const { adapter, device, context } = await initGpu(canvas);

  const mib = (bytes: number) => `${Math.round(bytes / 2 ** 20)} MiB`;
  const { vendor, architecture, description } = adapter.info;
  const gpuName = [vendor, architecture, description].filter(Boolean).join(' / ');
  // Optional features later milestones can use: GPU timings, faster sorts and
  // reductions, and half-precision math.
  const features = ['timestamp-query', 'subgroups', 'shader-f16']
    .map((name) => `${adapter.features.has(name) ? '✓' : '✗'} ${name}`)
    .join('  ');
  log.textContent = [
    `GPU: ${gpuName || '(hidden by this browser)'}`,
    `maxStorageBufferBindingSize: ${mib(device.limits.maxStorageBufferBindingSize)}`,
    `maxBufferSize: ${mib(device.limits.maxBufferSize)}`,
    `Features: ${features}`,
  ].join('\n');

  // The checks take a few seconds, so they only run when asked for (?check).
  let checkStatus = '';
  if (params.has('check')) {
    summary.textContent = 'Running the GPU checks…';
    const reports = [
      await runFirstCompute(device),
      await runFlatSplatsCheck(device),
      await runGaussiansCheck(device),
      await runDepthSortCheck(device),
      await runTilesCheck(device),
      await runRealSceneCheck(device),
      await runBackwardCheck(device),
      await runDatasetCheck(device),
      await runLossCheck(device),
      await runTrainingCheck(device),
      await runDensifyCheck(device),
      await runInsideCheck(device),
    ];
    log.textContent += '\n\n' + reports.join('\n\n');
    const failed = reports.some((report) => /\bFAIL\b/.test(report));
    checkStatus = failed ? '✗ Some GPU checks failed' : '✓ All GPU checks passed';
    panel.open = failed;
  } else {
    log.textContent += '\n\nAdd ?check to the address to run the GPU checks.';
  }

  // ?train=<name> trains on data/<name>'s photos instead of showing a finished scene.
  const trainName = params.get('train');
  if (trainName) {
    // &view=<mode> and &photo=<n> start in an Inside view mode, at a photo.
    const options = {
      totalSteps: Number(params.get('steps') ?? 30_000),
      mode: params.get('view') ?? undefined,
      photo: params.has('photo') ? Number(params.get('photo')) : undefined,
    };
    await runTrainingPage(device, context, { canvas, summary, log }, trainName, options, checkStatus);
    return;
  }

  const renderer = new GaussianRenderer(device);
  const sceneName = params.get('scene');
  let camera: ViewerCamera;
  let background: RGB;
  let sceneSize: number;
  let depthSample: Float32Array;
  if (sceneName) {
    const scene = await loadScene(sceneName, (message) => (summary.textContent = message));
    renderer.setGaussians(scene.gaussians, { rest: scene.shRest, degree: scene.shDegree });
    const { center, up, radius } = frameScene(scene.cameras);
    camera = new ViewerCamera(canvas, {
      target: center,
      up,
      radius,
      eye: scene.cameras[0].position,
      poses: scene.cameras.map((photo) => ({ position: photo.position, forward: photo.forward, fovY: verticalFov(photo) })),
    });
    background = [0, 0, 0]; // what the scene was trained against
    sceneSize = radius;
    depthSample = samplePoints(scene.gaussians, FLOATS_PER_GAUSSIAN);
    log.textContent +=
      `\n\nScene: ${sceneName}, ${scene.count.toLocaleString('en-US')} Gaussians, spherical harmonics up to degree ` +
      `${scene.shDegree}, ${scene.cameras.length} training photos.`;
  } else {
    renderer.setGaussians(packGaussians(trefoilKnot(2000)));
    camera = new ViewerCamera(canvas, { eye: [1.49, 1.44, 2.17] });
    background = KNOT_BACKGROUND;
    sceneSize = 3;
    depthSample = new Float32Array([0, 0, 0]);
    log.textContent +=
      '\n\nScene: the test knot. Add ?scene=train to load a real one (npm run download-scene fetches it), ' +
      'or ?train=train to train one from photos (npm run download-photos fetches them).';
  }
  log.textContent += `\n${CONTROLS}`;
  // The Inside view's modes that need no training (M4): color, depth and work.
  const inside = new InsideView(device, renderer, [0.1 * sceneSize, 10 * sceneSize]);
  const insidePanel = new InsidePanel(false);
  if (inside.modes.some(({ mode }) => mode === params.get('view'))) inside.mode = params.get('view') as InsideMode;
  window.addEventListener('keydown', (event) => inside.handleKey(event.code));

  let lastTime = performance.now();
  let statsStart = lastTime;
  let framesSinceStats = 0;
  let fps = 0;
  const showStatus = () => {
    summary.textContent = [checkStatus, sceneName ?? 'knot', camera.label, `${fps} fps`].filter(Boolean).join(' · ');
  };
  const frame = (time: number) => {
    camera.update(Math.max(0, time - lastTime) / 1000);
    lastTime = time;
    resizeToDisplay(canvas, device.limits.maxTextureDimension2D);
    const encoder = device.createCommandEncoder();
    const matrices = camera.matrices(canvas.width / canvas.height);
    if (inside.mode === 'depth' && sceneName) inside.depthRange = visibleDepthRange(depthSample, matrices.view);
    inside.encode(encoder, context.getCurrentTexture(), { ...matrices, viewport: [canvas.width, canvas.height] }, background);
    device.queue.submit([encoder.finish()]);
    // Grows the renderer's tile buffers if this frame needed more room. Nothing waits on it.
    void renderer.afterSubmit();

    framesSinceStats++;
    if (time - statsStart >= 500) {
      fps = Math.round((framesSinceStats * 1000) / (time - statsStart));
      framesSinceStats = 0;
      statsStart = time;
    }
    showStatus();
    insidePanel.update(inside);
    requestAnimationFrame(frame);
  };
  showStatus();
  requestAnimationFrame(frame);
  // Tells scripts/check.mjs the checks have finished and the scene is on screen.
  document.body.dataset.status = 'done';
}

main().catch((err: unknown) => {
  summary.textContent = '✗ Error';
  log.textContent = err instanceof Error ? err.message : String(err);
  panel.open = true;
  document.body.dataset.status = 'error';
});
