import { GaussianRenderer } from './gaussianRenderer';
import { FLOATS_PER_GAUSSIAN, packGaussians } from './gaussians';
import { resizeToDisplay } from './gpu';
import { InsidePanel } from './insidePanel';
import { InsideView, samplePoints, visibleDepthRange, type InsideMode } from './insideView';
import { loadPlyFromUrl, loadScene } from './loadScene';
import { parseGaussianPly, type GaussianScene } from './plyLoader';
import { frameGaussians, frameScene, parseCameras, verticalFov, type TrainingCamera } from './sceneCameras';
import { trefoilKnot } from './testScene';
import type { RGB } from './tileRasterizer';
import { ViewerCamera } from './viewerCamera';

// The viewer (M1.6, M5): walk through a trained scene. It shows data/<name>
// (?scene=<name>), a .ply from any URL (?ply=<url>, for sharing), or a .ply
// dropped onto the page, with a cameras.json alongside if there is one; or the
// test knot. The Inside view's color, depth and work modes work here too.

export const VIEWER_CONTROLS = [
  'Drag to turn, scroll to zoom.',
  'WASD or arrows to fly, Q/E down and up, Shift for speed.',
  '[ and ] step through the training photos, O orbits, R resets.',
  '1 to 3 switch the Inside view: color, depth, and work per pixel. H hides the panels.',
  'Drop a trained .ply (and its cameras.json) onto the page to view it.',
].join('\n');

const KNOT_BACKGROUND: RGB = [0.07, 0.08, 0.12];

export interface ViewerElements {
  canvas: HTMLCanvasElement;
  summary: HTMLElement;
  log: HTMLPreElement;
}

export interface ViewerOptions {
  sceneName?: string;
  plyUrl?: string;
  mode?: string;
}

type Scene = GaussianScene & { cameras: TrainingCamera[] };

export async function runViewerPage(
  device: GPUDevice,
  context: GPUCanvasContext,
  { canvas, summary, log }: ViewerElements,
  options: ViewerOptions,
  checkStatus: string,
): Promise<void> {
  const renderer = new GaussianRenderer(device);
  const inside = new InsideView(device, renderer, [0.1, 10]);
  const insidePanel = new InsidePanel(false);
  if (inside.modes.some(({ mode }) => mode === options.mode)) inside.mode = options.mode as InsideMode;
  window.addEventListener('keydown', (event) => inside.handleKey(event.code));

  let label = 'knot';
  let camera: ViewerCamera | undefined;
  let background: RGB = KNOT_BACKGROUND;
  let depthSample: Float32Array = new Float32Array(3);
  const show = (scene: Scene, name: string) => {
    renderer.setGaussians(scene.gaussians, { rest: scene.shRest, degree: scene.shDegree });
    // With cameras, frame the scene the way it was photographed; without, around the Gaussians.
    const { center, up, radius } = scene.cameras.length
      ? frameScene(scene.cameras)
      : frameGaussians(scene.gaussians, FLOATS_PER_GAUSSIAN);
    camera?.dispose();
    camera = new ViewerCamera(canvas, {
      target: center,
      up,
      radius,
      eye: scene.cameras[0]?.position,
      poses: scene.cameras.map((photo) => ({ position: photo.position, forward: photo.forward, fovY: verticalFov(photo) })),
    });
    background = [0, 0, 0]; // what scenes are trained against
    depthSample = samplePoints(scene.gaussians, FLOATS_PER_GAUSSIAN);
    label = name;
    log.textContent +=
      `\n\nScene: ${name}, ${scene.count.toLocaleString('en-US')} Gaussians, spherical harmonics up to degree ` +
      `${scene.shDegree}, ${scene.cameras.length ? `${scene.cameras.length} training photos` : 'no camera list'}.`;
  };

  if (options.sceneName) {
    show(await loadScene(options.sceneName, (message) => (summary.textContent = message)), options.sceneName);
  } else if (options.plyUrl) {
    show(await loadPlyFromUrl(options.plyUrl, (message) => (summary.textContent = message)), options.plyUrl.split('/').pop()!);
  } else {
    renderer.setGaussians(packGaussians(trefoilKnot(2000)));
    camera = new ViewerCamera(canvas, { eye: [1.49, 1.44, 2.17] });
    log.textContent +=
      '\n\nScene: the test knot. Add ?scene=train to load a real one (npm run download-scene fetches it), ' +
      '?train=train to train one from photos (npm run download-photos fetches them), or ?new to make one from yours.';
  }
  log.textContent += `\n${VIEWER_CONTROLS}`;

  // A dropped .ply replaces the scene; a cameras.json dropped with it adds the photo viewpoints.
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    const files = [...(event.dataTransfer?.files ?? [])];
    const ply = files.find((file) => file.name.toLowerCase().endsWith('.ply'));
    if (!ply) return;
    const camerasFile = files.find((file) => file.name.toLowerCase().endsWith('.json'));
    summary.textContent = `Reading ${ply.name}…`;
    (async () => {
      const scene = parseGaussianPly(await ply.arrayBuffer());
      const cameras = camerasFile ? parseCameras(JSON.parse(await camerasFile.text())) : [];
      show({ ...scene, cameras }, ply.name);
    })().catch((error: unknown) => {
      log.textContent += `\n\nCouldn't open ${ply.name}: ${error instanceof Error ? error.message : String(error)}`;
    });
  });

  let lastTime = performance.now();
  let statsStart = lastTime;
  let framesSinceStats = 0;
  let fps = 0;
  const frame = (time: number) => {
    const view = camera!;
    view.update(Math.max(0, time - lastTime) / 1000);
    lastTime = time;
    resizeToDisplay(canvas, device.limits.maxTextureDimension2D);
    const encoder = device.createCommandEncoder();
    const matrices = view.matrices(canvas.width / canvas.height);
    if (inside.mode === 'depth') inside.depthRange = visibleDepthRange(depthSample, matrices.view);
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
    summary.textContent = [checkStatus, label, view.label, `${fps} fps`].filter(Boolean).join(' · ');
    insidePanel.update(inside);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  // Tells scripts/check.mjs the checks have finished and the scene is on screen.
  document.body.dataset.status = 'done';
}
