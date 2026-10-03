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
import { runScenesCheck } from './checks/scenes';
import { runTilesCheck } from './checks/tiles';
import { runTrainingCheck } from './checks/training';
import { initGpu } from './gpu';
import { runNewScenePage } from './newScenePage';
import { runTrainingPage } from './trainingPage';
import { runViewerPage } from './viewerPage';

const params = new URLSearchParams(location.search);
const canvas = document.querySelector<HTMLCanvasElement>('#gpu-canvas')!;
const panel = document.querySelector<HTMLDetailsElement>('#panel')!;
const summary = document.querySelector<HTMLElement>('#summary')!;
const log = document.querySelector<HTMLPreElement>('#log')!;

async function main(): Promise<void> {
  // ?new makes a scene from photos (M5). It doesn't draw anything itself.
  if (params.has('new')) {
    canvas.hidden = true;
    panel.hidden = true;
    await runNewScenePage(document.querySelector<HTMLElement>('#new-scene')!);
    return;
  }
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
      await runScenesCheck(),
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

  await runViewerPage(
    device,
    context,
    { canvas, summary, log },
    { sceneName: params.get('scene') ?? undefined, plyUrl: params.get('ply') ?? undefined, mode: params.get('view') ?? undefined },
    checkStatus,
  );
}

main().catch((err: unknown) => {
  summary.textContent = '✗ Error';
  log.textContent = err instanceof Error ? err.message : String(err);
  panel.open = true;
  document.body.dataset.status = 'error';
});
