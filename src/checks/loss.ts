import { ssim } from '../metrics';
import { TrainingLoss } from '../trainingLoss';
import fixture from './fixtures/loss.json';
import { readBuffer, verdict } from './helpers';

// M3.2: the training loss against PyTorch. reference/loss_reference.py computes
// 0.8 × L1 + 0.2 × (1 − SSIM) for a random render and target, with its gradient
// from autograd (checked there against finite differences). This check runs the
// GPU loss on the same images and compares both. It also rebuilds the loss from
// metrics.ts's CPU SSIM, which measures the held-out photos.
// Notes: docs/steps/3-training.md

interface LossFixture {
  size: [number, number];
  render: number[];
  target: number[];
  loss: number;
  gradient: number[];
  finiteDifferenceError: number;
}

const data = fixture as unknown as LossFixture;

export async function runLossCheck(device: GPUDevice): Promise<string> {
  const [width, height] = data.size;
  const pixels = width * height;
  const rendered = device.createBuffer({ size: pixels * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(
    rendered,
    0,
    Float32Array.from({ length: pixels * 4 }, (_, k) => (k % 4 === 3 ? 0 : data.render[Math.floor(k / 4) * 3 + (k % 4)])),
  );
  const target = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: target },
    Uint8Array.from({ length: pixels * 4 }, (_, k) => (k % 4 === 3 ? 255 : data.target[Math.floor(k / 4) * 3 + (k % 4)])),
    { bytesPerRow: width * 4 },
    [width, height],
  );

  const loss = new TrainingLoss(device, width, height);
  const encoder = device.createCommandEncoder();
  loss.encode(encoder, rendered, target.createView());
  device.queue.submit([encoder.finish()]);
  const value = await loss.read();
  const gradient = new Float32Array(await readBuffer(device, loss.pixelGrads));
  loss.destroy();
  rendered.destroy();
  target.destroy();

  let worst = 0;
  let largest = 0;
  for (let p = 0; p < pixels; p++) {
    for (let channel = 0; channel < 3; channel++) {
      const expected = data.gradient[p * 3 + channel];
      largest = Math.max(largest, Math.abs(expected));
      worst = Math.max(worst, Math.abs(gradient[p * 4 + channel] - expected));
    }
  }
  const lossError = Math.abs(value - data.loss);
  const failures = [
    ...(lossError <= 1e-5 ? [] : [`  loss ${value} vs ${data.loss}`]),
    ...(worst / largest <= 2e-3 ? [] : [`  gradient off by ${(worst / largest).toExponential(1)} of the largest`]),
  ];

  // The same loss from the CPU's SSIM: 0.8 × mean |x − y| + 0.2 × (1 − SSIM).
  const asImage = (values: number[], scale: number) =>
    Float32Array.from({ length: pixels * 4 }, (_, k) => (k % 4 === 3 ? 0 : values[Math.floor(k / 4) * 3 + (k % 4)] / scale));
  const x = asImage(data.render, 1);
  const y = asImage(data.target, 255);
  let l1 = 0;
  for (let k = 0; k < x.length; k++) if (k % 4 !== 3) l1 += Math.abs(x[k] - y[k]);
  const cpuLoss = (0.8 * l1) / (3 * pixels) + 0.2 * (1 - ssim(x, y, width, height));
  const cpuError = Math.abs(cpuLoss - data.loss);
  const cpuFailures = cpuError <= 1e-6 ? [] : [`  loss ${cpuLoss} vs ${data.loss}`];
  return [
    'M3.2: training loss',
    `L1 + SSIM vs PyTorch: ${verdict(failures)} (${width}×${height}; loss off by ${lossError.toExponential(1)}, ` +
      `gradient by ${(worst / largest).toExponential(1)} of the largest; autograd vs finite differences ` +
      `${data.finiteDifferenceError.toExponential(1)})`,
    ...failures,
    `SSIM on the CPU vs PyTorch: ${verdict(cpuFailures)} (the same loss rebuilt from metrics.ts, off by ${cpuError.toExponential(1)})`,
    ...cpuFailures,
  ].join('\n');
}
