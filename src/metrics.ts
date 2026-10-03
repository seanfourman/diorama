// Image quality, measured as the reference's metrics.py measures it, on the CPU.
// Images hold 4 numbers per pixel (rgb used), from 0 to 1.

/** Peak signal-to-noise ratio in dB, over the rgb channels. */
export function psnr(a: Float32Array, b: Float32Array): number {
  let squaredError = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const difference = a[i + channel] - b[i + channel];
      squaredError += difference * difference;
    }
  }
  return 10 * Math.log10(1 / (squaredError / ((3 * a.length) / 4)));
}

/**
 * Structural similarity, averaged over pixels and channels: an 11×11 Gaussian
 * window (σ = 1.5) with zero padding at the edges, as in training's loss.wgsl.
 */
export function ssim(a: Float32Array, b: Float32Array, width: number, height: number): number {
  const C1 = 0.01 ** 2;
  const C2 = 0.03 ** 2;
  const pixels = width * height;
  const maps = Array.from({ length: 5 }, () => new Float32Array(pixels));
  const scratch = new Float32Array(pixels);
  let total = 0;
  for (let channel = 0; channel < 3; channel++) {
    const [mx, my, mxx, myy, mxy] = maps;
    for (let p = 0; p < pixels; p++) {
      const x = a[p * 4 + channel];
      const y = b[p * 4 + channel];
      mx[p] = x;
      my[p] = y;
      mxx[p] = x * x;
      myy[p] = y * y;
      mxy[p] = x * y;
    }
    for (const map of maps) {
      blur(map, scratch, width, height, 1, 0);
      blur(scratch, map, width, height, 0, 1);
    }
    for (let p = 0; p < pixels; p++) {
      const vx = mxx[p] - mx[p] * mx[p];
      const vy = myy[p] - my[p] * my[p];
      const cxy = mxy[p] - mx[p] * my[p];
      total +=
        ((2 * mx[p] * my[p] + C1) * (2 * cxy + C2)) / ((mx[p] * mx[p] + my[p] * my[p] + C1) * (vx + vy + C2));
    }
  }
  return total / (3 * pixels);
}

const WINDOW = (() => {
  const raw = Array.from({ length: 11 }, (_, k) => Math.exp(-((k - 5) ** 2) / (2 * 1.5 * 1.5)));
  const sum = raw.reduce((total, w) => total + w, 0);
  return raw.map((w) => w / sum);
})();

// One direction of the separable blur; pixels past the edge count as 0.
function blur(source: Float32Array, target: Float32Array, width: number, height: number, dx: number, dy: number): void {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -5; k <= 5; k++) {
        const sx = x + k * dx;
        const sy = y + k * dy;
        if (sx >= 0 && sx < width && sy >= 0 && sy < height) sum += WINDOW[k + 5] * source[sy * width + sx];
      }
      target[y * width + x] = sum;
    }
  }
}
