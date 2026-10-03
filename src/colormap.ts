// The Turbo colormap (Google, 2019): dark blue through green and yellow to dark
// red, readable and close to perceptually even. This is Ruofei Du's polynomial
// fit, the same one inspect.wgsl uses, so the legend and the checks match the GPU.

const RED = [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943];
const GREEN = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604];
const BLUE = [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973];

/** Turbo at x in [0, 1] (clamped), as rgb in [0, 1]. */
export function turbo(x: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, x));
  const powers = [1, t, t * t, t * t * t, t ** 4, t ** 5];
  const channel = (c: number[]) => Math.min(1, Math.max(0, c.reduce((sum, k, i) => sum + k * powers[i], 0)));
  return [channel(RED), channel(GREEN), channel(BLUE)];
}

/** A CSS gradient through Turbo, for legends. `reverse` runs it from red to blue. */
export function turboGradient(reverse = false): string {
  const stops = Array.from({ length: 11 }, (_, k) => {
    const [r, g, b] = turbo(reverse ? 1 - k / 10 : k / 10).map((v) => Math.round(v * 255));
    return `rgb(${r}, ${g}, ${b}) ${k * 10}%`;
  });
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
