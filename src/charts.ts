import type { Densification } from './trainer';

// Live training curves for the Inside view (M4), drawn with Canvas 2D: the loss,
// PSNR on the held-out photos, and the number of Gaussians with each
// densification's additions and removals. Dotted lines mark the opacity resets,
// and a solid one the end of densification.

const PANEL_HEIGHT = 74;
const GAP = 14;
const LEFT = 8;
const RIGHT = 8;
const TEXT = '#d8dce8';
const MUTED = 'rgba(216, 220, 232, 0.35)';
const LOSS_COLOR = '#ffb454';
const PSNR_COLOR = '#7fd1ff';
const COUNT_COLOR = '#c7a6ff';
const ADDED_COLOR = '#58d68d';
const REMOVED_COLOR = '#ff6b6b';

export interface ChartMarks {
  totalSteps: number;
  opacityResets: number[];
  densifyUntil: number;
}

export class TrainingCharts {
  private readonly canvas: HTMLCanvasElement;
  private readonly marks: ChartMarks;
  private readonly loss: [number, number][] = [];
  private readonly psnr: [number, number][] = [];
  private readonly count: [number, number][] = [];
  private smoothedLoss = NaN;

  constructor(canvas: HTMLCanvasElement, marks: ChartMarks) {
    this.canvas = canvas;
    this.marks = marks;
  }

  /** Adds a loss reading. Single steps are noisy (each is one photo), so the curve is a moving average. */
  addLoss(step: number, value: number): void {
    this.smoothedLoss = Number.isNaN(this.smoothedLoss) ? value : 0.9 * this.smoothedLoss + 0.1 * value;
    this.loss.push([step, this.smoothedLoss]);
  }

  addPsnr(step: number, value: number): void {
    this.psnr.push([step, value]);
  }

  addCount(step: number, value: number): void {
    const last = this.count.at(-1);
    if (!last || last[1] !== value || step - last[0] >= 200) this.count.push([step, value]);
  }

  draw(densifications: Densification[]): void {
    const { canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    // Three panels, then a strip for the densification bars.
    const axis = 3 * PANEL_HEIGHT + 2 * GAP + 10;
    const height = axis + 12;
    canvas.style.height = `${height}px`;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = '11px ui-monospace, Consolas, monospace';
    const x = (step: number) => LEFT + (step / this.marks.totalSteps) * (width - LEFT - RIGHT);

    const panels = [
      { title: 'loss (smoothed)', data: this.loss, color: LOSS_COLOR, log: true, format: (v: number) => v.toFixed(4) },
      { title: 'test PSNR', data: this.psnr, color: PSNR_COLOR, log: false, format: (v: number) => `${v.toFixed(2)} dB` },
      { title: 'Gaussians', data: this.count, color: COUNT_COLOR, log: false, format: (v: number) => v.toLocaleString('en-US') },
    ];
    panels.forEach((panel, k) => {
      const top = k * (PANEL_HEIGHT + GAP) + 14;
      const bottom = top + PANEL_HEIGHT - 16;
      // Schedule marks: opacity resets (dotted), the end of densification (solid).
      ctx.strokeStyle = MUTED;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      for (const step of this.marks.opacityResets) this.vertical(ctx, x(step), top, bottom);
      ctx.setLineDash([]);
      this.vertical(ctx, x(this.marks.densifyUntil), top, bottom);

      const latest = panel.data.at(-1);
      ctx.fillStyle = TEXT;
      ctx.fillText(`${panel.title}${latest ? `: ${panel.format(latest[1])}` : ''}`, LEFT, top - 3);
      if (panel.data.length === 0) return;
      const values = panel.data.map(([, v]) => (panel.log ? Math.log(v) : v));
      let lo = Math.min(...values);
      let hi = Math.max(...values);
      if (hi - lo < 1e-9) [lo, hi] = [lo - 0.5, hi + 0.5];
      const y = (value: number) => bottom - (((panel.log ? Math.log(value) : value) - lo) / (hi - lo)) * (bottom - top - 4);
      ctx.strokeStyle = panel.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      panel.data.forEach(([step, value], i) => (i ? ctx.lineTo(x(step), y(value)) : ctx.moveTo(x(step), y(value))));
      ctx.stroke();
      if (panel.data.length < 60) {
        ctx.fillStyle = panel.color;
        for (const [step, value] of panel.data) ctx.fillRect(x(step) - 1.5, y(value) - 1.5, 3, 3);
      }
    });

    // Under the Gaussian count: each densification's additions (up) and removals
    // (down). Each direction is scaled to its own largest, on a square-root scale,
    // so the huge prunes after opacity resets don't flatten the rest.
    const mostAdded = Math.max(1, ...densifications.map((d) => d.cloned + d.split));
    const mostRemoved = Math.max(1, ...densifications.map((d) => d.removed));
    for (const d of densifications) {
      const added = Math.sqrt((d.cloned + d.split) / mostAdded) * 10;
      const removed = Math.sqrt(d.removed / mostRemoved) * 10;
      ctx.fillStyle = ADDED_COLOR;
      ctx.fillRect(x(d.step) - 0.5, axis - added, 1.5, added);
      ctx.fillStyle = REMOVED_COLOR;
      ctx.fillRect(x(d.step) - 0.5, axis, 1.5, removed);
    }
  }

  private vertical(ctx: CanvasRenderingContext2D, at: number, top: number, bottom: number): void {
    ctx.beginPath();
    ctx.moveTo(at, top);
    ctx.lineTo(at, bottom);
    ctx.stroke();
  }
}
