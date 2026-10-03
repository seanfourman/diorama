import type { InsideView } from './insideView';

// The Inside view's panel (M4), bottom left: the modes and their keys, what the
// current one shows, its color scale, and (while training) the curves.

export class InsidePanel {
  private readonly root = document.querySelector<HTMLElement>('#inside')!;
  private readonly modes = document.querySelector<HTMLElement>('#modes')!;
  private readonly legend = document.querySelector<HTMLElement>('#legend')!;
  private readonly scale = document.querySelector<HTMLElement>('#scale')!;
  private readonly scaleBar = document.querySelector<HTMLElement>('#scale-bar')!;
  private readonly scaleLow = document.querySelector<HTMLElement>('#scale-low')!;
  private readonly scaleHigh = document.querySelector<HTMLElement>('#scale-high')!;
  readonly charts = document.querySelector<HTMLCanvasElement>('#charts')!;
  private shown = '';

  constructor(withCharts: boolean) {
    this.root.hidden = false;
    this.charts.hidden = !withCharts;
    // H hides both panels, for an unobstructed view.
    window.addEventListener('keydown', (event) => {
      if (event.code === 'KeyH') document.body.classList.toggle('bare');
    });
  }

  /** Shows `inside`'s current mode, with `note` (such as a hint or the latest numbers) after the legend. */
  update(inside: InsideView, note = ''): void {
    const info = inside.info;
    const key = `${info.mode}|${note}`;
    if (key === this.shown) return;
    this.shown = key;
    this.modes.replaceChildren(
      ...inside.modes.map(({ mode, key: code, name }) => {
        const item = document.createElement('span');
        item.textContent = `${code.replace('Digit', '')} ${name}`;
        if (mode === info.mode) item.className = 'current';
        return item;
      }),
    );
    this.legend.textContent = note ? `${info.legend} ${note}` : info.legend;
    this.scale.hidden = !info.scale;
    if (info.scale) {
      this.scaleBar.style.background = inside.scaleBackground;
      this.scaleLow.textContent = info.scale.low;
      this.scaleHigh.textContent = info.scale.high;
    }
  }
}
