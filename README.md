# Diorama

Photos in, a walkable 3D scene out. A 3D Gaussian Splatting renderer and trainer that runs entirely in the browser on WebGPU, with every GPU kernel, forward and backward, written from scratch.

> Work in progress: the renderer is done, and training is next. The plan and milestones are in [docs/PLAN.md](docs/PLAN.md).

## Run

```bash
npm install
npm run download-scene   # optional: a real trained scene (266 MB) into data/train
npm run dev
```

Open http://localhost:5173/?scene=train in a browser with WebGPU (current Chrome or Edge). Leave off `?scene=train` for the built-in test scene.

- Drag to turn, scroll to zoom.
- WASD or the arrow keys to fly, Q and E for down and up, Shift for speed.
- [ and ] step through the training photos' viewpoints, O goes back to orbiting, R resets.

## Checks

`npm run check` builds the app, runs its GPU checks in headless Edge or Chrome on your machine's GPU, and fails if any of them do. To run them in your own browser, add `?check` to the address.

The "train" scene comes from the original 3D Gaussian Splatting release (Inria) and is under that release's license: research and personal use.
