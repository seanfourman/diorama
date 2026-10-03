# Diorama

Photos in, a walkable 3D scene out. A 3D Gaussian Splatting renderer and trainer that runs entirely in the browser on WebGPU, with every GPU kernel, forward and backward, written from scratch.

> The renderer, its backward pass, training, the Inside view and training from your own photos all work. The plan and milestones are in [docs/PLAN.md](docs/PLAN.md).

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

## Train

```bash
npm run download-photos   # the Tanks and Temples photos with COLMAP poses (652 MB) into data/train
npm run dev
```

Open http://localhost:5173/?train=train to train the scene from its 263 training photos and watch it learn. The camera controls are the same as above.
- Space pauses and resumes training.
- P saves the Gaussians as a .ply in the standard 3DGS layout.

The held-out photos (every 8th) are measured at step 7,000 and at the end. Add `&steps=N` to stop sooner.

On an AMD Radeon RX 7900 XTX (Edge), the full 30,000 steps take 18 minutes and reach 22.04 dB PSNR and 0.809 SSIM on the held-out photos, with 1.1 million Gaussians.

### The Inside view

Keys 1 to 7 show what the optimizer is doing while it trains, with live curves for the loss, the held-out PSNR and the Gaussian count. H hides the panels.

| Key | Mode | What it shows |
|---|---|---|
| 1 | Color | the scene |
| 2 | Depth | distance from the camera |
| 3 | Work | how many splats each pixel walks through: what rendering and training cost |
| 4 | Pull | the average pull on each Gaussian's 2D center, the signal that triggers densification |
| 5 | Densification | Gaussians that were cloned (green) or split (orange), glowing while new |
| 6 | Compare | render and photo side by side, from a photo's exact viewpoint ([ and ] jump between photos; the held-out ones come last) |
| 7 | Error | per-pixel error against the photo |

The viewer (`?scene=`) has modes 1 to 3.

## Your own scene

```bash
npm run download-colmap   # COLMAP, for the camera poses (Windows; elsewhere, install it)
npm run dev
```

Open http://localhost:5173/?new and drop 50–200 photos of a place, or a video walking through it. Move around rather than turning on the spot, overlap the shots heavily, and keep the lighting steady. The page uploads them to the dev server, which runs COLMAP to find the camera poses, then opens the training page. If the photos overlap too little for a trustworthy reconstruction, it says so first. When training ends, the scene is saved to `data/<name>-trained/`. Open `?scene=<name>-trained` to walk through it.

From the command line: `npm run reconstruct -- <name> --from=<photos folder>`, then `?train=<name>`.

To share a scene, host its `point_cloud.ply` (and `cameras.json`) anywhere that allows cross-origin requests, and send `?ply=<url>`. A `.ply` dropped onto the viewer opens too.

## Checks

`npm run check` builds the app, runs its GPU checks in headless Edge or Chrome on your machine's GPU, and fails if any of them do. To run them in your own browser, add `?check` to the address.

The trained "train" scene comes from the original 3D Gaussian Splatting release (Inria) and is under that release's license: research and personal use. The photos are from the Tanks and Temples benchmark, under its own terms.
