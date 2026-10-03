# Diorama

Photos in, a walkable 3D scene out: 3D Gaussian Splatting trained and rendered in the browser with WebGPU, with every GPU kernel written from scratch. A portfolio project.

Read `docs/PLAN.md` before starting work. It holds the milestones, the stack decisions and the progress checklist. Tick the checklist as steps land, and record new decisions in the plan rather than only in chat.

## Stack

- TypeScript + Vite, with WebGPU used directly. No Three.js and no GPU or ML libraries in the app.
- WGSL for all GPU code. Shaders live in `src/shaders/*.wgsl` and are imported with Vite's `?raw` suffix.
- Python + PyTorch on the CPU, in a git-ignored `.venv`, for the gradient reference. To set it up: `python -m venv .venv`, then `.venv/Scripts/python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu`.
- The forward math lives in four places that must agree: `preprocess.wgsl` and `rasterize.wgsl`, their backward passes, `src/checks/reference.ts`, and `reference/backward_reference.py`. After changing it, update all four, then rerun `.venv/Scripts/python reference/backward_reference.py` to regenerate `src/checks/fixtures/backward.json`.
- The dev GPU is an AMD Radeon RX 7900 XTX. There's no CUDA, so don't suggest CUDA-only tools.
- Edge on the dev machine offers `timestamp-query` and `shader-f16` but not `subgroups` (checked 2026-09-30).

## Commands

- `npm run dev`: dev server at http://localhost:5173. Add `?scene=train` to load the real scene, `?train=train` to train it from its photos (`&steps=N` to stop early), and `?check` to run the GPU checks, which are off by default so the page opens fast.
- `npm run build`: typecheck with `tsc`, then a production build
- `npm run check`: build, open the page with `?check` in headless Edge on the real GPU, print the on-page log, and exit 1 if a check prints FAIL or the console shows an error. Run it after every change. `main.ts` signals completion by setting `<body data-status>` to `done` or `error`.
- `node scripts/check.mjs --scene=train --screenshot=<file>`: the same checks against the last build, then the real scene on screen, saved as a PNG. Look at it after any rendering change.
- `node scripts/check.mjs --query="train=train&steps=7000" --timeout=1800`: a headless training run without the checks. It prints the status line every 30 s and the held-out PSNR/SSIM at the end.
- Add `--screenshot=<file> --keys=Digit4,BracketRight,Digit7` to then press each key and save `<file>-<code>.png` after each: the way to look at the Inside view's modes.
- `npm run download-scene [-- <name>]`: fetches a trained scene (default "train", 266 MB) into `data/<name>/`, which is git-ignored. The scene's license covers research and personal use only, so check it before publishing a demo.
- `npm run download-photos [-- <name>]`: fetches the Tanks and Temples photos with COLMAP poses (`tandt_db.zip`, 652 MB, kept in `data/`) and unpacks one scene's into `data/<name>/images` and `data/<name>/sparse/0`.

## Layout

- `src/main.ts`: startup, then the GPU checks, then the render loop (or, with `?train=`, the training page).
- `src/gaussianRenderer.ts`: the renderer. First the preprocess turns 3D Gaussians into 2D splats. Then `src/tileRasterizer.ts`, a tile rasterizer in the reference's style, takes over: it counts tiles, prefix-sums the counts, writes (tile, depth) keys, sorts them, finds each tile's range, and blends each tile. Output goes to an rgba8unorm storage texture, and the canvas is configured for that.
- `src/gaussians.ts`: CPU-side layouts of the GPU structs, with packers.
- `src/radixSort.ts` and `src/prefixSum.ts`: GPU primitives, a stable radix sort of 32- or 64-bit keys with values, and an exclusive prefix sum. Both read their counts from GPU buffers, and neither relies on subgroups.
- `src/random.ts`: the seeded random number generator that scenes and checks use.
- `src/shaders/`: WGSL. `common.wgsl` holds the shared structs and tile helpers, and is prepended to the renderer's shaders. `gaussianMath.wgsl` holds the rotation and spherical-harmonic math, shared by the preprocess and its backward pass. `rasterizeBackward.wgsl` and `preprocessBackward.wgsl` are the backward pass (M2). `project.wgsl` is M1.1's kernel, now used only by its check.
- `reference/backward_reference.py`: the renderer in PyTorch (float64), whose autograd gradients are the ground truth for the backward pass. `reference/loss_reference.py` does the same for the loss. Each writes a fixture into `src/checks/fixtures/`.
- `src/checks/`: one GPU check per step. Each returns a report whose lines say PASS or FAIL. `reference.ts` is the renderer's math in plain TypeScript; `helpers.ts` holds the readback plumbing.
- `src/trainer.ts`: training (M3). Each step is render → loss → backward → Adam, in one command buffer with no readback. It densifies every 100 steps. `src/trainingLoss.ts` + `loss.wgsl` is L1 + D-SSIM and its per-pixel gradient. `adam.wgsl` is the optimizer, which works on raw parameters (log scale, opacity logit) and writes the activated copy the renderer draws. `densify.wgsl` clones, splits and prunes as a stream compaction (count, prefix sum, scatter).
- `src/trainingPage.ts`: the `?train=` page. It trains while drawing the current Gaussians from a free camera, and measures the held-out photos at step 7,000 and at the end, between `Trainer.optimize()` and `maintain()` (before densifying, like the reference). P saves a .ply (`src/plyWriter.ts`). `&view=<mode>&photo=<n>` start in an Inside view mode at a photo.
- `src/insideView.ts` + `inspect.wgsl` + `blit.wgsl`: the Inside view (M4), keys 1–7. Most modes recolor the 2D splats through `GaussianRenderer.encode`'s `afterPreprocess` hook; work and error are per-pixel heatmaps; the photo modes letterbox a photo-sized render. Also GPU PSNR for the curves. `src/insidePanel.ts` and `src/charts.ts` draw the panel and the curves; `src/colormap.ts` is Turbo, shared with the shader. Densification stamps each new Gaussian's kind and step into the raw parameters' padding (floats 7 and 15).
- `src/colmap.ts`, `src/loadDataset.ts`: reading a COLMAP reconstruction and its photos, the train/test split (every 8th photo held out), and the starting Gaussians, whose sizes come from nearest neighbors on the GPU (`nearestNeighbors.wgsl`). `src/metrics.ts` holds PSNR and SSIM on the CPU, as the reference measures them.
- `src/plyLoader.ts`, `src/sceneCameras.ts`, `src/loadScene.ts`: reading trained scenes. The loader converts raw .ply parameters to the renderer's layout, reads the training cameras, and frames the scene (center, up, radius).
- `src/mat4.ts` and `src/viewerCamera.ts`: column-major matrix math, and the orbit/walk camera that can jump to the training photos.
- `src/testScene.ts`: the stand-in trefoil-knot scene, shown when there's no `?scene=`.
- `src/readback.ts`: copying buffers and textures back to the CPU.
- `scripts/check.mjs`: the headless browser runner behind `npm run check`. `scripts/download-scene.mjs` and `scripts/download-photos.mjs`: the downloaders.

## Working agreement

Claude writes the code when Sean asks, then explains what it did in plain terms (Sean, 2026-09-30: "write the code and tell me overall what you did"). The explanations matter: Sean has to be able to explain every kernel and gradient in an interview, so walk through the key ideas after each step.

Each step gets a brief in `docs/steps/` and, where possible, a GPU check that prints PASS or FAIL. Before calling a step done, run `npm run check` and confirm that a planted bug makes it fail.
