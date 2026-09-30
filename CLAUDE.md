# Diorama

Photos in, a walkable 3D scene out: 3D Gaussian Splatting trained and rendered in the browser with WebGPU, with every GPU kernel written from scratch. A portfolio project.

Read `docs/PLAN.md` before starting work. It holds the milestones, the stack decisions and the progress checklist. Tick the checklist as steps land, and record new decisions in the plan rather than only in chat.

## Stack

- TypeScript + Vite, with WebGPU used directly. No Three.js and no GPU or ML libraries in the app.
- WGSL for all GPU code. Shaders live in `src/shaders/*.wgsl` and are imported with Vite's `?raw` suffix.
- Python + PyTorch on the CPU for gradient checks (M2, not set up yet).
- The dev GPU is an AMD Radeon RX 7900 XTX. There's no CUDA, so don't suggest CUDA-only tools.
- Edge on the dev machine offers `timestamp-query` and `shader-f16` but not `subgroups` (checked 2026-09-30).

## Commands

- `npm run dev`: dev server at http://localhost:5173. Add `?scene=train` to load the real scene, and `?check` to run the GPU checks, which are off by default so the page opens fast.
- `npm run build`: typecheck with `tsc`, then a production build
- `npm run check`: build, open the page with `?check` in headless Edge on the real GPU, print the on-page log, and exit 1 if a check prints FAIL or the console shows an error. Run it after every change. `main.ts` signals completion by setting `<body data-status>` to `done` or `error`.
- `node scripts/check.mjs --scene=train --screenshot=<file>`: the same checks against the last build, then the real scene on screen, saved as a PNG. Look at it after any rendering change.
- `npm run download-scene [-- <name>]`: fetches a trained scene (default "train", 266 MB) into `data/<name>/`, which is git-ignored. The scene's license covers research and personal use only, so check it before publishing a demo.

## Layout

- `src/main.ts`: startup, then the GPU checks, then the render loop.
- `src/gaussianRenderer.ts`: the renderer. First the preprocess turns 3D Gaussians into 2D splats. Then `src/tileRasterizer.ts`, a tile rasterizer in the reference's style, takes over: it counts tiles, prefix-sums the counts, writes (tile, depth) keys, sorts them, finds each tile's range, and blends each tile. Output goes to an rgba8unorm storage texture, and the canvas is configured for that.
- `src/gaussians.ts`: CPU-side layouts of the GPU structs, with packers.
- `src/radixSort.ts` and `src/prefixSum.ts`: GPU primitives, a stable radix sort of 32- or 64-bit keys with values, and an exclusive prefix sum. Both read their counts from GPU buffers, and neither relies on subgroups.
- `src/random.ts`: the seeded random number generator that scenes and checks use.
- `src/shaders/`: WGSL. `common.wgsl` holds the shared structs and tile helpers, and is prepended to the renderer's shaders. `project.wgsl` is M1.1's kernel, now used only by its check.
- `src/checks/`: one GPU check per step. Each returns a report whose lines say PASS or FAIL. `reference.ts` is the renderer's math in plain TypeScript; `helpers.ts` holds the readback plumbing.
- `src/plyLoader.ts`, `src/sceneCameras.ts`, `src/loadScene.ts`: reading trained scenes. The loader converts raw .ply parameters to the renderer's layout, reads the training cameras, and frames the scene (center, up, radius).
- `src/mat4.ts` and `src/viewerCamera.ts`: column-major matrix math, and the orbit/walk camera that can jump to the training photos.
- `src/testScene.ts`: the stand-in trefoil-knot scene, shown when there's no `?scene=`.
- `scripts/check.mjs`: the headless browser runner behind `npm run check`. `scripts/download-scene.mjs`: the scene downloader.

## Working agreement

Claude writes the code when Sean asks, then explains what it did in plain terms (Sean, 2026-09-30: "write the code and tell me overall what you did"). The explanations matter: Sean has to be able to explain every kernel and gradient in an interview, so walk through the key ideas after each step.

Each step gets a brief in `docs/steps/` and, where possible, a GPU check that prints PASS or FAIL. Before calling a step done, run `npm run check` and confirm that a planted bug makes it fail.
