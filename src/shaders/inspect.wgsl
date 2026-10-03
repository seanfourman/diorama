// The Inside view (M4): what the renderer and the optimizer are doing, as images.
// - colorize_*: recolor each 2D splat after the preprocess, before the tile stage
//   blends it: by depth, by the pull on its center (the densification signal), or
//   by when densification made it. The blending is untouched, so a Gaussian's
//   color shows through as much as it shows in the real image.
// - work_heatmap: how many splats each pixel walked through, from the
//   rasterizer's per-pixel record.
// - error_heatmap: |render − photo| per pixel, plus each tile's sum of squared
//   errors, which the CPU adds up into a PSNR.
// common.wgsl is prepended. Notes: docs/steps/4-inside-view.md

struct InspectParams {
  // error_heatmap: where this image's per-tile sums go in `partials`.
  first_tile: u32,
  // colorize_age: the current step.
  iteration: f32,
  // colorize_gradient: the densification threshold.
  grad_threshold: f32,
  // work_heatmap: the splat counts that map to the two ends of the scale.
  work_min: f32,
  // colorize_depth: the depths that map to the two ends of the scale.
  depth_range: vec2<f32>,
  // error_heatmap: the mean absolute error that maps to the top of the scale.
  error_max: f32,
  work_max: f32,
}

@group(0) @binding(0) var<uniform> inspect: InspectParams;
@group(0) @binding(1) var<storage, read_write> splats: array<Splat2D>;
// Training's densification statistics: (sum of 2D-center gradient norms, times seen, largest radius, unused).
@group(0) @binding(2) var<storage, read> stats: array<vec4<f32>>;
// Training's raw parameters, 16 per Gaussian. Densification records in the
// padding how each Gaussian came about (float 7: 0 from the points, 1 cloned,
// 2 split) and when (float 15: the step).
@group(0) @binding(3) var<storage, read> raw: array<f32>;
@group(0) @binding(4) var<storage, read> pixel_last: array<u32>;
@group(0) @binding(5) var<storage, read> pixel_color: array<vec4<f32>>;
@group(0) @binding(6) var photo: texture_2d<f32>;
@group(0) @binding(7) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var<storage, read_write> partials: array<f32>;

// Turbo, as in src/colormap.ts: dark blue (0) through green to dark red (1).
fn turbo(x: f32) -> vec3<f32> {
  let t = saturate(x);
  let v4 = vec4<f32>(1.0, t, t * t, t * t * t);
  let v2 = v4.zw * v4.z;
  return saturate(vec3<f32>(
    dot(v4, vec4<f32>(0.13572138, 4.6153926, -42.66032258, 132.13108234)) + dot(v2, vec2<f32>(-152.94239396, 59.28637943)),
    dot(v4, vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2<f32>(4.27729857, 2.82956604)),
    dot(v4, vec4<f32>(0.1066733, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2<f32>(-89.90310912, 27.34824973)),
  ));
}

const NEVER_SEEN = vec3<f32>(0.25);
const FROM_POINTS = vec3<f32>(0.35);
const CLONED = vec3<f32>(0.2, 0.95, 0.35);
const SPLIT = vec3<f32>(1.0, 0.5, 0.1);
// How fast a new Gaussian's glow fades, in steps.
const AGE_FADE = 1000.0;

// Near is red and far is blue, on a log scale.
@compute @workgroup_size(256)
fn colorize_depth(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&splats) || splats[i].radius == 0.0) { return; }
  let range = log(inspect.depth_range);
  let t = (log(splats[i].depth) - range.x) / (range.y - range.x);
  splats[i].color = vec4<f32>(turbo(1.0 - t), splats[i].color.a);
}

// The average pull on the 2D center, on a log scale from 1/100 of the
// densification threshold (0) to 10 times it (1). The threshold itself is at 2/3,
// in the oranges: anything that color or redder gets cloned or split next time.
@compute @workgroup_size(256)
fn colorize_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&splats) || splats[i].radius == 0.0) { return; }
  let s = stats[i];
  var rgb = NEVER_SEEN;
  if (s.y > 0.0) {
    let ratio = max(s.x / s.y / inspect.grad_threshold, 1e-6);
    rgb = turbo((log(ratio) / log(10.0) + 2.0) / 3.0);
  }
  splats[i].color = vec4<f32>(rgb, splats[i].color.a);
}

// Cloned Gaussians green, split ones orange, the starting ones gray. New ones
// glow, then fade to a third of their brightness.
@compute @workgroup_size(256)
fn colorize_age(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&splats) || splats[i].radius == 0.0) { return; }
  let kind = raw[i * 16u + 7u];
  var rgb = FROM_POINTS;
  if (kind > 0.5) {
    let age = max(inspect.iteration - raw[i * 16u + 15u], 0.0);
    let base = select(SPLIT, CLONED, kind < 1.5);
    rgb = base * mix(1.0 / 3.0, 1.0, exp(-age / AGE_FADE));
  }
  splats[i].color = vec4<f32>(rgb, splats[i].color.a);
}

// Log scale: work_min splats or fewer is 0, work_max or more is 1.
@compute @workgroup_size(16, 16)
fn work_heatmap(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(output);
  if (any(id.xy >= size)) { return; }
  let last = max(f32(pixel_last[id.y * size.x + id.x]), 1.0);
  let t = log2(last / inspect.work_min) / log2(inspect.work_max / inspect.work_min);
  textureStore(output, id.xy, vec4<f32>(turbo(t), 1.0));
}

var<workgroup> squared_errors: array<f32, 256>;

// Renders are measured as the reference measures them: clamped to [0, 1] and
// rounded to 8 bits, like the photo.
@compute @workgroup_size(16, 16)
fn error_heatmap(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let size = textureDimensions(output);
  var squared = 0.0;
  if (all(id.xy < size)) {
    let rendered = round(saturate(pixel_color[id.y * size.x + id.x].rgb) * 255.0) / 255.0;
    let difference = rendered - textureLoad(photo, id.xy, 0).rgb;
    squared = dot(difference, difference);
    let mean_error = (abs(difference.r) + abs(difference.g) + abs(difference.b)) / 3.0;
    textureStore(output, id.xy, vec4<f32>(turbo(mean_error / inspect.error_max), 1.0));
  }
  // The tile's sum of squared errors, by halving.
  squared_errors[tid] = squared;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (tid < stride) {
      squared_errors[tid] += squared_errors[tid + stride];
    }
  }
  if (tid == 0u) {
    partials[inspect.first_tile + wg.y * groups.x + wg.x] = squared_errors[0];
  }
}
