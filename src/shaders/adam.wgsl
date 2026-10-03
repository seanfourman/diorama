// The optimizer (M3): Adam on the raw training parameters, as in the reference.
// Training keeps scales as logarithms, opacity as a logit and color as the
// degree-0 spherical-harmonic coefficient, so they can move freely while their
// activated values stay valid. The renderer's gradients are for the activated
// values, so each step first applies the chain rule through exp, sigmoid and
// SH_C0, then writes the activated values back for the next render.
// Notes: docs/steps/3-training.md

struct AdamParams {
  // 1 − β₁ᵗ and 1 − β₂ᵗ at this step.
  bias_correction1: f32,
  bias_correction2: f32,
  position_lr: f32,
  opacity_lr: f32,
  scale_lr: f32,
  rotation_lr: f32,
  color_lr: f32,
  sh_lr: f32,
  // The training views' size in pixels, which turns pixel gradients into the
  // reference's NDC units for the densification statistics.
  viewport: vec2<f32>,
}

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPSILON = 1e-15; // as in the reference
const SH_C0 = 0.28209479177387814;
const PARAMS = 16u; // raw parameters per Gaussian, laid out like struct Gaussian
const SH = 45u;

@group(0) @binding(0) var<uniform> adam: AdamParams;
// Raw: position, opacity logit, log scale, (pad), rotation, degree-0 coefficient, (pad).
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
// The renderer's spherical-harmonic buffer, which holds raw coefficients already.
@group(0) @binding(2) var<storage, read_write> sh_params: array<f32>;
// Adam's running (mean, mean square) of the gradient: PARAMS + SH per Gaussian.
@group(0) @binding(3) var<storage, read_write> moments: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> gaussian_grads: array<f32>;
@group(0) @binding(5) var<storage, read> sh_grads: array<f32>;
// The renderer's activated Gaussians.
@group(0) @binding(6) var<storage, read_write> gaussians: array<f32>;
@group(0) @binding(7) var<storage, read> splats: array<Splat2D>;
@group(0) @binding(8) var<storage, read> splat_grads: array<f32>;
// Per Gaussian, for densification: (sum of 2D-center gradient norms, times seen, largest radius, unused).
@group(0) @binding(9) var<storage, read_write> stats: array<vec4<f32>>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// One Adam update of one number; returns the new value.
fn update(value: f32, grad: f32, slot: u32, lr: f32) -> f32 {
  var m = moments[slot];
  m = vec2<f32>(BETA1 * m.x + (1.0 - BETA1) * grad, BETA2 * m.y + (1.0 - BETA2) * grad * grad);
  moments[slot] = m;
  return value - lr * (m.x / adam.bias_correction1) / (sqrt(m.y / adam.bias_correction2) + EPSILON);
}

// Writes Gaussian i's activated values from its raw ones.
fn activate_one(i: u32) {
  let at = i * PARAMS;
  for (var k = 0u; k < 3u; k++) {
    gaussians[at + k] = params[at + k];
    gaussians[at + 4u + k] = exp(params[at + 4u + k]);
    gaussians[at + 12u + k] = 0.5 + SH_C0 * params[at + 12u + k];
  }
  gaussians[at + 3u] = sigmoid(params[at + 3u]);
  for (var k = 8u; k < 12u; k++) {
    gaussians[at + k] = params[at + k];
  }
}

@compute @workgroup_size(256)
fn activate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&params) / PARAMS) { return; }
  activate_one(id.x);
}

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&params) / PARAMS) { return; }
  let at = i * PARAMS;
  let slots = i * (PARAMS + SH);
  for (var k = 0u; k < 3u; k++) {
    params[at + k] = update(params[at + k], gaussian_grads[at + k], slots + k, adam.position_lr);
    // scale = exp(raw), so d/draw = d/dscale × scale.
    let scale_grad = gaussian_grads[at + 4u + k] * exp(params[at + 4u + k]);
    params[at + 4u + k] = update(params[at + 4u + k], scale_grad, slots + 4u + k, adam.scale_lr);
    // color = 0.5 + SH_C0 × raw.
    let color_grad = gaussian_grads[at + 12u + k] * SH_C0;
    params[at + 12u + k] = update(params[at + 12u + k], color_grad, slots + 12u + k, adam.color_lr);
  }
  // opacity = sigmoid(raw), so d/draw = d/dopacity × opacity × (1 − opacity).
  let opacity = sigmoid(params[at + 3u]);
  let opacity_grad = gaussian_grads[at + 3u] * opacity * (1.0 - opacity);
  params[at + 3u] = update(params[at + 3u], opacity_grad, slots + 3u, adam.opacity_lr);
  // The renderer normalizes the quaternion itself, and its gradient already accounts for that.
  for (var k = 8u; k < 12u; k++) {
    params[at + k] = update(params[at + k], gaussian_grads[at + k], slots + k, adam.rotation_lr);
  }
  for (var k = 0u; k < SH; k++) {
    let s = i * SH + k;
    sh_params[s] = update(sh_params[s], sh_grads[s], slots + PARAMS + k, adam.sh_lr);
  }
  activate_one(i);
}

// The reference's opacity reset, every 3000 steps while densifying: every opacity
// is capped at 0.01. Gaussians the photos need climb back up; the rest stay faint
// and get pruned. Adam's history for opacity starts over.
const RESET_LOGIT = -4.59511985013459; // logit(0.01)

@compute @workgroup_size(256)
fn reset_opacity(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&params) / PARAMS) { return; }
  params[i * PARAMS + 3u] = min(params[i * PARAMS + 3u], RESET_LOGIT);
  moments[i * (PARAMS + SH) + 3u] = vec2<f32>(0.0);
  activate_one(i);
}

// For densification: where the 2D center's gradient is large, a Gaussian is
// struggling to cover its area. Like the reference, the norm is taken in NDC units.
@compute @workgroup_size(256)
fn accumulate_stats(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&stats)) { return; }
  let radius = splats[i].radius;
  if (radius == 0.0) { return; }
  let ndc_grad = vec2<f32>(splat_grads[i * 9u], splat_grads[i * 9u + 1u]) * adam.viewport * 0.5;
  let s = stats[i];
  stats[i] = vec4<f32>(s.x + length(ndc_grad), s.y + 1.0, max(s.z, radius), 0.0);
}
