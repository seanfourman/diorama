// The training loss (M3): (1 − λ) × L1 + λ × (1 − SSIM), as in the reference
// (λ = 0.2), and its gradient with respect to every rendered pixel.
//
// SSIM compares local statistics under an 11×11 Gaussian window: the means μ, the
// second moments E[x²], E[y²] and E[xy], and from those the variances and the
// covariance. Blurring is separable, so each blur is a horizontal pass and then a
// vertical one, with zeros past the image's edges (as PyTorch's padded conv2d does).
//
// The gradient comes out of the same machinery. Each pixel's SSIM depends on x
// only through μx, E[x²] and E[xy], so per pixel we work out ∂S/∂μx, ∂S/∂E[x²] and
// ∂S/∂E[xy], blur those maps with the same window, and combine:
//   dS̄/dx = (blur(∂S/∂μx) + 2x × blur(∂S/∂E[x²]) + y × blur(∂S/∂E[xy])) / count.
// Notes: docs/steps/3-training.md

struct LossParams {
  size: vec2<u32>,
  // The blur: 1 for horizontal, 0 for vertical, and how many maps to blur.
  horizontal: u32,
  map_count: u32,
  // The normalized window: weights for offsets 0 to 5 (and by symmetry −5 to 0).
  weights: array<vec4<f32>, 2>,
}

const LAMBDA = 0.2;
const C1 = 0.0001; // 0.01²
const C2 = 0.0009; // 0.03²

@group(0) @binding(0) var<uniform> params: LossParams;
@group(0) @binding(1) var<storage, read> rendered: array<vec4<f32>>;
@group(0) @binding(2) var target_image: texture_2d<f32>;
// Five maps of `pixel_count` vec4s each: x, y, x², y², xy (rgb used).
@group(0) @binding(3) var<storage, read_write> moments: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> blur_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> blur_out: array<vec4<f32>>;
// Blurred moments, then (reused) the blurred partial derivatives.
@group(0) @binding(6) var<storage, read> blurred: array<vec4<f32>>;
// Three maps: ∂S/∂μx, ∂S/∂E[x²], ∂S/∂E[xy]; then SSIM itself in a fourth.
@group(0) @binding(7) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> pixel_grads: array<vec4<f32>>;
// Each pixel's share of the loss, without the constant λ.
@group(0) @binding(10) var<storage, read_write> pixel_loss: array<f32>;

fn pixel_count() -> u32 {
  return params.size.x * params.size.y;
}

fn weight(offset: i32) -> f32 {
  let k = u32(abs(offset));
  return params.weights[k / 4u][k % 4u];
}

@compute @workgroup_size(16, 16)
fn moments_pass(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= params.size)) { return; }
  let p = id.y * params.size.x + id.x;
  let n = pixel_count();
  let x = rendered[p].rgb;
  let y = textureLoad(target_image, id.xy, 0).rgb;
  moments[p] = vec4<f32>(x, 0.0);
  moments[n + p] = vec4<f32>(y, 0.0);
  moments[2u * n + p] = vec4<f32>(x * x, 0.0);
  moments[3u * n + p] = vec4<f32>(y * y, 0.0);
  moments[4u * n + p] = vec4<f32>(x * y, 0.0);
}

// One direction of the separable blur, over `map_count` maps.
@compute @workgroup_size(16, 16)
fn blur_pass(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= params.size)) { return; }
  let n = pixel_count();
  let step = select(vec2<i32>(0, 1), vec2<i32>(1, 0), params.horizontal == 1u);
  let size = vec2<i32>(params.size);
  for (var m = 0u; m < params.map_count; m++) {
    var sum = vec4<f32>(0.0);
    for (var k = -5; k <= 5; k++) {
      let at = vec2<i32>(id.xy) + step * k;
      if (all(at >= vec2<i32>(0)) && all(at < size)) {
        sum += weight(k) * blur_in[m * n + u32(at.y) * params.size.x + u32(at.x)];
      }
    }
    blur_out[m * n + id.y * params.size.x + id.x] = sum;
  }
}

@compute @workgroup_size(16, 16)
fn ssim_pass(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= params.size)) { return; }
  let p = id.y * params.size.x + id.x;
  let n = pixel_count();
  let mx = blurred[p].rgb;
  let my = blurred[n + p].rgb;
  let vx = blurred[2u * n + p].rgb - mx * mx; // σx²
  let vy = blurred[3u * n + p].rgb - my * my; // σy²
  let cxy = blurred[4u * n + p].rgb - mx * my; // σxy
  let a1 = 2.0 * mx * my + C1;
  let a2 = 2.0 * cxy + C2;
  let b1 = mx * mx + my * my + C1;
  let b2 = vx + vy + C2;
  let s = a1 * a2 / (b1 * b2);
  // σx² = E[x²] − μx² and σxy = E[xy] − μx μy also depend on μx.
  partials[p] = vec4<f32>(s * (2.0 * my / a1 - 2.0 * my / a2 - 2.0 * mx / b1 + 2.0 * mx / b2), 0.0);
  partials[n + p] = vec4<f32>(-s / b2, 0.0);
  partials[2u * n + p] = vec4<f32>(2.0 * s / a2, 0.0);
  partials[3u * n + p] = vec4<f32>(s, 0.0);
}

@compute @workgroup_size(16, 16)
fn gradient_pass(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= params.size)) { return; }
  let p = id.y * params.size.x + id.x;
  let n = pixel_count();
  // Both terms are means over every pixel and channel.
  let scale = 1.0 / f32(3u * n);
  let x = rendered[p].rgb;
  let y = textureLoad(target_image, id.xy, 0).rgb;
  let ssim_grad = blurred[p].rgb + 2.0 * x * blurred[n + p].rgb + y * blurred[2u * n + p].rgb;
  let grad = ((1.0 - LAMBDA) * sign(x - y) - LAMBDA * ssim_grad) * scale;
  pixel_grads[p] = vec4<f32>(grad, 0.0);
  let s = partials[3u * n + p].rgb;
  pixel_loss[p] = dot(vec3<f32>(1.0), (1.0 - LAMBDA) * abs(x - y) - LAMBDA * s) * scale;
}
