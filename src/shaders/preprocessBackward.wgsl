// The backward pass of preprocess.wgsl (M2), following preprocessCUDA's backward
// in the reference. It takes each splat's 2D gradients from rasterizeBackward and
// carries them back to the Gaussian's own parameters: position, scale, rotation,
// opacity, base color and spherical-harmonic coefficients.
//
// It recomputes the forward's intermediate values rather than storing them, so
// the forward steps below must match preprocess.wgsl exactly.
// Notes: docs/steps/2-backward-pass.md

struct SceneParams { sh_degree: u32 }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sh_rest: array<f32>;
@group(0) @binding(3) var<uniform> scene_params: SceneParams;
@group(0) @binding(4) var<storage, read> splats: array<Splat2D>;
// From rasterizeBackward, 9 per splat as float bits: d/d(center x, y), d/d(conic
// a, b, c), d/d(opacity), d/d(color r, g, b).
@group(0) @binding(5) var<storage, read> splat_grads: array<u32>;
// Laid out like struct Gaussian: 16 floats each, with the same padding.
@group(0) @binding(6) var<storage, read_write> gaussian_grads: array<f32>;
// Laid out like sh_rest: 45 floats each.
@group(0) @binding(7) var<storage, read_write> sh_grads: array<f32>;

const BLUR = 0.3; // must match preprocess.wgsl

@compute @workgroup_size(256)
fn preprocess_backward(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&gaussians)) { return; }
  for (var k = 0u; k < 16u; k++) { gaussian_grads[i * 16u + k] = 0.0; }
  for (var k = 0u; k < 45u; k++) { sh_grads[i * 45u + k] = 0.0; }
  // Culled splats drew nothing, so they have no gradient.
  if (splats[i].radius == 0.0) { return; }

  let grad = array<f32, 9>(
    bitcast<f32>(splat_grads[i * 9u]), bitcast<f32>(splat_grads[i * 9u + 1u]), bitcast<f32>(splat_grads[i * 9u + 2u]),
    bitcast<f32>(splat_grads[i * 9u + 3u]), bitcast<f32>(splat_grads[i * 9u + 4u]), bitcast<f32>(splat_grads[i * 9u + 5u]),
    bitcast<f32>(splat_grads[i * 9u + 6u]), bitcast<f32>(splat_grads[i * 9u + 7u]), bitcast<f32>(splat_grads[i * 9u + 8u]),
  );
  let mean_grad = vec2<f32>(grad[0], grad[1]);
  let conic_grad = vec3<f32>(grad[2], grad[3], grad[4]);
  let opacity_grad = grad[5];
  let color_grad = vec3<f32>(grad[6], grad[7], grad[8]);

  // The forward again, as in preprocess.wgsl.
  let g = gaussians[i];
  let t = (camera.view * vec4<f32>(g.position, 1.0)).xyz;
  let depth = -t.z;
  let q_length = length(g.rotation);
  let q = g.rotation / q_length;
  let r = rotation_matrix(g.rotation);
  let m = mat3x3<f32>(r[0] * g.scale.x, r[1] * g.scale.y, r[2] * g.scale.z);
  let cov3d = m * transpose(m);
  let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.viewport * 0.5;
  let limit = 1.3 * camera.viewport * 0.5 / focal;
  let ratio = t.xy / depth;
  // Where J's evaluation point was clamped, it doesn't move with the center.
  let free = select(vec2<f32>(1.0), vec2<f32>(0.0), abs(ratio) > limit);
  let tx = clamp(ratio.x, -limit.x, limit.x) * depth;
  let ty = clamp(ratio.y, -limit.y, limit.y) * depth;
  let depth2 = depth * depth;
  let depth3 = depth2 * depth;
  let j = mat3x2<f32>(
    focal.x / depth, 0.0,
    0.0, -focal.y / depth,
    focal.x * tx / depth2, -focal.y * ty / depth2,
  );
  let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let jw = j * w;
  let cov = jw * cov3d * transpose(jw);
  let a = cov[0][0] + BLUR;
  let b = cov[0][1];
  let c = cov[1][1] + BLUR;
  let det = a * c - b * b;

  // 1. Conic → 2D covariance. The conic is (c, −b, a) / det, where det = ac − b².
  let det2 = det * det;
  let a_grad = (-c * c * conic_grad.x + b * c * conic_grad.y - b * b * conic_grad.z) / det2;
  let b_grad = (2.0 * b * c * conic_grad.x - (det + 2.0 * b * b) * conic_grad.y + 2.0 * a * b * conic_grad.z) / det2;
  let c_grad = (-b * b * conic_grad.x + a * b * conic_grad.y - a * a * conic_grad.z) / det2;
  // As a symmetric matrix: b sits in both off-diagonal slots, so each gets half.
  let cov_grad = mat2x2<f32>(a_grad, 0.5 * b_grad, 0.5 * b_grad, c_grad);

  // 2. Σ' = T Σ Tᵀ with T = J W, so dL/dΣ = Tᵀ G T and dL/dT = 2 G T Σ.
  let cov3d_grad = transpose(jw) * cov_grad * jw;
  let j_grad = 2.0 * cov_grad * jw * cov3d * transpose(w);

  // 3. J moves with the view-space center t, through depth = −t.z and (unless
  // clamped) t.x and t.y.
  var t_grad = vec3<f32>(
    j_grad[2][0] * free.x * focal.x / depth2,
    j_grad[2][1] * free.y * -focal.y / depth2,
    0.0,
  );
  let depth_grad =
    j_grad[0][0] * (-focal.x / depth2) +
    j_grad[1][1] * (focal.y / depth2) +
    j_grad[2][0] * (-(1.0 + free.x) * focal.x * tx / depth3) +
    j_grad[2][1] * ((1.0 + free.y) * focal.y * ty / depth3);
  t_grad.z -= depth_grad;

  // 4. The 2D center: mean = (clip.xy / clip.w × (0.5, −0.5) + 0.5) × viewport.
  let clip = camera.proj * vec4<f32>(t, 1.0);
  let ndc_grad = mean_grad * vec2<f32>(0.5, -0.5) * camera.viewport;
  let clip_grad = vec4<f32>(ndc_grad / clip.w, 0.0, -dot(ndc_grad, clip.xy) / (clip.w * clip.w));
  t_grad += (transpose(camera.proj) * clip_grad).xyz;

  // 5. Color: base + Σ basisₖ(direction) × shₖ, clamped at zero.
  let camera_position = -(transpose(w) * camera.view[3].xyz);
  let offset = g.position - camera_position;
  let distance = length(offset);
  let direction = offset / distance;
  let basis_count = sh_count(scene_params.sh_degree);
  var basis = sh_basis(direction);
  var raw_color = g.color;
  for (var k = 0u; k < basis_count; k++) {
    raw_color += basis[k] * sh(i, k);
  }
  // No gradient flows through a channel the clamp cut off.
  let raw_grad = select(color_grad, vec3<f32>(0.0), raw_color < vec3<f32>(0.0));
  var direction_grad = vec3<f32>(0.0);
  var basis_gradient = sh_basis_gradient(direction);
  for (var k = 0u; k < basis_count; k++) {
    let coefficient_grad = basis[k] * raw_grad;
    sh_grads[i * 45u + k * 3u] = coefficient_grad.r;
    sh_grads[i * 45u + k * 3u + 1u] = coefficient_grad.g;
    sh_grads[i * 45u + k * 3u + 2u] = coefficient_grad.b;
    direction_grad += basis_gradient[k] * dot(sh(i, k), raw_grad);
  }
  // direction = offset / |offset|: remove the part along the direction, then scale.
  let color_position_grad = (direction_grad - direction * dot(direction, direction_grad)) / distance;

  // 6. Position: t = W × position + translation.
  let position_grad = transpose(w) * t_grad + color_position_grad;

  // 7. Σ = M Mᵀ with M = R S, so dL/dM = 2 dL/dΣ M. Column k of M is column k of
  // R times scale k.
  let m_grad = 2.0 * cov3d_grad * m;
  let scale_grad = vec3<f32>(dot(m_grad[0], r[0]), dot(m_grad[1], r[1]), dot(m_grad[2], r[2]));
  let r_grad = mat3x3<f32>(m_grad[0] * g.scale.x, m_grad[1] * g.scale.y, m_grad[2] * g.scale.z);
  // R from the normalized quaternion (w, x, y, z). gr(row, col) is dL/dR[row][col].
  let qw = q.x;
  let qx = q.y;
  let qy = q.z;
  let qz = q.w;
  let g00 = r_grad[0][0];
  let g01 = r_grad[1][0];
  let g02 = r_grad[2][0];
  let g10 = r_grad[0][1];
  let g11 = r_grad[1][1];
  let g12 = r_grad[2][1];
  let g20 = r_grad[0][2];
  let g21 = r_grad[1][2];
  let g22 = r_grad[2][2];
  let q_grad = 2.0 * vec4<f32>(
    -qz * g01 + qy * g02 + qz * g10 - qx * g12 - qy * g20 + qx * g21,
    qy * g01 + qz * g02 + qy * g10 - 2.0 * qx * g11 - qw * g12 + qz * g20 + qw * g21 - 2.0 * qx * g22,
    -2.0 * qy * g00 + qx * g01 + qw * g02 + qx * g10 + qz * g12 - qw * g20 + qz * g21 - 2.0 * qy * g22,
    -2.0 * qz * g00 - qw * g01 + qx * g02 + qw * g10 - 2.0 * qz * g11 + qy * g12 + qx * g20 + qy * g21,
  );
  // q = raw / |raw|: remove the part along q, then scale.
  let rotation_grad = (q_grad - q * dot(q, q_grad)) / q_length;

  // Laid out like struct Gaussian: position, opacity, scale, (pad), rotation, color, (pad).
  let out = i * 16u;
  gaussian_grads[out] = position_grad.x;
  gaussian_grads[out + 1u] = position_grad.y;
  gaussian_grads[out + 2u] = position_grad.z;
  gaussian_grads[out + 3u] = opacity_grad;
  gaussian_grads[out + 4u] = scale_grad.x;
  gaussian_grads[out + 5u] = scale_grad.y;
  gaussian_grads[out + 6u] = scale_grad.z;
  gaussian_grads[out + 8u] = rotation_grad.x;
  gaussian_grads[out + 9u] = rotation_grad.y;
  gaussian_grads[out + 10u] = rotation_grad.z;
  gaussian_grads[out + 11u] = rotation_grad.w;
  gaussian_grads[out + 12u] = raw_grad.r;
  gaussian_grads[out + 13u] = raw_grad.g;
  gaussian_grads[out + 14u] = raw_grad.b;
}

// Coefficient k + 1 (k from 0 to 14) of Gaussian i, rgb.
fn sh(i: u32, k: u32) -> vec3<f32> {
  let at = i * 45u + k * 3u;
  return vec3<f32>(sh_rest[at], sh_rest[at + 1u], sh_rest[at + 2u]);
}
