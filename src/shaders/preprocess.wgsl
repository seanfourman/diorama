// Turns each 3D Gaussian into a 2D splat: its center in pixels, its depth, the
// inverse of its projected 2D covariance (the "conic"), a bounding radius, and its
// color as seen from the camera. Follows preprocessCUDA in the reference
// rasterizer. Its backward pass is preprocessBackward.wgsl, which recomputes the
// same steps, so keep the two in step.
// Notes: docs/steps/1.3-3d-gaussians.md and docs/steps/1.6-real-scene.md

struct SceneParams { sh_degree: u32 }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> splats: array<Splat2D>;
// Spherical-harmonic coefficients 1 to 15 per Gaussian, rgb each, coefficient-major.
@group(0) @binding(3) var<storage, read> sh_rest: array<f32>;
@group(0) @binding(4) var<uniform> scene_params: SceneParams;

const NEAR_CULL = 0.2; // the reference skips Gaussians closer than this
const BLUR = 0.3;      // added to the 2D covariance so every splat covers about a pixel or more

@compute @workgroup_size(256)
fn preprocess(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&gaussians)) { return; }
  let g = gaussians[i];
  splats[i].radius = 0.0; // culled, unless it makes it to the end

  // View space: x right, y up, the camera looking down −z.
  let t = (camera.view * vec4<f32>(g.position, 1.0)).xyz;
  let depth = -t.z;
  if (depth <= NEAR_CULL) { return; }

  // 3D covariance: Σ = R S Sᵀ Rᵀ = M Mᵀ, with M = R S.
  let r = rotation_matrix(g.rotation);
  let m = mat3x3<f32>(r[0] * g.scale.x, r[1] * g.scale.y, r[2] * g.scale.z);
  let cov3d = m * transpose(m);

  // Project to 2D (EWA splatting): Σ' = J W Σ Wᵀ Jᵀ. W rotates world into view
  // space; J is the Jacobian of view space → pixels at this Gaussian's center.
  let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.viewport * 0.5;
  // Like the reference, evaluate J no further out than 1.3× the field of view,
  // which keeps Gaussians far outside the view from blowing up.
  let limit = 1.3 * camera.viewport * 0.5 / focal;
  let tx = clamp(t.x / depth, -limit.x, limit.x) * depth;
  let ty = clamp(t.y / depth, -limit.y, limit.y) * depth;
  // Pixel x = cx + fx·x/depth and pixel y = cy − fy·y/depth, where depth = −z and
  // pixel y points down. J's columns are their derivatives by x, y and z.
  let j = mat3x2<f32>(
    focal.x / depth, 0.0,
    0.0, -focal.y / depth,
    focal.x * tx / (depth * depth), -focal.y * ty / (depth * depth),
  );
  let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let jw = j * w;
  let cov = jw * cov3d * transpose(jw);
  let a = cov[0][0] + BLUR;
  let b = cov[0][1];
  let c = cov[1][1] + BLUR;

  let det = a * c - b * b;
  if (det <= 0.0) { return; }
  // 3σ along the ellipse's longer axis, from the larger eigenvalue.
  let mid = 0.5 * (a + c);
  let lambda = mid + sqrt(max(0.1, mid * mid - det));
  let radius = ceil(3.0 * sqrt(lambda));

  let clip = camera.proj * vec4<f32>(t, 1.0);
  let mean = (clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
  // Skip splats that are entirely off screen.
  if (any(mean + radius < vec2<f32>(0.0)) || any(mean - radius > camera.viewport)) { return; }

  // The color seen from this camera. The base color is the degree-0 part; higher
  // degrees add how it changes with the viewing direction.
  var color = g.color;
  let basis_count = sh_count(scene_params.sh_degree);
  if (basis_count > 0u) {
    // The view matrix is [W | t], so the camera sits at −Wᵀ t.
    let camera_position = -(transpose(w) * camera.view[3].xyz);
    var basis = sh_basis(normalize(g.position - camera_position));
    for (var k = 0u; k < basis_count; k++) {
      color += basis[k] * sh(i, k);
    }
  }
  // Like the reference, clamp at zero. The top end is clamped when the pixel is stored.
  color = max(color, vec3<f32>(0.0));

  splats[i] = Splat2D(mean, depth, radius, vec3<f32>(c, -b, a) / det, vec4<f32>(color, g.opacity));
}

// Coefficient k + 1 (k from 0 to 14) of Gaussian i, rgb.
fn sh(i: u32, k: u32) -> vec3<f32> {
  let at = i * 45u + k * 3u;
  return vec3<f32>(sh_rest[at], sh_rest[at + 1u], sh_rest[at + 2u]);
}
