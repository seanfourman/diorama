// Turns each 3D Gaussian into a 2D splat: its center in pixels, its depth, the
// inverse of its projected 2D covariance (the "conic"), a bounding radius, and its
// color as seen from the camera. Follows preprocessCUDA in the reference
// rasterizer. Notes: docs/steps/1.3-3d-gaussians.md and docs/steps/1.6-real-scene.md

struct SceneParams { sh_degree: u32 }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> splats: array<Splat2D>;
// Spherical-harmonic coefficients 1 to 15 per Gaussian, rgb each, coefficient-major.
@group(0) @binding(3) var<storage, read> sh_rest: array<f32>;
@group(0) @binding(4) var<uniform> scene_params: SceneParams;

const NEAR_CULL = 0.2; // the reference skips Gaussians closer than this
const BLUR = 0.3;      // added to the 2D covariance so every splat covers about a pixel or more

// Spherical-harmonic constants, as in the reference.
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152789498);
const SH_C3 = array<f32, 7>(
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
  -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
);

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
  if (scene_params.sh_degree > 0u) {
    // The view matrix is [W | t], so the camera sits at −Wᵀ t.
    let camera_position = -(transpose(w) * camera.view[3].xyz);
    color += sh_color(i, normalize(g.position - camera_position), scene_params.sh_degree);
  }
  // Like the reference, clamp at zero. The top end is clamped when the pixel is stored.
  color = max(color, vec3<f32>(0.0));

  splats[i] = Splat2D(mean, depth, radius, vec3<f32>(c, -b, a) / det, vec4<f32>(color, g.opacity));
}

// Coefficient k (1 to 15) of Gaussian i, rgb.
fn sh(i: u32, k: u32) -> vec3<f32> {
  let at = i * 45u + (k - 1u) * 3u;
  return vec3<f32>(sh_rest[at], sh_rest[at + 1u], sh_rest[at + 2u]);
}

// The view-dependent part of a Gaussian's color: spherical harmonics of degrees 1
// up to `degree` in direction `d`, as in computeColorFromSH in the reference.
fn sh_color(i: u32, d: vec3<f32>, degree: u32) -> vec3<f32> {
  let x = d.x;
  let y = d.y;
  let z = d.z;
  var result = SH_C1 * (-y * sh(i, 1u) + z * sh(i, 2u) - x * sh(i, 3u));
  if (degree > 1u) {
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    result += SH_C2[0] * x * y * sh(i, 4u)
      + SH_C2[1] * y * z * sh(i, 5u)
      + SH_C2[2] * (2.0 * zz - xx - yy) * sh(i, 6u)
      + SH_C2[3] * x * z * sh(i, 7u)
      + SH_C2[4] * (xx - yy) * sh(i, 8u);
    if (degree > 2u) {
      result += SH_C3[0] * y * (3.0 * xx - yy) * sh(i, 9u)
        + SH_C3[1] * x * y * z * sh(i, 10u)
        + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh(i, 11u)
        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh(i, 12u)
        + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh(i, 13u)
        + SH_C3[5] * z * (xx - yy) * sh(i, 14u)
        + SH_C3[6] * x * (xx - 3.0 * yy) * sh(i, 15u);
    }
  }
  return result;
}

// The rotation matrix of a quaternion stored as (w, x, y, z), normalized first.
fn rotation_matrix(raw: vec4<f32>) -> mat3x3<f32> {
  let q = normalize(raw);
  let qw = q.x;
  let qx = q.y;
  let qy = q.z;
  let qz = q.w;
  // Column-major: each line below is one column.
  return mat3x3<f32>(
    1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy),
    2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx),
    2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy),
  );
}
