// Math shared by the forward preprocess and its backward pass: prepended to both,
// so they can't drift apart.

// Spherical-harmonic constants, as in the reference.
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152789498);
const SH_C3 = array<f32, 7>(
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
  -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
);

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

// Spherical-harmonic basis functions 1 to 15 in direction d (unit length), as in
// computeColorFromSH in the reference. Degree 1 is the first 3, degree 2 the
// first 8, degree 3 all 15. Function 0 is a constant and lives in the base color.
fn sh_basis(d: vec3<f32>) -> array<f32, 15> {
  let x = d.x;
  let y = d.y;
  let z = d.z;
  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  return array<f32, 15>(
    -SH_C1 * y, SH_C1 * z, -SH_C1 * x,
    SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2.0 * zz - xx - yy), SH_C2[3] * x * z, SH_C2[4] * (xx - yy),
    SH_C3[0] * y * (3.0 * xx - yy), SH_C3[1] * x * y * z, SH_C3[2] * y * (4.0 * zz - xx - yy),
    SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), SH_C3[4] * x * (4.0 * zz - xx - yy),
    SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3.0 * yy),
  );
}

// The gradient of each basis function above with respect to (x, y, z).
fn sh_basis_gradient(d: vec3<f32>) -> array<vec3<f32>, 15> {
  let x = d.x;
  let y = d.y;
  let z = d.z;
  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  return array<vec3<f32>, 15>(
    vec3<f32>(0.0, -SH_C1, 0.0),
    vec3<f32>(0.0, 0.0, SH_C1),
    vec3<f32>(-SH_C1, 0.0, 0.0),
    SH_C2[0] * vec3<f32>(y, x, 0.0),
    SH_C2[1] * vec3<f32>(0.0, z, y),
    SH_C2[2] * vec3<f32>(-2.0 * x, -2.0 * y, 4.0 * z),
    SH_C2[3] * vec3<f32>(z, 0.0, x),
    SH_C2[4] * vec3<f32>(2.0 * x, -2.0 * y, 0.0),
    SH_C3[0] * vec3<f32>(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0),
    SH_C3[1] * vec3<f32>(y * z, x * z, x * y),
    SH_C3[2] * vec3<f32>(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z),
    SH_C3[3] * vec3<f32>(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy),
    SH_C3[4] * vec3<f32>(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z),
    SH_C3[5] * vec3<f32>(2.0 * x * z, -2.0 * y * z, xx - yy),
    SH_C3[6] * vec3<f32>(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0),
  );
}

// How many basis functions a degree uses: 0, 3, 8 or 15.
fn sh_count(degree: u32) -> u32 {
  return (degree + 1u) * (degree + 1u) - 1u;
}
