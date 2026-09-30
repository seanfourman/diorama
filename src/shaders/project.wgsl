// Projects each point's center to pixel coordinates. No covariance yet (M1.1).

struct Camera { view_proj: mat4x4<f32>, viewport: vec2<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> screen: array<vec2<f32>>;

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&positions)) { return; }
  let clip = camera.view_proj * vec4<f32>(positions[i].xyz, 1.0);
  let ndc = clip.xy / clip.w;
  screen[i] = (ndc * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
}
