// Shared by preprocess.wgsl and splat.wgsl: each is compiled with this file prepended.
// The CPU-side mirrors of these structs are in src/gaussians.ts.

struct Camera {
  view: mat4x4<f32>,      // world → view space: x right, y up, the camera looking down −z
  proj: mat4x4<f32>,      // view → clip space, with WebGPU's 0..1 depth range
  viewport: vec2<f32>,    // in pixels
}

// A 3D Gaussian as stored in GPU memory: 16 floats.
struct Gaussian {
  position: vec3<f32>,
  opacity: f32,
  scale: vec3<f32>,       // standard deviations along the Gaussian's own axes
  rotation: vec4<f32>,    // quaternion (w, x, y, z); the shader normalizes it
  color: vec3<f32>,
}

// A Gaussian projected to the screen, which is everything the draw needs: 12 floats.
struct Splat2D {
  mean: vec2<f32>,        // center in pixels, y down
  depth: f32,             // distance in front of the camera
  radius: f32,            // half-size of the bounding square in pixels; 0 means culled
  conic: vec3<f32>,       // inverse 2D covariance [[a, b], [b, c]] as (a, b, c)
  color: vec4<f32>,       // rgb + opacity
}

// The tile grid (M1.5): the screen is cut into TILE_SIZE × TILE_SIZE-pixel tiles.
const TILE_SIZE = 16u;

fn tile_count(viewport: vec2<f32>) -> vec2<u32> {
  return (vec2<u32>(viewport) + TILE_SIZE - 1u) / TILE_SIZE;
}

// The tiles a splat's bounding square touches, as (min x, min y, end x, end y),
// with the ends exclusive and everything clamped to the grid.
fn tile_rect(mean: vec2<f32>, radius: f32, tiles: vec2<u32>) -> vec4<u32> {
  let size = f32(TILE_SIZE);
  let lo = clamp(floor((mean - radius) / size), vec2<f32>(0.0), vec2<f32>(tiles));
  let hi = clamp(floor((mean + radius) / size) + 1.0, vec2<f32>(0.0), vec2<f32>(tiles));
  return vec4<u32>(vec2<u32>(lo), vec2<u32>(hi));
}
