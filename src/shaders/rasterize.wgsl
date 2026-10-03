// Draws every pixel of one 16×16 tile per workgroup (M1.5), following renderCUDA
// in the reference. The tile's splats, sorted front to back, are loaded into
// workgroup memory 256 at a time. Each pixel blends them in order until it's
// nearly opaque, then composites what's left over the background. Once every
// pixel in the tile is nearly opaque, the tile stops loading splats. It also
// saves, per pixel, what the backward pass needs (M2). Notes: docs/steps/1.5-tiles.md

struct RasterParams { background: vec4<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> raster: RasterParams;
@group(0) @binding(2) var<storage, read> splats: array<Splat2D>;
@group(0) @binding(3) var<storage, read> pair_splats: array<u32>;
@group(0) @binding(4) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;
// Per pixel, row by row: the unclamped color (rgb) and the transmittance left at
// the end (a), for the loss and the backward pass.
@group(0) @binding(6) var<storage, read_write> pixel_color: array<vec4<f32>>;
// Per pixel: how many splats down the tile's list the last one it blended was.
@group(0) @binding(7) var<storage, read_write> pixel_last: array<u32>;

const BATCH = 256u; // one splat per thread per batch

var<workgroup> batch_mean: array<vec2<f32>, 256>;
var<workgroup> batch_conic: array<vec4<f32>, 256>; // conic (a, b, c), then opacity
var<workgroup> batch_color: array<vec4<f32>, 256>;
// How many of the tile's pixels are done, and whether that's all of them.
var<workgroup> done_count: atomic<u32>;
var<workgroup> tile_done: u32;

@compute @workgroup_size(16, 16)
fn rasterize(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let tiles = tile_count(camera.viewport);
  let run = tile_ranges[wg.y * tiles.x + wg.x];
  let pixel = wg.xy * TILE_SIZE + lid.xy;
  let inside = all(vec2<f32>(pixel) < camera.viewport);
  // Sample at the pixel's center, in the same pixel space as the splat means.
  let center = vec2<f32>(pixel) + 0.5;

  var transmittance = 1.0;
  var color = vec3<f32>(0.0);
  var done = !inside;
  var visited = 0u; // splats this pixel has looked at so far
  var last = 0u;
  var counted = false; // whether this pixel is in done_count yet
  if (tid == 0u) { atomicStore(&done_count, 0u); }
  for (var start = run.x; start < run.y; start += BATCH) {
    // Every thread loads one splat of the next batch into workgroup memory.
    workgroupBarrier();
    let k = start + tid;
    if (k < run.y) {
      let s = splats[pair_splats[k]];
      batch_mean[tid] = s.mean;
      batch_conic[tid] = vec4<f32>(s.conic, s.color.a);
      batch_color[tid] = s.color;
    }
    workgroupBarrier();

    let batch_size = min(BATCH, run.y - start);
    for (var j = 0u; j < batch_size && !done; j++) {
      visited++;
      let d = center - batch_mean[j];
      let conic = batch_conic[j];
      let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
      if (power > 0.0) { continue; }
      let alpha = min(0.99, conic.w * exp(power));
      if (alpha < 1.0 / 255.0) { continue; }
      let remaining = transmittance * (1.0 - alpha);
      // Like the reference, stop once the pixel is nearly opaque, without adding
      // this splat.
      if (remaining < 0.0001) {
        done = true;
        continue;
      }
      color += batch_color[j].rgb * alpha * transmittance;
      transmittance = remaining;
      last = visited;
    }

    // Like the reference, the whole tile stops once every pixel is done.
    if (done && !counted) {
      atomicAdd(&done_count, 1u);
      counted = true;
    }
    workgroupBarrier();
    if (tid == 0u) { tile_done = u32(atomicLoad(&done_count) == 256u); }
    if (workgroupUniformLoad(&tile_done) == 1u) { break; }
  }
  if (inside) {
    let final_color = color + transmittance * raster.background.rgb;
    textureStore(output, pixel, vec4<f32>(final_color, 1.0));
    let index = pixel.y * u32(camera.viewport.x) + pixel.x;
    pixel_color[index] = vec4<f32>(final_color, transmittance);
    pixel_last[index] = last;
  }
}
