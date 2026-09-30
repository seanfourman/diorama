// Draws every pixel of one 16×16 tile per workgroup (M1.5), following renderCUDA
// in the reference. The tile's splats, sorted front to back, are loaded into
// workgroup memory 256 at a time. Each pixel blends them in order until it's
// nearly opaque, then composites what's left over the background.
// Notes: docs/steps/1.5-tiles.md

struct RasterParams { background: vec4<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> raster: RasterParams;
@group(0) @binding(2) var<storage, read> splats: array<Splat2D>;
@group(0) @binding(3) var<storage, read> pair_splats: array<u32>;
@group(0) @binding(4) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;

const BATCH = 256u; // one splat per thread per batch

var<workgroup> batch_mean: array<vec2<f32>, 256>;
var<workgroup> batch_conic: array<vec4<f32>, 256>; // conic (a, b, c), then opacity
var<workgroup> batch_color: array<vec4<f32>, 256>;

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
    }
  }
  if (inside) {
    textureStore(output, pixel, vec4<f32>(color + transmittance * raster.background.rgb, 1.0));
  }
}
