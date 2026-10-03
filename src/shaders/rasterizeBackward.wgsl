// The backward pass of rasterize.wgsl (M2), following renderCUDA's backward in the
// reference. Given dL/d(pixel color), it works out every splat's share of the
// gradient: with respect to its 2D center, its conic, its opacity and its color.
//
// Each pixel walks its tile's list back to front, starting from its last blended
// splat, and undoes the blending as it goes. Dividing the final transmittance by
// (1 − alpha) recovers the transmittance in front of each splat. The 256 pixels'
// shares of a splat are summed in workgroup memory, and only the tile's total
// goes to the splat's global gradient. WebGPU has no float atomics, so that last
// add is a compare-and-swap loop, and summing first keeps its contention low.
// Notes: docs/steps/2-backward-pass.md

struct RasterParams { background: vec4<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> raster: RasterParams;
@group(0) @binding(2) var<storage, read> splats: array<Splat2D>;
@group(0) @binding(3) var<storage, read> pair_splats: array<u32>;
@group(0) @binding(4) var<storage, read> tile_ranges: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> pixel_color: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> pixel_last: array<u32>;
// dL/d(pixel rgb), row by row.
@group(0) @binding(7) var<storage, read> pixel_grads: array<vec4<f32>>;
// Per splat, GRADS floats (as u32 bits, so they can be added atomically):
// d/d(center x, center y), d/d(conic a, b, c), d/d(opacity), d/d(color r, g, b).
@group(0) @binding(8) var<storage, read_write> splat_grads: array<atomic<u32>>;

const BATCH = 128u; // smaller than the forward's, to leave workgroup memory for the sums
const GRADS = 9u;

var<workgroup> batch_mean: array<vec2<f32>, 128>;
var<workgroup> batch_conic: array<vec4<f32>, 128>;
var<workgroup> batch_color: array<vec4<f32>, 128>;
var<workgroup> batch_index: array<u32, 128>;
// Each pixel's share of the current splat's gradient, then each column's sum.
var<workgroup> shares: array<array<f32, 256>, 9>;
var<workgroup> column_sums: array<array<f32, 16>, 9>;

// A float add built from a compare-and-swap loop.
fn atomic_add(index: u32, value: f32) {
  var old = atomicLoad(&splat_grads[index]);
  loop {
    let swap = atomicCompareExchangeWeak(&splat_grads[index], old, bitcast<u32>(bitcast<f32>(old) + value));
    if (swap.exchanged) { break; }
    old = swap.old_value;
  }
}

@compute @workgroup_size(16, 16)
fn rasterize_backward(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let tiles = tile_count(camera.viewport);
  let run = tile_ranges[wg.y * tiles.x + wg.x];
  let pixel = wg.xy * TILE_SIZE + lid.xy;
  let inside = all(vec2<f32>(pixel) < camera.viewport);
  let center = vec2<f32>(pixel) + 0.5;

  var final_transmittance = 0.0;
  var last = 0u;
  var pixel_grad = vec3<f32>(0.0);
  if (inside) {
    let index = pixel.y * u32(camera.viewport.x) + pixel.x;
    final_transmittance = pixel_color[index].a;
    last = pixel_last[index];
    pixel_grad = pixel_grads[index].rgb;
  }
  // How much the background term moves the loss per unit of transmittance.
  let background_grad = dot(raster.background.rgb, pixel_grad);

  var transmittance = final_transmittance;
  // The color of everything behind the current splat, as seen through it.
  var behind = vec3<f32>(0.0);
  var next_alpha = 0.0;
  var next_color = vec3<f32>(0.0);

  // Walk the tile's list back to front, a batch at a time.
  var batch_end = run.y;
  loop {
    if (batch_end <= run.x) { break; }
    let batch_start = select(run.x, batch_end - BATCH, batch_end - run.x > BATCH);
    workgroupBarrier();
    if (tid < batch_end - batch_start) {
      let index = pair_splats[batch_start + tid];
      let s = splats[index];
      batch_mean[tid] = s.mean;
      batch_conic[tid] = vec4<f32>(s.conic, s.color.a);
      batch_color[tid] = s.color;
      batch_index[tid] = index;
    }
    workgroupBarrier();

    for (var slot = batch_end - batch_start; slot > 0u; slot--) {
      let j = slot - 1u;
      var share: array<f32, 9>;
      // Only splats this pixel actually blended count: those before its last one.
      if (inside && batch_start + j - run.x < last) {
        let d = center - batch_mean[j];
        let conic = batch_conic[j];
        let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
        if (power <= 0.0) {
          let gaussian = exp(power);
          let alpha = min(0.99, conic.w * gaussian);
          if (alpha >= 1.0 / 255.0) {
            // The transmittance in front of this splat.
            transmittance /= 1.0 - alpha;
            let color = batch_color[j].rgb;
            behind = next_alpha * next_color + (1.0 - next_alpha) * behind;
            next_alpha = alpha;
            next_color = color;
            // color = Σ cᵢ αᵢ Tᵢ + T_final × background, so
            // dC/dαᵢ = Tᵢ (cᵢ − behind) − T_final × background / (1 − αᵢ).
            let alpha_grad =
              transmittance * dot(color - behind, pixel_grad) -
              final_transmittance / (1.0 - alpha) * background_grad;
            // alpha = opacity × G and G = exp(power), with d = pixel − center.
            let power_grad = alpha_grad * conic.w * gaussian;
            share[0] = power_grad * (conic.x * d.x + conic.y * d.y);
            share[1] = power_grad * (conic.y * d.x + conic.z * d.y);
            share[2] = power_grad * (-0.5 * d.x * d.x);
            share[3] = power_grad * (-d.x * d.y);
            share[4] = power_grad * (-0.5 * d.y * d.y);
            share[5] = alpha_grad * gaussian;
            let color_grad = alpha * transmittance * pixel_grad;
            share[6] = color_grad.r;
            share[7] = color_grad.g;
            share[8] = color_grad.b;
          }
        }
      }

      // Sum the 256 shares: each column of 16, then the 16 column sums. Summing
      // columns, neighboring threads read neighboring words of workgroup memory,
      // which avoids bank conflicts.
      for (var v = 0u; v < GRADS; v++) {
        shares[v][tid] = share[v];
      }
      workgroupBarrier();
      if (tid < GRADS * 16u) {
        let v = tid / 16u;
        let column = tid % 16u;
        var sum = 0.0;
        for (var row = 0u; row < 16u; row++) {
          sum += shares[v][row * 16u + column];
        }
        column_sums[v][column] = sum;
      }
      workgroupBarrier();
      if (tid < GRADS) {
        var total = 0.0;
        for (var column = 0u; column < 16u; column++) {
          total += column_sums[tid][column];
        }
        if (total != 0.0) {
          atomic_add(batch_index[j] * GRADS + tid, total);
        }
      }
    }
    batch_end = batch_start;
  }
}
