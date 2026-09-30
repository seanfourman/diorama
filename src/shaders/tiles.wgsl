// The tile stage (M1.5), following the reference. Count the tiles each splat's
// bounding square touches; write one (depth, tile) sort key per (splat, tile)
// pair, into the slots a prefix sum over the counts gives each splat. Once the
// pairs are sorted by tile and then depth, find where each tile's run starts and
// ends. Notes: docs/steps/1.5-tiles.md

struct TileParams { pair_capacity: u32 }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> splats: array<Splat2D>;
// Tiles touched per splat. After the prefix sum, each splat's first pair slot.
@group(0) @binding(2) var<storage, read_write> tile_offsets: array<u32>;
// Element 0: how many pairs this frame needs (the prefix sum's total).
@group(0) @binding(3) var<storage, read> pair_total: array<u32>;
// One sort key per pair, as (low word, high word) = (depth bits, tile index),
// and the pair's splat.
@group(0) @binding(4) var<storage, read_write> pair_keys: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> pair_splats: array<u32>;
// Each tile's run of the sorted pairs: [start, end).
@group(0) @binding(6) var<storage, read_write> tile_ranges: array<vec2<u32>>;
@group(0) @binding(7) var<uniform> tile_params: TileParams;

@compute @workgroup_size(256)
fn count_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&splats)) { return; }
  let s = splats[i];
  var touched = 0u;
  if (s.radius > 0.0) {
    let rect = tile_rect(s.mean, s.radius, tile_count(camera.viewport));
    touched = (rect.z - rect.x) * (rect.w - rect.y);
  }
  tile_offsets[i] = touched;
}

@compute @workgroup_size(256)
fn duplicate(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&splats)) { return; }
  let s = splats[i];
  if (s.radius == 0.0) { return; }
  let tiles = tile_count(camera.viewport);
  let rect = tile_rect(s.mean, s.radius, tiles);
  // A positive float's bits sort like the number, so within each tile the pairs
  // end up front to back.
  let depth_bits = bitcast<u32>(s.depth);
  var slot = tile_offsets[i];
  for (var y = rect.y; y < rect.w; y++) {
    for (var x = rect.x; x < rect.z; x++) {
      // Pairs past the capacity are dropped. The renderer grows its buffers for
      // the next frame.
      if (slot < tile_params.pair_capacity) {
        pair_keys[slot] = vec2<u32>(depth_bits, y * tiles.x + x);
        pair_splats[slot] = i;
      }
      slot++;
    }
  }
}

// Runs over the sorted pairs. Wherever the tile index changes, one tile's run
// ends and the next one's starts. Dispatched in 2D when there are more pairs
// than 65,535 workgroups can cover in one row.
@compute @workgroup_size(256)
fn find_ranges(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let pair_count = min(pair_total[0], tile_params.pair_capacity);
  let i = (wg.y * groups.x + wg.x) * 256u + tid;
  if (i >= pair_count) { return; }
  let tile_id = pair_keys[i].y;
  if (i == 0u) {
    tile_ranges[tile_id].x = 0u;
  } else {
    let previous = pair_keys[i - 1u].y;
    if (tile_id != previous) {
      tile_ranges[previous].y = i;
      tile_ranges[tile_id].x = i;
    }
  }
  if (i == pair_count - 1u) { tile_ranges[tile_id].y = pair_count; }
}
