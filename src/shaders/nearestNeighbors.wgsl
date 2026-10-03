// For each point, the mean squared distance to its 3 nearest neighbors, by brute
// force (M3). Training's initial Gaussians are sized by this, as in the reference
// (its distCUDA2). Every thread checks every other point, so a workgroup loads the
// points 256 at a time into workgroup memory and all its threads share them.
// About 180,000 points means 3 × 10¹⁰ distances, a fraction of a second here.

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>; // xyz + padding
@group(0) @binding(1) var<storage, read_write> mean_distance2: array<f32>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn nearest(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let n = arrayLength(&points);
  let i = id.x;
  let me = points[min(i, n - 1u)].xyz;
  // The three smallest squared distances so far, smallest first.
  var best = vec3<f32>(3.4e38);
  for (var start = 0u; start < n; start += 256u) {
    workgroupBarrier();
    if (start + tid < n) { tile[tid] = points[start + tid]; }
    workgroupBarrier();
    let batch = min(256u, n - start);
    for (var j = 0u; j < batch; j++) {
      if (start + j == i) { continue; }
      let d = tile[j].xyz - me;
      let d2 = dot(d, d);
      if (d2 < best.z) {
        if (d2 < best.y) {
          best.z = best.y;
          if (d2 < best.x) {
            best.y = best.x;
            best.x = d2;
          } else {
            best.y = d2;
          }
        } else {
          best.z = d2;
        }
      }
    }
  }
  if (i < n) { mean_distance2[i] = (best.x + best.y + best.z) / 3.0; }
}
