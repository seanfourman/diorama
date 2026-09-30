// Exclusive prefix sum in place, one block of 512 values per workgroup, using
// Blelloch's work-efficient scan. Each block's total goes to `sums`, so the next
// level up can scan the totals and add_block_sums can offset each block by the
// total of the blocks before it. How many values there are comes from a GPU
// buffer, so it can change every frame without the CPU knowing it (M1.5): blocks
// past the end exit straight away. Notes: docs/steps/1.4-depth-sort.md

const SCAN_BLOCK = 512u; // two values per thread

struct ScanParams { level: u32, capacity: u32 }

@group(0) @binding(0) var<uniform> params: ScanParams;
// Element 0: how many values to scan (at most the capacity).
@group(0) @binding(1) var<storage, read> scan_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;

var<workgroup> temp: array<u32, 512>;

// This level's value count. Level 0 scans the values themselves; each level
// after that scans one total per block of the level below.
fn level_count() -> u32 {
  var n = min(scan_count[0], params.capacity);
  for (var k = 0u; k < params.level; k++) {
    n = (n + SCAN_BLOCK - 1u) / SCAN_BLOCK;
  }
  return n;
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let n = level_count();
  // Block 0 always runs, so the total gets written (as 0) even with nothing to scan.
  if (wg.x > 0u && wg.x * SCAN_BLOCK >= n) { return; }
  let base = wg.x * SCAN_BLOCK;
  for (var k = 0u; k < 2u; k++) {
    let i = base + k * 256u + tid;
    var value = 0u;
    if (i < n) { value = data[i]; }
    temp[k * 256u + tid] = value;
  }

  // Up-sweep: build a tree of partial sums in place. Afterwards the last slot
  // holds the block's total.
  var offset = 1u;
  for (var d = SCAN_BLOCK / 2u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (tid < d) {
      let left = offset * (2u * tid + 1u) - 1u;
      let right = offset * (2u * tid + 2u) - 1u;
      temp[right] += temp[left];
    }
    offset <<= 1u;
  }
  if (tid == 0u) {
    sums[wg.x] = temp[SCAN_BLOCK - 1u];
    temp[SCAN_BLOCK - 1u] = 0u;
  }

  // Down-sweep: push the partial sums back down the tree, which leaves each slot
  // holding the sum of everything before it.
  for (var d = 1u; d < SCAN_BLOCK; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (tid < d) {
      let left = offset * (2u * tid + 1u) - 1u;
      let right = offset * (2u * tid + 2u) - 1u;
      let t = temp[left];
      temp[left] = temp[right];
      temp[right] += t;
    }
  }
  workgroupBarrier();

  for (var k = 0u; k < 2u; k++) {
    let i = base + k * 256u + tid;
    if (i < n) { data[i] = temp[k * 256u + tid]; }
  }
}

// Once `sums` holds the scanned block totals, offset every value by its block's.
@compute @workgroup_size(256)
fn add_block_sums(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let n = level_count();
  if (wg.x * SCAN_BLOCK >= n) { return; }
  let offset = sums[wg.x];
  for (var k = 0u; k < 2u; k++) {
    let i = wg.x * SCAN_BLOCK + k * 256u + tid;
    if (i < n) { data[i] += offset; }
  }
}
