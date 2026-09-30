// Stable radix sort, one 8-bit digit per pass (M1.4). count_digits builds a
// histogram of digits for each block of 1024 keys. A prefix sum over those counts
// (digit-major) turns each count into the output position where that block's run
// of that digit starts. scatter then moves every key to its place.
//
// The key type comes from a prelude that radixSort.ts prepends: u32, or for
// 64-bit keys, vec2<u32> as (low word, high word). How many keys to sort comes
// from a GPU buffer, so it can change every frame without the CPU knowing it
// (M1.5): blocks past the end exit straight away. Notes: docs/steps/1.4-depth-sort.md

const WORKGROUP_SIZE = 256u;
const ITEMS_PER_THREAD = 4u;
const BLOCK_SIZE = 1024u; // WORKGROUP_SIZE × ITEMS_PER_THREAD

struct SortParams { shift: u32, word: u32, capacity: u32 }

@group(0) @binding(0) var<uniform> params: SortParams;
// Element 0: how many keys to sort (at most the capacity).
@group(0) @binding(1) var<storage, read> sort_count: array<u32>;
@group(0) @binding(2) var<storage, read> keys_in: array<Key>;
@group(0) @binding(3) var<storage, read> values_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<Key>;
@group(0) @binding(5) var<storage, read_write> values_out: array<u32>;
// counts[digit × blocks + block]: how many keys of each digit each block has, and
// after the prefix sum, where that block's run of that digit starts.
@group(0) @binding(6) var<storage, read_write> counts: array<u32>;
// Element 0: how many entries of `counts` are in use, for the prefix sum to read.
@group(0) @binding(7) var<storage, read_write> counts_length: array<u32>;

fn key_count() -> u32 {
  return min(sort_count[0], params.capacity);
}

fn block_count() -> u32 {
  return (key_count() + BLOCK_SIZE - 1u) / BLOCK_SIZE;
}

fn digit_of(key: Key) -> u32 {
  return (key_word(key, params.word) >> params.shift) & 0xffu;
}

var<workgroup> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn count_digits(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blocks = block_count();
  if (wg.x == 0u && tid == 0u) {
    counts_length[0] = 256u * blocks;
  }
  if (wg.x >= blocks) { return; }

  atomicStore(&histogram[tid], 0u);
  workgroupBarrier();
  let n = key_count();
  for (var chunk = 0u; chunk < ITEMS_PER_THREAD; chunk++) {
    let i = wg.x * BLOCK_SIZE + chunk * WORKGROUP_SIZE + tid;
    if (i < n) {
      atomicAdd(&histogram[digit_of(keys_in[i])], 1u);
    }
  }
  workgroupBarrier();
  counts[tid * blocks + wg.x] = atomicLoad(&histogram[tid]);
}

// For each digit, a 256-bit mask (8 words) of which threads hold that digit in
// the current chunk. A key's rank among equal digits is the number of set bits
// below its own thread's bit. That keeps equal keys in their original order
// (stable) without needing subgroups.
var<workgroup> digit_masks: array<atomic<u32>, 2048>;
// The next free output position for each digit, in this block.
var<workgroup> next_position: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blocks = block_count();
  if (wg.x >= blocks) { return; }
  let n = key_count();
  next_position[tid] = counts[tid * blocks + wg.x];
  let word = tid / 32u;
  let bit = 1u << (tid % 32u);

  for (var chunk = 0u; chunk < ITEMS_PER_THREAD; chunk++) {
    for (var w = 0u; w < 8u; w++) {
      atomicStore(&digit_masks[tid * 8u + w], 0u);
    }
    workgroupBarrier();

    let i = wg.x * BLOCK_SIZE + chunk * WORKGROUP_SIZE + tid;
    let valid = i < n;
    var key: Key;
    var digit = 0u;
    if (valid) {
      key = keys_in[i];
      digit = digit_of(key);
      atomicOr(&digit_masks[digit * 8u + word], bit);
    }
    workgroupBarrier();

    if (valid) {
      // Threads before this one holding the same digit: the lower bits of its
      // own word, plus every bit of the words before it.
      var rank = countOneBits(atomicLoad(&digit_masks[digit * 8u + word]) & (bit - 1u));
      for (var w = 0u; w < word; w++) {
        rank += countOneBits(atomicLoad(&digit_masks[digit * 8u + w]));
      }
      let destination = next_position[digit] + rank;
      keys_out[destination] = key;
      values_out[destination] = values_in[i];
    }
    workgroupBarrier();

    // Move each digit's next free position past this chunk's keys of that digit.
    var chunk_count = 0u;
    for (var w = 0u; w < 8u; w++) {
      chunk_count += countOneBits(atomicLoad(&digit_masks[tid * 8u + w]));
    }
    next_position[tid] += chunk_count;
    workgroupBarrier();
  }
}
