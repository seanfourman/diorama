// Densification and pruning (M3.5), as in the reference's densify_and_prune. Every
// 100 steps, training reshapes the set of Gaussians:
// - A Gaussian whose 2D center kept getting pulled hard (a large average gradient)
//   is struggling to cover its area. A small one is cloned (a second copy that
//   can move away); a large one is split into two smaller ones placed at random
//   inside it.
// - Gaussians that have faded out, or grown too large in the world, are removed.
// It runs in two passes around a prefix sum: count_outputs writes how many
// Gaussians each one becomes (0, 1 or 2), the prefix sum turns the counts into
// output positions, and scatter writes the new set there. count_outputs also
// tallies what happened, and scatter stamps each new Gaussian with how and when
// it was made, for the Inside view (M4).
// gaussianMath.wgsl is prepended (for rotation_matrix). Notes: docs/steps/3-training.md

struct DensifyParams {
  // The average 2D-center gradient, in NDC units, from which a Gaussian is densified.
  grad_threshold: f32,
  // At most this large (its largest scale), a Gaussian is cloned; above it, split.
  clone_max_size: f32,
  // Below this opacity, a Gaussian is removed.
  min_opacity: f32,
  // Above this size, a Gaussian is removed; 0 turns the test off.
  max_size: f32,
  // The current step: varies the random split positions from one densification
  // to the next, and stamps the new Gaussians.
  seed: u32,
}

const PARAMS = 16u; // raw parameters per Gaussian, as in adam.wgsl
const SH = 45u;
const SLOTS = PARAMS + SH; // Adam moments per Gaussian
// The reference splits each Gaussian into 2, each 1/(0.8 × 2) the size.
const SPLIT_SHRINK = 1.6;

const DROP = 0u;
const KEEP = 1u;
const CLONE = 2u;
const SPLIT = 3u;

@group(0) @binding(0) var<uniform> settings: DensifyParams;
@group(0) @binding(1) var<storage, read> params: array<f32>;
// Per Gaussian: (sum of 2D-center gradient norms, times seen, largest radius, unused).
@group(0) @binding(2) var<storage, read> stats: array<vec4<f32>>;
// count_outputs writes each Gaussian's output count here; after the prefix sum,
// it holds where each Gaussian's outputs start.
@group(0) @binding(3) var<storage, read_write> outputs: array<u32>;
@group(0) @binding(4) var<storage, read> sh: array<f32>;
@group(0) @binding(5) var<storage, read> moments: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> new_params: array<f32>;
@group(0) @binding(7) var<storage, read_write> new_sh: array<f32>;
@group(0) @binding(8) var<storage, read_write> new_moments: array<vec2<f32>>;
// How many Gaussians were removed, kept, cloned and split, indexed by action.
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>, 4>;

// Densification's record in each Gaussian's padding: float 7 says how it came
// about (0 from the starting points, 1 cloned, 2 split), float 15 at which step.
const KIND_SLOT = 7u;
const BIRTH_SLOT = 15u;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn removed(opacity: f32, size: f32) -> bool {
  return opacity < settings.min_opacity || (settings.max_size > 0.0 && size > settings.max_size);
}

// What happens to Gaussian i. The reference clones, then splits, then prunes the
// result, so a clone or split child is removed if it would fail the pruning test.
// Both clone copies match the original, and both split children have the same
// opacity and size, so the test is the same for every output.
fn action(i: u32) -> u32 {
  let at = i * PARAMS;
  let s = stats[i];
  let grad = select(0.0, s.x / s.y, s.y > 0.0); // never seen: no gradient
  let size = exp(max(params[at + 4u], max(params[at + 5u], params[at + 6u])));
  let opacity = sigmoid(params[at + 3u]);
  if (grad >= settings.grad_threshold) {
    if (size <= settings.clone_max_size) {
      return select(CLONE, DROP, removed(opacity, size));
    }
    return select(SPLIT, DROP, removed(opacity, size / SPLIT_SHRINK));
  }
  return select(KEEP, DROP, removed(opacity, size));
}

fn gaussian_count() -> u32 {
  return arrayLength(&params) / PARAMS;
}

// Each workgroup tallies its own Gaussians first (workgroup memory starts at
// zero), so only 4 adds per workgroup reach the global counters.
var<workgroup> local_counts: array<atomic<u32>, 4>;

@compute @workgroup_size(256)
fn count_outputs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let i = id.x;
  if (i < gaussian_count()) {
    let a = action(i);
    outputs[i] = select(select(1u, 2u, a == CLONE || a == SPLIT), 0u, a == DROP);
    atomicAdd(&local_counts[a], 1u);
  }
  workgroupBarrier();
  if (tid < 4u) {
    atomicAdd(&counters[tid], atomicLoad(&local_counts[tid]));
  }
}

// Copies Gaussian i to slot o. A Gaussian that carries on keeps its Adam history;
// a new one starts without, as in the reference.
fn copy_gaussian(i: u32, o: u32, keep_moments: bool) {
  for (var k = 0u; k < PARAMS; k++) {
    new_params[o * PARAMS + k] = params[i * PARAMS + k];
  }
  for (var k = 0u; k < SH; k++) {
    new_sh[o * SH + k] = sh[i * SH + k];
  }
  for (var k = 0u; k < SLOTS; k++) {
    new_moments[o * SLOTS + k] = select(vec2<f32>(0.0), moments[i * SLOTS + k], keep_moments);
  }
}

// A well-mixed 32-bit hash (PCG), for random numbers without any state.
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Uniform on (0, 1), from a hash's top 24 bits.
fn unit(h: u32) -> f32 {
  return (f32(h >> 8u) + 0.5) / 16777216.0;
}

// Three independent standard normal numbers, by the Box–Muller transform.
fn normal3(seed: u32) -> vec3<f32> {
  let h1 = pcg(seed);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  let tau = 6.283185307179586;
  let r1 = sqrt(-2.0 * log(unit(h1)));
  let r2 = sqrt(-2.0 * log(unit(h3)));
  return vec3<f32>(r1 * cos(tau * unit(h2)), r1 * sin(tau * unit(h2)), r2 * cos(tau * unit(h4)));
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= gaussian_count()) { return; }
  let a = action(i);
  let o = outputs[i];
  if (a == KEEP || a == CLONE) {
    copy_gaussian(i, o, true);
  }
  if (a == CLONE) {
    copy_gaussian(i, o + 1u, false);
    new_params[(o + 1u) * PARAMS + KIND_SLOT] = 1.0;
    new_params[(o + 1u) * PARAMS + BIRTH_SLOT] = f32(settings.seed);
  }
  if (a == SPLIT) {
    // Each child is placed at a random point drawn from the Gaussian itself, and
    // shrunk: position = center + R (scale ⊙ n), n ~ N(0, I).
    let at = i * PARAMS;
    let center = vec3<f32>(params[at], params[at + 1u], params[at + 2u]);
    let log_scale = vec3<f32>(params[at + 4u], params[at + 5u], params[at + 6u]);
    let rotation = rotation_matrix(vec4<f32>(params[at + 8u], params[at + 9u], params[at + 10u], params[at + 11u]));
    for (var c = 0u; c < 2u; c++) {
      let child = o + c;
      copy_gaussian(i, child, false);
      let position = center + rotation * (exp(log_scale) * normal3(pcg(settings.seed ^ pcg(2u * i + c))));
      for (var k = 0u; k < 3u; k++) {
        new_params[child * PARAMS + k] = position[k];
        new_params[child * PARAMS + 4u + k] = log_scale[k] - log(SPLIT_SHRINK);
      }
      new_params[child * PARAMS + KIND_SLOT] = 2.0;
      new_params[child * PARAMS + BIRTH_SLOT] = f32(settings.seed);
    }
  }
}
