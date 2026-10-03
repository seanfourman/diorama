// Draws an image onto the canvas at its own aspect ratio, letterboxed (M4). For
// side-by-side comparisons, pixels left of `split` come from one image and the
// rest from another, with a thin line between them.

struct BlitParams {
  // Where the image goes on the target, in pixels: left, top, right, bottom.
  rect: vec4<f32>,
  // The x, in pixels, where `left_image` gives way to `right_image`.
  split: f32,
}

@group(0) @binding(0) var<uniform> blit: BlitParams;
@group(0) @binding(1) var left_image: texture_2d<f32>;
@group(0) @binding(2) var right_image: texture_2d<f32>;
@group(0) @binding(3) var image_sampler: sampler;

const BACKDROP = vec3<f32>(0.07, 0.08, 0.12);

// One triangle that covers the whole target.
@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let corner = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn draw(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let p = position.xy; // pixel centers, y down
  if (any(p < blit.rect.xy) || any(p >= blit.rect.zw)) {
    return vec4<f32>(BACKDROP, 1.0);
  }
  if (abs(p.x - blit.split) < 1.0) {
    return vec4<f32>(1.0);
  }
  let uv = (p - blit.rect.xy) / (blit.rect.zw - blit.rect.xy);
  // textureSample would need uniform control flow, which the early returns above
  // break; with an explicit level, it doesn't.
  let left = textureSampleLevel(left_image, image_sampler, uv, 0.0);
  let right = textureSampleLevel(right_image, image_sampler, uv, 0.0);
  return vec4<f32>(select(right.rgb, left.rgb, p.x < blit.split), 1.0);
}
