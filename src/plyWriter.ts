import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS } from './gaussians';
import type { RawGaussians } from './trainer';

// Saves trained Gaussians as a .ply in the reference's layout (M3), the format
// plyLoader.ts reads and other 3DGS viewers open. Values are stored raw: log
// scales, opacity logits and unnormalized quaternions.

const PROPERTIES = [
  'x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
  ...Array.from({ length: SH_REST_FLOATS }, (_, k) => `f_rest_${k}`),
  'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
];

export function writeGaussianPly({ params, sh }: RawGaussians): Blob {
  const count = params.length / FLOATS_PER_GAUSSIAN;
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${count}`,
    ...PROPERTIES.map((name) => `property float ${name}`),
    'end_header',
    '',
  ].join('\n');
  const stride = PROPERTIES.length;
  const body = new Float32Array(count * stride);
  for (let i = 0; i < count; i++) {
    const at = i * FLOATS_PER_GAUSSIAN;
    const row = i * stride;
    body.set(params.subarray(at, at + 3), row); // the normals stay 0
    body.set(params.subarray(at + 12, at + 15), row + 6);
    // The file stores the coefficients channel by channel; training keeps them coefficient by coefficient.
    for (let channel = 0; channel < 3; channel++) {
      for (let c = 0; c < SH_REST_FLOATS / 3; c++) {
        body[row + 9 + channel * (SH_REST_FLOATS / 3) + c] = sh[i * SH_REST_FLOATS + c * 3 + channel];
      }
    }
    body[row + 9 + SH_REST_FLOATS] = params[at + 3];
    body.set(params.subarray(at + 4, at + 7), row + 10 + SH_REST_FLOATS);
    body.set(params.subarray(at + 8, at + 12), row + 13 + SH_REST_FLOATS);
  }
  // Float32Array holds the machine's byte order, which is little-endian everywhere WebGPU runs.
  return new Blob([header, body]);
}
