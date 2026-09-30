import { FLOATS_PER_GAUSSIAN, SH_REST_FLOATS } from './gaussians';

// Reads the .ply files that 3D Gaussian Splatting training writes, and converts
// them to this renderer's layout (M1.6). The file stores raw training parameters;
// the renderer wants them activated. Scales are stored as logarithms, opacity as
// a logit, and color as spherical-harmonic coefficients.

/** The degree-0 spherical harmonic, 1 / (2√π). */
export const SH_C0 = 0.28209479177387814;

export interface GaussianScene {
  count: number;
  /** Packed like packGaussians(), with the degree-0 color as `color`. */
  gaussians: Float32Array<ArrayBuffer>;
  /** SH_REST_FLOATS per Gaussian (see gaussians.ts). Zero past the file's degree. */
  shRest: Float32Array<ArrayBuffer>;
  /** The highest spherical-harmonic degree in the file, 0 to 3. */
  shDegree: number;
}

export function parseGaussianPly(buffer: ArrayBuffer): GaussianScene {
  const bytes = new Uint8Array(buffer);
  const marker = new TextEncoder().encode('end_header\n');
  const markerAt = indexOf(bytes, marker, 64 * 1024);
  if (markerAt < 0) throw new Error('No .ply header found.');
  const headerEnd = markerAt + marker.length;
  const header = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  if (!header.startsWith('ply\n')) throw new Error('Not a .ply file.');
  if (!/^format binary_little_endian 1\.0$/m.test(header)) {
    throw new Error('Only binary little-endian .ply files are supported.');
  }

  const elements: string[] = [];
  const properties: string[] = [];
  let count = 0;
  for (const line of header.split('\n')) {
    const [keyword, ...words] = line.trim().split(/\s+/);
    if (keyword === 'element') {
      elements.push(words[0]);
      if (words[0] === 'vertex') count = Number(words[1]);
    } else if (keyword === 'property' && elements.at(-1) === 'vertex') {
      if (words[0] !== 'float') throw new Error(`Property ${words[1]} is ${words[0]}; only float properties are supported.`);
      properties.push(words[1]);
    }
  }
  if (elements[0] !== 'vertex') throw new Error('Expected the .ply to start with its vertex element.');

  const column = (name: string) => {
    const index = properties.indexOf(name);
    if (index < 0) throw new Error(`The .ply has no "${name}" property. Is it a 3D Gaussian Splatting file?`);
    return index;
  };
  const position = ['x', 'y', 'z'].map(column);
  const dc = ['f_dc_0', 'f_dc_1', 'f_dc_2'].map(column);
  const opacity = column('opacity');
  const scale = ['scale_0', 'scale_1', 'scale_2'].map(column);
  const rotation = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].map(column);
  const restCount = properties.filter((name) => name.startsWith('f_rest_')).length;
  const perChannel = restCount / 3;
  const shDegree = Math.round(Math.sqrt(perChannel + 1)) - 1;
  if ((shDegree + 1) ** 2 - 1 !== perChannel || shDegree > 3) {
    throw new Error(`Unexpected number of spherical-harmonic coefficients (${restCount}).`);
  }
  const rest = Array.from({ length: restCount }, (_, k) => column(`f_rest_${k}`));

  const stride = properties.length;
  const byteLength = count * stride * 4;
  if (headerEnd + byteLength > buffer.byteLength) throw new Error('The .ply file is shorter than its header says.');
  // A Float32Array has to start on a multiple of 4 bytes, so copy the body if the
  // header didn't end on one. (Float32Array reads in the machine's byte order,
  // which is little-endian on every platform browsers run on.)
  const data =
    headerEnd % 4 === 0
      ? new Float32Array(buffer, headerEnd, count * stride)
      : new Float32Array(buffer.slice(headerEnd, headerEnd + byteLength));

  const gaussians = new Float32Array(count * FLOATS_PER_GAUSSIAN);
  const shRest = new Float32Array(count * SH_REST_FLOATS);
  for (let i = 0; i < count; i++) {
    const row = i * stride;
    const out = i * FLOATS_PER_GAUSSIAN;
    for (let k = 0; k < 3; k++) {
      gaussians[out + k] = data[row + position[k]];
      gaussians[out + 4 + k] = Math.exp(data[row + scale[k]]);
      gaussians[out + 12 + k] = 0.5 + SH_C0 * data[row + dc[k]];
    }
    gaussians[out + 3] = 1 / (1 + Math.exp(-data[row + opacity]));
    for (let k = 0; k < 4; k++) gaussians[out + 8 + k] = data[row + rotation[k]];
    // The file stores the higher-order coefficients channel by channel (all the
    // red ones, then green, then blue); the shader reads them coefficient by coefficient.
    for (let c = 0; c < perChannel; c++) {
      for (let channel = 0; channel < 3; channel++) {
        shRest[i * SH_REST_FLOATS + c * 3 + channel] = data[row + rest[channel * perChannel + c]];
      }
    }
  }
  return { count, gaussians, shRest, shDegree };
}

function indexOf(haystack: Uint8Array, needle: Uint8Array, searchLimit: number): number {
  const end = Math.min(haystack.length, searchLimit) - needle.length;
  outer: for (let i = 0; i <= end; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) continue outer;
    }
    return i;
  }
  return -1;
}
