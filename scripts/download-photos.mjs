// Downloads the Tanks and Temples / Deep Blending photo sets that the original
// 3DGS paper trained on (tandt_db.zip, 683 MB, mirrored on Hugging Face) and
// extracts one scene's photos and COLMAP reconstruction into data/<scene>/:
// images/*.jpg and sparse/0/{cameras,images,points3D}.bin.
// Usage: npm run download-photos [-- <scene>]   (train, truck, drjohnson or playroom)
//
// The zip is kept in data/ so other scenes can be extracted without downloading
// it again. It's covered by the original datasets' licenses.
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { download, extract, readCentralDirectory } from './lib/files.mjs';

const URL = 'https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip';
const DATA = path.join(import.meta.dirname, '..', 'data');
const ZIP = path.join(DATA, 'tandt_db.zip');
const scene = process.argv[2] ?? 'train';

await mkdir(DATA, { recursive: true });
if (!existsSync(ZIP)) await download(URL, ZIP);

// Scenes sit at tandt/<scene>/ or db/<scene>/ inside the zip.
const names = readCentralDirectory(ZIP).map(({ name }) => name);
const prefix = names.find((name) => new RegExp(`^(tandt|db)/${scene}/$`).test(name));
if (!prefix) {
  const scenes = new Set(names.map((name) => name.split('/')[1]).filter(Boolean));
  console.error(`No scene "${scene}" in the zip. It has: ${[...scenes].join(', ')}.`);
  process.exit(1);
}
const extracted = await extract(ZIP, prefix, path.join(DATA, scene));
console.log(`Extracted ${extracted} files to ${path.join(DATA, scene)}`);
