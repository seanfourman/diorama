// Downloads the Tanks and Temples / Deep Blending photo sets that the original
// 3DGS paper trained on (tandt_db.zip, 683 MB, mirrored on Hugging Face) and
// extracts one scene's photos and COLMAP reconstruction into data/<scene>/:
// images/*.jpg and sparse/0/{cameras,images,points3D}.bin.
// Usage: npm run download-photos [-- <scene>]   (train, truck, drjohnson or playroom)
//
// The zip is kept in data/ so other scenes can be extracted without downloading
// it again. It's covered by the original datasets' licenses.
import { createWriteStream, existsSync, closeSync, openSync, readSync, statSync } from 'node:fs';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import { inflateRawSync } from 'node:zlib';
import path from 'node:path';

const URL = 'https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip';
const DATA = path.join(import.meta.dirname, '..', 'data');
const ZIP = path.join(DATA, 'tandt_db.zip');
const scene = process.argv[2] ?? 'train';

await mkdir(DATA, { recursive: true });
if (!existsSync(ZIP)) await download(URL, ZIP);

const entries = readCentralDirectory(ZIP);
// Scenes sit at tandt/<scene>/ or db/<scene>/ inside the zip.
const prefix = entries.map(({ name }) => name).find((name) => new RegExp(`^(tandt|db)/${scene}/$`).test(name));
if (!prefix) {
  const scenes = new Set(entries.map(({ name }) => name.split('/')[1]).filter(Boolean));
  console.error(`No scene "${scene}" in the zip. It has: ${[...scenes].join(', ')}.`);
  process.exit(1);
}
const file = openSync(ZIP, 'r');
let extracted = 0;
for (const entry of entries) {
  if (!entry.name.startsWith(prefix) || entry.name.endsWith('/')) continue;
  const target = path.join(DATA, scene, entry.name.slice(prefix.length));
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, readEntry(file, entry));
  extracted++;
}
closeSync(file);
console.log(`Extracted ${extracted} files to ${path.join(DATA, scene)}`);

async function download(url, target) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  const total = Number(response.headers.get('content-length')) || 0;
  const out = createWriteStream(`${target}.part`);
  let received = 0;
  let nextReport = 0.1;
  for await (const chunk of response.body) {
    if (!out.write(chunk)) await once(out, 'drain');
    received += chunk.length;
    if (total && received / total >= nextReport) {
      console.log(`tandt_db.zip: ${Math.round((100 * received) / total)}%`);
      nextReport += 0.1;
    }
  }
  out.end();
  await once(out, 'finish');
  await rename(`${target}.part`, target);
}

// The zip's table of contents: it sits at the end of the file, found through the
// end-of-central-directory record in the last few KB.
function readCentralDirectory(zipPath) {
  const file = openSync(zipPath, 'r');
  const read = (position, length) => {
    const buffer = Buffer.alloc(length);
    readSync(file, buffer, 0, length, position);
    return buffer;
  };
  const { size } = statSync(zipPath);
  const tailLength = Math.min(size, 65557);
  const tail = read(size - tailLength, tailLength);
  const end = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  if (end < 0) throw new Error('Not a zip file.');
  const count = tail.readUInt16LE(end + 10);
  const directory = read(tail.readUInt32LE(end + 16), tail.readUInt32LE(end + 12));
  const entries = [];
  for (let offset = 0, k = 0; k < count; k++) {
    if (directory.readUInt32LE(offset) !== 0x02014b50) throw new Error('Damaged zip directory.');
    const nameLength = directory.readUInt16LE(offset + 28);
    entries.push({
      method: directory.readUInt16LE(offset + 10),
      compressedSize: directory.readUInt32LE(offset + 20),
      localHeader: directory.readUInt32LE(offset + 42),
      name: directory.toString('utf8', offset + 46, offset + 46 + nameLength),
    });
    offset += 46 + nameLength + directory.readUInt16LE(offset + 30) + directory.readUInt16LE(offset + 32);
  }
  closeSync(file);
  return entries;
}

function readEntry(file, { method, compressedSize, localHeader }) {
  const header = Buffer.alloc(30);
  readSync(file, header, 0, 30, localHeader);
  const start = localHeader + 30 + header.readUInt16LE(26) + header.readUInt16LE(28);
  const data = Buffer.alloc(compressedSize);
  readSync(file, data, 0, compressedSize, start);
  if (method === 0) return data;
  if (method === 8) return inflateRawSync(data);
  throw new Error(`Unsupported zip compression method ${method}.`);
}
