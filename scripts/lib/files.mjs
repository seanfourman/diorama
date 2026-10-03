// Downloading and unzipping, for the download scripts. No dependencies: zips are
// read straight from their central directory and inflated with node:zlib.
import { closeSync, createWriteStream, openSync, readSync, statSync } from 'node:fs';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { once } from 'node:events';
import { inflateRawSync } from 'node:zlib';
import path from 'node:path';

/** Downloads `url` to `target`, through a .part file, printing progress every 10%. */
export async function download(url, target) {
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
      console.log(`${path.basename(target)}: ${Math.round((100 * received) / total)}%`);
      nextReport += 0.1;
    }
  }
  out.end();
  await once(out, 'finish');
  await rename(`${target}.part`, target);
}

/**
 * The zip's table of contents: it sits at the end of the file, found through the
 * end-of-central-directory record in the last few KB.
 */
export function readCentralDirectory(zipPath) {
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

/** One entry's contents. `file` is an open file descriptor of the zip. */
export function readEntry(file, { method, compressedSize, localHeader }) {
  const header = Buffer.alloc(30);
  readSync(file, header, 0, 30, localHeader);
  const start = localHeader + 30 + header.readUInt16LE(26) + header.readUInt16LE(28);
  const data = Buffer.alloc(compressedSize);
  readSync(file, data, 0, compressedSize, start);
  if (method === 0) return data;
  if (method === 8) return inflateRawSync(data);
  throw new Error(`Unsupported zip compression method ${method}.`);
}

/**
 * Extracts the zip's entries under `prefix` into `targetDir`, without the prefix.
 * Entries that would land outside `targetDir` are refused. Returns how many files.
 */
export async function extract(zipPath, prefix, targetDir) {
  const root = path.resolve(targetDir);
  const file = openSync(zipPath, 'r');
  let extracted = 0;
  try {
    for (const entry of readCentralDirectory(zipPath)) {
      if (!entry.name.startsWith(prefix) || entry.name.endsWith('/')) continue;
      const target = path.resolve(root, entry.name.slice(prefix.length));
      if (!target.startsWith(root + path.sep)) throw new Error(`The zip entry ${entry.name} points outside ${root}.`);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, readEntry(file, entry));
      extracted++;
    }
  } finally {
    closeSync(file);
  }
  return extracted;
}
