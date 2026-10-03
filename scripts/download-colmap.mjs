// Downloads COLMAP, the structure-from-motion tool that turns photos into camera
// poses and a sparse point cloud (M5), into tools/colmap/. It takes the latest
// official Windows build without CUDA (the dev GPU is AMD), from
// github.com/colmap/colmap. Usage: npm run download-colmap
//
// On macOS or Linux, install COLMAP with your package manager instead; the
// reconstruction script also looks for `colmap` on the PATH.
import { existsSync } from 'node:fs';
import { mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { download, extract, readCentralDirectory } from './lib/files.mjs';

const TOOLS = path.join(import.meta.dirname, '..', 'tools');
const TARGET = path.join(TOOLS, 'colmap');

if (process.platform !== 'win32') {
  console.error('This downloads the Windows build. Install COLMAP with your package manager instead.');
  process.exit(1);
}
const release = await (await fetch('https://api.github.com/repos/colmap/colmap/releases/latest')).json();
const asset = release.assets?.find(({ name }) => /windows-nocuda\.zip$/.test(name));
if (!asset) throw new Error(`COLMAP's latest release (${release.tag_name}) has no Windows build without CUDA.`);

await mkdir(TOOLS, { recursive: true });
const zip = path.join(TOOLS, asset.name);
if (!existsSync(zip)) {
  console.log(`Downloading COLMAP ${release.tag_name} (${Math.round(asset.size / 1e6)} MB)…`);
  await download(asset.browser_download_url, zip);
}
// Drop the zip's top-level folder, if everything sits in one.
const names = readCentralDirectory(zip).map(({ name }) => name);
const first = names[0].split('/')[0];
const top = names.every((name) => name.startsWith(`${first}/`)) ? `${first}/` : '';
await rm(TARGET, { recursive: true, force: true });
const count = await extract(zip, top, TARGET);
await rm(zip);
console.log(`Extracted ${count} files to ${TARGET}`);
