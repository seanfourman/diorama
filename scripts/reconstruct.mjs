// Turns a folder of photos into a scene the training page can learn (M5): copies
// them into data/<name>/input/, runs COLMAP (scripts/lib/reconstruct.mjs), and
// leaves data/<name>/images/ and data/<name>/sparse/0/. Then open ?train=<name>.
// Usage: npm run reconstruct -- <name> [--from=<photos folder>] [--sequential] [--resume=<stage>]
//   --from        copy the photos from here first (otherwise data/<name>/input/ must have them)
//   --sequential  match each photo only with its next ones: for video frames and
//                 walks, where the file names follow the path. Much faster.
//   --resume      start at a later stage (matching, mapping or undistortion),
//                 reusing the earlier ones' results, after a failure or to retry.
import { copyFile, mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';
import { cleanLogLine, reconstruct } from './lib/reconstruct.mjs';

const DATA = path.join(import.meta.dirname, '..', 'data');
const [name] = process.argv.slice(2).filter((arg) => !arg.startsWith('--'));
const option = (key) => process.argv.find((arg) => arg.startsWith(`--${key}=`))?.slice(key.length + 3);
if (!name || !/^[\w-]+$/.test(name)) {
  console.error('Usage: npm run reconstruct -- <name> [--from=<photos folder>] [--sequential]   (name: letters, digits, - and _)');
  process.exit(1);
}

const sceneDir = path.join(DATA, name);
const from = option('from');
if (from) {
  const input = path.join(sceneDir, 'input');
  await mkdir(input, { recursive: true });
  const photos = (await readdir(from)).filter((file) => /\.(jpe?g|png)$/i.test(file));
  for (const photo of photos) await copyFile(path.join(from, photo), path.join(input, photo));
  console.log(`Copied ${photos.length} photos into ${input}`);
}

const start = Date.now();
const elapsed = () => `${Math.round((Date.now() - start) / 1000)} s`;
// COLMAP is chatty; show the lines that say how far along it is.
const progress = /Processed file|Matching block|Processing image|Registering image|Global positioning|Rotation averaging|Bundle adjustment|Undistorting image|Elapsed time/i;
const { placed, total, quality, warnings } = await reconstruct(sceneDir, {
  matcher: process.argv.includes('--sequential') ? 'sequential' : 'exhaustive',
  resume: option('resume') ?? 'features',
  onStage: (stage) => console.log(`\n[${elapsed()}] ${stage}…`),
  onLog: (line) => {
    if (progress.test(line)) console.log(`  ${cleanLogLine(line)}`);
  },
});
console.log(`\n[${elapsed()}] Done: COLMAP placed ${placed} of ${total} photos. Open ?train=${name} to train it.`);
console.log(
  `Quality: ${quality.points.toLocaleString('en-US')} points, each seen by ${quality.trackLength.toFixed(1)} photos on average; ` +
    `${Math.round(quality.observationsPerImage)} points per photo; reprojection error ${quality.reprojectionError.toFixed(2)} px.`,
);
for (const warning of warnings) console.log(`Warning: ${warning}`);
