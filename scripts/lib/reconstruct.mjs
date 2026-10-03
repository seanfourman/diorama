// Photos in, camera poses out (M5): runs COLMAP on data/<name>/input/ and leaves
// what the training page reads: data/<name>/images/ (undistorted photos) and
// data/<name>/sparse/0/ (cameras.bin, images.bin, points3D.bin). It follows the
// reference's convert.py: features with one shared OPENCV camera, matching, a
// sparse reconstruction, then undistortion to pinhole photos. Structure from
// motion is its own project, so COLMAP does it; we only drive it.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..', '..');
const BUNDLED = path.join(ROOT, 'tools', 'colmap', 'bin', process.platform === 'win32' ? 'colmap.exe' : 'colmap');
const PHOTO = /\.(jpe?g|png)$/i;

/** The photos' longer side after undistortion, like the reference's 1,600-pixel default. */
export const MAX_IMAGE_SIZE = 1600;

/** The pipeline's stages, in order; `resume` starts at one of them. */
export const STAGES = ['features', 'matching', 'mapping', 'undistortion'];

/**
 * Reconstructs data/<name>. `matcher` picks which photo pairs COLMAP compares:
 * 'exhaustive' (every pair; best for up to ~150 photos in any order) or
 * 'sequential' (each photo with its next few; for video frames and walks, and
 * much faster). `resume` skips the stages before it, reusing their results in
 * data/<name>/colmap/. `onLog` gets COLMAP's output lines, `onStage` each
 * stage's name. Resolves to how many photos were placed, out of how many.
 */
export async function reconstruct(
  sceneDir,
  { matcher = 'exhaustive', resume = 'features', onLog = () => {}, onStage = () => {} } = {},
) {
  const input = path.join(sceneDir, 'input');
  const work = path.join(sceneDir, 'colmap');
  const photos = (await readdir(input)).filter((name) => PHOTO.test(name));
  if (photos.length < 3) throw new Error(`Found ${photos.length} photos in ${input}; COLMAP needs at least 3 (50 to 200 is best).`);
  const first = STAGES.indexOf(resume);
  if (first < 0) throw new Error(`No stage "${resume}"; the stages are ${STAGES.join(', ')}.`);
  if (first === 0) await rm(work, { recursive: true, force: true });
  await mkdir(path.join(work, 'sparse'), { recursive: true });
  const database = path.join(work, 'database.db');
  const run = (stage, args) => {
    onStage(stage);
    return runColmap(args, onLog);
  };

  if (first <= 0) await run('Finding features in each photo', [
    'feature_extractor',
    '--database_path', database,
    '--image_path', input,
    '--ImageReader.single_camera', '1',
    '--ImageReader.camera_model', 'OPENCV',
    '--FeatureExtraction.use_gpu', '0',
  ]);
  if (first <= 1) await run(
    `Matching features between photos (${matcher})`,
    matcher === 'sequential'
      ? ['sequential_matcher', '--database_path', database, '--FeatureMatching.use_gpu', '0', '--SequentialMatching.overlap', '20']
      : ['exhaustive_matcher', '--database_path', database, '--FeatureMatching.use_gpu', '0'],
  );
  // The global mapper solves for every camera at once (rotation averaging, then
  // positions), which is much faster than adding them one by one.
  if (first <= 2) await run('Solving for the cameras and the 3D points', [
    'global_mapper',
    '--database_path', database,
    '--image_path', input,
    '--output_path', path.join(work, 'sparse'),
  ]);
  // Several disconnected models can come out, in numbered folders (next to a
  // project.ini); keep the one with the most photos.
  const models = (await readdir(path.join(work, 'sparse'))).filter((model) => existsSync(path.join(work, 'sparse', model, 'images.bin')));
  if (models.length === 0) throw new Error("COLMAP couldn't connect the photos. They need more overlap.");
  const sizes = await Promise.all(models.map(async (model) => (await stat(path.join(work, 'sparse', model, 'images.bin'))).size));
  const best = models[sizes.indexOf(Math.max(...sizes))];
  const quality = await measureQuality(path.join(work, 'sparse', best));
  await rm(path.join(work, 'undistorted'), { recursive: true, force: true });
  await run('Undistorting the photos', [
    'image_undistorter',
    '--image_path', input,
    '--input_path', path.join(work, 'sparse', best),
    '--output_path', path.join(work, 'undistorted'),
    '--output_type', 'COLMAP',
    '--max_image_size', String(MAX_IMAGE_SIZE),
  ]);

  // Into the layout loadDataset reads.
  onStage('Moving the results into place');
  const images = path.join(sceneDir, 'images');
  const sparse = path.join(sceneDir, 'sparse', '0');
  await rm(images, { recursive: true, force: true });
  await rm(path.join(sceneDir, 'sparse'), { recursive: true, force: true });
  await mkdir(path.dirname(sparse), { recursive: true });
  await rename(path.join(work, 'undistorted', 'images'), images);
  await rename(path.join(work, 'undistorted', 'sparse'), sparse);
  const placed = (await readdir(images)).filter((name) => PHOTO.test(name)).length;
  return { placed, total: photos.length, quality, warnings: qualityWarnings(quality, placed, photos.length) };
}

/**
 * COLMAP's statistics for a model (model_analyzer): how many 3D points, how many
 * photos see each point on average (track length), how many points each photo
 * sees, and the mean reprojection error in pixels.
 */
async function measureQuality(model) {
  const lines = [];
  await runColmap(['model_analyzer', '--path', model], (line) => lines.push(cleanLogLine(line)));
  const value = (label) => Number(lines.find((line) => line.startsWith(`${label}:`))?.split(':')[1]);
  return {
    points: value('Points'),
    trackLength: value('Mean track length'),
    observationsPerImage: value('Mean observations per image'),
    reprojectionError: value('Mean reprojection error'),
  };
}

/**
 * Plain-language warnings for a reconstruction that is likely distorted. A low
 * reprojection error alone doesn't prove much: with too little overlap, COLMAP
 * can explain every photo well and still bend the scene as a whole, which shows
 * up as short tracks and few points per photo. The thresholds sit between the
 * "train" photos (301 photos: tracks of 6.1 photos, 2,000 points per photo; the
 * poses match the dataset's to 0.23%) and every 6th of them (51 photos: 4.0 and
 * 600; off by 30% or more).
 */
export function qualityWarnings({ trackLength, observationsPerImage }, placed, total) {
  const warnings = [];
  if (placed < 0.8 * total) {
    warnings.push(`COLMAP could place only ${placed} of ${total} photos; the rest didn't connect to the others.`);
  }
  if (trackLength < 4.5 || observationsPerImage < 1000) {
    warnings.push(
      `The photos overlap too little: each 3D point is seen by ${trackLength.toFixed(1)} photos on average (5 or more is ` +
        `healthy) and each photo sees ${Math.round(observationsPerImage)} points (1,000 or more). The scene may come out ` +
        'bent. More photos, closer together, fix this.',
    );
  }
  return warnings;
}

/** A COLMAP log line without its prefix (severity, date, time, thread, source file). */
export function cleanLogLine(line) {
  return line.replace(/^[IWEF]\d{8} [\d:.]+\s+\d+ [\w.]+:\d+\] /, '');
}

/** COLMAP's executable: the one npm run download-colmap fetched, or `colmap` on the PATH. */
export function colmapPath() {
  return existsSync(BUNDLED) ? BUNDLED : 'colmap';
}

function runColmap(args, onLog) {
  return new Promise((resolve, reject) => {
    const executable = colmapPath();
    const child = spawn(executable, [...args, '--log_target', 'stderr', '--log_color', '0'], {
      env: { ...process.env, PATH: `${path.dirname(executable)}${path.delimiter}${process.env.PATH}` },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const lines = [];
    let partial = '';
    const take = (chunk) => {
      const text = partial + chunk.toString();
      const split = text.split(/\r?\n/);
      partial = split.pop();
      for (const line of split) {
        if (!line.trim()) continue;
        lines.push(line);
        if (lines.length > 50) lines.shift();
        onLog(line);
      }
    };
    child.stdout.on('data', take);
    child.stderr.on('data', take);
    child.on('error', (error) =>
      reject(
        error.code === 'ENOENT'
          ? new Error('COLMAP not found. Run "npm run download-colmap" (Windows) or install it so `colmap` is on the PATH.')
          : error,
      ),
    );
    child.on('close', (code) =>
      code === 0 ? resolve() : reject(new Error(`colmap ${args[0]} failed (exit code ${code}):\n${lines.slice(-10).join('\n')}`)),
    );
  });
}
