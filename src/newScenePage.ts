// The new-scene page (?new, M5): drop photos or a video of a place, and get a
// scene to train. The browser prepares the images (photos shrunk to at most
// 1,600 pixels with their EXIF rotation applied, or a video cut into frames),
// uploads them to the dev server, which runs COLMAP (scripts/lib/sceneApi.mjs),
// then opens the training page. Needs `npm run dev` and COLMAP
// (`npm run download-colmap`).

const MAX_SIZE = 1600;
const MAX_FRAMES = 200;
const FRAMES_PER_SECOND = 2;
/** Up to this many photos in no particular order, every pair is matched; past it, or for video, only neighbors. */
const EXHAUSTIVE_LIMIT = 150;

interface SceneInfo {
  name: string;
  photos: number;
  trainable: boolean;
  trained: boolean;
}

interface Status {
  state: 'idle' | 'running' | 'done' | 'error';
  stage?: string;
  lines?: string[];
  placed?: number;
  total?: number;
  error?: string;
  seconds?: number;
  quality?: { points: number; trackLength: number; observationsPerImage: number; reprojectionError: number };
  warnings?: string[];
}

export async function runNewScenePage(root: HTMLElement): Promise<void> {
  root.hidden = false;
  root.innerHTML = `
    <h1>New scene</h1>
    <p>Drop 50 to 200 photos of a place, or a short video walking through it. Move around rather than turning on
      the spot, overlap each photo heavily with the last, and keep the lighting steady.</p>
    <label class="drop" id="drop">
      <input type="file" id="files" multiple accept="image/*,video/*" hidden>
      <span id="drop-text">Drop photos or a video here, or click to choose.</span>
    </label>
    <p><label>Name <input id="name" pattern="[A-Za-z0-9_-]+" maxlength="64"></label>
      <button id="start" disabled>Make the scene</button></p>
    <pre id="progress"></pre>
    <h2>Scenes in data/</h2>
    <ul id="scenes"><li>Loading…</li></ul>`;
  const $ = <T extends HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;
  const drop = $('drop');
  const input = $<HTMLInputElement>('files');
  const nameInput = $<HTMLInputElement>('name');
  const start = $<HTMLButtonElement>('start');
  const progress = $<HTMLPreElement>('progress');
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  nameInput.value = `scene-${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  void listScenes($('scenes'));

  let files: File[] = [];
  const choose = (chosen: File[]) => {
    const video = chosen.find((file) => file.type.startsWith('video/'));
    files = video ? [video] : chosen.filter((file) => file.type.startsWith('image/'));
    $('drop-text').textContent = video
      ? `Video: ${video.name}. It will be cut into up to ${MAX_FRAMES} frames, ${FRAMES_PER_SECOND} a second.`
      : `${files.length} photos.${files.length > 300 ? ' That many will be slow; 50 to 200 is plenty.' : ''}`;
    start.disabled = files.length === 0 || (!video && files.length < 3);
  };
  input.addEventListener('change', () => choose([...(input.files ?? [])]));
  drop.addEventListener('dragover', (event) => {
    event.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    choose([...(event.dataTransfer?.files ?? [])]);
  });

  start.addEventListener('click', () => {
    const name = nameInput.value.trim();
    if (!/^[\w-]{1,64}$/.test(name)) {
      progress.textContent = 'The name can only have letters, digits, - and _.';
      return;
    }
    start.disabled = true;
    makeScene(name, files, (text) => (progress.textContent = text)).catch((error: unknown) => {
      progress.textContent += `\n\n✗ ${error instanceof Error ? error.message : String(error)}`;
      start.disabled = false;
    });
  });
  document.body.dataset.status = 'done';
}

async function makeScene(name: string, files: File[], show: (text: string) => void): Promise<void> {
  // Never mix new photos into an existing scene. Uploads left from an attempt
  // that never got as far as COLMAP's results are cleared and replaced.
  const scenes = await fetch('/api/scenes').then((response) => (response.ok ? (response.json() as Promise<SceneInfo[]>) : []));
  const existing = scenes.find((scene) => scene.name === name);
  if (existing?.trainable || existing?.trained) throw new Error(`A scene called "${name}" already exists in data/; pick another name.`);
  if (existing) await fetch(`/api/scenes/${name}/input`, { method: 'DELETE' });
  const isVideo = files.length === 1 && files[0].type.startsWith('video/');
  // 1. The images, as JPEGs at most MAX_SIZE across, named in order so
  //    sequential matching pairs neighbors.
  const images: { file: string; blob: Blob }[] = [];
  if (isVideo) {
    for await (const [k, count, blob] of videoFrames(files[0])) {
      images.push({ file: `frame_${String(k + 1).padStart(4, '0')}.jpg`, blob });
      show(`Cutting the video into frames… ${k + 1} of ${count}`);
    }
  } else {
    const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    for (const [k, file] of sorted.entries()) {
      show(`Preparing photos… ${k + 1} of ${sorted.length}`);
      images.push({ file: `photo_${String(k + 1).padStart(4, '0')}.jpg`, blob: await shrink(await createImageBitmap(file)) });
    }
  }

  // 2. Upload.
  for (const [k, { file, blob }] of images.entries()) {
    show(`Uploading… ${k + 1} of ${images.length}`);
    const response = await fetch(`/api/scenes/${name}/input/${file}`, { method: 'PUT', body: blob });
    if (!response.ok) throw new Error(await apiError(response, 'The upload failed. Is this the dev server (npm run dev)?'));
  }

  // 3. COLMAP, polled once a second.
  const matcher = isVideo || images.length > EXHAUSTIVE_LIMIT ? 'sequential' : 'exhaustive';
  const response = await fetch(`/api/scenes/${name}/reconstruct?matcher=${matcher}`, { method: 'POST' });
  if (!response.ok) throw new Error(await apiError(response, "Couldn't start COLMAP."));
  const began = performance.now();
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const status = (await (await fetch(`/api/scenes/${name}/status`)).json()) as Status;
    const seconds = Math.round((performance.now() - began) / 1000);
    if (status.state === 'error') throw new Error(status.error);
    if (status.state === 'done') {
      const quality = status.quality!;
      // Carry &steps= over, for a quick try.
      const steps = new URLSearchParams(location.search).get('steps');
      const training = `?train=${encodeURIComponent(name)}${steps ? `&steps=${encodeURIComponent(steps)}` : ''}`;
      const summary =
        `COLMAP placed ${status.placed} of ${status.total} photos in ${status.seconds} s: ` +
        `${quality.points.toLocaleString('en-US')} points, each seen by ${quality.trackLength.toFixed(1)} photos on average.`;
      // A weak reconstruction trains into a bent scene, so say so and let the person decide.
      if (status.warnings?.length) {
        show(`${summary}\n\n⚠ ${status.warnings.join('\n⚠ ')}`);
        const link = document.createElement('a');
        link.href = training;
        link.textContent = 'Train it anyway';
        document.querySelector('#progress')!.after(link);
        return;
      }
      show(`${summary}\n\nOpening the training page…`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      location.search = training;
      return;
    }
    show(`COLMAP, ${seconds} s: ${status.stage}…\n\n${(status.lines ?? []).join('\n')}`);
  }
}

// Frames spread evenly through the video, FRAMES_PER_SECOND of them up to
// MAX_FRAMES, by seeking a video element and drawing it.
async function* videoFrames(file: File): AsyncGenerator<[number, number, Blob]> {
  const video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';
  video.src = URL.createObjectURL(file);
  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error(`This browser can't decode ${file.name}. Try an MP4 (H.264).`));
  });
  const count = Math.min(MAX_FRAMES, Math.max(3, Math.floor(video.duration * FRAMES_PER_SECOND)));
  for (let k = 0; k < count; k++) {
    video.currentTime = ((k + 0.5) / count) * video.duration;
    await new Promise((resolve) => (video.onseeked = resolve));
    yield [k, count, await shrink(video, video.videoWidth, video.videoHeight)];
  }
  URL.revokeObjectURL(video.src);
}

// A JPEG of `source`, scaled down to at most MAX_SIZE across.
async function shrink(source: CanvasImageSource & { width?: number; height?: number }, width?: number, height?: number): Promise<Blob> {
  const [w, h] = [width ?? (source.width as number), height ?? (source.height as number)];
  const scale = Math.min(1, MAX_SIZE / Math.max(w, h));
  const canvas = new OffscreenCanvas(Math.round(w * scale), Math.round(h * scale));
  canvas.getContext('2d')!.drawImage(source, 0, 0, canvas.width, canvas.height);
  if (source instanceof ImageBitmap) source.close();
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
}

async function listScenes(list: HTMLElement): Promise<void> {
  try {
    const response = await fetch('/api/scenes');
    if (!response.ok) throw new Error();
    const scenes = (await response.json()) as SceneInfo[];
    list.replaceChildren(
      ...scenes.map(({ name, photos, trainable, trained }) => {
        const item = document.createElement('li');
        const link = (label: string, query: string) => `<a href="?${query}=${encodeURIComponent(name)}">${label}</a>`;
        item.innerHTML =
          `<b>${name}</b>: ${photos ? `${photos} photos uploaded` : 'no uploads'}` +
          `${trainable ? ` · ${link('train it', 'train')}` : ''}${trained ? ` · ${link('view it', 'scene')}` : ''}`;
        return item;
      }),
    );
    if (scenes.length === 0) list.innerHTML = '<li>None yet.</li>';
  } catch {
    list.innerHTML = '<li>The scene list needs the dev server (npm run dev).</li>';
  }
}

async function apiError(response: Response, fallback: string): Promise<string> {
  try {
    return ((await response.json()) as { error?: string }).error ?? fallback;
  } catch {
    return fallback;
  }
}
