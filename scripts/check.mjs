// Opens the built app in headless Edge or Chrome on this machine's GPU with
// ?check, waits for the page's GPU checks to finish, prints the on-page log and
// console output, and exits with code 1 if anything failed. Run it with
// `npm run check`, which builds first. data/ is served too, for the scenes.
// Options: --headed shows the browser window; --scene=<name> shows that scene
// after the checks; --screenshot=<file> saves a PNG of the page once everything
// has loaded; BROWSER=<path> picks the browser.
import { spawn, execSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const DIST = path.join(ROOT, 'dist');
const HTTP_PORT = 4179;
const DEBUG_PORT = 9339;
const TIMEOUT_MS = 90_000; // loading and checking a real scene takes a while
const CONTENT_TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const BROWSERS = [
  process.env.BROWSER,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const browserPath = BROWSERS.find((p) => p && existsSync(p));
if (!browserPath) {
  console.error('No Edge or Chrome found. Set BROWSER to the browser executable.');
  process.exit(1);
}

const server = createServer(async (req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);
  const base = urlPath.startsWith('/data/') ? ROOT : DIST;
  const file = path.join(base, urlPath === '/' ? '/index.html' : urlPath);
  try {
    // Never serve anything outside dist/ and data/, whatever "../" the path holds.
    if (!file.startsWith(base === ROOT ? path.join(ROOT, 'data') : DIST)) throw new Error('outside');
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': CONTENT_TYPES[path.extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(HTTP_PORT, '127.0.0.1', resolve));

const profile = mkdtempSync(path.join(tmpdir(), 'diorama-check-'));
const browser = spawn(browserPath, [
  `--remote-debugging-port=${DEBUG_PORT}`,
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--window-size=1280,800',
  ...(process.argv.includes('--headed') ? [] : ['--headless=new']),
  'about:blank',
], { stdio: 'ignore' });

let socket;
let passed = false;
try {
  const { send, consoleLines } = await attachToPage();
  await send('Runtime.enable');
  await send('Log.enable');
  const scene = process.argv.find((arg) => arg.startsWith('--scene='))?.slice('--scene='.length);
  const query = scene ? `?check&scene=${encodeURIComponent(scene)}` : '?check';
  const navigation = await send('Page.navigate', { url: `http://127.0.0.1:${HTTP_PORT}/${query}` });
  const navigationError = navigation.error?.message ?? navigation.result?.errorText;
  if (navigationError) console.log(`Couldn't open the page: ${navigationError}`);

  // main.ts sets <body data-status> to "done" or "error" once its checks finish.
  let status = '';
  const deadline = Date.now() + TIMEOUT_MS;
  while (status !== 'done' && status !== 'error' && Date.now() < deadline) {
    await sleep(250);
    status = await evaluate(send, 'document.body?.dataset.status ?? ""');
  }
  const log = await evaluate(send, "document.querySelector('#log')?.textContent ?? ''");

  if (status !== 'done') {
    const where = await evaluate(send, "`${location.href}: ${document.querySelector('#summary')?.textContent ?? ''}`");
    console.log(`Status line at the end: ${where}`);
  }
  console.log(log || '(the page log is empty)');
  if (consoleLines.length) console.log('\nConsole:\n' + consoleLines.join('\n'));

  const screenshotFile = process.argv.find((arg) => arg.startsWith('--screenshot='))?.slice('--screenshot='.length);
  if (screenshotFile) {
    await sleep(1000); // let a few frames render first
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    await writeFile(screenshotFile, Buffer.from(shot.result.data, 'base64'));
    console.log(`\nScreenshot saved to ${screenshotFile}`);
  }

  const problems = [];
  if (status !== 'done') {
    problems.push(status === 'error' ? 'the page reported an error' : `no result after ${TIMEOUT_MS / 1000}s`);
  }
  if (/\bFAIL\b/.test(log)) problems.push('a check failed');
  if (consoleLines.some((line) => line.startsWith('[error]'))) problems.push('the console has errors');
  passed = problems.length === 0;
  console.log(passed ? '\nAll checks passed.' : `\nFailed: ${problems.join(', ')}.`);
} finally {
  socket?.close();
  if (process.platform === 'win32') {
    try { execSync(`taskkill /PID ${browser.pid} /T /F`, { stdio: 'ignore' }); } catch {}
  } else {
    browser.kill('SIGKILL');
  }
  server.close();
  await sleep(1000);
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {}
}
process.exit(passed ? 0 : 1);

// Connects to the browser's DevTools protocol and returns a request function plus
// the console output collected so far.
async function attachToPage() {
  let target;
  for (let i = 0; i < 75 && !target; i++) {
    await sleep(200);
    try {
      const targets = await (await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`)).json();
      target = targets.find((t) => t.type === 'page');
    } catch {
      // The browser isn't listening yet.
    }
  }
  if (!target) throw new Error('The browser never exposed a page to attach to.');

  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });

  let nextId = 1;
  const pending = new Map();
  const consoleLines = [];
  socket.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    } else if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
      consoleLines.push(`[${msg.params.type}] ${text}`);
    } else if (msg.method === 'Runtime.exceptionThrown') {
      const details = msg.params.exceptionDetails;
      consoleLines.push(`[error] ${details.exception?.description ?? details.text}`);
    } else if (msg.method === 'Log.entryAdded') {
      const { level, text, url } = msg.params.entry;
      consoleLines.push(`[${level}] ${text}${url ? ` (${url})` : ''}`);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      socket.send(JSON.stringify({ id, method, params }));
    });
  return { send, consoleLines };
}

async function evaluate(send, expression) {
  const response = await send('Runtime.evaluate', { expression, returnByValue: true });
  return response.result?.result?.value ?? '';
}
