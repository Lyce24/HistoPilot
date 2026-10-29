/**
 * Record docs/assets/demo.gif: a tour of the synthetic BLCA demo as a demo service serves it.
 *
 *   bash serve.sh --demo --port 8795       # in another terminal, or histopilot serve --demo
 *   node web/scripts/record-demo-gif.mjs [--url http://127.0.0.1:8795] [--output FILE] [--keep]
 *
 * It refuses any service that is not a demo service, so only the synthetic demo is recorded.
 * Frames come from Chromium's screencast with their timestamps; ffmpeg turns them into a
 * 10 fps, 960-pixel-wide GIF in two passes (palettegen, then paletteuse). --keep keeps the
 * frames and prints their folder. Needs ffmpeg and a Playwright Chromium headless shell under
 * ~/.cache/ms-playwright, or its path in HISTOPILOT_CHROMIUM.
 */
import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argument = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
};
const url = argument('--url', 'http://127.0.0.1:8795').replace(/\/+$/, '');
const output = resolve(argument('--output', join(web, '../docs/assets/demo.gif')));
const keep = process.argv.includes('--keep');
const VIEWPORT = { width: 1200, height: 750 };
const GIF = { width: 960, fps: 10 };
const delay = (ms) => new Promise((done) => setTimeout(done, ms));

// Only a demo service may be recorded: anything else could show a real study.
const session = await (await fetch(url + '/api/v1/session')).json();
const listing = await (await fetch(url + '/api/v1/projects', { headers: { 'X-HistoPilot-Token': session.token } })).json();
assert.equal(listing.demoOnly, true, url + ' is not a demo service; start one with histopilot serve --demo');
assert.deepEqual(listing.projects.map((project) => project.id), ['blca-demo-v1']);

const frames = await mkdtemp(join(tmpdir(), 'histopilot-demo-gif-'));
const cache = join(homedir(), '.cache/ms-playwright');
let executable = process.env.HISTOPILOT_CHROMIUM;
if (!executable) {
  const candidate = (await readdir(cache).catch(() => [])).filter((name) => name.startsWith('chromium_headless_shell-')).sort().at(-1);
  if (!candidate) throw new Error('Install a Playwright Chromium headless shell or set HISTOPILOT_CHROMIUM to a Chromium executable.');
  executable = join(cache, candidate, 'chrome-headless-shell-linux64/chrome-headless-shell');
}
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--hide-scrollbars', '--remote-debugging-pipe', '--user-data-dir=' + join(frames, 'browser-profile')],
  { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
let stderr = '';
browser.stderr.on('data', (data) => { stderr += data.toString(); });
const requests = new Map();
let nextId = 0, buffer = '', sessionId;
const exceptions = [];
const captured = [];
const writes = [];
const rejectPending = (error) => { for (const { reject } of requests.values()) reject(error); requests.clear(); };
browser.on('error', rejectPending);
browser.on('exit', (code) => rejectPending(new Error('Chromium exited (' + code + '): ' + stderr)));
browser.stdio[3].on('error', rejectPending);
browser.stdio[4].on('error', rejectPending);
browser.stdio[4].on('data', (data) => {
  buffer += data.toString();
  let end;
  while ((end = buffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
    if (message.id) {
      const pending = requests.get(message.id); requests.delete(message.id);
      if (message.error) pending?.reject(new Error(JSON.stringify(message.error))); else pending?.resolve(message.result);
    } else if (message.method === 'Page.screencastFrame') {
      const { data, metadata, sessionId: frame } = message.params;
      const file = join(frames, 'frame-' + String(captured.length).padStart(5, '0') + '.png');
      captured.push({ file, at: metadata.timestamp ?? Date.now() / 1000 });
      writes.push(writeFile(file, Buffer.from(data, 'base64')));
      cdp('Page.screencastFrameAck', { sessionId: frame }).catch(() => {});
    } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
  }
});
function cdp(method, params = {}, session = sessionId) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    requests.set(id, { resolve, reject });
    browser.stdio[3].write(JSON.stringify({ id, method, params, ...(session ? { sessionId: session } : {}) }) + '\0');
  });
}
async function evaluate(expression) {
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function waitFor(expression, description = expression) {
  for (let attempt = 0; attempt < 150; attempt += 1) {
    try {
      if (await evaluate('Boolean(' + expression + ')')) return;
    } catch (error) {
      if (!/navigated|execution context|Cannot find context/i.test(String(error))) throw error;
    }
    await delay(100);
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}

// A visible pointer and a ring on each click: headless Chromium draws no cursor of its own.
const POINTER = `(() => {
  const install = () => {
    if (document.getElementById('demo-pointer')) return;
    const style = document.createElement('style');
    style.textContent = '#demo-pointer{position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;transform:translate(-60px,-60px)}'
      + '.demo-click{position:fixed;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;border:3px solid #0f766e;z-index:2147483646;pointer-events:none;animation:demo-click .5s ease-out forwards}'
      + '@keyframes demo-click{from{transform:scale(.4);opacity:.95}to{transform:scale(1.5);opacity:0}}';
    document.head.appendChild(style);
    const pointer = document.createElement('div');
    pointer.id = 'demo-pointer';
    pointer.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M3 2v17l4.6-4.4 3.1 7 3.2-1.4-3.1-6.9H17z" fill="#1f2328" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>';
    document.body.appendChild(pointer);
    addEventListener('mousemove', (event) => { pointer.style.transform = 'translate(' + (event.clientX - 3) + 'px,' + (event.clientY - 2) + 'px)'; }, true);
    addEventListener('mousedown', (event) => {
      const ring = document.createElement('div');
      ring.className = 'demo-click';
      ring.style.left = event.clientX + 'px';
      ring.style.top = event.clientY + 'px';
      document.body.appendChild(ring);
      setTimeout(() => ring.remove(), 600);
    }, true);
  };
  if (document.body) install(); else addEventListener('DOMContentLoaded', install);
})()`;
// It starts below the start page's cards, so neither shows its hover state at first.
const pointer = { x: VIEWPORT.width * 0.5, y: VIEWPORT.height * 0.86 };
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
async function moveTo(x, y, ms = 550) {
  const from = { ...pointer };
  const steps = Math.max(6, Math.round(ms / 20));
  for (let step = 1; step <= steps; step += 1) {
    const t = ease(step / steps);
    pointer.x = from.x + (x - from.x) * t; pointer.y = from.y + (y - from.y) * t;
    await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', x: pointer.x, y: pointer.y });
    await delay(ms / steps);
  }
}
/** Bring an element into view smoothly, then move to it and click it, as a person would. */
async function click(target, description = target) {
  await waitFor(target, description);
  const top = await evaluate(`(${target}).getBoundingClientRect().top`);
  const bottom = await evaluate(`(${target}).getBoundingClientRect().bottom`);
  if (top < 90 || bottom > VIEWPORT.height - 30) await scrollBy(top - VIEWPORT.height * 0.35);
  const box = await evaluate(`(() => { const box = (${target}).getBoundingClientRect(); return { x: box.x + Math.min(box.width / 2, 90), y: box.y + box.height / 2 }; })()`);
  await moveTo(box.x, box.y);
  await delay(180);
  await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', x: box.x, y: box.y, button: 'left', clickCount: 1 });
  await delay(70);
  await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', x: box.x, y: box.y, button: 'left', clickCount: 1 });
}
async function scrollBy(distance, ms = 900) {
  await evaluate(`new Promise((done) => {
    const start = scrollY, target = Math.max(0, Math.min(start + ${distance}, document.documentElement.scrollHeight - innerHeight));
    const began = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - began) / ${ms});
      scrollTo(0, start + (target - start) * (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2));
      if (t < 1) requestAnimationFrame(frame); else done();
    };
    requestAnimationFrame(frame);
  })`);
}
async function scrollToElement(selector, offset = 90) {
  await waitFor(`document.querySelector(${JSON.stringify(selector)})`);
  await scrollBy(await evaluate(`document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect().top`) - offset);
}
const byText = (selector, text) => `[...document.querySelectorAll(${JSON.stringify(selector)})].find((element) => element.textContent.trim() === ${JSON.stringify(text)})`;
async function openModule(module) {
  await click(`document.querySelector('.sidebar a[href="#${module}"]')`, 'module ' + module);
  await waitFor('document.querySelector(".stage-library .blca-demo-record-link")', 'library of ' + module);
}
async function openRecord(record, name) {
  await click(`document.querySelector('[data-demo-record="${record}"]')`, 'record ' + record);
  await waitFor(`document.querySelector('.blca-demo-record-heading h2')?.textContent === ${JSON.stringify(name)}`);
}
async function openStep(title) {
  await click(`[...document.querySelectorAll('.stage-steps button')].find((button) => button.querySelector('strong')?.textContent === ${JSON.stringify(title)})`, 'step ' + title);
  await waitFor(`document.querySelector('.blca-demo-step > h2')?.textContent === ${JSON.stringify(title)}`);
}

let started, finished = false;
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });
  // Page transitions in a few frames rather than many: smaller and steadier GIF.
  await cdp('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }, { name: 'prefers-color-scheme', value: 'light' }] });
  await cdp('Page.addScriptToEvaluateOnNewDocument', { source: POINTER });
  await cdp('Page.navigate', { url: url + '/' });
  await waitFor(byText('.start-action-copy strong', 'Open the BLCA demo'), 'the demo service start page');
  await evaluate('document.fonts.ready.then(() => true)');
  await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', ...pointer });
  await delay(400);
  started = Date.now() / 1000;
  await cdp('Page.startScreencast', { format: 'png', everyNthFrame: 1, maxWidth: VIEWPORT.width, maxHeight: VIEWPORT.height });

  // Start page: a demo service offers the demo alone.
  await delay(2400);
  await click(`${byText('.start-action-copy strong', 'Open the BLCA demo')}.closest('button')`, 'the demo card');
  await waitFor('document.querySelector(".blca-demo-hero h1")?.textContent === "BLCA demo"');

  // Overview: what the demo is, then the pipeline it walks through.
  await delay(2600);
  await scrollToElement('.blca-demo-pipeline', 150);
  await delay(2000);

  await openModule('dataset');
  await delay(700);
  await openRecord('blca-dataset', 'BLCA synthetic slides');
  await delay(2600);

  await openModule('features');
  await delay(600);
  await openRecord('blca-features', 'UNI v1 feature bundle');
  await delay(700);
  await openStep('Validation');
  await delay(2400);

  await openModule('cohort');
  await delay(600);
  await openRecord('blca-protocol', 'Grade 1/3 training · Grade 2 testing');
  await delay(700);
  await openStep('Training and testing sets');
  await delay(2600);

  // Experiments: the frozen design, then its run histories and training curves.
  await openModule('experiments');
  await delay(600);
  await openRecord('blca-baseline-v2', 'BLCA baseline v2');
  await delay(600);
  await openStep('Training folds');
  await delay(2200);
  await openStep('Runs');
  await waitFor('document.querySelector(".experiment-history-charts svg")');
  await delay(1600);
  await scrollToElement('.experiment-run-detail', 100);
  await delay(3000);
  await scrollToElement('.run-resource-usage', 100);
  await delay(2200);

  // Apply models: a scored run with its ROC curve, then its clinical utility.
  await openModule('apply');
  await delay(700);
  await openRecord('blca-evaluation', 'Grade 2 scored run');
  await delay(600);
  await openStep('Metrics');
  await delay(1600);
  await scrollToElement('.blca-demo-chart', 110);
  await delay(2800);
  await click(byText('.blca-demo-record-heading button', '← Back to records'), 'back to records');
  await waitFor('document.querySelector(".stage-library .blca-demo-record-link")');
  await delay(500);
  await openRecord('blca-clinical-utility', 'Synthetic threshold review');
  await delay(600);
  await openStep('Thresholds');
  await delay(1300);
  await scrollToElement('.blca-demo-chart', 110);
  await delay(2700);

  await openModule('interpretation');
  await delay(600);
  await openRecord('blca-interpretation', 'Attention interpretation guide');
  await delay(700);
  await openStep('Schematic');
  await delay(1400);
  await scrollToElement('.blca-demo-step', 100);
  await delay(2800);

  await cdp('Page.stopScreencast');
  const stopped = Date.now() / 1000;
  await Promise.all(writes);
  assert.deepEqual(exceptions, [], 'The page raised errors while recording');
  assert.ok(captured.length > 50, 'Too few frames: ' + captured.length);

  // Each frame lasts until the next one; the last until the recording stopped.
  const lines = captured.flatMap((frame, index) => {
    const until = index + 1 < captured.length ? captured[index + 1].at : stopped;
    return ["file '" + frame.file + "'", 'duration ' + Math.max(0.001, until - frame.at).toFixed(4)];
  });
  lines.push("file '" + captured.at(-1).file + "'");
  const list = join(frames, 'frames.txt');
  await writeFile(list, lines.join('\n') + '\n');
  const palette = join(frames, 'palette.png');
  const scale = `fps=${GIF.fps},scale=${GIF.width}:-1:flags=lanczos`;
  await run('ffmpeg', ['-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-vf', scale + ',palettegen=stats_mode=diff', palette]);
  await mkdir(dirname(output), { recursive: true });
  await run('ffmpeg', ['-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-i', palette, '-lavfi',
    scale + '[frames];[frames][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle', '-loop', '0', output]);
  const { stdout } = await run('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames', '-show_entries', 'format=duration', '-of', 'json', output]);
  const probe = JSON.parse(stdout);
  const size = (await stat(output)).size;
  console.log(JSON.stringify({
    output, url, recordedSeconds: Number((stopped - started).toFixed(1)), screencastFrames: captured.length,
    gif: { width: probe.streams[0].width, height: probe.streams[0].height, frames: Number(probe.streams[0].nb_read_frames), seconds: Number(Number(probe.format.duration).toFixed(1)), megabytes: Number((size / 1e6).toFixed(2)) },
    frames: keep ? frames : undefined,
  }, null, 2));
  if (size > 8e6) console.warn('The GIF is larger than 8 MB; GitHub may render it slowly.');
  finished = true;
} catch (error) {
  console.error('Frames kept in', frames);
  throw error;
} finally {
  browser.kill();
  if (finished && !keep) await rm(frames, { recursive: true, force: true });
}
