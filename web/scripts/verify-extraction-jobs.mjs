/** Run with node web/scripts/verify-extraction-jobs.mjs. Uses file:// and local Chromium; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-extraction-jobs-'));
const artifacts = output;
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalFeatures from ${source('pages/LocalFeatures.tsx')};
import JobTray from ${source('components/JobTray.tsx')};
import { scientific } from ${source('api/scientific.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { trident } from ${source('api/trident.ts')};
import { taskCenter } from ${source('api/taskCenter.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const state = window.workflow = { calls: [], errors: [], extractions: [] };
window.fetch = async (...args) => { const message = 'Unexpected request: ' + args[0]; state.errors.push(message); throw new Error(message); };
function extraction(id, model, folder, completed) {
  return { id, state: 'running', createdAt: '2026-09-25T10:00:00Z', updatedAt: '2026-09-25T10:05:00Z',
    spec: { datasetId: null, slideRoot: '/slides/' + folder, outputPath: '/features/' + folder, options: { task: 'all', patch_encoder: model, mag: 20, patch_size: 256 } },
    outputPath: '/features/' + folder, logPath: '/logs/' + folder, sessionName: 'extract-' + folder, logs: 'Processing ' + folder,
    progress: { stage: 'patch_features', stages: [
      { id: 'preparing', label: 'Prepare', status: 'complete' }, { id: 'segmentation', label: 'Segment', status: 'complete' },
      { id: 'coordinates', label: 'Coordinates', status: 'complete' }, { id: 'patch_features', label: 'Features', status: 'active' },
      { id: 'validation', label: 'Validate', status: 'pending' } ],
      label: 'Extracting patch features', detail: 'Encoding slides', completed, total: 100, unit: 'slides', percent: completed,
      currentSlide: folder + '.svs', elapsedSeconds: 300, stageElapsedSeconds: 120, etaSeconds: 600,
      ratePerSecond: 0.2, scope: 'stage', warnings: [] } };
}
state.extractions = [extraction('extract/first', 'uni_v2', 'first', 12), extraction('extract/second', 'uni_v1', 'second', 7)];
scientific.datasets = async () => ({ datasets: [] });
scientific.configurations = async () => ({ configurations: [] });
bundles.list = async () => ({ items: [] });
taskCenter.rollup = async (scope) => ({ scope: scope ?? {}, state: 'not-started', counts: {}, byKind: {}, progress: null, live: 0, active: 0, pending: 0, held: false, position: null, queuePosition: null, waitingReason: null, eta: null, runnerAlive: true, paused: false, stopRequest: null, lastFailure: null, recentFailures: 0, current: null, startedAt: null, finishedAt: null, ownerKey: null, ownerKind: null, ownerId: null, title: null, projectId: null, projectName: null, href: '#task-center', updatedAt: '2026-09-25T10:05:00Z' });
taskCenter.tasks = async () => ({ tasks: [] });
trident.catalog = async () => ({ source: '#catalog', schemaVersion: 1, defaults: { task: 'all', patch_encoder: 'uni_v2' },
  options: [{ name: 'task', flag: '--task', label: 'Task', group: 'execution', type: 'string', default: 'all', advanced: false, description: '' }],
  patchEncoders: ['uni_v1', 'uni_v2'], slideEncoders: [], runtime: { ready: true, available: true } });
trident.jobs = async () => { state.calls.push({ method: 'jobs', at: performance.now() }); return { jobs: copy(state.extractions) }; };
trident.job = async (_, id) => { state.calls.push({ method: 'job', id, at: performance.now() }); return copy(state.extractions.find(item => item.id === id)); };
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="stage-workspace"><LocalFeatures workspace={{ project: { id: 'project', storagePath: '/project' }, sources: [] }} /></main><JobTray projectId="project" /></QueryClientProvider>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
    outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
    lib: { entry: fixture, name: 'ExtractionJobsFixture', formats: ['iife'], fileName: () => 'fixture.js' },
  } });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{padding:24px 24px 100px}#app{max-width:1320px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter((name) => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')],
  { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
let stderr = '';
browser.stderr.on('data', (data) => { stderr += data.toString(); });
const requests = new Map();
let nextId = 0, buffer = '', sessionId;
const exceptions = [];
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
    } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
    else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') exceptions.push(message.params.args);
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
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate('Boolean(' + expression + ')')) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}

const visible = (selector) => `[...document.querySelectorAll(${JSON.stringify(selector)})].find(el => el.checkVisibility())`;
const button = (text) => `[...document.querySelectorAll('button')].find(el => el.checkVisibility() && el.textContent.trim() === ${JSON.stringify(text)})`;
const folder = `[...document.querySelectorAll('label')].find(el => el.checkVisibility() && el.textContent.trim().startsWith('Slide folder'))?.querySelector('input')`;
async function click(expression, label = expression) {
  await waitFor(expression + ' && !' + expression + '.matches(":disabled")', label);
  await evaluate(expression + '.click()');
}
async function job(id) {
  const href = '#features?extraction=' + encodeURIComponent(id);
  await click(visible('.job-tray a[href="' + href + '"]'), 'compute link ' + id);
  await waitFor(visible('.trident-job-detail') + '?.textContent.includes(' + JSON.stringify(id) + ')', 'progress for ' + id);
  assert.equal(await evaluate('window.location.hash'), href);
}
async function settings() {
  const step = `[...document.querySelectorAll('nav[aria-label="Extraction steps"] button')].find(el => el.checkVisibility() && el.querySelector('strong')?.textContent === 'Extraction settings')`;
  await click(step, 'extraction settings step');
  await waitFor(folder, 'slide folder input');
}
async function screenshot(name) {
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  await evaluate('Promise.allSettled(document.getAnimations().filter(animation => animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished))');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(artifacts, name + '.png'), Buffer.from(data, 'base64'));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href + '#features' });
  await waitFor('document.querySelectorAll(".pfm-extraction-run").length === 2', 'active extractions in empty feature library');
  await waitFor('document.querySelector(".job-tray").innerText.includes("0 running · 0 queued · 2 extractions")');
  assert.equal(await evaluate('document.body.innerText.includes("Create your first feature bundle")'), false, 'Active extraction must replace the first-bundle empty state');
  assert.equal(await evaluate('document.querySelector(".pfm-extraction-runs").innerText.includes("12 of 100 slides processed")'), true, 'Landing page shows extraction progress');
  await screenshot('01-extraction-library');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('01-extraction-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Extraction library overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await click(visible('.job-tray-toggle'), 'Compute jobs');
  await job('extract/first');
  assert.equal(await evaluate('document.querySelector(".trident-job-detail").innerText.includes("/slides/first")'), true);
  await job('extract/second');
  assert.equal(await evaluate('document.querySelector(".trident-job-detail").innerText.includes("/slides/second")'), true);
  await job('extract/first');
  await settings();
  await evaluate(`(() => { const el = ${folder}; Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '/slides/my-unsaved-selection'); el.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  // A background list refresh must not send the user back to progress or reset settings.
  const listCount = await evaluate('window.workflow.calls.filter(call => call.method === "jobs").length');
  await waitFor('window.workflow.calls.filter(call => call.method === "jobs").length > ' + listCount, 'three-second background extraction polling');
  assert.equal(await evaluate(folder + '?.value'), '/slides/my-unsaved-selection');
  // Same URL clicks do not fire hashchange. They still must reopen the progress panel.
  await job('extract/first');
  await settings();
  assert.equal(await evaluate(folder + '.value'), '/slides/my-unsaved-selection');
  await click(button('Back to feature bundles'));
  await waitFor(visible('.pfm-extraction-runs'), 'return to extraction library');
  assert.equal(await evaluate('window.location.hash'), '#features');
  await job('extract/second');
  await settings();
  assert.equal(await evaluate(folder + '.value'), '/slides/my-unsaved-selection', 'Switching jobs and returning from the library retains unsaved settings');
  await job('extract/first');
  await screenshot('02-selected-extraction');
  const updateAt = await evaluate(`(() => { const item = window.workflow.extractions[0]; item.state = 'succeeded'; item.progress.completed = 100; item.progress.percent = 100; item.progress.label = 'Complete'; item.progress.stages = item.progress.stages.map(stage => ({ ...stage, status: 'complete' })); return performance.now(); })()`);
  await waitFor('document.querySelector(".trident-job-detail").innerText.includes("Extraction complete")', 'polled terminal extraction status');
  await waitFor('document.querySelector(".job-tray").innerText.includes("0 running · 0 queued · 1 extraction")', 'tray active count refresh');
  const lastPoll = await evaluate('window.workflow.calls.filter(call => call.method === "job" && call.id === "extract/first").at(-1).at');
  assert.ok(lastPoll > updateAt && lastPoll - updateAt < 5000, 'Job detail automatically refreshes on the three-second timer');
  await screenshot('03-completed-extraction');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('04-mobile-extraction');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Extraction progress overflows mobile viewport');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(artifacts, 'verification.json'), JSON.stringify({ passed: true, scope: 'Bundled real React components and Chromium using mocked read-only APIs; no HistoPilot server started.', checks: ['active extraction before any bundle', 'compute job exact link', 'switch between two jobs', 'same URL reopens progress after settings', 'settings survive background polling', 'back to library', 'settings retained across library and job navigation', 'three-second automatic status and tray refresh', 'mobile viewport'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: extraction visibility, exact job navigation, same-URL reopening, preserved settings, polling and mobile layout.');
  console.log('Artifacts: ' + artifacts);
} catch (error) {
  try { await writeFile(join(artifacts, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have started. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + artifacts);
  throw error;
} finally {
  browser.kill();
}
