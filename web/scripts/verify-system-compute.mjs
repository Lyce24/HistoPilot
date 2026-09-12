/** Run with node scripts/verify-system-compute.mjs. Uses file:// Chromium fixtures; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-system-compute-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React, { useState } from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import System from ${source('pages/System.tsx')};
import { api, ApiError } from ${source('api/client.ts')};
import ${source('styles.css')};
const gib = 1024 ** 3;
const state = window.workflow = { computeCalls: 0, systemCalls: 0, failure: null, errors: [] };
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
api.system = async () => { state.systemCalls++; return { mode: 'local', workspace: '/home/research/projects', storage: { engine: 'SQLite', journalMode: 'wal', schemaVersion: 3 }, control: { process: 'Control service', cudaModelsLoaded: false }, sourcesReadOnly: true, workers: { executionEnabled: true, nativeExecutionImplemented: true, tmuxAvailable: true, status: 'Runtime checks are available in each workflow.' } }; };
api.systemCompute = async () => {
  state.computeCalls++;
  if (state.failure === 404) throw new ApiError('Not Found', 404);
  if (state.failure) throw new Error('Connection interrupted');
  return {
    sampledAt: new Date().toISOString(), sampleIntervalSeconds: 5,
    host: { hostname: 'pathology-workstation', platform: 'Linux', release: '6.8.0', uptimeSeconds: 90061 },
    cpu: { model: 'Intel Xeon W-2295 CPU @ 3.00GHz', logicalCores: 36, physicalCores: 18, availableCores: 36, utilizationPercent: 37.2, loadAverage: [4.2, 3.1, 2.8], status: 'available', message: null },
    memory: { totalBytes: 128 * gib, usedBytes: 42 * gib, availableBytes: 86 * gib, utilizationPercent: 32.8, swapTotalBytes: 8 * gib, swapUsedBytes: 0, status: 'available', message: null },
    gpu: { status: 'available', message: null, devices: [{ index: 0, name: 'NVIDIA RTX A5000', uuid: 'GPU-0', driverVersion: '596.71', utilizationPercent: 87, memoryTotalBytes: 24 * gib, memoryUsedBytes: 18 * gib, memoryFreeBytes: 6 * gib, memoryUtilizationPercent: 75, temperatureCelsius: 64, powerWatts: 175.5, powerLimitWatts: 230 }] },
    disks: [{ role: 'workspace', path: '/home/research/projects', totalBytes: 1024 * gib, usedBytes: 256 * gib, freeBytes: 768 * gib, utilizationPercent: 25, status: 'available', message: null }, { role: 'data', path: '/mnt/slides', totalBytes: 4 * 1024 * gib, usedBytes: 1.5 * 1024 * gib, freeBytes: 2.5 * 1024 * gib, utilizationPercent: 37.5, status: 'available', message: null }],
  };
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
function Fixture() { const [show, setShow] = useState(true); window.hideSystem = () => setShow(false); return <QueryClientProvider client={client}><main>{show ? <System /> : <p>System page closed</p>}</main></QueryClientProvider>; }
createRoot(document.getElementById('app')).render(<Fixture />);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()], define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: { outDir: join(output, 'dist'), emptyOutDir: true, minify: false, lib: { entry: fixture, name: 'SystemComputeFixture', formats: ['iife'], fileName: () => 'fixture.js' } } });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('') + '<style>body{padding:24px}#app{max-width:1240px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');
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
const button = (text) => '[...document.querySelectorAll("button")].find(el => !el.closest("[hidden]") && el.textContent.trim() === ' + JSON.stringify(text) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function screenshot(name) {
  await new Promise(resolve => setTimeout(resolve, 220));
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('document.body?.innerText.includes("NVIDIA RTX A5000")');
  assert.equal(await evaluate('document.querySelectorAll("progress").length'), 6);
  await screenshot('00-system-desktop');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('01-system-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'System dashboard overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await waitFor('window.workflow.computeCalls >= 2', 'compute polls every five seconds');
  assert.equal(await evaluate('window.workflow.systemCalls'), 1, 'expensive runtime diagnostics must not poll');
  await click('Pause live updates');
  await waitFor('document.body.innerText.includes("Updates paused")');
  const beforePause = await evaluate('window.workflow.computeCalls');
  await delay(5400);
  assert.equal(await evaluate('window.workflow.computeCalls'), beforePause, 'paused polling must stop');
  await click('Refresh status');
  await waitFor('window.workflow.computeCalls > ' + beforePause);
  assert.equal(await evaluate('window.workflow.systemCalls'), 2);
  assert.equal(await evaluate('document.body.innerText.includes("Updates paused")'), true);
  await click('Resume live updates');
  await evaluate('window.workflow.failure = "connection"');
  await click('Refresh status');
  await waitFor('document.body.innerText.includes("Connection interrupted")');
  assert.equal(await evaluate('document.body.innerText.includes("Last known sample")'), true);
  assert.equal(await evaluate('document.body.innerText.includes("NVIDIA RTX A5000")'), true, 'failed refresh retains the last sample');
  await evaluate('window.workflow.failure = 404');
  await click('Refresh status');
  await waitFor('document.body.innerText.includes("Restart HistoPilot manually")');
  const before404 = await evaluate('window.workflow.computeCalls');
  await delay(5400);
  assert.equal(await evaluate('window.workflow.computeCalls'), before404, 'old backend must not be polled until manual refresh');
  await screenshot('02-system-old-service');
  await evaluate('window.workflow.failure = null');
  await click('Refresh status');
  await waitFor('!document.querySelector("[role=alert]")');
  await evaluate('window.hideSystem()');
  await waitFor('document.body.innerText.includes("System page closed")');
  const beforeUnmount = await evaluate('window.workflow.computeCalls');
  await delay(5400);
  assert.equal(await evaluate('window.workflow.computeCalls'), beforeUnmount, 'compute query must stop when System is unmounted');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  const result = { passed: true, checks: ['CPU, RAM, GPU and disk metrics', 'desktop/mobile layout', '5-second compute-only polling', 'pause and manual refresh', 'stale sample retention', 'old-service404 instructions and polling stop', 'unmount stops polling'], output };
  await writeFile(join(output, 'verification.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally { browser.kill(); }
