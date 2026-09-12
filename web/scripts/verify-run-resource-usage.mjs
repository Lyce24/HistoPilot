/** Run with node scripts/verify-run-resource-usage.mjs. Uses file:// Chromium fixtures; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-run-resource-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React, { useState } from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import { ResourceCards } from ${source('components/RunResourceUsage.tsx')};
import { development } from ${source('api/development.ts')};
import { ApiError } from ${source('api/client.ts')};
import ${source('styles.css')};
import ${source('clinical-workspace.css')};
import ${source('theme.css')};
const state = window.workflow = { calls: 0, errors: [], mode: null, gpu: 60, aborted: 0 };
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const start = Date.now() - 15000;
const sample = (index, at = start + index*3000) => ({at: new Date(at).toISOString(), host: {cpuCount:36,cpuUtilizationPercent:index===2?null:index*7,totalRamGb:192,availableRamGb:180-index,bootId:'fixture',kernel:'Linux'},gpus:[{index:0,uuid:'gpu-one',name:'NVIDIA RTX A5000',driverVersion:'596.71',totalMemoryGb:24,usedMemoryGb:index,freeMemoryGb:24-index,utilizationPercent:state.gpu}],runs:[{runId:'run-one',pid:123,rssGb:1+index/8}]});
const rows = () => [...Array.from({length:5},(_,index)=>sample(index)),sample(5, Date.now())];
const initial = sample(5);
const execution = {batchId:'batch-one',status:'running',sessionName:'fixture',logPath:'/fixture/log',outputPath:'/fixture/output',createdAt:new Date(start).toISOString(),updatedAt:initial.at,findings:[],runs:[],runCounts:{total:1,queued:0,running:1,completed:0,failed:0,cancelled:0},telemetry:{path:'/fixture/telemetry.jsonl',intervalSeconds:3,latest:initial,peak:{hostUsedRamGb:17,runRssGb:{'run-one':2},gpuUsedMemoryGb:{'0':5}}}};
development.resourceHistory = async (_, __, signal) => {
  state.calls++;
  if(state.mode === 404) throw new ApiError('Not Found', 404);
  if(state.mode === 'transient') throw new Error('Resource endpoint temporarily offline');
  if(state.mode === 'pending') return new Promise((resolve, reject) => {signal.addEventListener('abort',()=>{state.aborted++; reject(new DOMException('Aborted','AbortError'));},{once:true});});
  return {batchId:'batch-one', rows:rows(),totalRows:6,truncated:false};
};
const client = new QueryClient({defaultOptions:{queries:{retry:false}}});
function Fixture(){const [show,setShow]=useState(true);const [record,setRecord]=useState(execution);window.hideResources=()=>setShow(false);window.showResources=()=>setShow(true);window.finishResources=()=>setRecord({...execution,status:'completed',updatedAt:new Date().toISOString()});window.resumeResources=()=>setRecord({...execution,status:'running',updatedAt:new Date().toISOString()});return <QueryClientProvider client={client}><main className='clinical-workspace'>{show?<ResourceCards project='fixture' execution={record}/>:<p>Resources closed</p>}</main></QueryClientProvider>;}
createRoot(document.getElementById('app')).render(<Fixture/>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()], define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: { outDir: join(output, 'dist'), emptyOutDir: true, minify: false, lib: { entry: fixture, name: 'RunResourceFixture', formats: ['iife'], fileName: () => 'fixture.js' } } });
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
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('document.querySelectorAll(".run-resource-chart").length === 4');
  assert.equal(await evaluate('document.querySelector(".tone-brown svg").querySelectorAll(".run-resource-line").length'), 2, 'missing CPU measurements must split the curve');
  await evaluate('document.querySelector(".tone-brown details").open = true');
  await waitFor('document.querySelector(".tone-brown tbody")');
  assert.equal(await evaluate('document.querySelector(".tone-brown tbody").innerText.includes("Not recorded")'), true);
  assert.equal(await evaluate('[...document.querySelectorAll(".tone-brown tbody td")].some(el=>el.textContent === "0.00")'), true, 'true zero CPU usage remains available');
  await screenshot('00-resource-desktop');
  await waitFor('window.workflow.calls >= 2', 'active history polling');
  await evaluate('window.workflow.mode = "transient"');
  await click('Refresh');
  await waitFor('document.body.innerText.includes("Resource history could not refresh")');
  assert.equal(await evaluate('document.querySelectorAll(".run-resource-chart").length'), 4, 'transient failure retains the recorded history');
  assert.equal(await evaluate('document.body.innerText.includes("60% utilization")'), true);
  await evaluate('window.workflow.mode = null');
  await waitFor('!document.body.innerText.includes("Resource history could not refresh")', 'active polling recovers from transient errors');
  await evaluate('window.workflow.mode = 404');
  await click('Refresh');
  await waitFor('document.body.innerText.includes("Restart HistoPilot manually")');
  const before404 = await evaluate('window.workflow.calls');
  await delay(3400);
  await evaluate('window.dispatchEvent(new Event("focus")); document.dispatchEvent(new Event("visibilitychange"))');
  await delay(300);
  assert.equal(await evaluate('window.workflow.calls'), before404, '404 must stop interval and focus retries');
  await screenshot('01-resource-old-service');
  await evaluate('window.workflow.mode = null');
  await click('Retry resource history');
  await waitFor('!document.body.innerText.includes("Restart HistoPilot manually")');
  const beforeFinish = await evaluate('window.workflow.calls');
  await evaluate('window.workflow.gpu = 95; window.finishResources()');
  await waitFor('window.workflow.calls > ' + beforeFinish + ' && document.body.innerText.includes("95% utilization")', 'terminal transition fetches final measurements');
  assert.equal(await evaluate('document.body.innerText.includes("recorded measurements, not live device usage")'), true);
  const terminalCalls = await evaluate('window.workflow.calls');
  await delay(3400);
  assert.equal(await evaluate('window.workflow.calls'), terminalCalls, 'completed batches must not continue polling');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('02-resource-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'resource dashboard must not overflow mobile viewport');
  await evaluate('window.resumeResources()');
  await waitFor('document.body.innerText.includes("Updating")');
  await evaluate('window.workflow.mode = "pending"');
  await click('Refresh');
  await waitFor('document.body.innerText.includes("Refreshing…")');
  await evaluate('window.hideResources()');
  await waitFor('window.workflow.aborted === 1 && document.body.innerText.includes("Resources closed")', 'unmount aborts resource request');
  const unmountedCalls = await evaluate('window.workflow.calls');
  await delay(3400);
  assert.equal(await evaluate('window.workflow.calls'), unmountedCalls, 'unmounted resources must stop polling');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  const result = {passed:true,checks:['four recorded metric histories','missing CPU chart gaps and numeric fallback','3-second active polling','transient failure preserves and recovers history','404 stops interval and focus retries','manual retry restores history','terminal transition fetches final sample and stops polling','mobile layout','unmount aborts request and stops polling'],output};
  await writeFile(join(output,'verification.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
} finally {browser.kill();}
