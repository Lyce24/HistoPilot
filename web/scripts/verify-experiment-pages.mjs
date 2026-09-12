/** Run with node scripts/verify-experiment-pages.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-experiment-pages-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalExperiments from ${source('pages/LocalExperiments.tsx')};
import { experiments } from ${source('api/experiments.ts')};
import { scientific } from ${source('api/scientific.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { mil } from ${source('api/mil.ts')};
import { development } from ${source('api/development.ts')};
import { predictors } from ${source('api/predictors.ts')};
import { lifecycle } from ${source('api/lifecycle.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const state = window.workflow = { calls: [], errors: [], records: [], execution: null, resourceCalls: 0, resourceFailure: false };
lifecycle.inventory = async project => ({ projectId: project, revision: 1, projectState: 'active', items: state.records.map(record => ({ key: record.key, id: record.id, type: 'draft', kind: 'model-experiment', name: record.name, state: record.state, createdAt: record.createdAt, dependsOn: [], usedBy: [] })), audit: [], note: '' });
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const protocol = { id: 'protocol', versionLabel: { tag: 'Development protocol' }, manifest: { kind: 'protocol', datasetId: 'dataset', spec: { datasetId: 'dataset', target: { task: 'classification', field: 'grade', unit: 'slide', labels: { low: 'low', high: 'high' }, classOrder: ['low', 'high'] }, split: { mode: 'kfold', folds: 2, seeds: [21] } } } };
const bundle = { id: 'bundle', versionLabel: { tag: 'Feature bundle' }, current: true, findings: [], manifest: { kind: 'feature-bundle', datasetId: 'dataset', spec: { featureSetId: 'features', packArtifactIds: [] }, packs: [], summary: { packCount: 0, slideCount: 10, patchCount: 100, dimensions: 8, dtype: 'float32' } } };
const resolvedInputs = { canPlan: true, findings: [], resolvedLoadingPolicy: 'native', packArtifactId: null, featureSetId: 'features', bundleId: 'bundle', executionImplemented: false };
function resourceRows() { const now = Date.now(); return Array.from({length: 9}, (_, index) => ({at: new Date(now - (8-index)*15000).toISOString(), host: {cpuCount: 36, totalRamGb: 192, availableRamGb: 175-index, cpuUtilizationPercent: index===3 ? null : 12+index*4, kernel: 'Linux 6.8', bootId:'fixture'}, gpus:[{index:0,uuid:'fixture-gpu',name:'NVIDIA RTX A5000',driverVersion:'596.71',totalMemoryGb:24,usedMemoryGb:1+index/2,freeMemoryGb:23-index/2,utilizationPercent:15+index*7}], runs: [{runId:'run-42-split-0',pid:123,rssGb:1+index/8}]})); }
function record(input) { return { id: 'experiment', key: 'draft:experiment', name: input.name, notes: input.notes ?? '', tags: input.tags ?? [], revision: 1, state: 'active', status: 'created', stage: 'planning', legacy: false, createdAt: '2026-09-12T00:00:00Z', updatedAt: '2026-09-12T00:00:00Z', inputs: null, batches: [], drafts: [], batchPlans: [], predictorId: null, executionImplemented: true }; }
function manifest(spec) { const splitPlans = [0, 1].map(fold => ({ id: 'split-' + fold, planId: 'plan-' + fold, seed: 21, fold, slideCount: 10, partitions: { training: 6, validation: 2, assessment: 2 } })); return { kind: 'mil-batch', version: 1, datasetId: 'dataset', spec: copy(spec), configurations: [{ id: 'configuration', number: 1, recipe: copy(spec.recipe) }], splitPlans, runs: spec.trainingSeeds.flatMap(seed => splitPlans.map(split => ({ id: 'run-' + seed + '-' + split.id, candidateId: 'configuration', trainingSeed: seed, splitPlanId: split.id, status: 'planned' }))), summary: { configurationCount: 1, trainingSeedCount: spec.trainingSeeds.length, splitPlanCount: 2, runCount: spec.trainingSeeds.length * 2 }, executionImplemented: true, previewHash: 'batch-preview', resolvedInputs }; }
experiments.summaries = async () => ({ items: copy(state.records) });
experiments.get = async (_, id) => copy(state.records.find(item => item.id === id));
experiments.create = async (_, input) => { state.calls.push({ method: 'create', input: copy(input) }); const result = record(input); state.records.push(result); return copy(result); };
experiments.update = async (_, id, input) => { state.calls.push({ method: 'update', input: copy(input) }); const original = state.records.find(item => item.id === id); if (input.expectedRevision !== original.revision) throw new Error('Revision mismatch'); const result = { ...original, ...copy(input), revision: original.revision + 1 }; state.records = [result]; return copy(result); };
experiments.submit = async (_, id, input) => { state.calls.push({ method: 'submit', input: copy(input) }); const original = state.records.find(item => item.id === id); if (original.revision !== input.expectedRevision || !original.batchPlans.length || !original.inputs) throw new Error('Invalid submit'); const batch = { id: 'batch', key: 'configuration:batch', name: original.batchPlans[0].spec.batchName, state: 'active', status: 'running', createdAt: '2026-09-12T00:00:00Z', manifest: manifest(original.batchPlans[0].spec) }; state.execution = { batchId: 'batch', status: 'running', sessionName: 'fixture-only', logPath: '/fixture/logs', outputPath: '/fixture/output', findings: [], runCounts: { total: batch.manifest.runs.length, queued: batch.manifest.runs.length - 1, running: 1, completed: 0, failed: 0, cancelled: 0 }, runs: batch.manifest.runs.map((run, index) => ({ ...run, status: index ? 'queued' : 'running' })), createdAt: '2026-09-12T00:00:00Z', updatedAt: '2026-09-12T00:00:01Z' }; state.resourceRows = resourceRows(); state.execution.telemetry = {path:'/fixture/telemetry.jsonl',intervalSeconds:15,latest:state.resourceRows.at(-1),peak:{hostUsedRamGb:25,gpuUsedMemoryGb:{'0':5},runRssGb:{'run-42-split-0':2}}}; state.execution.runs[0].progress = {epoch:9,maxEpochs:40,globalStep:90,trainingLoss:0.021,validation:{loss:0.041},learningRate:0.0003,cudaPeakAllocatedBytes:1073741824,cudaPeakReservedBytes:2147483648}; state.execution.runs[0].checkpointPath = '/fixture/run/best.ckpt'; state.execution.runs[0].outputPath = '/fixture/run'; batch.execution = copy(state.execution); const result = { ...original, revision: original.revision + 1, stage: 'running', status: 'running', configurationLocked: true, batches: [batch], predictorPolicies: { batch: original.batchPlans[0].spec.predictorPolicy }, submission: { ...input, status: 'submitted', batchIds: ['batch'], submittedAt: '2026-09-12T00:00:01Z', error: null, retryable: false } }; state.records = [result]; return copy(result); };
scientific.configurations = async () => ({ configurations: [protocol] });
bundles.list = async () => ({ items: [bundle] });
mil.preview = async (_, spec) => { state.calls.push({ method: 'inputPreview', spec: copy(spec) }); return resolvedInputs; };
development.runtime = async () => ({ available: true, python: '/fixture/python', versions: {}, cudaAvailable: false, gpuCount: 0, host: { cpuCount: 8, totalRamGb: 32, availableRamGb: 24 }, gpus: [], findings: [] });
development.preview = async (_, spec) => { state.calls.push({ method: 'batchPreview', spec: copy(spec) }); return { ...manifest(spec), findings: [], canFreeze: true }; };
development.execution = async () => copy(state.execution);
development.results = async () => ({ batchId: 'batch', status: 'completed', findings: [], oof: [], candidates: [42, 43].map(trainingSeed => ({ candidateId: 'configuration', trainingSeed, splitSeed: 21, complete: true, completedRuns: 2, totalRuns: 2, metrics: { available: true, auroc: 0.85, accuracy: 0.8 } })) });
development.history = async (_, __, runId) => ({runId,rows:Array.from({length:9},(_,i)=>({epoch:i+1,trainingLoss:0.7/(i+1),validation:{loss:0.9/(i+1)},learningRate:0.0003,checkpointUnit:'patient'})),totalRows:9,truncated:false});
development.resourceHistory = async () => { state.resourceCalls++; if(state.resourceFailure) throw new Error('Fixture resource refresh unavailable'); return {batchId:'batch',rows:copy(state.resourceRows),totalRows:state.resourceRows.length,truncated:false}; };
predictors.list = async () => ({ items: [] });
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
window.refreshExperimentFixture = () => client.invalidateQueries();
window.finishExperiment = async () => { state.execution.status = 'completed'; state.execution.updatedAt = '2026-09-12T00:00:03Z'; state.execution.runCounts = { ...state.execution.runCounts, completed: state.execution.runCounts.total, running: 0, queued: 0 }; state.execution.runs.forEach(run => run.status = 'completed'); state.records[0].stage = 'finished'; state.records[0].status = 'completed'; state.records[0].batches[0].status = 'completed'; state.records[0].batches[0].execution = copy(state.execution); await client.invalidateQueries(); };
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="stage-workspace"><LocalExperiments workspace={{ project: { id: 'project', name: 'Bladder study', config: {} } }} /></main></QueryClientProvider>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'ExperimentPagesFixture', formats: ['iife'], fileName: () => 'fixture.js' },
} });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{padding:24px}#app{max-width:1320px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

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
const dialogs = [];
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
    } else if (message.method === 'Page.javascriptDialogOpening') { dialogs.push(message.params.message); void cdp('Page.handleJavaScriptDialog', { accept: true }); }
    else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
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
const field = (text, selector = 'input,select,textarea') => '[...document.querySelectorAll("label")].find(el => !el.closest("[hidden]") && el.textContent.trim().startsWith(' + JSON.stringify(text) + '))?.querySelector(' + JSON.stringify(selector) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function fill(text, value, selector = 'input,select,textarea') {
  const element = field(text, selector);
  await waitFor(element, 'field ' + text);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function screenshot(name) {
  await new Promise(resolve => setTimeout(resolve, 220));
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
async function step(label, title) {
  const element = '[...document.querySelectorAll("nav")].find(el => !el.closest("[hidden]") && el.getAttribute("aria-label") === ' + JSON.stringify(label) + ')';
  const match = '[...' + element + '.querySelectorAll("button")].find(el => el.querySelector("strong")?.textContent === ' + JSON.stringify(title) + ')';
  await waitFor(match + ' && !' + match + '.matches(":disabled")');
  await evaluate(match + '.click()');
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('document.body?.innerText.includes("Create your first experiment")');
  assert.equal(await evaluate('document.querySelector(".experiment-create-fields") === null'), true);
  assert.equal(await evaluate('document.querySelector(".stage-library h2") === null'), true, 'The library must not repeat the page heading');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library-toolbar input[type=search]").length'), 1);
  assert.equal(await evaluate('document.body.innerText.includes("Your experiments")'), false);
  await screenshot('00-library');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('00-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Library overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await click('Create experiment');
  await waitFor('document.querySelector(".experiment-create-fields") !== null');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await fill('Experiment name', 'Unified experiment');
  await fill('Tags', 'baseline, ui');
  await fill('Notes', 'Verify the separate workflow pages', 'textarea');
  await click('Create & open inputs');
  await waitFor('document.body.innerText.includes("Verify inputs")');
  assert.equal(await evaluate('document.querySelector("#development-tab-batches").disabled'), true);
  await click('Check & continue to batches');
  await waitFor('document.body.innerText.includes("Batch name")');
  assert.equal(await evaluate('document.body.innerText.includes("Verify inputs")'), false);
  await fill('Batch name', 'Two-seed baseline');
  await fill('Training seeds', '42, 43');
  await click('Continue to training settings');
  await waitFor('[...document.querySelectorAll("[data-batch-step]")].find(el => el.dataset.batchStep === "2")?.hidden === false');
  assert.equal(await evaluate('document.body.innerText.includes("Training seeds")'), false);
  await fill('Maximum epochs', '0');
  await click('Continue to compute & predictors');
  await waitFor('document.body.innerText.includes("Maximum epochs must be at least 1")');
  assert.equal(await evaluate('[...document.querySelectorAll("[data-batch-step]")].find(el => el.dataset.batchStep === "2").hidden'), false, 'Invalid epoch value allowed next page');
  await fill('Maximum epochs', '17');
  await click('Back');
  await waitFor('document.body.innerText.includes("Training seeds")');
  assert.equal(await evaluate(field('Batch name') + '.value'), 'Two-seed baseline');
  assert.equal(await evaluate(field('Training seeds') + '.value'), '42, 43');
  await click('Continue to training settings');
  assert.equal(await evaluate(field('Maximum epochs') + '.value'), '17');
  await click('Continue to compute & predictors');
  await waitFor('document.body.innerText.includes("Compute & parallelism")');
  assert.equal(await evaluate('document.body.innerText.includes("Maximum epochs")'), false);
  await fill('Run on', 'cpu', 'select');
  await click('Continue to batch review');
  await waitFor('document.body.innerText.includes("Review Two-seed baseline")');
  assert.equal(await evaluate('document.body.innerText.includes("Compute & parallelism")'), false);
  await click('Check batch');
  await waitFor('document.body.innerText.includes("Resolved batch")');
  const spec = await evaluate('window.workflow.calls.find(call => call.method === "batchPreview").spec');
  assert.equal(spec.recipe.maxEpochs, 17); assert.deepEqual(spec.trainingSeeds, [42, 43]); assert.deepEqual(spec.resources.gpuIds, []);
  await screenshot('01-batch-review');
  await click('Add batch to plan');
  await waitFor('document.body.innerText.includes("Batch plans (1)")');
  assert.equal(await evaluate('document.body.innerText.includes("Maximum epochs")'), false);
  await click('Continue to review & submit');
  await waitFor('document.body.innerText.includes("Freeze & submit experiment")');
  assert.equal(await evaluate('document.body.innerText.includes("Add a training batch")'), false);
  await screenshot('02-submit-review');
  await click('Freeze & submit experiment');
  await waitFor('document.body.innerText.includes("Inputs, batches and predictor choices are locked.")');
  assert.equal(await evaluate('document.querySelector("#development-tab-results").disabled'), true);
  assert.equal(await evaluate('document.body.innerText.includes("Freeze & submit experiment")'), false);
  await step('Experiment steps', 'Inputs');
  assert.equal(await evaluate('document.querySelector(".mil-plan-fields").disabled'), true);
  await step('Experiment steps', 'Batches');
  assert.equal(await evaluate('document.body.innerText.includes("Add training batch")'), false);
  assert.equal(await evaluate('document.body.innerText.includes("Locked")'), true);
  await step('Experiment steps', 'Runs');
  await waitFor('document.querySelector(".run-resource-metric") || document.querySelector(".run-resource-grid")');
  const order = await evaluate(`(() => { const h2=[...document.querySelectorAll('h2')].filter(el=>el.checkVisibility()); const runs=h2.find(el=>el.textContent==='Runs'), predictors=h2.find(el=>el.textContent==='Predictor creation'); const list=document.querySelector('.experiment-run-tracker'), resources=document.querySelector('.experiment-execution-resources');return {runsFirst:Boolean(runs.compareDocumentPosition(predictors)&4),resourcesAfter:Boolean(list.compareDocumentPosition(resources)&4)}; })()`);
  assert.equal(order.runsFirst, true, 'Runs must precede predictor creation');
  assert.equal(order.resourcesAfter, true, 'Resource usage must follow the actual run details');
  await waitFor('document.body.innerText.includes("CPU") && document.body.innerText.includes("VRAM")');
  await screenshot('03-running');
  await click('Checkpoints');
  await waitFor('document.body.innerText.includes("Checkpoint validation")');
  await screenshot('03-checkpoints');
  await click('Diagnostics');
  await waitFor('document.body.innerText.includes("Learning rate")');
  await click('Artifacts');
  await waitFor('document.body.innerText.includes("/fixture/run/best.ckpt")');
  await evaluate('document.querySelector(".experiment-run-tabs [aria-selected=true]").dispatchEvent(new KeyboardEvent("keydown", {key:"Home", bubbles:true}))');
  await waitFor('document.querySelector(".experiment-run-tabs [aria-selected=true]")?.textContent === "Overview"');
  assert.equal(await evaluate('document.activeElement.textContent'), 'Overview', 'Run tab keyboard navigation must move focus');
  await fill('Search runs', 'seed 43');
  await waitFor('document.querySelectorAll(".experiment-runs-table tbody tr").length === 2');
  await evaluate('document.querySelectorAll(".experiment-run-select")[1].click()');
  await waitFor('document.querySelector(".experiment-run-detail h3")?.textContent.includes("Fold 2")');
  assert.equal(await evaluate('document.querySelector(".experiment-run-detail h3")?.textContent.includes("seed 43")'), true);
  await fill('Search runs', 'no-matching-run');
  await waitFor('document.body.innerText.includes("No runs match these filters")');
  assert.equal(await evaluate('document.querySelector(".experiment-run-detail") === null'), true, 'An empty run filter must not retain unrelated run details');
  await fill('Search runs', '');
  await fill('Status', 'running', 'select');
  await waitFor('document.querySelectorAll(".experiment-runs-table tbody tr").length === 1');
  await fill('Status', 'all', 'select');
  await evaluate('document.querySelector(".experiment-run-select").click()');
  await waitFor('document.querySelector(".experiment-run-detail h3")?.textContent.includes("seed 42")');
  await click('Overview');
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await screenshot('03-running-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),true,'Run monitor overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1080,deviceScaleFactor:1,mobile:false});
  await evaluate('window.finishExperiment()');
  await waitFor('document.querySelector("#development-tab-results").disabled === false');
  await step('Experiment steps', 'Results');
  await waitFor('document.body.innerText.includes("OOF AUROC")');
  assert.equal(await evaluate('document.body.innerText.includes("2 complete configuration / seed groups")'), true);
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('04-results-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Finished record overflows mobile viewport');
  await click('Back to experiments');
  await waitFor('document.querySelector(".stage-library")?.textContent.includes("Unified experiment")');
  await screenshot('05-saved-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Saved library overflows mobile viewport');
  await click('Unified experiment');
  await waitFor('document.body.innerText.includes("This experiment is finished.")');
  await evaluate("window.dispatchEvent(new Event('histopilot:stage-library'))");
  await waitFor('document.querySelector(".stage-library")?.textContent.includes("Unified experiment")');
  assert.deepEqual(dialogs, [], 'Sidebar return from a finished experiment must not prompt to discard edits');

  const beforeManage = await evaluate('JSON.stringify(window.workflow.records[0])');
  await click('Manage');
  await waitFor('document.querySelector(".record-management-page h1")?.textContent === "Manage Unified experiment"');
  assert.equal(await evaluate('document.querySelector(".stage-library")?.checkVisibility()'), false, 'Manage must replace the visible library while retaining its state');
  await waitFor(button('Archive'));
  assert.equal(await evaluate(button('Delete…') + '?.checkVisibility()'), true);
  await screenshot('06-manage-mobile');
  await click('Back to experiments');
  await waitFor('document.querySelector(".stage-library")?.checkVisibility() && document.querySelector(".stage-library")?.textContent.includes("Unified experiment")');
  assert.equal(await evaluate('JSON.stringify(window.workflow.records[0])'), beforeManage, 'Opening and closing Manage mutated the saved record');

  await evaluate('(() => { const other = structuredClone(window.workflow.records[0]); other.id = "experiment-two"; other.key = "draft:experiment-two"; other.name = "Comparison experiment"; other.inputs.loadingPolicy = "native"; window.workflow.records.push(other); return window.refreshExperimentFixture(); })()');
  await waitFor('document.querySelector(".stage-library")?.textContent.includes("Comparison experiment")');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  assert.equal(await evaluate('document.body.innerText.includes("Compare selected experiments")'), false, 'Comparison controls should appear only when records are selected');
  await fill('Search', 'no-such-experiment');
  await waitFor('document.body.innerText.includes("No experiments match this view")');
  await click('Clear filters');
  await waitFor('document.querySelectorAll(".experiment-record-table tbody tr").length === 2');
  await fill('State', 'archived', 'select');
  await waitFor('document.body.innerText.includes("No experiments match this view")');
  await click('Clear filters');
  await fill('Stage', 'running', 'select');
  await waitFor('document.body.innerText.includes("No experiments match this view")');
  await click('Clear filters');
  await fill('Sort', 'name', 'select');
  await waitFor('document.querySelector(".experiment-record-table tbody .stage-record-name")?.textContent === "Comparison experiment"');
  await fill('Search', 'Unified');
  await waitFor('document.querySelectorAll(".experiment-record-table tbody tr").length === 1');
  await click('Unified experiment');
  await waitFor('document.body.innerText.includes("This experiment is finished.")');
  await click('Back to experiments');
  await waitFor('document.querySelector(".stage-library") !== null');
  assert.equal(await evaluate(field('Search') + '.value'), 'Unified', 'Opening a record must retain the library search');
  assert.equal(await evaluate(field('Sort', 'select') + '.value'), 'name', 'Opening a record must retain sorting');
  await fill('Search', '');
  await waitFor('document.querySelectorAll(".experiment-record-table tbody tr").length === 2');
  await screenshot('07-clean-library-desktop');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('07-clean-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Library filters overflow the mobile viewport');
  const compareChoices = '[...document.querySelectorAll(".experiment-record-table input")].filter(el => el.type === "checkbox")';
  await evaluate(compareChoices + '[0].click()');
  await waitFor('document.body.innerText.includes("1 selected")');
  await evaluate(compareChoices + '[1].click()');
  await waitFor('document.body.innerText.includes("2 selected")');
  assert.equal(await evaluate('document.querySelector(".experiment-comparison") === null'), true, 'Selecting rows must not append the comparison below the library');
  await click('Compare selected experiments');
  await waitFor('document.body.innerText.includes("Compare experiment inputs")');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true, 'Comparison must replace the library');
  assert.equal(await evaluate('document.querySelectorAll(".experiment-comparison thead th").length'), 3);
  await screenshot('07-comparison-mobile');
  await click('Back to experiment selection');
  await waitFor('document.querySelector(".stage-library")?.textContent.includes("2 selected")');
  assert.equal(await evaluate(compareChoices + '.filter(el => el.checked).length'), 2, 'Comparison return lost selected records');
  assert.equal(await evaluate('document.querySelector(".experiment-comparison") === null'), true);

  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  assert.deepEqual(dialogs, [], 'Saved submitted experiments should not prompt to discard edits');
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'React and Chromium with mocked experiment/development APIs, no HistoPilot server or training jobs.', calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: experiment library/create, input validation, four batch pages, numeric guards, preserved settings, submission, running locks, run/resource/predictor order, run search/filter/selection, keyboard detail tabs, resource charts, finished results, manage/compare pages, sidebar return, and mobile overflow.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Browser may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally { browser.kill(); }
