/** Real recipe editing and save payloads in offline Chromium. Starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-mil-recipes-'));
const artifacts = resolve(web, '../docs/dev-review/2026-09-14-mil-recipes');
await mkdir(artifacts, { recursive: true });
const source = (path) => JSON.stringify(join(web, 'src', path));
const fixture = join(output, 'fixture.tsx');
await writeFile(fixture, `
import React, { useState } from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import DevelopmentBatches from ${source('components/DevelopmentBatches.tsx')};
import { development } from ${source('api/development.ts')};
import { experiments } from ${source('api/experiments.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
const state = window.recipeCheck = { saved: [], previews: [], errors: [] };
window.fetch = async (...args) => { const error = 'Unexpected network request: ' + args[0]; state.errors.push(error); throw new Error(error); };
window.confirm = () => true;
const copy = value => structuredClone(value);
const inputs = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native', packArtifactId: null };
const protocol = { target: { classes: ['Wild type', 'Mutant'], positiveClass: 'Mutant', task: 'binary_classification', unit: 'patient' }, split: { version: 4, mode: 'kfold', seeds: [42], folds: 2 } };
development.runtime = async () => ({ available: true, findings: [], versions: {}, cudaAvailable: false, gpuCount: 0, host: { cpuCount: 16, totalRamGb: 64, availableRamGb: 32 }, gpus: [] });
development.preview = async (_, spec) => {
  state.previews.push(copy(spec));
  return { kind: 'mil-batch', version: 1, datasetId: 'dataset', spec, previewHash: 'offline-preview', canFreeze: true, findings: [],
    configurations: [{ id: 'candidate', number: 1, recipe: spec.recipe }], splitPlans: [], runs: [], summary: { configurationCount: 1, trainingSeedCount: 1, splitPlanCount: 2, runCount: 2 }, resolvedInputs: { canPlan: true, findings: [] } };
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
function Fixture() {
  const [record, setRecord] = useState({ id: 'experiment', name: 'MIL controls', revision: 1, batchPlans: [], stage: 'planning', state: 'active', inputs });
  experiments.update = async (_, id, update) => {
    state.saved.push(copy(update));
    const saved = { ...record, ...update, revision: record.revision + 1 };
    setRecord(saved); return copy(saved);
  };
  return <QueryClientProvider client={client}><DevelopmentBatches project="offline-mil" inputs={inputs} experimentName={record.name}
    experimentId={record.id} experimentRevision={record.revision} ownedBatches={[]} ownedDrafts={[]} record={record} experimentStage="planning"
    protocol={protocol} tab="batches" onOpenSetup={() => {}} /></QueryClientProvider>;
}
createRoot(document.getElementById('app')).render(<Fixture />);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()], define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
    outDir: join(output, 'dist'), emptyOutDir: true, minify: false, lib: { entry: fixture, name: 'MILRecipeFixture', formats: ['iife'], fileName: () => 'fixture.js' },
  } });
const dist = join(output, 'dist');
const css = (await readdir(dist)).filter((name) => name.endsWith('.css'));
await writeFile(join(dist, 'index.html'), '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
  + css.map((name) => '<link rel="stylesheet" href="' + name + '">').join('')
  + '<style>body{padding:24px}#app{max-width:1250px;margin:auto}</style></head><body><div id="app"></div><script src="fixture.js"></script></body></html>');

// Reuse the same Chromium pipe transport as the preparation verification script.
const cache = join(homedir(), '.cache/ms-playwright');
const candidate = (await readdir(cache)).filter((name) => name.startsWith('chromium_headless_shell-')).sort().at(-1);
const executable = process.env.HISTOPILOT_CHROMIUM ?? join(cache, candidate ?? '', 'chrome-headless-shell-linux64/chrome-headless-shell');
const browser = spawn(executable, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--remote-debugging-pipe', '--user-data-dir=' + join(output, 'browser-profile')], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });
const requests = new Map(), exceptions = [];
let nextId = 0, buffer = '', sessionId, stderr = '';
browser.stderr.on('data', data => { stderr += data.toString(); });
const rejectPending = error => { for (const { reject } of requests.values()) reject(error); requests.clear(); };
browser.on('error', rejectPending);
browser.on('exit', code => rejectPending(new Error('Chromium exited (' + code + '): ' + stderr)));
browser.stdio[3].on('error', rejectPending);
browser.stdio[4].on('error', rejectPending);
browser.stdio[4].on('data', data => {
  buffer += data.toString(); let end;
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
  return new Promise((resolve, reject) => { requests.set(id, { resolve, reject }); browser.stdio[3].write(JSON.stringify({ id, method, params, ...(session ? { sessionId: session } : {}) }) + '\0'); });
}
async function evaluate(expression) {
  const response = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function waitFor(expression, description = expression) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (await evaluate('Boolean(' + expression + ')')) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Timed out: ' + description + '\n' + await evaluate('document.body.innerText'));
}
const button = text => '[...document.querySelectorAll("button")].find(el => el.checkVisibility() && el.textContent.trim() === ' + JSON.stringify(text) + ')';
const field = text => '(() => { const label = [...document.querySelectorAll("label")].find(el => el.checkVisibility() && el.textContent.trim().startsWith(' + JSON.stringify(text) + ')); return label?.control ?? label?.querySelector("input,select,textarea"); })()';
async function click(text) { await waitFor(button(text) + ' && !' + button(text) + '.disabled'); await evaluate(button(text) + '.click()'); }
async function fill(label, value) {
  const element = field(label); await waitFor(element, label);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(String(value)) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function openSettings() {
  await click('Continue to training settings');
  await waitFor(field('Learning rate'));
  await evaluate('document.querySelectorAll("details.batch-settings-details").forEach(el => el.open = true)');
}
async function save() {
  await click('Continue to compute & predictors');
  await click('Continue to batch review');
  await waitFor(button('Add batch to plan'));
  await click('Add batch to plan');
}
try {
  const { targetId } = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId, flatten: true }, null));
  await cdp('Runtime.enable'); await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor(field('Batch name'));
  await fill('Batch name', 'Experimental recipe'); await openSettings();
  assert.equal(await evaluate(field('Learning rate') + '.value'), '0.0001');
  assert.equal(await evaluate(field('Weight decay') + '.value'), '0.005');
  await fill('Training loss', 'bce');
  await fill('Class loss weights', 'inverse_prevalence');
  await fill('Patient prediction aggregation', 'mean_logits');
  await fill('Ensemble member aggregation', 'mean_logit');
  await fill('Training record sampling', 'patient_natural');
  await fill('Epoch selection policy', 'fallback');
  await fill('Fixed epoch budget', '20'); await fill('Minimum validation positives', '5');
  await save(); await waitFor('window.recipeCheck.saved.length === 1');
  const saved = await evaluate('window.recipeCheck.saved[0].batchPlans[0].spec');
  assert.equal(saved.recipe.learningRate, 0.0001); assert.equal(saved.recipe.weightDecay, 0.005);
  for (const [key, value] of Object.entries({ lossType: 'bce', classWeighting: 'inverse_prevalence', patientAggregation: 'mean_logits', ensembleAggregation: 'mean_logit', samplingStrategy: 'patient_natural', minValidationPositives: 5, fixedEpochBudget: 20 })) assert.equal(saved.recipe[key], value, key);
  await click('Edit batch'); await openSettings();
  assert.equal(await evaluate(field('Training loss') + '.value'), 'bce');
  assert.equal(await evaluate(field('Fixed epoch budget') + '.value'), '20');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'Recipe controls overflow mobile viewport');
  const mobile = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(artifacts, 'recipe-mobile.png'), Buffer.from(mobile.data, 'base64'));
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await click('Back'); await fill('Start from a template', 'oceanpath'); await openSettings();
  assert.equal(await evaluate(field('Learning rate') + '.value'), '0.0001');
  assert.equal(await evaluate(field('Weight decay') + '.value'), '0.005');
  assert.equal(await evaluate(field('Maximum epochs') + '.value'), '20');
  assert.equal(await evaluate(field('Minimum training epochs') + '.value'), '10');
  assert.equal(await evaluate(field('Patient prediction aggregation') + '.value'), 'mean_logits');
  const desktop = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(artifacts, 'oceanpath-preset.png'), Buffer.from(desktop.data, 'base64'));
  await save(); await waitFor('window.recipeCheck.saved.length === 2');
  const preset = await evaluate('window.recipeCheck.saved[1].batchPlans.at(-1).spec.recipe');
  assert.equal(preset.bagSize, null); assert.equal(preset.lrScheduler, 'cosine');
  assert.equal(preset.finalLrFraction, 0.01); assert.equal(preset.ensembleAggregation, 'mean_logit');
  assert.deepEqual(await evaluate('window.recipeCheck.errors'), []); assert.deepEqual(exceptions, []);
  await writeFile(join(artifacts, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real React editor and save payloads, offline Chromium; no HistoPilot server or training.', checks: ['new LR/WD defaults', 'BCE', 'automatic class weights', 'patient sampling', 'patient and ensemble logit averaging', 'explicit validation-positive fallback budget', 'save and reopen', 'OceanPath preset', 'mobile viewport'], saved, preset }, null, 2));
  console.log('PASS: offline MIL recipe editing, save/reopen, OceanPath defaults and preset, mobile layout.');
  console.log('Artifacts: ' + artifacts);
} catch (error) {
  try { await writeFile(join(artifacts, 'failure.txt'), await evaluate('document.body.innerText')); } catch { /* Browser startup can fail. */ }
  throw error;
} finally { browser.kill(); }
