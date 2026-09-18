/** Run with node scripts/verify-feature-workflow.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-feature-workflow-'));
const fixture = join(output, 'fixture.tsx');
const artifacts = resolve(web, '../docs/dev-review/2026-09-14-dataset-bundle-workflow-assets');
await mkdir(artifacts, { recursive: true });
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalFeatures from ${source('pages/LocalFeatures.tsx')};
import { scientific } from ${source('api/scientific.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { packing } from ${source('api/packing.ts')};
import { trident } from ${source('api/trident.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};
const copy = value => structuredClone(value);
const state = window.workflow = { calls: [], errors: [], sources: [], bundles: [], jobs: [], extractions: [] };
window.fetch = async (...args) => { state.errors.push('Unexpected request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const summary = { slideCount: 2, matchedSlides: 2, missingSlides: 0, orphanFiles: 0, dimensions: 8, patchCount: 12 };
const spec = { datasetId: 'dataset', path: '/features', encoderId: 'uni', fileSuffix: '.h5', idSuffix: '', recursive: false };
const sourceRecord = { id: 'source', projectId: 'project', createdAt: '2026-09-12T00:00:00Z', contentHash: 'source-hash', versionLabel: { tag: 'UNI source' }, manifest: { kind: 'feature', datasetId: 'dataset', spec, summary, files: [], findings: [] } };
state.sources.push(sourceRecord);
const validation = { valid: true, tensorValidationComplete: true, provenanceComplete: true, featureSetId: 'source', sourceContentHash: 'hash', slideCount: 2, totalPatches: 12, dimensions: 8, sourceDtype: 'float32', current: true };
const artifact = { id: 'pack', materializationId: 'materialization', featureSetId: 'source', outputPath: '/packs/verified', sourceContentHash: 'hash', slideCount: 2, totalPatches: 12, dimensions: 8, outputDtype: 'float32', dtypePolicy: 'preserve', verification: 'full', validation, current: true };
function savedBundle(id, versionLabel, bundleSpec) { return { id, createdAt: '2026-09-12T00:00:00Z', versionLabel, current: true, findings: [], manifest: { kind: 'feature-bundle', datasetId: 'dataset', spec: bundleSpec, summary: { slideCount: 2, patchCount: 12, dimensions: 8, packCount: bundleSpec.packArtifactIds.length }, feature: { id: 'source' }, packs: bundleSpec.packArtifactIds.length ? [artifact] : [] } }; }
state.bundles.push(savedBundle('baseline', { tag: 'Baseline bundle' }, { featureSetId: 'source', packArtifactIds: [] }));
state.bundles.push({ ...savedBundle('needs-review', { tag: 'Attention bundle', note: 'Recheck the source files' }, { featureSetId: 'source', packArtifactIds: [] }), current: false, createdAt: '2026-09-10T00:00:00Z' });
scientific.datasets = async () => ({ datasets: [{ id: 'dataset', versionLabel: { tag: 'Slides' }, manifest: { name: 'Slides', summary: { slideCount: 2, includedSlides: 2 } } }] });
scientific.configurations = async () => ({ configurations: copy(state.sources) });
scientific.featurePreview = async (_, next) => { state.calls.push({ method: 'sourcePreview', spec: copy(next) }); return { spec: copy(next), summary, files: [], findings: [], canFreeze: true, previewHash: 'preview-source' }; };
scientific.featureFreeze = async (_, next, hash) => { state.calls.push({ method: 'sourceFreeze', spec: copy(next), hash }); const result = { ...sourceRecord, manifest: { ...sourceRecord.manifest, spec: copy(next), previewHash: hash } }; state.sources = [result]; return copy(result); };
bundles.list = async () => { state.calls.push({ method: 'bundleList' }); return { items: copy(state.bundles) }; };
bundles.preview = async (_, next) => { state.calls.push({ method: 'bundlePreview', spec: copy(next) }); return { spec: copy(next), summary: { slideCount: 2, patchCount: 12, dimensions: 8, packCount: next.packArtifactIds.length }, packs: next.packArtifactIds.length ? [artifact] : [], findings: [], canFreeze: true, previewHash: 'preview-bundle-' + state.calls.length }; };
bundles.freeze = async (_, next, previewHash, operationId, versionLabel) => { state.calls.push({ method: 'bundleFreeze', spec: copy(next), previewHash, operationId, versionLabel: copy(versionLabel) }); const result = savedBundle('new-bundle', copy(versionLabel), copy(next)); state.bundles.push(result); return copy(result); };
packing.jobs = async () => ({ jobs: copy(state.jobs), artifacts: [artifact], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
packing.validation = async () => validation;
packing.preview = async (_, next) => { state.calls.push({ method: 'packPreview', spec: copy(next) }); return { spec: copy(next), slideCount: 2, patchCount: 12, dimensions: 8, sourceDtype: 'float32', findings: [], canRun: true, previewHash: 'job-review' }; };
packing.start = async (_, next) => { state.calls.push({ method: 'packStart', spec: copy(next) }); const job = { id: 'validation', featureSetId: 'source', state: 'succeeded', spec: copy(next), createdAt: '2026-09-12T00:00:00Z', updatedAt: '2026-09-12T00:00:01Z', result: { validation }, logPath: '/logs/validation', sessionName: 'hp-validate' }; state.jobs = [job]; return copy(job); };
packing.job = async (_, id) => copy(state.jobs.find(job => job.id === id));
const outputLayout = { jobDir: '/trident', coordsDir: '/trident/coords', patchesDir: '/trident/patches', featuresDir: '/trident/features_uni', contoursDir: '/trident/contours', geojsonDir: '/trident/geojson', thumbnailsDir: '/trident/thumbnails', coordinatePattern: '/trident/patches/*.h5', featurePattern: '/trident/features_uni/*.h5', featureKind: 'patch' };
trident.catalog = async () => ({ source: '#catalog', schemaVersion: 1, defaults: { task: 'all' }, options: [{ name: 'task', type: 'string', default: 'all' }], patchEncoders: [], slideEncoders: [] });
trident.jobs = async () => ({ jobs: copy(state.extractions) });
trident.job = async (_, id) => copy(state.extractions.find(item => item.id === id));
trident.preview = async (_, next) => { state.calls.push({ method: 'extractionPreview', spec: copy(next) }); return { spec: copy(next), slideCount: 2, outputLayout, findings: [], canRun: true, previewHash: 'extraction-review', runtime: { ready: true }, command: ['trident'] }; };
trident.start = async (_, next) => { state.calls.push({ method: 'extractionStart', spec: copy(next) }); const job = { id: 'extraction', state: 'succeeded', createdAt: '2026-09-12T00:00:00Z', spec: copy(next), outputPath: next.outputPath, outputLayout, logPath: '/logs/extraction', sessionName: 'hp-extract', result: { outputLayout, featurePath: '/trident/features_uni', completedSlides: 2, findings: [] } }; state.extractions.push(job); return copy(job); };
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><main className="stage-workspace"><LocalFeatures workspace={{ project: { id: 'project', storagePath: '/project' }, dataset: { id: 'dataset' }, sources: [] }} /></main></QueryClientProvider>);
`);
await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'FeatureWorkflowFixture', formats: ['iife'], fileName: () => 'fixture.js' },
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
const field = (text, selector = 'input,select,textarea') => '[...document.querySelectorAll("label")].find(el => el.checkVisibility() && el.textContent.trim().startsWith(' + JSON.stringify(text) + '))?.querySelector(' + JSON.stringify(selector) + ')';
async function click(text) {
  if (text === 'Create feature bundle') await verifyHeader(true);
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
  if (text === 'Back to feature bundles') {
    await waitFor('document.querySelector(".stage-library")');
    await verifyHeader(true);
  } else if (text === 'Create feature bundle') {
    await waitFor('document.querySelector(".page-header [data-stage-action=back]")');
    await verifyHeader(false);
  }
}
async function fill(text, value, selector = 'input,select,textarea') {
  const element = field(text, selector);
  await waitFor(element, 'field ' + text);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function verifyHeader(library) {
  const header = await evaluate(`({
    library: Boolean(document.querySelector('.stage-library')),
    actions: [...document.querySelectorAll('.page-header [data-stage-action]')].filter(el=>el.checkVisibility()).map(el=>({kind:el.dataset.stageAction,text:el.textContent.trim()}))
  })`);
  assert.equal(header.library, library, 'New bundles must start from their library');
  assert.deepEqual(header.actions, [{kind: library ? 'create' : 'back', text: library ? 'Create feature bundle' : 'Back to feature bundles'}], 'Header must have one Create or Back action');
}
async function screenshot(name) {
  await evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
  await evaluate('Promise.allSettled(document.getAnimations().filter(animation=>animation.effect?.getTiming().iterations !== Infinity).map(animation=>animation.finished))');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(artifacts, 'features-' + name + '.png'), Buffer.from(data, 'base64'));
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
  await waitFor('document.body?.innerText.includes("Baseline bundle")');
  assert.equal(await evaluate('document.querySelector(".feature-packing") === null'), true);
  await verifyHeader(true);
  await screenshot('00-library');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('00-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Feature library overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  assert.equal(await evaluate('document.querySelectorAll(".stage-library h2, .stage-library-eyebrow").length'), 0);
  assert.equal(await evaluate('[...document.querySelectorAll("button")].filter(el => el.textContent.trim() === "Create feature bundle").length'), 1);
  assert.equal(await evaluate('document.querySelector("button[aria-label=\\"Manage Baseline bundle\\"]").dataset.recordKey'), 'configuration:baseline');
  const listsBeforeRefresh = await evaluate('window.workflow.calls.filter(call => call.method === "bundleList").length');
  await click('Refresh');
  await waitFor('window.workflow.calls.filter(call => call.method === "bundleList").length > ' + listsBeforeRefresh);
  await fill('Search', 'no matching bundle');
  await waitFor('document.body.innerText.includes("No matching bundles")');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 0);
  await click('Clear filters');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 2');
  await fill('Status', 'attention', 'select');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 1');
  assert.equal(await evaluate('document.querySelector(".stage-record-name").textContent'), 'Attention bundle');
  await fill('Search', 'Recheck');
  assert.equal(await evaluate('document.querySelector(".stage-record-name").textContent'), 'Attention bundle');
  await fill('Status', 'verified', 'select');
  await waitFor('document.body.innerText.includes("No matching bundles")');
  await click('Clear filters');
  await fill('Sort', 'oldest', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === "Attention bundle"');
  await fill('Sort', 'recent', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === "Baseline bundle"');
  await fill('Sort', 'name', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === "Attention bundle"');
  await click('Clear filters');
  await fill('Search', 'Baseline');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 1');
  await click('Baseline bundle');
  await waitFor('document.body.innerText.includes("No pack is included in this bundle.")');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await verifyHeader(false);
  await evaluate("window.dispatchEvent(new Event('histopilot:stage-library'))");
  await waitFor('document.querySelector(".stage-library") !== null');
  assert.equal(await evaluate('document.querySelector("input[type=search]").value'), 'Baseline');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 1);
  await click('Clear filters');
  await click('Baseline bundle');
  await click('Back to feature bundles');
  await fill('Search', 'Baseline');
  await fill('Status', 'verified', 'select');
  await fill('Sort', 'name', 'select');
  await click('Create feature bundle');
  await waitFor('document.body.innerText.includes("Choose a feature source")');
  await click('Back to feature bundles');
  await waitFor('document.querySelector(".stage-library")');
  assert.equal(await evaluate('document.querySelector("input[type=search]").value'), 'Baseline');
  assert.equal(await evaluate(field('Status', 'select') + '.value'), 'verified');
  assert.equal(await evaluate(field('Sort', 'select') + '.value'), 'name');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 1);
  await click('Create feature bundle');
  await waitFor('document.body.innerText.includes("Choose a feature source")');
  await click('Add feature source');
  await evaluate('[...document.querySelectorAll("button")].find(el => el.querySelector("strong")?.textContent === "Extract with a PFM").click()');
  await waitFor(field('Slide folder'));
  assert.equal(await evaluate(field('Dataset filter (optional)', 'select') + '.value'), '');
  await fill('Slide folder', '/slides/all');
  await click('Preview extraction');
  await waitFor('document.body.innerText.includes("Extraction preflight")');
  assert.deepEqual(await evaluate('window.workflow.calls.filter(call => call.method === "extractionPreview").at(-1).spec.datasetId'), null);
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "extractionPreview").at(-1).spec.recursive'), true);
  assert.equal(await evaluate('document.body.innerText.includes("TRIDENT output directory")'), false);
  assert.equal(await evaluate(button('Back to extraction settings') + '?.dataset.stageAction'), 'back');
  await click('Back to extraction settings');
  assert.equal(await evaluate(field('TRIDENT output directory') + '.value'), '/project/trident');
  await fill('Slide selection', 'upload', 'select');
  await evaluate(`(() => {
    const input = [...document.querySelectorAll('input[type="file"]')].find(el => el.checkVisibility());
    const transfer = new DataTransfer();
    transfer.items.add(new File([${JSON.stringify('wsi,mpp\ncohort/A.svs,0.25\ncohort/B.svs,0.5\n')}], 'slides.csv', {type:'text/csv'}));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', {bubbles:true}));
  })()`);
  await waitFor(button('Preview extraction') + ' && !' + button('Preview extraction') + '.matches(":disabled")');
  await screenshot('01-extraction-selection');
  await click('Preview extraction');
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "extractionPreview").at(-1).spec.slideList.filename'), 'slides.csv');
  assert.equal(await evaluate('atob(window.workflow.calls.filter(call => call.method === "extractionPreview").at(-1).spec.slideList.contentBase64)'), 'wsi,mpp\ncohort/A.svs,0.25\ncohort/B.svs,0.5\n');
  await click('Start extraction');
  await waitFor('document.body.innerText.includes("Extraction complete")');
  assert.equal(await evaluate('document.body.innerText.includes("Extraction preflight")'), false);
  await click('Inspect features for a bundle');
  await waitFor(field('Feature directory or TRIDENT job root') + '?.value === "/trident/features_uni"');
  await click('Inspect & review features');
  await waitFor('document.body.innerText.includes("Review feature coverage")');
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "sourcePreview").at(-1).spec.sourceExtractionJobId'), 'extraction');
  await click('Back to feature settings');
  await fill('Feature directory or TRIDENT job root', '/features/new');
  assert.equal(await evaluate(field('Dataset filter (optional)', 'select') + '.value'), '');
  await fill('Slide selection', 'server', 'select');
  await fill('Slide list CSV path', '/slides/reusable-selection.csv');
  await click('Inspect & review features');
  assert.deepEqual(await evaluate('window.workflow.calls.filter(call => call.method === "sourcePreview").at(-1).spec.slideList'), {path:'/slides/reusable-selection.csv'});
  await waitFor('document.body.innerText.includes("Review feature coverage")');
  assert.equal(await evaluate('document.body.innerText.includes("Choose your existing features")'), false);
  await verifyHeader(false);
  await screenshot('01-coverage');
  await click('Back to feature settings');
  assert.equal(await evaluate(field('Feature directory or TRIDENT job root') + '.value'), '/features/new');
  assert.equal(await evaluate('document.body.innerText.includes("Review feature coverage")'), false);
  await click('Inspect & review features');
  await click('Continue to validation & packs');
  await waitFor('document.body.innerText.includes("Features only — skip packing")');
  await click('Validate feature contents');
  await waitFor('document.body.innerText.includes("Review content validation")');
  assert.equal(await evaluate('document.body.innerText.includes("Choose what to include")'), false);
  await click('Start validation');
  await waitFor('document.body.innerText.includes("Content validation complete")');
  assert.equal(await evaluate('document.body.innerText.includes("Review content validation")'), false);
  await screenshot('02-validation-result');
  await click('Review bundle');
  await waitFor('document.body.innerText.includes("Name & freeze bundle")');
  assert.equal(await evaluate('document.body.innerText.includes("Choose what to include")'), false);
  assert.equal(await evaluate('document.body.innerText.includes("Content validation complete")'), false);
  await click('Back to validation & packs');
  await step('Feature validation and packing steps', 'Contents & validation');
  await evaluate('[...document.querySelectorAll("button")].find(el => el.querySelector("strong")?.textContent === "Features + existing pack").click()');
  await click('Add pack to bundle');
  await waitFor('document.body.innerText.includes("Features + 1 pack")');
  await step('Feature validation and packing steps', 'Runs & results');
  await click('Review bundle');
  await waitFor('document.body.innerText.includes("Name & freeze bundle")');
  assert.deepEqual(await evaluate('window.workflow.calls.filter(call => call.method === "bundlePreview").at(-1).spec.packArtifactIds'), ['pack']);
  await verifyHeader(false);
  assert.equal(await evaluate(button('Back to validation & packs') + '?.dataset.stageAction'), 'back');
  await screenshot('03-bundle-review');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('03-bundle-review-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), true, 'Bundle review overflows mobile viewport');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await click('Name & freeze bundle');
  await fill('Version tag', 'Unified navigation bundle');
  await click('Freeze feature bundle version');
  await waitFor('window.workflow.calls.some(call => call.method === "bundleFreeze")');
  assert.deepEqual(await evaluate('window.workflow.calls.find(call => call.method === "bundleFreeze").spec.packArtifactIds'), ['pack']);
  assert.equal(await evaluate('window.workflow.calls.find(call => call.method === "sourceFreeze").spec.path'), '/features/new');
  await waitFor('window.location.hash.startsWith("#cohort?")');
  assert.equal(await evaluate('new URLSearchParams(window.location.hash.split("?")[1]).get("bundle")'), 'new-bundle');
  assert.equal(await evaluate('new URLSearchParams(window.location.hash.split("?")[1]).has("dataset")'), false);
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(artifacts, 'features-verification.json'), JSON.stringify({ passed: true, scope: 'React and Chromium with mocked feature APIs. No HistoPilot server started.', steps: ['dataset-independent folder extraction', 'CSV upload with exact MPP values', 'shared server slide list input', 'named bundle handoff to targets and splits', 'one Create action in library header', 'one Back action with no competing Create in editors/details', 'return to library before creating another bundle', 'clean library', 'search and clear', 'combined search and readiness filters', 'recent/oldest/name sorting', 'exact Manage record key', 'open/back retains library filters', 'create', 'source settings', 'extraction preflight', 'extraction activity', 'attach extraction outputs', 'coverage', 'back preserves settings', 'register source', 'job review', 'validation activity', 'bundle review', 'back preserves pack selection', 'tagged freeze'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: feature bundle library, source coverage, job and bundle page transitions, preserved settings, validation, pack inclusions, and tagged freeze.');
  console.log('Artifacts: ' + artifacts);
} catch (error) {
  try { await writeFile(join(artifacts, 'features-failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Browser launch may have failed. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + artifacts);
  throw error;
} finally {
  browser.kill();
}
