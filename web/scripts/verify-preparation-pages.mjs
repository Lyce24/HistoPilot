/** Run with node scripts/verify-preparation-pages.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-preparation-pages-'));
const fixture = join(output, 'fixture.tsx');
const artifacts = resolve(web, '../docs/dev-review/2026-09-14-dataset-bundle-workflow');
await mkdir(artifacts, { recursive: true });
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalDataset from ${source('pages/LocalDataset.tsx')};
import LocalProtocol from ${source('pages/LocalProtocol.tsx')};
import { scientific } from ${source('api/scientific.ts')};
import { packing } from ${source('api/packing.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { newSplit } from ${source('components/SplitStrategy.tsx')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
import ${source('components/StageWorkflow.css')};

const copy = (value) => structuredClone(value);
const state = window.workflow = { calls: [], drafts: [], datasets: [], protocols: [], bundles: [], errors: [], revision: 0 };
window.fetch = async (...args) => { state.errors.push('Unexpected network request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const rows = ['low', 'high', 'low', 'high'].map((grade, index) => ({ slideId: 'slide-' + index, patientId: 'patient-' + index, patientIdSource: 'source', slidePath: '/slides/' + index + '.svs', attributes: { grade, cohort: ['TCGA', 'SurGen', 'RIH', 'TCGA'][index] } }));
const dictionary = [{ key: 'grade', sourceColumn: 'grade', owner: 'patient', type: 'categorical' }, { key: 'cohort', sourceColumn: 'cohort', owner: 'patient', type: 'categorical' }];
const importSpec = { source: { path: '/metadata.csv' }, slideIdColumn: 'Slide_ID', patientIdColumn: 'Patient_ID', patientIdFallback: 'unresolved', slideRoot: '/slides', recursive: true, includeMissingSlides: false, missingValues: [''], attributes: copy(dictionary) };
const importSummary = { sourceRowCount: 4, slideCount: 4, mappedPatientCount: 4, verifiedPatientCount: 4, unlinkedSlideCount: 0, matchedSlideCount: 4, missingSlideCount: 0, unmatchedFileCount: 0, excludedRowCount: 0, scannedFileCount: 4 };
const protocolSpec = (datasetId) => ({ datasetId, target: { field: 'grade', task: 'binary_classification', unit: 'patient', classes: ['low', 'high'], labels: { low: 'low', high: 'high' }, positiveClass: 'high', missing: 'block', unmapped: 'block' }, predictors: [], eligibility: [], split: newSplit([42], 2), constraints: { minPatientsPerClass: 1, minPatientsPerPartition: 1 }, featureBundleId: 'bundle-shared' });
const datasets = ['a', 'b'].map((key, index) => ({ id: 'dataset-' + key, projectId: 'project', createdAt: '2026-09-' + (10 + index) + 'T12:00:00Z', contentHash: key, artifacts: {}, versionLabel: { tag: 'Hospital ' + key.toUpperCase(), note: 'Saved dataset ' + key, revision: 1 }, manifest: { name: key, dictionary: copy(dictionary), summary: copy(importSummary), provenance: { mapping: copy(importSpec) } } }));
const featureBundles = [{ id: 'bundle-shared', current: true, findings: [], versionLabel: { tag: 'All slides · UNI' }, manifest: { kind: 'feature-bundle', datasetId: 'another-dataset', spec: { featureSetId: 'source', packArtifactIds: [] }, summary: { slideCount: 6, patchCount: 600, dimensions: 1024, dtype: 'float32', packCount: 0 }, feature: { validation: { tensorValidationComplete: true } }, packs: [] } }];
const protocols = ['a', 'b'].map((key, index) => ({ id: 'protocol-' + key, projectId: 'project', createdAt: '2026-09-' + (10 + index) + 'T12:00:00Z', versionLabel: { tag: 'Protocol ' + key.toUpperCase(), note: 'Saved protocol ' + key, revision: 1 }, manifest: { kind: 'protocol', datasetId: 'dataset-' + key, spec: protocolSpec('dataset-' + key) } }));
const drafts = [
  { id: 'import-draft', name: 'Existing import draft', kind: 'import', payload: { type: 'dataset-import', spec: { ...copy(importSpec), source: { path: '/saved-import.csv' } } } },
  { id: 'protocol-draft', name: 'Existing protocol draft', kind: 'experiment', payload: { type: 'analysis-protocol', spec: protocolSpec('dataset-b') } },
].map((draft) => ({ ...draft, projectId: 'project', status: 'editable', revision: 3, createdAt: '2026-09-12T12:00:00Z', updatedAt: '2026-09-12T12:00:00Z' }));
const values = (field) => field ? [...new Set(rows.map((row) => row.attributes[field]))].map((value) => ({ value, count: rows.filter((row) => row.attributes[field] === value).length })) : [];
const stats = { totalSlides: 4, patientCount: 4, fallbackSlideCount: 0, groupCount: 4, unlinkedSlideCount: 0, sample: rows };
scientific.datasets = async () => ({ datasets: copy(state.datasets) });
scientific.drafts = async () => ({ drafts: copy(state.drafts) });
scientific.configurations = async (_, kind) => ({ configurations: kind === 'protocol' ? copy(state.protocols) : [] });
packing.jobs = async () => ({ jobs: [], artifacts: [] });
bundles.list = async () => ({ items: copy(state.bundles) });
scientific.draft = async (_, id) => { state.calls.push({ method: 'draft', id }); return copy(state.drafts.find((draft) => draft.id === id)); };
scientific.queryDataset = async (_, id, query) => {
  state.calls.push({ method: 'queryDataset', id });
  return { records: copy(rows), total: 4, totalSlides: 4, offset: 0, limit: 200, summary: copy(importSummary), valueCounts: values(query.field), valuesTruncated: false, distribution: { kind: 'categorical', unit: 'patient', counts: values(query.field), total: 4, missingCount: 0 } };
};
scientific.exploreProtocol = async (_, request) => {
  state.calls.push({ method: 'exploreProtocol', request: copy(request) });
  return { datasetId: request.datasetId, splitMode: request.splitMode, valid: true, dataset: copy(stats), cohort: copy(stats), ...(request.featureBundleId ? { populationSource: 'dataset_and_bundle', matchedSlides: 4, bundleSlides: 6 } : {}), partitions: null, unassigned: null,
    target: request.targetField ? { field: request.targetField, values: values(request.targetField).map(({ value, count }) => ({ value, slides: count })), distinctCount: 2 } : null, findings: [] };
};
scientific.inspect = async (_, source) => { state.calls.push({ method: 'inspect', source: copy(source) }); return { headers: ['Slide_ID', 'Patient_ID', 'grade'], sheets: [], sheet: null, rows: rows.map((row) => ({ Slide_ID: row.slideId, Patient_ID: row.patientId, grade: row.attributes.grade })), rowCount: 4, fingerprint: source.path, findings: [] }; };
scientific.saveDraft = async (_, input, current) => {
  state.calls.push({ method: 'saveDraft', input: copy(input), current: copy(current) });
  const draft = { ...copy(input), id: current?.id ?? 'created-' + (state.drafts.length + 1), revision: (current?.revision ?? 0) + 1, status: 'editable', projectId: 'project', createdAt: '2026-09-12T12:00:00Z', updatedAt: '2026-09-12T12:00:00Z' };
  state.drafts = [...state.drafts.filter((item) => item.id !== draft.id), draft]; return copy(draft);
};
scientific.importPreview = async (_, draftId, revision) => { state.calls.push({ method: 'importPreview', draftId, revision }); return { draftId, revision, previewHash: 'import-' + revision, canFreeze: true, findings: [], summary: copy(importSummary), dictionary: copy(dictionary), records: copy(rows), recordsTruncated: false }; };
scientific.protocolPreview = async (_, draftId, revision) => { state.calls.push({ method: 'protocolPreview', draftId, revision }); return { draftId, revision, previewHash: 'protocol-' + revision, canFreeze: true, findings: [], partitions: [], summary: { splitVersion: 4, includedPatients: 4, includedSlides: 4, totalSlides: 4, excludedSlides: 0, includedGroups: 4, strategy: 'kfold', evaluationPlanCount: 2 } }; };
scientific.protocolPreflight = async (_, protocolId) => { state.calls.push({ method: 'protocolPreflight', protocolId }); return { protocolId, findings: [] }; };
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
const root = createRoot(document.getElementById('app'));
window.configure = (page, empty = false, hash = '') => {
  state.datasets = empty ? [] : copy(datasets); state.protocols = empty ? [] : copy(protocols); state.drafts = empty ? [] : copy(drafts); state.bundles = empty ? [] : copy(featureBundles);
  state.revision += 1; client.clear(); window.history.replaceState({}, '', location.pathname + hash);
  const workspace = { project: { id: 'project', name: 'Preparation study', config: { seed: 42, folds: 2 } }, dataset: { id: 'dataset-a' }, sources: [{ role: 'slides', path: '/slides' }] };
  root.render(<QueryClientProvider client={client}><div className="stage-workspace" key={state.revision}>{page === 'data' ? <LocalDataset workspace={workspace} /> : <LocalProtocol workspace={workspace} />}</div></QueryClientProvider>);
};
window.configure('data', true);
`);

await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'PreparationWorkflowFixture', formats: ['iife'], fileName: () => 'fixture.js' },
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
const button = (text) => '[...document.querySelectorAll("button")].find(el => el.checkVisibility() && el.textContent.trim() === ' + JSON.stringify(text) + ')';
const field = (text, selector = 'input,select,textarea') => '(() => { const label = [...document.querySelectorAll("label")].find(el => el.checkVisibility() && el.textContent.trim().startsWith(' + JSON.stringify(text) + ')); return label?.control ?? label?.querySelector(' + JSON.stringify(selector) + '); })()';
async function click(text) {
  if (text === 'Create dataset' || text === 'Create protocol') await verifyHeader(true);
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
  if (text === 'Back to datasets' || text === 'Back to protocols') {
    await waitFor('document.querySelector(".stage-library")');
    await verifyHeader(true);
  }
}
async function fill(text, value, selector = 'input,select,textarea') {
  const element = field(text, selector);
  await waitFor(element, 'field ' + text);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function screenshot(name) {
  await evaluate('Promise.allSettled(document.getAnimations().map(animation => animation.finished))');
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(artifacts, 'preparation-' + name + '.png'), Buffer.from(data, 'base64'));
}
const visible = (selector) => '[...document.querySelectorAll(' + JSON.stringify(selector) + ')].filter(el => el.checkVisibility()).length';
async function verifyHeader(library) {
  const header = await evaluate(`({
    page: document.querySelector('.dataset-workspace') ? 'dataset' : 'protocol',
    library: Boolean(document.querySelector('.stage-library')),
    actions: [...document.querySelectorAll('.page-header [data-stage-action]')].filter(el => el.checkVisibility()).map(el => ({ kind: el.dataset.stageAction, text: el.textContent.trim() }))
  })`);
  assert.equal(header.library, library, 'Creation must begin from the library; return there before creating another record');
  assert.deepEqual(header.actions, [{ kind: library ? 'create' : 'back', text: library ? 'Create ' + header.page : 'Back to ' + (header.page === 'dataset' ? 'datasets' : 'protocols') }], 'Page header must show one consistent Create or Back action');
}
async function currentPage(key) {
  await waitFor('document.querySelector("[data-stage-page]")?.dataset.stagePage === ' + JSON.stringify(key));
  await waitFor('document.activeElement?.classList.contains("stage-page")', 'transition focuses the active page');
  assert.ok(await evaluate('window.scrollY < 100'), 'transition returns to the page heading');
  await verifyHeader(false);
}
async function openRecord(label) {
  const selector = '[...document.querySelectorAll("button")].find(el => el.getAttribute("aria-label") === ' + JSON.stringify(label) + ')';
  await waitFor(selector); await evaluate(selector + '.click()');
  await waitFor('document.querySelector(".page-header [data-stage-action=back]")');
  await verifyHeader(false);
}
async function configure(page, empty = false, hash = '') {
  await evaluate('window.configure(' + [page, empty, hash].map(JSON.stringify).join(', ') + ')');
  await waitFor('document.querySelector(".stage-library") && !document.body.innerText.includes("Loading ")');
  await verifyHeader(true);
}
async function verifyLibraryControls({ emptyTitle, names, draftName, managementKeys }) {
  assert.equal(await evaluate('document.querySelectorAll(".stage-library h2, .stage-library-eyebrow").length'), 0);
  for (const [name, key] of managementKeys) {
    const selector = '[...document.querySelectorAll(".stage-library button[data-record-key]")].find(el => el.getAttribute("aria-label") === ' + JSON.stringify('Manage ' + name) + ')';
    assert.equal(await evaluate(selector + '?.dataset.recordKey'), key);
  }
  await fill('Search', 'no matching record');
  await waitFor('document.body.innerText.includes(' + JSON.stringify(emptyTitle) + ')');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 0);
  await click('Clear filters');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 3');
  await fill('Status', 'editable', 'select');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 1');
  assert.equal(await evaluate('document.querySelector(".stage-record-name").textContent'), draftName);
  await fill('Status', 'frozen', 'select');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 2');
  await fill('Search', draftName);
  await waitFor('document.body.innerText.includes(' + JSON.stringify(emptyTitle) + ')');
  await fill('Search', '');
  await fill('Sort', 'oldest', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === ' + JSON.stringify(names[0]));
  await fill('Sort', 'recent', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === ' + JSON.stringify(names[1]));
  await fill('Sort', 'name', 'select');
  await waitFor('document.querySelector(".stage-record-name")?.textContent === ' + JSON.stringify(names[0]));
  await click('Clear filters');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 3');
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('document.body?.innerText.includes("No datasets or import drafts yet")');
  assert.equal(await evaluate(visible('fieldset')), 0);
  assert.equal(await evaluate(visible('.stage-steps')), 0);
  await verifyHeader(true);
  await screenshot('00-empty-datasets');
  await click('Create dataset');
  await currentPage('import-1');
  assert.equal(await evaluate(button('Continue to column mapping') + '.matches(":disabled")'), true);
  assert.equal(await evaluate(visible('.dataset-section')), 1);
  await fill('Dataset name', 'New mapping with retained edits');
  await fill('Metadata file path', '/new-metadata.csv');
  await click('Read file & show columns');
  await click('Continue to column mapping');
  await currentPage('import-2');
  assert.equal(await evaluate(visible('.dataset-section')), 1);
  assert.equal(await evaluate(field('Slide_ID source column', 'select') + '.value'), 'Slide_ID');
  await click('Back to datasets');
  await waitFor('document.querySelector(".stage-library")');
  await click('Return to current import');
  await currentPage('import-2');
  await click('Back to files');
  await currentPage('import-1');
  assert.equal(await evaluate(field('Dataset name') + '.value'), 'New mapping with retained edits');
  assert.equal(await evaluate(field('Metadata file path') + '.value'), '/new-metadata.csv');
  await click('Continue to column mapping');
  await click('Preview dataset');
  await currentPage('import-3');
  assert.equal(await evaluate(visible('.dataset-section')), 1);
  await screenshot('01-dataset-review');
  await click('Name & freeze dataset');
  assert.equal(await evaluate(field('Version tag', 'input') + '.value'), 'New mapping with retained edits');
  assert.equal(await evaluate(button('Freeze dataset version') + '.matches(":disabled")'), false);
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  assert.equal(await evaluate('Boolean(document.querySelector("[role=dialog]"))'), true);
  assert.equal(await evaluate('Boolean(document.querySelector(".stage-library"))'), false);
  await click('Back to review');
  await click('Back to mapping');
  await fill('Slide_ID source column', '', 'select');
  assert.equal(await evaluate(button('Preview dataset') + '.matches(":disabled")'), true);
  assert.equal(await evaluate('[...document.querySelectorAll(".stage-step")].find(el => el.textContent.includes("Review & freeze")).disabled'), true);
  await configure('data');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 3);
  await screenshot('02-saved-datasets');
  await verifyLibraryControls({ emptyTitle: 'No matching datasets or imports', names: ['Hospital A', 'Hospital B'], draftName: 'Existing import draft', managementKeys: [['Hospital B', 'dataset:dataset-b'], ['Existing import draft', 'draft:import-draft']] });
  await fill('Search', 'Hospital B');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 1');
  await openRecord('Open dataset Hospital B');
  await waitFor('document.querySelector(".science-version")?.textContent.includes("Hospital B")');
  await waitFor('window.workflow.calls.some(call => call.method === "queryDataset" && call.id === "dataset-b")');
  assert.ok(await evaluate('[...document.querySelectorAll("a")].some(el => el.getAttribute("href") === "#cohort?dataset=dataset-b")'));
  await click('Back to datasets');
  assert.equal(await evaluate('document.querySelector("input[type=search]").value'), 'Hospital B');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 1);
  await click('Clear filters');
  await openRecord('Open import Existing import draft');
  await waitFor(field('Dataset name') + '?.value === "Existing import draft"');
  assert.equal(await evaluate(field('Metadata file path') + '.value'), '/saved-import.csv');
  await fill('Dataset name', 'Unsaved import edit');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor('document.querySelector(".stage-library")');
  await openRecord('Open import Existing import draft');
  await waitFor(field('Dataset name') + '?.value === "Unsaved import edit"');
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "draft" && call.id === "import-draft").length'), 1);
  await configure('targets', true);
  assert.ok(await evaluate('document.body.innerText.includes("No protocols or drafts yet")'));
  assert.equal(await evaluate(visible('fieldset')), 0);
  await click('Create protocol');
  await currentPage('protocol-1');
  assert.equal(await evaluate(button('Continue to target') + '.matches(":disabled")'), true);
  await configure('targets', false, '#cohort?dataset=dataset-b&saved=dataset');
  await screenshot('03-protocol-library');
  assert.equal(await evaluate('document.querySelectorAll(".stage-library tbody tr").length'), 3);
  await verifyLibraryControls({ emptyTitle: 'No matching protocols or drafts', names: ['Protocol A', 'Protocol B'], draftName: 'Existing protocol draft', managementKeys: [['Protocol B', 'configuration:protocol-b'], ['Existing protocol draft', 'draft:protocol-draft']] });
  await fill('Search', 'Hospital B');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 2');
  await fill('Search', 'grade');
  await waitFor('document.querySelectorAll(".stage-library tbody tr").length === 3');
  await click('Clear filters');
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "exploreProtocol").length'), 0);
  await click('Create protocol');
  await currentPage('protocol-1');
  assert.equal(await evaluate(field('Dataset version', 'select') + '.value'), 'dataset-b');
  assert.equal(await evaluate(visible('.protocol-section')), 1);
  await fill('Protocol name', 'Retained development draft');
  assert.equal(await evaluate(button('Continue to target') + '.matches(":disabled")'), true);
  await fill('Feature bundle', 'bundle-shared', 'select');
  await waitFor('document.querySelector(".protocol-bundle-source .science-metrics")?.innerText.includes("Shared slides")');
  assert.equal(await evaluate(visible('select') + ' >= 1'), true);
  assert.equal(await evaluate('document.body.innerText.includes("Training set selection")'), false);
  assert.equal(await evaluate('document.body.innerText.includes("Training slide filters")'), true);
  await fill('Dataset version', 'dataset-a', 'select');
  assert.equal(await evaluate(field('Feature bundle', 'select') + '.value'), 'bundle-shared');
  await fill('Dataset version', 'dataset-b', 'select');
  await waitFor('window.workflow.calls.some(call => call.method === "exploreProtocol" && call.request.featureBundleId === "bundle-shared" && call.request.datasetId === "dataset-b")');
  await screenshot('03a-dataset-bundle-selection');
  await click('Add condition');
  await waitFor(field('Field', 'select') + '?.value === "cohort"');
  await waitFor(field('TCGA', 'input'));
  await evaluate(field('TCGA', 'input') + '.click()');
  await evaluate(field('SurGen', 'input') + '.click()');
  await waitFor('window.workflow.calls.some(call => call.method === "exploreProtocol" && call.request.split?.pools?.rules?.train?.[0]?.value?.join(",") === "TCGA,SurGen")');
  const filtered = await evaluate('window.workflow.calls.filter(call => call.method === "exploreProtocol").at(-1).request');
  assert.equal(filtered.split.pools.trainSelection, 'rules');
  assert.deepEqual(filtered.split.pools.rules.train, [{ field: 'cohort', op: 'in', value: ['TCGA', 'SurGen'] }]);
  await screenshot('03b-cohort-value-filter');
  await evaluate('[...document.querySelectorAll("button")].find(el => el.getAttribute("aria-label") === "Remove Training slide filters condition 1").click()');
  await waitFor('document.querySelectorAll("#protocol-cohort .protocol-condition").length === 0');
  await click('Continue to target');
  await currentPage('protocol-2');
  assert.equal(await evaluate(button('Continue to split design') + '.matches(":disabled")'), true);
  assert.equal(await evaluate(visible('.protocol-section')), 1);
  await fill('Target attribute', 'grade', 'select');
  await waitFor(field('Class names', 'input') + '?.value.includes("high")');
  await fill('Positive class', 'high', 'select');
  await click('Continue to split design');
  await currentPage('protocol-3');
  assert.equal(await evaluate(visible('.protocol-section')), 1);
  await fill('Split seeds', 'invalid', 'input');
  assert.equal(await evaluate(button('Continue to review') + '.matches(":disabled")'), true);
  await fill('Split seeds', '13, 27', 'input');
  await click('Back to protocols');
  await click('Return to current protocol');
  await currentPage('protocol-3');
  assert.equal(await evaluate(field('Split seeds', 'input') + '.value'), '13, 27');
  await click('Back');
  await currentPage('protocol-2');
  assert.equal(await evaluate(field('Positive class', 'select') + '.value'), 'high');
  await click('Back');
  await currentPage('protocol-1');
  assert.equal(await evaluate(field('Protocol name', 'input') + '.value'), 'Retained development draft');
  await click('Continue to target'); await click('Continue to split design'); await click('Continue to review');
  await currentPage('protocol-4');
  await click('Preview & preflight');
  await waitFor(button('Name & freeze protocol'));
  await currentPage('protocol-4-preview-protocol-1');
  assert.equal(await evaluate(visible('.protocol-section')), 1);
  assert.equal(await evaluate('document.getElementById("protocol-review").checkVisibility()'), false);
  assert.equal(await evaluate('document.querySelector(".setup-step-actions").checkVisibility()'), false);
  assert.equal(await evaluate(button('Preview & preflight') + ' === undefined'), true);
  assert.equal(await evaluate(button('Back to split design') + '?.checkVisibility()'), true);
  const saved = await evaluate('window.workflow.calls.filter(call => call.method === "saveDraft" && call.input.payload.type === "analysis-protocol").at(-1).input');
  assert.equal(saved.name, 'Retained development draft');
  assert.equal(saved.payload.spec.datasetId, 'dataset-b');
  assert.equal(saved.payload.spec.featureBundleId, 'bundle-shared');
  assert.equal(saved.payload.spec.split.pools.trainSelection, 'remaining');
  assert.deepEqual(saved.payload.spec.split.pools.rules.train, []);
  assert.ok(await evaluate('window.workflow.calls.some(call => call.method === "exploreProtocol" && call.request.targetField === "grade" && call.request.featureBundleId === "bundle-shared")'));
  assert.deepEqual(saved.payload.spec.split.seeds, [13, 27]);
  assert.equal(saved.payload.spec.target.positiveClass, 'high');
  await screenshot('04-protocol-review');
  await click('Name & freeze protocol');
  await waitFor('document.querySelector("[role=dialog]")');
  assert.equal(await evaluate(field('Version tag', 'input') + '.value'), 'Retained development draft');
  await click('Back to review');
  await evaluate('[...document.querySelectorAll("summary")].find(el => el.textContent === "Review settings and rerun checks").click()');
  await click('Preview & preflight');
  await waitFor('window.workflow.calls.filter(call => call.method === "protocolPreview").length === 2');
  await currentPage('protocol-4-preview-protocol-1');
  await click('Back to split design');
  await currentPage('protocol-3');
  assert.equal(await evaluate(field('Split seeds', 'input') + '.value'), '13, 27');
  await click('Continue to review');
  await currentPage('protocol-4-preview-protocol-1');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor('document.querySelector(".stage-library")');
  await openRecord('Open protocol Protocol B');
  await waitFor('document.querySelector(".panel-header")?.textContent.includes("Protocol B")');
  assert.ok(await evaluate('document.querySelector(".detail-list")?.textContent.includes("Hospital B")'));
  await click('Recheck input preflight');
  await waitFor('window.workflow.calls.some(call => call.method === "protocolPreflight" && call.protocolId === "protocol-b")');
  await click('Copy into a new draft');
  await currentPage('protocol-1');
  assert.equal(await evaluate(field('Dataset version', 'select') + '.value'), 'dataset-b');
  assert.equal(await evaluate(field('Protocol name', 'input') + '.value'), 'Protocol B copy');
  assert.equal(await evaluate('window.workflow.protocols.find(item => item.id === "protocol-b").manifest.spec.datasetId'), 'dataset-b');
  await click('Back to protocols');
  await openRecord('Open protocol draft Existing protocol draft');
  await waitFor(field('Protocol name') + '?.value === "Existing protocol draft"');
  assert.equal(await evaluate(field('Dataset version', 'select') + '.value'), 'dataset-b');
  await fill('Protocol name', 'Unsaved protocol edit');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor('document.querySelector(".stage-library")');
  await openRecord('Open protocol draft Existing protocol draft');
  await waitFor(field('Protocol name') + '?.value === "Unsaved protocol edit"');
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "draft" && call.id === "protocol-draft").length'), 1);
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await waitFor('!document.body.innerText.includes("Updating selection…")');
  await verifyHeader(false);
  await screenshot('05-mobile-protocol');
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'Protocol editor overflows the mobile viewport');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(artifacts, 'preparation-verification.json'), JSON.stringify({ passed: true, scope: 'Real React components and Chromium DOM; in-memory scientific APIs; no HistoPilot server.', checks: ['one Create action in library headers', 'one Back action and no competing Create in editor/detail headers', 'return to library before creating another record', 'empty and saved libraries', 'search and clear', 'combined draft/frozen filtering', 'recent/oldest/name sorting', 'search source dataset and target', 'exact Manage record keys', 'open/back retains search', 'exact dataset and protocol IDs', 'named cross-dataset bundle selection and shared-slide coverage', 'bundle retained across dataset changes', 'bundle passed to target-value exploration', 'direct training filters without duplicate selection gate', 'TCGA and SurGen selected as inclusive cohort values', 'removing final filter restores all shared slides', 'draft reopening', 'create and copy', 'one active substage', 'forward validation gates', 'review and freeze gates', 'version name reused when freezing', 'back/forward retained form state', 'same-module library event', 'focus and scroll reset', 'mobile snapshot'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: dataset/protocol libraries, transitions, exact records, validation gates, draft state, and same-module entry.');
  console.log('Artifacts: ' + artifacts);
} catch (error) {
  try { await writeFile(join(artifacts, 'preparation-failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + artifacts);
  throw error;
} finally {
  browser.kill();
}
