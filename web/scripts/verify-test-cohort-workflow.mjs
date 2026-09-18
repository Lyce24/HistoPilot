/** Run with node scripts/verify-test-cohort-workflow.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-cohort-workflow-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalEvaluationSetup from ${source('pages/LocalEvaluationSetup.tsx')};
import { scientific } from ${source('api/scientific.ts')};
import { evaluation } from ${source('api/evaluation.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};

const copy = (value) => structuredClone(value);
const state = window.workflow = { calls: [], drafts: [], cohorts: [], errors: [] };
window.fetch = async (...args) => { state.errors.push('Unexpected network request: ' + args[0]); throw new Error(state.errors.at(-1)); };
const records = Object.fromEntries(['dataset-a', 'dataset-b'].map((id) => [id,
  ['low', 'high', 'low'].map((grade, index) => ({ slideId: id + '-' + index, patientId: id + '-patient-' + index,
    patientIdSource: 'supplied', attributes: { site: index < 2 ? 'external' : 'development', grade } }))]));
const datasets = Object.entries(records).map(([id, rows]) => ({ id, projectId: 'project',
  versionLabel: { tag: id === 'dataset-a' ? 'Hospital A' : 'Hospital B', note: '' },
  manifest: { name: id, dictionary: ['site', 'grade'].map((key) => ({ key, owner: 'patient', type: 'text' })),
    summary: { slideCount: rows.length, includedSlides: rows.length, unlinkedSlideCount: 0 } } }));
const selected = (spec) => (spec.datasetIds?.length ? spec.datasetIds : [spec.datasetId]).flatMap((id) => records[id] ?? [])
  .filter((row) => spec.eligibility.every((condition) => condition.op !== 'eq' || row.attributes[condition.field] === condition.value));
const values = (rows, field) => [...new Set(rows.map((row) => row.attributes[field]))]
  .map((value) => ({ value, count: rows.filter((row) => row.attributes[field] === value).length }));
const stats = (rows) => ({ totalSlides: rows.length, patientCount: rows.length, fallbackSlideCount: 0,
  groupCount: rows.length, unlinkedSlideCount: 0, sample: rows });
scientific.datasets = async () => ({ datasets });
scientific.queryDataset = async (_, id, query) => ({ records: records[id], valueCounts: values(records[id], query.field), valuesTruncated: false });
scientific.exploreProtocol = async (_, request) => {
  state.calls.push({ method: 'explore', request: copy(request) });
  const rows = selected(request);
  return { datasetId: request.datasetId, splitMode: request.splitMode, valid: true,
    dataset: stats(records[request.datasetId]), cohort: stats(rows), partitions: null, unassigned: null,
    target: request.targetField ? { field: request.targetField, values: values(rows, request.targetField).map(({ value, count }) => ({ value, slides: count })), distinctCount: values(rows, request.targetField).length } : null,
    findings: [] };
};
evaluation.drafts = async () => ({ drafts: copy(state.drafts) });
evaluation.list = async () => ({ items: copy(state.cohorts) });
evaluation.draft = async (_, id) => copy(state.drafts.find((draft) => draft.id === id));
evaluation.get = async (_, id) => copy(state.cohorts.find((cohort) => cohort.id === id));
evaluation.saveDraft = async (_, name, spec, current) => {
  state.calls.push({ method: 'saveDraft', name, spec: copy(spec), current: copy(current) });
  const draft = { id: current?.id ?? 'draft-' + (state.drafts.length + 1), projectId: 'project', kind: 'experiment',
    name, revision: (current?.revision ?? 0) + 1, status: 'editable', payload: { type: 'evaluation-cohort', spec: copy(spec) } };
  state.drafts = [...state.drafts.filter((item) => item.id !== draft.id), draft];
  return copy(draft);
};
evaluation.preview = async (_, draft) => {
  const spec = state.drafts.find((item) => item.id === draft.id).payload.spec;
  state.calls.push({ method: 'preview', draft: copy(draft) });
  const rows = selected(spec), classCounts = Object.fromEntries(values(rows, spec.target?.field).map(({ value, count }) => [value, count]));
  return { spec: copy(spec), target: copy(spec.target), summary: { includedSlides: rows.length, includedPatients: rows.length,
    excludedSlides: 6 - rows.length, labeledSlides: rows.length, classCounts, developmentSlideOverlap: 0, developmentPatientOverlap: 0 },
    coverage: { selectedSlideIds: rows.map((row) => row.slideId), featureSlideCount: 0, missingFeatureSlideIds: [], missingPackSlideIds: [], packChecked: false },
    overlap: { slideIds: [], patientIds: [], patientsComparable: false },
    compatibility: { development: { dimensions: null, encoderId: null }, evaluation: { dimensions: null, encoderId: null } },
    findings: [], canFreeze: true, executionEnabled: false, previewHash: 'review-' + draft.revision };
};
evaluation.freeze = async (_, draft, previewHash, operationId, versionLabel) => {
  state.calls.push({ method: 'freeze', draft: copy(draft), previewHash, operationId, versionLabel: copy(versionLabel) });
  const preview = await evaluation.preview(_, draft);
  const cohort = { id: 'cohort-' + (state.cohorts.length + 1), projectId: 'project', contentHash: 'hash', createdAt: '2026-09-12T00:00:00Z',
    versionLabel, current: true, findings: [], manifest: { kind: 'evaluation-cohort', datasetId: preview.spec.datasetId, spec: preview.spec,
      target: preview.target, summary: preview.summary, coverage: preview.coverage, findings: [] } };
  state.cohorts.push(cohort);
  state.drafts.find((item) => item.id === draft.id).status = 'frozen';
  return copy(cohort);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
createRoot(document.getElementById('app')).render(<QueryClientProvider client={client}><LocalEvaluationSetup workspace={{ project: { id: 'project', name: 'Bladder study' } }} /></QueryClientProvider>);
`);

await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'CohortWorkflowFixture', formats: ['iife'], fileName: () => 'fixture.js' },
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
const button = (text) => '[...document.querySelectorAll("button")].find(el => el.textContent.trim() === ' + JSON.stringify(text) + ')';
const field = (text, selector = 'input,select,textarea') => '[...document.querySelectorAll("label")].find(el => el.textContent.trim().startsWith(' + JSON.stringify(text) + '))?.querySelector(' + JSON.stringify(selector) + ')';
async function click(text) {
  await waitFor(button(text) + ' && !' + button(text) + '.matches(":disabled")', 'enabled button ' + text);
  await evaluate(button(text) + '.click()');
}
async function fill(text, value, selector = 'input,select,textarea') {
  const element = field(text, selector);
  await waitFor(element, 'field ' + text);
  await evaluate('(() => { const el = ' + element + '; const prototype = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(prototype, "value").set.call(el, ' + JSON.stringify(value) + '); el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true })); })()');
}
async function assertAction(label, kind) {
  const target = button(label);
  await waitFor(target);
  assert.equal(await evaluate(target + '.dataset.stageAction'), kind, label + ' uses its shared control');
  assert.equal(await evaluate(target + '.querySelectorAll(".stage-action-icon svg").length'), 1, label + ' has one consistent icon');
}
async function screenshot(name) {
  const { data } = await cdp('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await writeFile(join(output, name + '.png'), Buffer.from(data, 'base64'));
}
try {
  const target = await cdp('Target.createTarget', { url: 'about:blank' }, null);
  ({ sessionId } = await cdp('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null));
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.navigate', { url: pathToFileURL(join(dist, 'index.html')).href });
  await waitFor('Boolean(document.body)');
  await waitFor('document.body.innerText.includes("No test cohorts yet")');
  assert.equal(await evaluate('document.querySelectorAll(".test-cohort-stage").length'), 0);
  await assertAction('Create test cohort', 'create');
  assert.equal(await evaluate('document.querySelectorAll(".page-header [data-stage-action=create]").length'), 1);
  await screenshot('00-list');
  await click('Create test cohort');
  await fill('Cohort name', 'External validation');
  await evaluate(field('Hospital A', 'input') + '.click()');
  await evaluate(field('Hospital B', 'input') + '.click()');
  await click('Add condition');
  await fill('Field', 'site', 'select');
  await fill('Value', 'external', 'input');
  await waitFor('window.workflow.calls.some(call => call.method === "explore" && call.request.datasetId === "dataset-b" && call.request.eligibility[0]?.value === "external")');
  await waitFor('[...document.querySelectorAll("progress")].filter(el => el.value === 2).length === 2');
  assert.equal(await evaluate('document.querySelectorAll(".test-cohort-stage").length'), 1);
  await screenshot('01-test-data');
  await assertAction('Back to test cohorts', 'back');
  await assertAction('Continue to prediction targets', 'continue');
  await click('Continue to prediction targets');
  assert.equal(await evaluate('document.querySelector(".test-cohort-datasets") === null'), true);
  await fill('Target attribute', 'grade', 'select');
  await waitFor(field('Class names', 'input') + '?.value.includes("high")');
  await fill('Positive class', 'high', 'select');
  await waitFor('document.body.innerText.includes("grade · selected test records")');
  await screenshot('02-targets');
  // A duplicate source edit must preserve both mappings rather than silently collapse one.
  await fill('Source value', 'high', 'input');
  await waitFor('document.body.innerText.includes("already has a mapping")');
  assert.equal(await evaluate('document.querySelectorAll(".science-label-row").length'), 2);
  await click('Continue to review and freeze');
  await waitFor('document.body.innerText.includes("Selected test slides")');
  const saved = await evaluate('window.workflow.calls.filter(call => call.method === "saveDraft").at(-1).spec');
  assert.deepEqual(saved.datasetIds, ['dataset-a', 'dataset-b']);
  assert.deepEqual(saved.eligibility, [{ field: 'site', op: 'eq', value: 'external' }]);
  assert.equal(saved.target.positiveClass, 'high');
  assert.deepEqual(Object.keys(saved.target.labels), ['low', 'high']);
  assert.ok(!saved.protocolId && !saved.featureBundleId && !saved.developmentFeatureBundleId);
  await screenshot('03-review');
  assert.equal(await evaluate(button('Freeze test cohort') + '.dataset.stageAction'), undefined);
  await click('Freeze test cohort');
  await fill('Version tag', 'external-validation-v1');
  await click('Freeze test cohort version');
  await waitFor('document.body.innerText.includes("Frozen test cohort") && !document.querySelector("[role=dialog]")');
  assert.equal(await evaluate('window.workflow.cohorts[0].manifest.summary.includedSlides'), 4);
  await screenshot('04-frozen');
  await click('Back to test cohorts');
  await waitFor('document.querySelector(".test-cohort-registry")?.textContent.includes("external-validation-v1")');
  await click('Create test cohort');
  await fill('Cohort name', 'Planned follow-up cohort');
  await evaluate(field('Hospital A', 'input') + '.click()');
  await click('Save draft');
  await waitFor('document.body.innerText.includes("Test cohort draft saved.")');
  await click('Back to test cohorts');
  await waitFor('document.querySelector(".test-cohort-registry")?.textContent.includes("Planned follow-up cohort")');
  await screenshot('05-frozen-and-planned-list');
  assert.equal(await evaluate('document.body.innerText.includes("Stage 0 · Saved records")'), false);
  assert.equal(await evaluate(`document.querySelector('button[data-record-key="configuration:cohort-1"]')?.textContent`), 'Manage');
  assert.equal(await evaluate(`document.querySelector('button[data-record-key="draft:draft-2"]')?.textContent`), 'Manage');
  await fill('Status', 'frozen', 'select');
  await waitFor('document.querySelectorAll(".test-cohort-registry tbody tr").length === 1');
  assert.equal(await evaluate('document.querySelector(".test-cohort-registry").textContent.includes("Planned follow-up cohort")'), false);
  await fill('Status', 'planned', 'select');
  await waitFor('document.querySelector(".test-cohort-registry").textContent.includes("Planned follow-up cohort")');
  await fill('Search', 'missing cohort');
  await waitFor('document.body.innerText.includes("No matching test cohorts")');
  await click('Clear filters');
  await waitFor('document.querySelectorAll(".test-cohort-registry tbody tr").length === 2');
  await click('Planned follow-up cohort');
  await waitFor(field('Cohort name', 'input') + '?.value === "Planned follow-up cohort"');
  assert.equal(await evaluate(field('Hospital A', 'input') + '.checked'), true);
  assert.equal(await evaluate(field('Hospital B', 'input') + '.checked'), false);
  assert.equal(await evaluate('window.workflow.drafts.filter(draft => draft.status === "editable").length'), 1);
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real React component and Chromium DOM with in-memory mocked scientific/evaluation APIs; no backend or HistoPilot server.', steps: ['shared create/back/continue controls preserve explicit freeze', 'list', 'create', 'multiple datasets', 'conditions', 'live distribution', 'prediction target', 'duplicate mapping guard', 'review', 'tagged freeze', 'list frozen and planned', 'search, status filters, reset and exact Manage record keys', 'resume draft'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: real React/Chromium staged test-cohort workflow with mocked APIs, including freeze without development/features and draft resume.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally {
  browser.kill();
}
