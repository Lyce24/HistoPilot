/** Run with node scripts/verify-evaluation-stage-navigation.mjs. Uses local Chromium and file://; starts no server. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-evaluation-workflow-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalModelEvaluation from ${source('pages/LocalModelEvaluation.tsx')};
import LocalClinicalUtility from ${source('pages/LocalClinicalUtility.tsx')};
import { fixturePredictor } from ${source('testFixtures/predictors.ts')};
import { fixtureExperiment, fixtureEvaluation } from ${source('testFixtures/evaluations.ts')};
import { predictors, modelEvaluations } from ${source('api/predictors.ts')};
import { evaluation } from ${source('api/evaluation.ts')};
import { bulkEvaluations } from ${source('api/bulkEvaluations.ts')};
import { clinicalAnalyses } from ${source('api/clinicalUtility.ts')};
import { experiments } from ${source('api/experiments.ts')};
import { ApiError } from ${source('api/client.ts')};
import { useReviewedPublication } from ${source('components/useReviewedPublication.ts')};
import { bundles } from ${source('api/bundles.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
const copy = (value) => structuredClone(value);
const models = [fixturePredictor(1, 11, 'ensemble'), fixturePredictor(1, 11, 'refit')];
const complete = fixtureEvaluation(models[0], .8, .7, 'cohort', 'patient', 'completed-eval');
complete.manifest.name = 'Existing evaluation';
const cohort = { id: 'cohort', current: false, versionLabel: { tag: 'Independent hospital' }, manifest: { target: models[0].manifest.target, spec: { datasetId: 'test', target: models[0].manifest.target, eligibility: [], patientIdentifiers: 'shared' }, summary: { includedSlides: 180 } } };
const state = window.workflow = { calls: [], evaluations: [complete], batches: [], reports: [], errors: [] };
window.fetch = async (...args) => { state.errors.push('Unexpected network request: ' + args[0]); throw new Error(state.errors.at(-1)); };
predictors.list = async () => ({ items: models });
experiments.summaries = async () => ({ items: [fixtureExperiment('study')] });
evaluation.list = async () => ({ items: [cohort] });
bundles.list = async () => ({ items: [] });
modelEvaluations.list = async () => ({ items: copy(state.evaluations) });
modelEvaluations.execution = async (_, id) => copy(state.evaluations.find((record) => record.id === id)?.execution ?? { status: 'not_started' });
const evalManifest = (selection) => ({ ...complete.manifest, ...copy(selection), coverage: { selectedSlideIds: ['a', 'b'], missingFeatureSlideIds: [], missingPackSlideIds: [], packChecked: false } });
modelEvaluations.preview = async (_, selection) => { state.calls.push({ method: 'single-preview', selection: copy(selection) }); return { canSave: true, previewHash: 'single-hash', findings: [], manifest: evalManifest(selection) }; };
modelEvaluations.save = async (_, selection, hash, operation) => {
  state.calls.push({ method: 'single-save', selection: copy(selection), hash, operation });
  const accepted = state.evaluationOperations ??= {};
  if (!accepted[operation]) {
    accepted[operation] = { ...complete, id: 'new-eval', manifest: evalManifest(selection), execution: { status: 'not_started' } };
    state.evaluations.push(accepted[operation]);
  }
  const attempts = state.calls.filter((call) => call.method === 'single-save').length;
  if (attempts === 1) throw new Error('Simulated lost save response.');
  if (attempts === 2) throw new ApiError('Simulated HTTP 503 after acceptance.', 503);
  return copy(accepted[operation]);
};
bulkEvaluations.list = async () => ({ items: copy(state.batches) });
bulkEvaluations.get = async (_, id) => copy(state.batches.find((batch) => batch.id === id));
bulkEvaluations.preview = async (_, selection) => {
  state.calls.push({ method: 'bulk-preview', selection: copy(selection) });
  return { canRun: true, previewHash: 'batch-hash', reviewedPredictorIds: copy(selection.predictorIds), eligibleCount: selection.predictorIds.length, blockedCount: 0,
    items: selection.predictorIds.map((id) => ({ predictorId: id, predictorName: models.find((model) => model.id === id).manifest.name, method: models.find((model) => model.id === id).manifest.method, eligible: true, findings: [], evaluationManifest: evalManifest({ predictorId: id }) })) };
};
bulkEvaluations.run = async (_, selection, review, operation) => {
  state.calls.push({ method: 'bulk-run', selection: copy(selection), review: copy(review), operation });
  const partial = state.calls.filter((call) => call.method === 'bulk-run').length === 1;
  const result = { id: 'batch-one', name: selection.namePrefix, status: 'running', cohortId: selection.cohortId,
    items: selection.predictorIds.map((id, index) => ({ predictorId: id, predictorName: models[index].manifest.name, method: models[index].manifest.method, status: partial && index ? 'failed' : 'running', evaluationId: 'completed-eval', execution: { status: partial && index ? 'not_started' : 'running' } })) };
  state.batches = [result]; return copy(result);
};
const point = { threshold: .5, tp: 40, fp: 10, tn: 50, fn: 20, sensitivity: .67, specificity: .83, ppv: .8, npv: .71, accuracy: .75, balancedAccuracy: .75, f1: .73, positiveLikelihoodRatio: 4, negativeLikelihoodRatio: .4, predictedPositive: 50, predictedNegative: 70, netBenefit: .25, treatAllNetBenefit: 0, treatNoneNetBenefit: 0, standardizedNetBenefit: .5, netInterventionsAvoidedPer100: 25, highRiskPer100: 42, truePositivePer100: 33, falsePositivePer100: 8, missedPositivePer100: 17 };
const report = { unit: 'patient', frozenUnit: 'patient', positiveClass: 'b', classOrder: ['a','b'], multiclass: false, decisionThreshold: .5, frozenDecisionThreshold: .5, thresholdSource: 'frozen_evaluation', counts: { total: 120, labeled: 120, unlabeled: 0, positive: 60, negative: 60, patients: 120, slides: 180, missingPatientIds: 0 }, metrics: { prevalence: .5, brierScore: .2, brierReference: .25, brierSkillScore: .2, logLoss: .4, multiclassBrierScore: null, rocAuc: .8, averagePrecision: .7, ece: .1, mce: .2 }, operatingPoint: point, operatingCurve: [point], rocCurve: [{ threshold: .5, falsePositiveRate: .17, truePositiveRate: .67 }], precisionRecallCurve: [{ threshold: .5, recall: .67, precision: .8 }], calibration: [{ lower: 0, upper: 1, count: 120, meanPredicted: .5, observedFraction: .5, absoluteError: 0 }], warnings: [], definitions: {}, sources: [] };
const clinicalManifest = (selection) => ({ kind: 'clinical-analysis', name: selection.name, datasetId: 'test', experimentId: 'study', predictorId: models[0].id, evaluationId: selection.evaluationId, selection: copy(selection), target: models[0].manifest.target, report });
clinicalAnalyses.list = async () => ({ items: copy(state.reports) });
clinicalAnalyses.preview = async (_, selection) => { state.calls.push({ method: 'clinical-preview', selection: copy(selection) }); return { canSave: true, previewHash: 'clinical-hash', findings: [], manifest: clinicalManifest(selection) }; };
clinicalAnalyses.save = async (_, selection, hash, operation) => { state.calls.push({ method: 'clinical-save', selection: copy(selection), hash, operation }); const saved = { id: 'report-one', createdAt: '', lifecycleState: 'active', contentHash: 'saved', manifest: clinicalManifest(selection) }; state.reports.push(saved); return copy(saved); };
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
function PublicationProbe() {
  const publication = useReviewedPublication(
    async selection => ({ canSave: true, previewHash: 'probe-hash' }),
    async (selection, hash, operation) => {
      state.calls.push({ method: 'probe-save', selection: copy(selection), hash, operation });
      if (state.calls.filter(call => call.method === 'probe-save').length === 1) {
        await new Promise(resolve => { window.releasePublication = resolve; });
        throw new Error('Uncertain publication probe');
      }
      return { id: 'original-probe-record' };
    },
    preview => preview.canSave,
    async saved => { state.probeSaved = saved; },
  );
  window.publicationProbe = publication;
  return <p id="publication-probe">Publication guard verification</p>;
}
const root = createRoot(document.getElementById('app'));
window.renderStage = (stage) => root.render(<QueryClientProvider client={client}><div className="stage-workspace">{stage === 'publication-probe' ? <PublicationProbe /> : stage === 'clinical' ? <LocalClinicalUtility workspace={{ project: { id: 'project', name: 'Evidence study' } }} /> : <LocalModelEvaluation workspace={{ project: { id: 'project', name: 'Evidence study' } }} />}</div></QueryClientProvider>);
window.renderStage('evaluation');
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
  await new Promise((resolve) => setTimeout(resolve, 200));
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
  await waitFor('document.body.innerText.includes("Existing evaluation")');
  assert.equal(await evaluate('document.querySelector(".stage-library") !== null'), true);
  assert.equal(await evaluate('document.querySelector("#evaluation-experiments-title") === null'), true);
  assert.equal(await evaluate('document.body.innerText.includes("Stage 0 · Saved records")'), false);
  assert.equal(await evaluate(`document.querySelector('.stage-library button[data-record-key="configuration:completed-eval"]')?.textContent`), 'Manage');
  await assertAction('Create evaluation', 'create');
  assert.equal(await evaluate('document.querySelectorAll(".page-header [data-stage-action=create]").length'), 1);
  await screenshot('00-evaluation-library');
  await fill('Search', 'no-such-evaluation');
  await waitFor('document.body.innerText.includes("No matching evaluations")');
  await click('Clear filters');
  await fill('Method', 'refit', 'select');
  await waitFor('document.body.innerText.includes("No matching evaluations")');
  await click('Clear filters');
  await fill('Search', 'Existing');
  await click('Existing evaluation');
  await waitFor('document.body.innerText.includes("Download metrics")');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Back to evaluations');
  assert.equal(await evaluate(field('Search', 'input') + '.value'), 'Existing');
  await click('Clear filters');
  await fill('Test cohort', 'cohort', 'select');
  await click('Compare methods');
  await waitFor('document.querySelector(".evaluation-comparison")');
  assert.equal(await evaluate(field('Test cohort', 'select') + '.value'), 'cohort');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Back to evaluations');
  await click('Create evaluation');
  await waitFor('document.querySelector("#evaluation-experiments-title")');
  assert.equal(await evaluate(field('Test cohort for selected experiments') + ' == null'), true);
  await evaluate('[...document.querySelectorAll("input")].find(el => el.getAttribute("aria-label") === "Evaluate experiment Three-seed ABMIL comparison (study)")?.click()');
  await assertAction('Continue to evaluation inputs', 'continue');
  await click('Continue to evaluation inputs');
  assert.equal(await evaluate('document.querySelector("#evaluation-experiments-title") === null'), true);
  await fill('Test cohort for selected experiments', 'cohort', 'select');
  await fill('Evaluation batch name', 'Method comparison', 'input');
  await waitFor('document.body.innerText.includes("No frozen feature bundles are available")');
  await evaluate('[...document.querySelectorAll("details")].find(el => el.querySelector("summary")?.textContent === "Inference settings").open = true');
  await fill('Decision threshold', '0', 'input');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor(button('Resume evaluation setup'));
  await click('Resume evaluation setup');
  assert.equal(await evaluate(field('Evaluation batch name', 'input') + '.value'), 'Method comparison');
  await click('Back');
  await assertAction('Continue to evaluation inputs', 'continue');
  await click('Continue to evaluation inputs');
  assert.equal(await evaluate(field('Evaluation batch name', 'input') + '.value'), 'Method comparison');
  assert.equal(await evaluate(field('Decision threshold', 'input') + '.value'), '0');
  await click('Review experiment evaluation');
  await waitFor('document.body.innerText.includes("compatible predictors will run")');
  assert.equal(await evaluate(field('Test cohort for selected experiments') + ' == null'), true);
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await assertAction('Back to evaluation inputs', 'back');
  assert.equal(await evaluate(button('Run reviewed predictors') + '.dataset.stageAction'), undefined);
  await screenshot('01-batch-review');
  await evaluate(field('I reviewed the test cohort', 'input') + '.click()');
  await click('Run reviewed predictors');
  await waitFor(button('Retry unfinished submissions'));
  assert.equal(await evaluate(button('Keep this batch and view submitted results') + ' !== undefined'), true);
  assert.equal(await evaluate(button('Back to evaluations') + '.disabled'), true);
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Retry unfinished submissions');
  await waitFor('document.querySelector("[data-stage-page=results]")');
  const submissions = await evaluate('window.workflow.calls.filter(call => call.method === "bulk-run")');
  assert.equal(submissions.length, 2);
  assert.deepEqual(submissions[0], submissions[1]);
  assert.equal(submissions[0].selection.inference.decisionThreshold, 0);
  await click('Open evaluation');
  await waitFor('document.body.innerText.includes("Download metrics")');
  await click('Back to evaluations');
  await waitFor('document.querySelector(".stage-library-tabs")');
  await evaluate('[...document.querySelectorAll(".stage-library-tabs button")].find(el => el.textContent.startsWith("Batches")).click()');
  await waitFor(button('Method comparison'));
  assert.equal(await evaluate(`document.querySelector('.stage-library button[data-record-key="configuration:batch-one"]')?.textContent`), 'Manage');
  await fill('Search', 'unknown batch');
  await waitFor('document.body.innerText.includes("No matching evaluation batches")');
  await click('Clear filters');
  await click('Method comparison');
  await waitFor('document.querySelector(".run-table")');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Back to evaluations');
  await click('Create evaluation');
  await click('Advanced: single predictor plan');
  await fill('Predictor', 'study-1-11-ensemble', 'select');
  await fill('Test cohort', 'cohort', 'select');
  await fill('Evaluation name', 'Single predictor plan', 'input');
  await click('Review evaluation');
  await waitFor(button('Save evaluation plan'));
  assert.equal(await evaluate(field('Evaluation name') + ' == null'), true);
  await evaluate(field('I reviewed the selected inputs', 'input') + '.click()');
  await click('Save evaluation plan');
  await waitFor(button('Retry this save'));
  assert.equal(await evaluate(button('Back to review') + ' === undefined'), true, 'Uncertain publication must not offer a reset escape');
  assert.equal(await evaluate(button('Back to evaluations') + '.disabled'), true);
  assert.equal(await evaluate(button('Back to evaluation inputs') + '.disabled'), true);
  await screenshot('02-uncertain-single-save');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Retry this save');
  await waitFor('document.body.innerText.includes("Simulated HTTP 503 after acceptance.")');
  assert.equal(await evaluate(button('Back to evaluations') + '.disabled'), true);
  assert.equal(await evaluate(button('Back to review') + ' === undefined'), true);
  await click('Retry this save');
  await waitFor(button('Run evaluation'));
  const saves = await evaluate('window.workflow.calls.filter(call => call.method === "single-save")');
  assert.equal(saves.length, 3);
  assert.equal(await evaluate('Object.keys(window.workflow.evaluationOperations).length'), 1);
  assert.equal(await evaluate('window.workflow.evaluations.filter(record => record.id === "new-eval").length'), 1);
  assert.deepEqual(saves[0], saves[1]);
  assert.deepEqual(saves[0], saves[2]);
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await evaluate('window.renderStage("clinical")');
  await waitFor(button('Create clinical analysis'));
  assert.equal(await evaluate(field('Completed evaluation') + ' == null'), true);
  await assertAction('Create clinical analysis', 'create');
  assert.equal(await evaluate('document.querySelectorAll(".page-header [data-stage-action=create]").length'), 1);
  await click('Create clinical analysis');
  await assertAction('Back to clinical analyses', 'back');
  await fill('Completed evaluation', 'completed-eval', 'select');
  await fill('Report name', 'Clinical utility evidence', 'input');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor(button('Resume clinical analysis'));
  await click('Resume clinical analysis');
  assert.equal(await evaluate(field('Report name', 'input') + '.value'), 'Clinical utility evidence');
  await assertAction('Analyze clinical utility', 'continue');
  await click('Analyze clinical utility');
  await waitFor(button('Save clinical utility report'));
  assert.equal(await evaluate(field('Completed evaluation') + ' == null'), true);
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  await click('Back to analysis settings');
  assert.equal(await evaluate(field('Report name', 'input') + '.value'), 'Clinical utility evidence');
  await assertAction('Analyze clinical utility', 'continue');
  await click('Analyze clinical utility');
  await waitFor(button('Save clinical utility report'));
  await evaluate(field('I reviewed the selected inputs', 'input') + '.click()');
  await click('Save clinical utility report');
  await waitFor(button('Copy into a new analysis'));
  assert.equal(await evaluate('document.querySelector("a[data-stage-action=continue]").textContent'), 'Continue to model interpretation');
  await click('Copy into a new analysis');
  assert.equal(await evaluate(field('Report name', 'input') + '.value'), 'Clinical utility evidence copy');
  await click('Back to clinical analyses');
  await waitFor(button('Clinical utility evidence'));
  assert.equal(await evaluate(`document.querySelector('.stage-library button[data-record-key="configuration:report-one"]')?.textContent`), 'Manage');
  await fill('Search', 'missing report');
  await waitFor('document.body.innerText.includes("No matching clinical analyses")');
  await click('Clear filters');
  await fill('State', 'archived', 'select');
  await waitFor('document.body.innerText.includes("No matching clinical analyses")');
  await click('Clear filters');
  await fill('Prediction unit', 'slide', 'select');
  await waitFor('document.body.innerText.includes("No matching clinical analyses")');
  await click('Clear filters');
  await screenshot('03-clinical-library');
  await click('Clinical utility evidence');
  await waitFor(button('Download report JSON'));
  assert.equal(await evaluate(field('Completed evaluation') + ' == null'), true);
  await click('Back to clinical analyses');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('04-clinical-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true);
  await evaluate('window.renderStage("publication-probe")');
  await waitFor('window.publicationProbe');
  await evaluate('window.publicationProbe.preview({ name: "Deliberate review" })');
  await waitFor('window.publicationProbe.review');
  await evaluate('window.publicationProbe.reset()');
  await waitFor('window.publicationProbe.review === null');
  await evaluate('window.publicationProbe.preview({ name: "Preserve this exact request" })');
  await waitFor('window.publicationProbe.review');
  await evaluate('window.publicationProbe.setAcknowledged(true)');
  await waitFor('window.publicationProbe.acknowledged');
  const reviewed = await evaluate('window.publicationProbe.review');
  // Call reset synchronously after publish, before React can disable the UI.
  await evaluate('window.beforeFailureReset = window.publicationProbe.reset; window.pendingPublication = window.publicationProbe.publish(); window.publicationProbe.reset(); void window.publicationProbe.publish()');
  await waitFor('window.releasePublication');
  assert.deepEqual(await evaluate('window.publicationProbe.review'), reviewed);
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "probe-save").length'), 1);
  await evaluate('(async () => { window.releasePublication(); await window.pendingPublication; window.beforeFailureReset(); })()');
  await waitFor('window.publicationProbe.review?.uncertain');
  await evaluate('window.publicationProbe.reset(); void window.publicationProbe.preview({ name: "Replacement request" })');
  assert.deepEqual(await evaluate('window.publicationProbe.review.selection'), reviewed.selection);
  assert.equal(await evaluate('window.publicationProbe.review.operationId'), reviewed.operationId);
  await evaluate('window.publicationProbe.publish()');
  await waitFor('window.publicationProbe.saved?.id === "original-probe-record"');
  const probeSaves = await evaluate('window.workflow.calls.filter(call => call.method === "probe-save")');
  assert.equal(probeSaves.length, 2);
  assert.deepEqual(probeSaves[0], probeSaves[1]);
  await evaluate('window.publicationProbe.reset()');
  await waitFor('window.publicationProbe.saved === null');
  assert.deepEqual(await evaluate('window.workflow.errors'), []);
  assert.deepEqual(exceptions, []);
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real React and local Chromium with mocked APIs; no backend or HistoPilot server.', checks: ['shared create/back/continue controls and execution action distinction', 'library search, filters, reset and exact Manage record keys', 'library and detail separation', 'batch page transitions and retained inputs', 'independent cohort compatibility review', 'partial batch retries preserve request identity', 'single review and transport/503 uncertain-save locks', 'pending and uncertain reset cannot erase operation identity; confirmed preview reset still works', 'clinical create, review, save, copy and library', 'mobile library containment'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: evaluation, batch and clinical library/page transitions, input preservation and publication retry locks.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally {
  browser.kill();
}
