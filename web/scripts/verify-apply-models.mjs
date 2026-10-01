/**
 * Run with node scripts/verify-apply-models.mjs. Uses local Chromium and file://; starts no server.
 * Drives the real Apply models page with mocked APIs: the runs library, a labeled run's views and
 * its clinical analysis, a reference standard added to its cohort (scores, agreement, cases and
 * clinical utility against it), applying predictors in a batch (with a lost submission),
 * batches, cohorts, method comparison, hash links between views and the mobile layout.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'vite';
import react from '@vitejs/plugin-react';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = await mkdtemp(join(tmpdir(), 'histopilot-apply-models-'));
const fixture = join(output, 'fixture.tsx');
const source = (path) => JSON.stringify(join(web, 'src', path));
await writeFile(fixture, `
import React from ${JSON.stringify(join(web, 'node_modules/react/index.js'))};
import { createRoot } from ${JSON.stringify(join(web, 'node_modules/react-dom/client.js'))};
import { QueryClient, QueryClientProvider } from ${JSON.stringify(join(web, 'node_modules/@tanstack/react-query/build/modern/index.js'))};
import LocalApplyModels from ${source('pages/LocalApplyModels.tsx')};
import { fixturePredictor } from ${source('testFixtures/predictors.ts')};
import { fixtureExperiment, fixtureEvaluation } from ${source('testFixtures/evaluations.ts')};
import { fixtureRollup } from ${source('testFixtures/taskCenter.ts')};
import { predictors, modelEvaluations } from ${source('api/predictors.ts')};
import { evaluation } from ${source('api/evaluation.ts')};
import { bulkEvaluations } from ${source('api/bulkEvaluations.ts')};
import { clinicalAnalyses } from ${source('api/clinicalUtility.ts')};
import { experiments } from ${source('api/experiments.ts')};
import { inferenceRuns } from ${source('api/inference.ts')};
import { caseReviews } from ${source('api/caseReview.ts')};
import { scientific } from ${source('api/scientific.ts')};
import { references } from ${source('api/references.ts')};
import { recalibration } from ${source('api/recalibration.ts')};
import { taskCenter } from ${source('api/taskCenter.ts')};
import { bundles } from ${source('api/bundles.ts')};
import { useReviewedPublication } from ${source('components/useReviewedPublication.ts')};
import ${source('styles.css')};
import ${source('local-workspace.css')};
import ${source('scientific.css')};
import ${source('clinical-workspace.css')};
const copy = (value) => structuredClone(value);
const models = [fixturePredictor(1, 11, 'ensemble'), fixturePredictor(1, 11, 'refit')];
const complete = fixtureEvaluation(models[0], .8, .7, 'cohort', 'patient', 'completed-eval');
complete.manifest.name = 'Existing evaluation';
complete.execution.result.metrics.selected.confusionMatrix = [[50, 10], [20, 40]];
complete.manifest.target = models[0].manifest.target;
const cohort = { id: 'cohort', current: false, createdAt: '2026-09-28', versionLabel: { tag: 'Independent hospital' }, manifest: { datasetId: 'test', target: models[0].manifest.target, spec: { datasetId: 'test', target: models[0].manifest.target, eligibility: [], patientIdentifiers: 'shared' }, summary: { includedSlides: 180, includedPatients: 120, excludedSlides: 0, labeledSlides: 180, classCounts: { a: 90, b: 90 }, developmentSlideOverlap: 0, developmentPatientOverlap: 0 }, findings: [] }, findings: [] };
const state = window.workflow = { calls: [], evaluations: [complete], batches: [], reports: [], standards: [], errors: [] };
window.fetch = async (...args) => { state.errors.push('Unexpected network request: ' + args[0]); throw new Error(state.errors.at(-1)); };
predictors.list = async () => ({ items: models });
experiments.summaries = async () => ({ items: [fixtureExperiment('study')] });
evaluation.list = async () => ({ items: [cohort] });
evaluation.get = async () => copy(cohort);
evaluation.drafts = async () => ({ drafts: [] });
scientific.datasets = async () => ({ datasets: [
  { id: 'test', projectId: 'project', createdAt: '2026-09-01', contentHash: 'test', versionLabel: { tag: 'Hospital slides' }, manifest: { dictionary: [{ key: 'reader_a', sourceColumn: 'Reader A grade', owner: 'slide', type: 'text' }] }, artifacts: {} },
  { id: 'later', projectId: 'project', createdAt: '2026-09-20', contentHash: 'later', versionLabel: { tag: 'Hospital slides with consensus' }, manifest: { dictionary: [{ key: 'reader_a', sourceColumn: 'Reader A grade', owner: 'slide', type: 'text' }, { key: 'consensus', sourceColumn: 'Consensus', owner: 'slide', type: 'text' }] }, artifacts: {} },
] });
// A reader's column over the cohort's 180 slides: 100 A, 70 B and 10 unreadable.
const readerValues = [{ value: 'A', count: 100 }, { value: 'B', count: 70 }, { value: '?', count: 10 }];
const referenceManifest = (selection) => {
  const classCounts = Object.fromEntries(selection.classes.map((name) => [name, readerValues.filter((item) => selection.labels[item.value] === name).reduce((sum, item) => sum + item.count, 0)]));
  const labeledSlides = Object.values(classCounts).reduce((sum, count) => sum + count, 0);
  return { kind: 'reference-standard', name: selection.name, cohortId: selection.cohortId, cohort: { id: selection.cohortId, contentHash: 'cohort' }, datasetId: selection.datasetIds[0], datasets: selection.datasetIds.map((id) => ({ id, contentHash: id })),
    field: selection.field, matchedBy: 'slideId', classes: copy(selection.classes), labels: copy(selection.labels),
    summary: { slides: 180, labeledSlides, missingSlides: 0, unmappedSlides: 180 - labeledSlides, unmatchedSlides: 0, classCounts, conflictingPatients: 0, values: readerValues.map((item) => ({ ...item, label: selection.labels[item.value] ?? null })), valuesTruncated: false },
    findings: 180 > labeledSlides ? [{ severity: 'warning', code: 'REFERENCE_UNMAPPED_VALUES', message: (180 - labeledSlides) + ' slides have values the mapping leaves out; they stay unlabeled.' }] : [] };
};
const calibrationMetrics = (brierScore, calibrationSlope) => ({ count: 120, brierScore, logLoss: .5, ece: .08, meanPredictedRisk: .45, observedFraction: .5, observedExpectedRatio: 1.1, calibrationSlope, calibrationIntercept: .1,
  bins: [{ lower: 0, upper: .1, count: 30, meanPredicted: .05, observedFraction: .15 }, { lower: .5, upper: .6, count: 40, meanPredicted: .55, observedFraction: .5 }, { lower: .9, upper: 1, count: 50, meanPredicted: .95, observedFraction: .82 }] });
recalibration.get = async (_, id, unit, referenceId) => {
  state.calls.push({ method: 'recalibration', id, unit, referenceId });
  return { evaluationId: id, unit: 'patient', classOrder: ['a', 'b'], positiveClass: 'b', method: 'platt', parameters: { slope: .62, intercept: -.05 },
    development: { units: 360, seedGroups: 1, original: calibrationMetrics(.2, .6), recalibrated: calibrationMetrics(.18, 1) },
    cohort: { units: 120, original: calibrationMetrics(.21, .58), recalibrated: calibrationMetrics(.19, .97) },
    developmentExcluded: 0, reference: referenceId ? { id: referenceId, name: 'Reader A' } : null, developmentSources: [] };
};
references.list = async (_, cohortId) => ({ items: copy(state.standards).filter((item) => !cohortId || item.manifest.cohortId === cohortId) });
references.preview = async (_, selection) => {
  state.calls.push({ method: 'reference-preview', selection: copy(selection) });
  const manifest = referenceManifest(selection);
  const empty = !manifest.summary.labeledSlides;
  return { canSave: !empty, previewHash: empty ? null : 'reference-hash', manifest, findings: empty ? [{ severity: 'error', code: 'REFERENCE_NO_LABELS', message: 'No cohort slide receives a label.' }] : manifest.findings };
};
references.save = async (_, selection, hash, operation) => {
  state.calls.push({ method: 'reference-save', selection: copy(selection), hash, operation });
  const saved = { id: 'reader-a', createdAt: '2026-09-29', contentHash: 'reader-a', lifecycleState: 'active', manifest: referenceManifest(selection) };
  state.standards.push(saved); return copy(saved);
};
references.scores = async (_, id, referenceId) => {
  state.calls.push({ method: 'reference-scores', id, referenceId });
  const metrics = copy(complete.execution.result.metrics);
  metrics.selected = { ...metrics.selected, auroc: .74, confusionMatrix: [[45, 15], [25, 35]] };
  return { ...metrics, scoredBy: { method: 'control_service_reference_join_v1', predictionsSha256: 'f'.repeat(64), reference: { id: referenceId, contentHash: referenceId } }, reference: { id: referenceId, name: 'Reader A' } };
};
references.agreement = async (_, id, unit) => {
  state.calls.push({ method: 'agreement', id, unit });
  return { evaluationId: id, unit: 'patient', classOrder: ['a', 'b'], developmentExcluded: 0, source: { predictionsSha256: 'f'.repeat(64) },
    sources: [{ id: 'run', name: 'Existing evaluation', kind: 'run', labeled: 120 }, { id: 'cohort', name: 'Cohort labels', kind: 'cohort', labeled: 120 }, ...state.standards.map((item) => ({ id: item.id, name: item.manifest.name, kind: 'reference', labeled: 110 }))],
    pairs: [
      { left: 'run', right: 'cohort', count: 120, agreement: .75, kappa: .5, disagreements: 30, matrix: [[50, 10], [20, 40]] },
      ...state.standards.flatMap((item) => [
        { left: 'run', right: item.id, count: 110, agreement: .7, kappa: .4, disagreements: 33, matrix: [[45, 15], [18, 32]] },
        { left: 'cohort', right: item.id, count: 110, agreement: .9, kappa: .8, disagreements: 11, matrix: [[55, 5], [6, 44]] },
      ]),
    ] };
};
scientific.configurations = async () => ({ configurations: [] });
bundles.list = async () => ({ items: [] });
taskCenter.rollup = async (scope) => fixtureRollup({ scope, state: 'finished', counts: { succeeded: 1 }, live: 0, active: 0, pending: 0 });
modelEvaluations.list = async () => ({ items: copy(state.evaluations) });
modelEvaluations.execution = async (_, id) => copy(state.evaluations.find((record) => record.id === id)?.execution ?? { status: 'not_started' });
const edges = Array.from({ length: 21 }, (_, index) => index / 20);
const bins = (at, count) => Array.from({ length: 20 }, (_, index) => index === at ? count : 0);
const quantiles = { p10: .55, p25: .6, median: .72, p75: .85, p90: .92 };
inferenceRuns.summary = async (_, id, query) => {
  state.calls.push({ method: 'summary', id, query: copy(query) });
  return { evaluationId: id, name: 'Existing evaluation', purpose: 'evaluation', predictorId: models[0].id, cohortId: 'cohort', unit: 'patient', task: 'binary_classification', classOrder: ['a', 'b'], positiveClass: 'b', decisionThreshold: .5,
    patientAggregation: 'mean', patients: 120, source: { predictionsSha256: 'f'.repeat(64) }, count: 120,
    predicted: [{ label: 'a', count: 70, fraction: 70 / 120, meanConfidence: .8 }, { label: 'b', count: 50, fraction: 50 / 120, meanConfidence: .7 }],
    confidence: { mean: .72, quantiles, edges, counts: { a: bins(16, 70), b: bins(14, 50) } }, margin: { mean: .4, quantiles, edges, counts: { a: bins(9, 70), b: bins(7, 50) } },
    development: { comparable: false }, attributes: [{ key: 'site', label: 'Site' }] };
};
caseReviews.query = async (_, id, query) => {
  state.calls.push({ method: 'cases', id, query: copy(query) });
  return { purpose: 'evaluation', evaluationId: id, name: 'Existing evaluation', predictorId: models[0].id, cohortId: 'cohort', featureBundleId: null, classOrder: ['a', 'b'], positiveClass: 'b', decisionThreshold: .5, unit: 'patient',
    source: { predictionsSha256: 'f'.repeat(64), comparisonSha256: null }, comparison: null, attributes: [], summary: {}, items: [], total: 0, offset: 0, hasMore: false };
};
const evalManifest = (selection) => ({ ...complete.manifest, ...copy(selection), coverage: { selectedSlideIds: ['a', 'b'], missingFeatureSlideIds: [], missingPackSlideIds: [], packChecked: false } });
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
  const result = { id: 'batch-one', name: selection.namePrefix, status: 'running', cohortId: selection.cohortId, createdAt: '2026-09-29',
    items: selection.predictorIds.map((id, index) => ({ predictorId: id, predictorName: models[index].manifest.name, method: models[index].manifest.method, status: partial && index ? 'failed' : 'running', evaluationId: 'completed-eval', execution: { status: partial && index ? 'not_started' : 'running' } })) };
  state.batches = [result]; return copy(result);
};
const point = { threshold: .5, tp: 40, fp: 10, tn: 50, fn: 20, sensitivity: .67, specificity: .83, ppv: .8, npv: .71, accuracy: .75, balancedAccuracy: .75, f1: .73, positiveLikelihoodRatio: 4, negativeLikelihoodRatio: .4, predictedPositive: 50, predictedNegative: 70, netBenefit: .25, treatAllNetBenefit: 0, treatNoneNetBenefit: 0, standardizedNetBenefit: .5, netInterventionsAvoidedPer100: 25, highRiskPer100: 42, truePositivePer100: 33, falsePositivePer100: 8, missedPositivePer100: 17 };
const report = { unit: 'patient', frozenUnit: 'patient', positiveClass: 'b', classOrder: ['a','b'], multiclass: false, decisionThreshold: .5, frozenDecisionThreshold: .5, thresholdSource: 'frozen_evaluation', counts: { total: 120, labeled: 120, unlabeled: 0, positive: 60, negative: 60, patients: 120, slides: 180, missingPatientIds: 0 }, metrics: { prevalence: .5, brierScore: .2, brierReference: .25, brierSkillScore: .2, logLoss: .4, multiclassBrierScore: null, rocAuc: .8, averagePrecision: .7, ece: .1, mce: .2 }, operatingPoint: point, operatingCurve: [point], rocCurve: [{ threshold: .5, falsePositiveRate: .17, truePositiveRate: .67 }], precisionRecallCurve: [{ threshold: .5, recall: .67, precision: .8 }], calibration: [{ lower: 0, upper: 1, count: 120, meanPredicted: .5, observedFraction: .5, absoluteError: 0 }], warnings: [], definitions: {}, sources: [] };
const clinicalManifest = (selection) => ({ kind: 'clinical-analysis', name: selection.name, datasetId: 'test', experimentId: 'study', predictorId: models[0].id, evaluationId: selection.evaluationId, selection: copy(selection), target: models[0].manifest.target, report });
clinicalAnalyses.list = async () => ({ items: copy(state.reports) });
clinicalAnalyses.get = async (_, id) => copy(state.reports.find((item) => item.id === id));
clinicalAnalyses.preview = async (_, selection) => { state.calls.push({ method: 'clinical-preview', selection: copy(selection) }); return { canSave: true, previewHash: 'clinical-hash', findings: [], manifest: clinicalManifest(selection) }; };
clinicalAnalyses.save = async (_, selection, hash, operation) => { state.calls.push({ method: 'clinical-save', selection: copy(selection), hash, operation }); const saved = { id: 'report-one', createdAt: '2026-09-29', lifecycleState: 'active', contentHash: 'saved', manifest: clinicalManifest(selection) }; state.reports.push(saved); return copy(saved); };
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
window.renderStage = (stage) => root.render(<QueryClientProvider client={client}><div className="stage-workspace">{stage === 'publication-probe' ? <PublicationProbe /> : <LocalApplyModels workspace={{ project: { id: 'project', name: 'Evidence study' } }} />}</div></QueryClientProvider>);
window.renderStage('apply');
`);

await build({ configFile: false, root: web, logLevel: 'error', plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  resolve: { alias: { 'react/jsx-runtime': join(web, 'node_modules/react/jsx-runtime.js') } }, build: {
  outDir: join(output, 'dist'), emptyOutDir: true, minify: false,
  lib: { entry: fixture, name: 'ApplyModelsFixture', formats: ['iife'], fileName: () => 'fixture.js' },
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
  const hash = () => evaluate('window.location.hash');
  const tab = (text) => '[...document.querySelectorAll(".stage-library-tabs button")].find(el => el.textContent.startsWith(' + JSON.stringify(text) + '))';
  const view = (text) => '[...document.querySelectorAll(".stage-views button")].find(el => el.textContent.trim() === ' + JSON.stringify(text) + ')';

  // Runs library: one table of labeled and unlabeled runs, with shared controls.
  await waitFor('document.body.innerText.includes("Existing evaluation")');
  assert.equal(await evaluate('document.querySelector(".stage-library") !== null'), true);
  assert.equal(await evaluate('document.querySelector("#evaluation-experiments-title") === null'), true);
  assert.equal(await evaluate(`document.querySelector('.stage-library button[data-record-key="configuration:completed-eval"]')?.textContent`), 'Manage');
  await assertAction('Apply predictors', 'create');
  assert.equal(await evaluate('document.querySelectorAll(".page-header [data-stage-action=create]").length'), 1);
  assert.equal(await evaluate(tab('Runs') + '.getAttribute("aria-pressed")'), 'true');
  await screenshot('00-runs-library');
  await fill('Search', 'no-such-run');
  await waitFor('document.body.innerText.includes("No matching runs")');
  await click('Clear filters');
  await fill('Method', 'refit', 'select');
  await waitFor('document.body.innerText.includes("No matching runs")');
  await click('Clear filters');
  await fill('Search', 'Existing');

  // A labeled run opens on Performance, its tab follows the link, and clinical utility lives inside it.
  await click('Existing evaluation');
  await waitFor('document.body.innerText.includes("Metrics (JSON)")');
  assert.equal(await hash(), '#apply?run=completed-eval');
  assert.equal(await evaluate('document.querySelector(".stage-library") === null'), true);
  assert.equal(await evaluate(view('Performance') + '.getAttribute("aria-current")'), 'page');
  await waitFor('document.body.innerText.includes("Performance by subgroup")');
  // Recalibration is fitted on development predictions and shown beside the metrics.
  await waitFor('document.body.innerText.includes("Platt scaling")');
  assert.equal(await evaluate('window.workflow.calls.find(call => call.method === "recalibration").referenceId'), null);
  await screenshot('01-run-performance');
  await evaluate(view('Predictions') + '.click()');
  await waitFor('document.body.innerText.includes("Predictions with metadata (CSV)")');
  assert.equal(await hash(), '#apply?run=completed-eval&tab=predictions');
  await evaluate(view('Performance') + '.click()');
  await click('Review errors');
  await waitFor(view('Cases') + '?.getAttribute("aria-current") === "page"');
  await waitFor('window.workflow.calls.some(call => call.method === "cases" && call.query.outcome === "error")', 'error review query');
  await evaluate(view('Performance') + '.click()');
  await click('New analysis');
  await fill('Report name', 'Clinical utility evidence', 'input');
  await click('Analyze clinical utility');
  await waitFor(button('Save clinical utility report'));
  await evaluate(field('I reviewed the selected inputs', 'input') + '.click()');
  await click('Save clinical utility report');
  await waitFor(button('Copy into a new analysis'));
  assert.equal(await evaluate('window.workflow.reports.length'), 1);
  assert.equal(await evaluate('window.workflow.calls.find(call => call.method === "clinical-save").selection.evaluationId'), 'completed-eval');
  await screenshot('02-run-clinical-utility');

  // Labels arriving later: a reference standard scores the same saved predictions.
  assert.equal(await evaluate('document.querySelector(".run-labels select").value'), '');
  await click('Add reference standard');
  await waitFor('document.body.innerText.includes("Add a reference standard")');
  assert.equal(await evaluate('[...document.querySelectorAll(".reference-datasets input")].map(el => el.checked).join()'), 'true,false');
  await fill('Column', 'reader_a', 'select');
  await waitFor('document.querySelector("table[aria-label=\\"Value mapping\\"]")');
  // Values naming a class are mapped to it; the rest stay unlabeled until mapped.
  assert.equal(await evaluate('document.querySelector("select[aria-label=\\"Class for A\\"]").value'), 'a');
  assert.equal(await evaluate('document.querySelector("select[aria-label=\\"Class for ?\\"]").value'), '');
  assert.equal(await evaluate(field('Name', 'input') + '.value'), 'Reader A grade');
  await fill('Name', 'Reader A', 'input');
  await screenshot('03-reference-mapping');
  await click('Review reference standard');
  await waitFor('document.body.innerText.includes("170 of 180 cohort slides labeled")');
  assert.equal(await evaluate('document.body.innerText.includes("10 slides have values the mapping leaves out")'), true);
  await evaluate(field('I reviewed the selected inputs', 'input') + '.click()');
  await click('Save reference standard');
  await waitFor('document.body.innerText.includes("Scored against the reference standard Reader A")');
  const savedReference = await evaluate('window.workflow.calls.find(call => call.method === "reference-save").selection');
  assert.deepEqual(savedReference.labels, { A: 'a', B: 'b' });
  assert.deepEqual(savedReference.datasetIds, ['test']);
  assert.equal(await hash(), '#apply?run=completed-eval&tab=performance&reference=reader-a');
  assert.equal(await evaluate('document.querySelector(".run-labels select").value'), 'reader-a');
  await waitFor('document.body.innerText.includes("with outcomes from Reader A")');
  await waitFor('window.workflow.calls.some(call => call.method === "recalibration" && call.referenceId === "reader-a")', 'recalibration against the reference');
  await screenshot('04-reference-performance');
  await click('New analysis');
  await click('Analyze clinical utility');
  await waitFor(button('Save clinical utility report'));
  assert.equal(await evaluate('window.workflow.calls.filter(call => call.method === "clinical-preview").at(-1).selection.referenceId'), 'reader-a');
  await click('Back to analysis settings');
  await click('Cancel');
  // Agreement pairs the run with every label source; a pair opens its errors against that source.
  await evaluate(view('Agreement') + '.click()');
  await waitFor('document.querySelector(".agreement-matrix")');
  assert.equal(await hash(), '#apply?run=completed-eval&tab=agreement&reference=reader-a');
  await evaluate('document.querySelector("button[aria-label^=\\"Existing evaluation and Reader A\\"]").click()');
  await waitFor('document.body.innerText.includes("33 disagreements")');
  await screenshot('05-agreement');
  await click('Review the run’s errors against Reader A');
  await waitFor(view('Cases') + '?.getAttribute("aria-current") === "page"');
  await waitFor('window.workflow.calls.some(call => call.method === "cases" && call.query.referenceId === "reader-a" && call.query.outcome === "error")', 'error review against the reference');
  // Back to the cohort's own labels: the link drops the reference.
  await evaluate('(() => { const el = document.querySelector(".run-labels select"); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(el, ""); el.dispatchEvent(new Event("change", { bubbles: true })); })()');
  await waitFor('!window.location.hash.includes("reference=")');
  // The same query against the cohort's labels was answered earlier, so the cached cases return.
  await waitFor('document.body.innerText.includes("Inspect predictions with their labels, original slides")', 'cases against the cohort labels');
  assert.equal(await evaluate('document.querySelector(".run-labels select").value'), '');
  await assertAction('Back to runs', 'back');
  await click('Back to runs');
  await waitFor(tab('Runs'));
  assert.equal(await hash(), '#apply');
  assert.equal(await evaluate(field('Search', 'input') + '.value'), 'Existing');
  await click('Clear filters');

  // Cohorts and method comparison are views of the same library.
  await evaluate(tab('Cohorts') + '.click()');
  await waitFor(button('Create labeled cohort'));
  assert.equal(await hash(), '#apply?view=cohorts');
  assert.equal(await evaluate(button('Create unlabeled cohort') + ' !== undefined'), true);
  assert.equal(await evaluate('document.querySelectorAll(".page-header").length'), 1);
  await screenshot('06-cohorts');
  // A frozen cohort lists its reference standards and the runs each one scores.
  await click('Independent hospital');
  await waitFor('document.body.innerText.includes("Reference standards")');
  await waitFor('document.querySelector("a[href=\\"#apply?run=completed-eval&tab=performance&reference=reader-a\\"]")');
  assert.equal(await evaluate('document.body.innerText.includes("170 of 180")'), true);
  await click('Back to cohorts');
  await waitFor(button('Create labeled cohort'));
  await evaluate(tab('Compare methods') + '.click()');
  await waitFor('document.querySelector(".evaluation-comparison")');
  await fill('Cohort', 'cohort', 'select');
  await evaluate(tab('Runs') + '.click()');
  await waitFor('document.body.innerText.includes("Existing evaluation")');

  // Applying predictors: selection, retained inputs across the library, review, lost submission.
  await click('Apply predictors');
  await waitFor('document.querySelector("#evaluation-experiments-title")');
  assert.equal(await hash(), '#apply?view=new');
  await evaluate('[...document.querySelectorAll("input")].find(el => el.getAttribute("aria-label") === "Apply experiment Three-seed ABMIL comparison (study)")?.click()');
  await assertAction('Continue to methods and cohort', 'continue');
  await click('Continue to methods and cohort');
  await fill('Cohort', 'cohort', 'select');
  await waitFor('document.body.innerText.includes("scored against the cohort’s labels")');
  await fill('Batch name (optional)', 'Method comparison', 'input');
  await waitFor('document.body.innerText.includes("No frozen feature bundles are available")');
  await evaluate('[...document.querySelectorAll("details")].find(el => el.querySelector("summary")?.textContent === "Inference settings").open = true');
  await fill('Threshold policy', 'explicit', 'select');
  await fill('Decision threshold', '0.35', 'input');
  await evaluate('window.dispatchEvent(new Event("histopilot:stage-library"))');
  await waitFor(button('Resume setup'));
  await click('Resume setup');
  await waitFor(field('Batch name (optional)', 'input'));
  assert.equal(await evaluate(field('Batch name (optional)', 'input') + '.value'), 'Method comparison');
  await click('Back');
  await click('Continue to methods and cohort');
  assert.equal(await evaluate(field('Batch name (optional)', 'input') + '.value'), 'Method comparison');
  assert.equal(await evaluate(field('Decision threshold', 'input') + '.value'), '0.35');
  await click('Review runs');
  await waitFor('document.body.innerText.includes("compatible predictors will run")');
  await assertAction('Back to methods and cohort', 'back');
  assert.equal(await evaluate(button('Run reviewed predictors') + '.dataset.stageAction'), undefined);
  await screenshot('07-batch-review');
  await evaluate(field('I reviewed the cohort', 'input') + '.click()');
  await click('Run reviewed predictors');
  await waitFor(button('Retry unfinished submissions'));
  assert.equal(await evaluate(button('Back to runs') + '.disabled'), true);
  await click('Retry unfinished submissions');
  await waitFor('document.querySelector("[data-stage-page=results]")');
  const submissions = await evaluate('window.workflow.calls.filter(call => call.method === "bulk-run")');
  assert.equal(submissions.length, 2);
  assert.deepEqual(submissions[0], submissions[1]);
  assert.equal(submissions[0].selection.inference.decisionThreshold, 0.35);
  assert.equal(submissions[0].selection.namePrefix, 'Method comparison');
  await click('Open run');
  await waitFor('document.body.innerText.includes("Metrics (JSON)")');
  assert.equal(await hash(), '#apply?run=completed-eval');

  // Batches, and links that name a view.
  await click('Back to runs');
  await evaluate(tab('Batches') + '.click()');
  await waitFor(button('Method comparison'));
  assert.equal(await evaluate(`document.querySelector('.stage-library button[data-record-key="configuration:batch-one"]')?.textContent`), 'Manage');
  await fill('Search', 'unknown batch');
  await waitFor('document.body.innerText.includes("No matching batches")');
  await click('Clear filters');
  await click('Method comparison');
  await waitFor('document.body.innerText.includes("Runs in this batch")');
  assert.equal(await hash(), '#apply?batch=batch-one');
  await click('Back to batches');
  await waitFor(tab('Batches') + '?.getAttribute("aria-pressed") === "true"');
  await evaluate('window.location.hash = "#apply?run=completed-eval&tab=compare"');
  await waitFor(view('Compare') + '?.getAttribute("aria-current") === "page"');
  await evaluate('window.location.hash = "#apply?view=new&predictor=study-1-11-refit"');
  await waitFor('document.body.innerText.includes("2. Choose methods and cohort")');
  assert.equal(await evaluate('[...document.querySelectorAll("input[name=evaluation-method]")].find(el => el.checked)?.value'), 'refit');
  await evaluate('window.location.hash = "#apply"');
  await waitFor('document.body.innerText.includes("Existing evaluation")');

  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await screenshot('08-runs-library-mobile');
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true);
  await evaluate('window.location.hash = "#apply?run=completed-eval"');
  await waitFor('document.body.innerText.includes("Metrics (JSON)")');
  await screenshot('09-run-mobile');
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
  await writeFile(join(output, 'verification.json'), JSON.stringify({ passed: true, scope: 'Real React and local Chromium with mocked APIs; no backend or HistoPilot server.', checks: ['shared create/back/continue controls', 'runs library search, filters, reset and exact Manage record keys', 'library and run separation, with links naming each view', 'labeled run tabs, error review, recalibration and clinical utility inside Performance', 'a reference standard added later: mapping, review, scores, clinical utility, agreement and cases against it, and back to the cohort labels', 'a frozen cohort lists its references and the runs they score', 'cohorts and method comparison as library views', 'setup inputs retained across the library; lost submissions retried with one request identity', 'batches library and batch runs', 'a linked predictor opens on its methods and cohort', 'pending and uncertain reset cannot erase operation identity', 'mobile containment'], calls: await evaluate('window.workflow.calls') }, null, 2));
  console.log('PASS: Apply models library, run views, clinical utility, reference standards, batch setup and retry locks, and linked views.');
  console.log('Artifacts: ' + output);
} catch (error) {
  try { await writeFile(join(output, 'failure.txt'), await evaluate('document.body.innerText')); await screenshot('failure'); } catch { /* Chromium may not have launched. */ }
  if (exceptions.length) console.error(JSON.stringify(exceptions, null, 2));
  console.error('Artifacts: ' + output);
  throw error;
} finally {
  browser.kill();
}
