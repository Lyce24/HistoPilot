/** Interactive offline fixture: real experiment UI, invented records, no network or workers. */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalExperiments from '../src/pages/LocalExperiments';
import type { Workspace } from '../src/api/types';
import type { ExperimentBatch, ExperimentBatchPlan, ModelExperiment } from '../src/api/experiments';
import { defaultRecipe, defaultResources, type BatchPreview, type DevelopmentBatchSpec, type TrainingExecution } from '../src/api/development';
import '../src/styles.css';
import '../src/local-workspace.css';
import '../src/scientific.css';
import '../src/clinical-workspace.css';

const project = 'offline-experiments';
const stamp = new Date().toISOString();
const inputs = { protocolId: 'protocol-review', featureBundleId: 'bundle-review', loadingPolicy: 'native' as const, packArtifactId: null };
const workspace = { mode: 'local', project: { id: project, name: 'Experiment review', config: {} } } as Workspace;
const protocol = { id: inputs.protocolId, createdAt: stamp, versionLabel: { tag: 'Cancer subtype · 5-fold CV', note: '' }, manifest: {
  kind: 'protocol', version: 4, datasetId: 'dataset-review', spec: { target: { task: 'classification', field: 'subtype', unit: 'patient' }, split: { mode: 'k-fold', seeds: [42], folds: 5 } },
} };
const bundle = { id: inputs.featureBundleId, createdAt: stamp, current: true, findings: [], versionLabel: { tag: 'Verified slide features', note: '' }, manifest: {
  datasetId: 'dataset-review', spec: { featureSetId: 'features-review', packArtifactIds: [] }, packs: [], summary: { slideCount: 180, patchCount: 240000, dimensions: 1024, dtype: 'float32', packCount: 0 },
} };
const resolvedInputs = { canPlan: true, findings: [], resolvedLoadingPolicy: 'native' as const, packArtifactId: null, featureSetId: 'features-review', bundleId: inputs.featureBundleId, executionImplemented: true };
const host = { cpuCount: 24, totalRamGb: 128, availableRamGb: 91, bootId: 'fixture', kernel: 'offline fixture' };
const gpu = { index: 0, uuid: 'fixture-gpu', name: 'Fixture GPU', driverVersion: 'offline', totalMemoryGb: 24, usedMemoryGb: 8, freeMemoryGb: 16, utilizationPercent: 74 };
const runtime = { available: true, python: '/offline/python', versions: { torch: 'fixture' }, cudaAvailable: true, gpuCount: 1, findings: [], host, gpus: [gpu] };
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } });
const records: Record<string, ModelExperiment> = {};
const traffic: { path: string; method: string; matched: boolean; body?: unknown }[] = [];
const submissions: { operationId: string; expectedRevision: number }[] = [];
let loseSubmission = false;

function plan(id: string, name: string, rate = 0.0003): ExperimentBatchPlan {
  return { id, spec: { version: 1, experimentName: 'Fixture', batchName: name, inputs, recipe: { ...defaultRecipe(), learningRate: rate, maxEpochs: 20 }, mode: 'single', grid: { learningRates: [rate], weightDecays: [0.0001], maxEpochs: [20] }, configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' } };
}
function preview(spec: DevelopmentBatchSpec): BatchPreview {
  return { kind: 'mil-batch', version: 1, datasetId: 'dataset-review', spec, configurations: [{ id: 'candidate-1', number: 1, recipe: spec.recipe }], splitPlans: Array.from({ length: 5 }, (_, i) => ({ id: `split-${i}`, planId: `split-${i}`, seed: 42, fold: i, phase: 'development', slideCount: 180, partitions: { training: 108, validation: 36, assessment: 36 } })), runs: Array.from({ length: 5 }, (_, i) => ({ id: `run-${i}`, candidateId: 'candidate-1', trainingSeed: 42, splitPlanId: `split-${i}`, status: 'planned' })), summary: { configurationCount: 1, trainingSeedCount: 1, splitPlanCount: 5, runCount: 5 }, executionImplemented: true, previewHash: 'offline-reviewed', resolvedInputs, canFreeze: true, findings: [] };
}
function execution(batchId: string, finished: boolean): TrainingExecution {
  return { batchId, status: finished ? 'completed' : 'running', sessionName: 'offline-no-worker', logPath: '/offline/worker.log', outputPath: '/offline/output', findings: [], createdAt: stamp, updatedAt: stamp,
    runCounts: { total: 5, queued: finished ? 0 : 2, running: finished ? 0 : 1, completed: finished ? 5 : 2, failed: 0, cancelled: 0, interrupted: 0 },
    runs: Array.from({ length: 5 }, (_, i) => ({ id: `run-${i}`, candidateId: 'candidate-1', trainingSeed: 42, splitPlanId: `split-${i}`, status: finished || i < 2 ? 'completed' : i === 2 ? 'running' : 'queued', progress: finished || i < 2 ? { epoch: 20, maxEpochs: 20, globalStep: 2000, trainingLoss: 0.18, validation: { loss: 0.3, accuracy: 0.82, auroc: 0.89 }, learningRate: 0.0003 } : i === 2 ? { epoch: 8, maxEpochs: 20, globalStep: 800, trainingLoss: 0.42, validation: { loss: 0.48, accuracy: 0.76, auroc: 0.84 }, learningRate: 0.0003 } : null })),
    resourcePlan: { requestedConcurrency: 1, effectiveConcurrency: 1, cpuSlotsPerRun: 4, cpuLimit: 6, ramLimit: 10, gpuSlotLimit: 1, note: 'Fixture resources' },
    telemetry: { path: '/offline/telemetry.jsonl', intervalSeconds: 3, latest: { at: stamp, host, gpus: [gpu], runs: finished ? [] : [{ runId: 'run-2', pid: 999, rssGb: 4.1 }] }, peak: { hostUsedRamGb: 37, runRssGb: { 'run-2': 4.8 }, gpuUsedMemoryGb: { '0': 8 } } },
  };
}
function batch(owner: string, source: ExperimentBatchPlan, finished: boolean): ExperimentBatch {
  const id = `${owner}-${source.id}`;
  return { id, key: `configuration:${id}`, name: source.spec.batchName, state: 'active', status: finished ? 'completed' : 'running', createdAt: stamp, manifest: preview({ ...source.spec, experimentId: owner }), execution: execution(id, finished) };
}
function make(id: string, name: string, stage: 'planning' | 'running' | 'finished'): ModelExperiment {
  const plans = [plan('baseline', 'Baseline'), plan('low-rate', 'Lower learning rate', 0.0001)];
  const batches = stage === 'planning' ? [] : plans.map((item) => batch(id, item, stage === 'finished'));
  return { id, key: `draft:${id}`, name, notes: 'Compare learning rates using the same frozen development splits.', tags: ['abmil', 'baseline'], revision: 1, state: 'active', status: stage === 'planning' ? 'planned' : stage === 'finished' ? 'completed' : 'running', stage, configurationLocked: stage !== 'planning', createdAt: stamp, updatedAt: stamp, inputs, batches, batchPlans: plans, drafts: [], legacy: false, predictorId: null, executionImplemented: true, submission: stage === 'planning' ? null : { operationId: `submit-${id}`, expectedRevision: 1, submittedAt: stamp, status: 'submitted', batchIds: batches.map((item) => item.id), error: null, retryable: false } };
}
records.planning = make('planning', 'Learning-rate comparison', 'planning');
records.running = make('running', 'ABMIL running example', 'running');
records.finished = make('finished', 'ABMIL finished template', 'finished');

function refresh() { void client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) }); }
function open(id: string) { window.location.hash = id ? `experiments?experiment=${id}` : 'experiments'; }
function finish() {
  const id = new URLSearchParams(window.location.hash.split('?')[1]).get('experiment') ?? '';
  const record = records[id];
  if (!record || record.stage !== 'running') return;
  record.stage = 'finished'; record.status = 'completed';
  record.batches.forEach((item) => { item.status = 'completed'; item.execution = execution(item.id, true); });
  refresh();
}
Object.assign(window, { __experimentReview: { records, traffic, submissions, finish, refresh } });
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
window.fetch = async (input, init) => {
  const path = String(input).replace(/^\/api\/v1/, '');
  const method = init?.method ?? 'GET';
  const body = init?.body ? JSON.parse(String(init.body)) : undefined;
  const request = { path, method, body, matched: true }; traffic.push(request);
  const base = `/projects/${project}`;
  if (path === '/session') return response({ token: 'offline-only' });
  if (path === `${base}/configurations?kind=protocol`) return response({ configurations: [protocol] });
  if (path === `${base}/feature-bundles`) return response({ items: [bundle] });
  if (path === `${base}/mil-experiments/runtime`) return response(runtime);
  if (path === `${base}/mil-experiments/preview`) return response(resolvedInputs);
  if (path === `${base}/mil-experiments/batches/preview`) return response(preview(body));
  if (path === `${base}/model-experiments?summary=true`) return response({ items: Object.values(records) });
  if (path === `${base}/model-experiments` && method === 'POST') {
    const id = `copy-${body.operationId}`;
    if (records[id]) return response(records[id]);
    const source = records[body.sourceExperimentId];
    records[id] = { ...make(id, body.name, 'planning'), notes: body.notes ?? '', tags: body.tags ?? [], inputs: source ? structuredClone(source.inputs) : null, batchPlans: source ? structuredClone(source.batchPlans ?? []) : [] };
    return response(records[id]);
  }
  const selected = new RegExp(`^${base}/model-experiments/([^/]+)(/submit)?$`).exec(path);
  if (selected) {
    const record = records[decodeURIComponent(selected[1])];
    if (!record) return response({ code: 'NOT_FOUND', detail: 'Fixture experiment not found' }, 404);
    if (selected[2] && method === 'POST') {
      submissions.push(body);
      if (!record.submission) {
        if (body.expectedRevision !== record.revision) return response({ code: 'STALE_EXPERIMENT', detail: 'Saved experiment changed' }, 409);
        record.batches = (record.batchPlans ?? []).map((item) => batch(record.id, item, false));
        record.stage = 'running'; record.status = 'running'; record.configurationLocked = true; record.revision += 1;
        record.submission = { ...body, submittedAt: stamp, status: 'submitted', batchIds: record.batches.map((item) => item.id), error: null, retryable: false };
      }
      if (loseSubmission) { loseSubmission = false; throw new TypeError('Fixture accepted submission; response deliberately lost.'); }
      return response(record);
    }
    if (method === 'PATCH') {
      if (body.expectedRevision !== record.revision) return response({ code: 'STALE_EXPERIMENT', detail: 'Saved experiment changed' }, 409);
      if (record.configurationLocked && (body.inputs || body.batchPlans)) return response({ code: 'EXPERIMENT_LOCKED', detail: 'Submitted configuration is immutable' }, 409);
      for (const field of ['name', 'notes', 'tags', 'inputs', 'batchPlans'] as const) if (field in body) Object.assign(record, { [field]: body[field] });
      record.revision += 1;
    }
    return response(record);
  }
  const run = new RegExp(`^${base}/mil-experiments/batches/([^/]+)/(execution|results|runs/[^/]+/history)$`).exec(path);
  if (run) {
    const item = Object.values(records).flatMap((record) => record.batches).find((candidate) => candidate.id === run[1]);
    if (!item) return response({ detail: 'Fixture batch not found' }, 404);
    if (run[2] === 'execution') return response(item.execution);
    if (run[2] === 'results') return response({ batchId: item.id, status: item.status, oof: [], findings: [], candidates: [{ candidateId: 'candidate-1', trainingSeed: 42, splitSeed: 42, complete: true, completedRuns: 5, totalRuns: 5, metrics: { available: true, auroc: 0.891, accuracy: 0.822 } }] });
    return response({ runId: run[2].split('/')[1], totalRows: 8, truncated: false, rows: Array.from({ length: 8 }, (_, i) => ({ epoch: i + 1, trainingLoss: 0.88 - i * 0.06, validation: { loss: 0.9 - i * 0.055, accuracy: 0.6 + i * 0.025, auroc: 0.65 + i * 0.025 }, learningRate: 0.0003, checkpointUnit: 'patient' })) });
  }
  request.matched = false;
  return response({ detail: `Unmocked fixture request: ${method} ${path}` }, 404);
};
function Fixture() {
  return <QueryClientProvider client={client}><nav aria-label="Offline experiment fixtures" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: 12, background: '#e8eef2' }}>
    <strong style={{ padding: 8 }}>Offline fixtures · no server or training</strong>
    <button className="btn btn-secondary btn-small" onClick={() => open('')}>Experiment list</button>
    <button className="btn btn-secondary btn-small" onClick={() => open('planning')}>Planning example</button>
    <button className="btn btn-secondary btn-small" onClick={() => open('running')}>Running example</button>
    <button className="btn btn-secondary btn-small" onClick={() => open('finished')}>Finished example</button>
    <button className="btn btn-secondary btn-small" onClick={finish}>Finish selected fixture</button>
    <button className="btn btn-secondary btn-small" onClick={() => { loseSubmission = true; }}>Lose next submission response</button>
  </nav><main style={{ padding: '24px', maxWidth: 1350, margin: '0 auto', minWidth: 0 }}><LocalExperiments workspace={workspace} /></main></QueryClientProvider>;
}
window.location.hash = 'experiments';
createRoot(document.getElementById('app')!).render(<Fixture />);
