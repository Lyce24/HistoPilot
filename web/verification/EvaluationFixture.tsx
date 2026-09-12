/** Real evaluation UI with invented predictors and mocked requests; no service or compute. */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalModelEvaluation from '../src/pages/LocalModelEvaluation';
import type { Workspace } from '../src/api/types';
import type { EvaluationCohort } from '../src/api/evaluation';
import type { EvaluationBatch } from '../src/api/bulkEvaluations';
import { fixturePredictor } from '../src/testFixtures/predictors';
import { fixtureExperiment, fixtureEvaluation } from '../src/testFixtures/evaluations';
import '../src/styles.css';
import '../src/local-workspace.css';
import '../src/scientific.css';
import '../src/clinical-workspace.css';

const project = 'offline-evaluations';
const workspace = { mode: 'local', project: { id: project, name: 'Evaluation review', config: {} } } as Workspace;
const predictors = Array.from({ length: 15 }, (_, configuration) => [11, 22, 33].flatMap((seed) => (['ensemble', 'refit'] as const).map((method) => fixturePredictor(configuration + 1, seed, method)))).flat();
predictors.push(fixturePredictor(1, 11, 'ensemble', 'other'), fixturePredictor(1, 11, 'refit', 'other'));
predictors.forEach((item) => { item.manifest.experiment!.name = item.manifest.experimentId === 'study' ? 'Three-seed ABMIL comparison' : 'Independent study'; item.manifest.candidateNumber = Number(item.manifest.candidateId.split('-').at(-1)); });
const target = predictors[0].manifest.target;
const cohort: EvaluationCohort = { id: 'cohort', projectId: project, contentHash: 'fixture', createdAt: '', current: true, findings: [], versionLabel: { tag: 'External test cohort', note: '', revision: 1, createdAt: '', updatedAt: '' }, manifest: {
  kind: 'evaluation-cohort', datasetId: 'test-data', target, findings: [], spec: { protocolId: 'protocol', developmentFeatureBundleId: 'features', datasetId: 'test-data', featureBundleId: 'test-features', target, eligibility: [], patientIdentifiers: 'shared', inference: { loadingPolicy: 'per_slide', packArtifactId: null, batchSize: 1, numWorkers: 0, device: 'cpu', precision: 'float32', patientAggregation: 'mean', decisionThreshold: 0.5 } }, summary: { includedSlides: 180, includedPatients: 120, excludedSlides: 0, labeledSlides: 180, classCounts: { a: 90, b: 90 }, developmentSlideOverlap: 0, developmentPatientOverlap: 0 },
} };
const experimentRecords = [fixtureExperiment('study'), fixtureExperiment('other'), fixtureExperiment('pending', 'running'), fixtureExperiment('skip-policy')];
experimentRecords[2].name = 'Predictors still running'; experimentRecords[3].name = 'Skip predictor study';
const secondCohort = { ...cohort, id: 'second-cohort', versionLabel: { ...cohort.versionLabel!, tag: 'Independent confirmation cohort' } };
const findPredictor = (id: string) => predictors.find((item) => item.id === id)!;
const results = [
  fixtureEvaluation(findPredictor('study-1-11-ensemble'), 0.7, 0.6), fixtureEvaluation(findPredictor('study-1-11-refit'), 0.8, 0.7),
  fixtureEvaluation(findPredictor('other-1-11-ensemble'), 0.6, 0.5), fixtureEvaluation(findPredictor('other-1-11-refit'), 0.65, 0.55),
  fixtureEvaluation(findPredictor('study-2-11-ensemble'), 0.99, 0.99),
  fixtureEvaluation(findPredictor('study-1-11-ensemble'), 0.5, 0.6, 'second-cohort'), fixtureEvaluation(findPredictor('study-1-11-refit'), 0.55, 0.7, 'second-cohort'),
  fixtureEvaluation(findPredictor('study-1-11-ensemble'), 0.9, 0.9, 'cohort', 'slide'), fixtureEvaluation(findPredictor('study-1-11-refit'), 0.85, 0.85, 'cohort', 'slide'),
];
let loseNextAck = false;
const traffic: { path: string; method: string; body?: Record<string, unknown>; matched: boolean }[] = [];
const records: EvaluationBatch[] = [];
Object.assign(window, { __evaluationReview: { predictors, traffic, records, results, loseNextAck: () => { loseNextAck = true; }, addPredictor: async () => { const item = fixturePredictor(16, 11, 'refit'); item.manifest.candidateNumber = 16; predictors.push(item); await client.invalidateQueries({ queryKey: ['predictors', project] }); } } });
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
window.fetch = async (input, init) => {
  const path = String(input).replace(/^\/api\/v1/, '');
  const method = init?.method ?? 'GET';
  const body = init?.body ? JSON.parse(String(init.body)) : undefined;
  const request = { path, method, body, matched: true }; traffic.push(request);
  const base = `/projects/${project}`;
  const bulk = `${base}/evaluation-runs/bulk`;
  if (path === '/session') return response({ token: 'offline-only' });
  if (path === `${base}/predictors?include_inactive=true`) return response({ items: predictors, executionEnabled: true });
  if (path === `${base}/model-experiments?summary=true`) return response({ items: experimentRecords });
  if (path === `${base}/evaluation-cohorts`) return response({ items: [cohort, secondCohort] });
  if (path === `${base}/evaluation-runs?include_inactive=true`) return response({ items: results, executionEnabled: true });
  if (path === bulk && method === 'GET') return response({ items: records });
  if (path === `${bulk}/preview`) {
    const ids: string[] = body.scope === 'all' ? predictors.map((item) => item.id) : body.predictorIds;
    return response({ canRun: ids.length > 0 && ids.length <= 256, previewHash: 'offline-review', reviewedPredictorIds: ids, eligibleCount: ids.length, blockedCount: 0, items: ids.map((id) => { const item = predictors.find((candidate) => candidate.id === id)!; return { predictorId: id, predictorName: item.manifest.name, method: item.manifest.method, eligible: true, findings: [] }; }) });
  }
  if (path === bulk && method === 'POST') {
    let record = records.find((item) => item.id === body.operationId);
    if (!record) {
      record = { id: body.operationId, status: 'queued', cohortId: body.cohortId, name: body.namePrefix || 'Offline evaluation', items: body.reviewedPredictorIds.map((id: string) => { const item = predictors.find((candidate) => candidate.id === id)!; return { predictorId: id, predictorName: item.manifest.name, method: item.manifest.method, status: 'queued', execution: { status: 'queued' } }; }) };
      records.push(record);
    }
    if (loseNextAck) { loseNextAck = false; return response({ detail: 'Offline fixture lost acknowledgement. Retry the same reviewed request.' }, 503); }
    return response(record);
  }
  if (path.startsWith(`${bulk}/`) && method === 'GET') return response(records.find((item) => path.endsWith(`/${item.id}`)) ?? null);
  request.matched = false;
  return response({ detail: `Unmocked evaluation fixture request: ${method} ${path}` }, 404);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity }, mutations: { retry: false } } });
function Fixture() {
  return <QueryClientProvider client={client}><nav aria-label="Offline evaluation fixtures" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: 12, background: '#e8eef2' }}>
    <strong style={{ padding: 8 }}>Offline evaluation · invented predictors · no compute</strong>
    <button className="btn btn-secondary btn-small" onClick={() => { location.hash = 'evaluate-models?experiment=study'; }}>Linked experiment</button>
    <button className="btn btn-secondary btn-small" onClick={() => { location.hash = 'evaluation?experiment=skip-policy'; }}>Skipped experiment</button>
    <button className="btn btn-secondary btn-small" onClick={() => { location.hash = 'evaluation'; }}>Choose experiments</button>
  </nav><main style={{ padding: 24, maxWidth: 1350, minWidth: 0, margin: '0 auto' }}><LocalModelEvaluation workspace={workspace} /></main></QueryClientProvider>;
}
location.hash = 'evaluate-models?experiment=study';
createRoot(document.getElementById('app')!).render(<Fixture />);
