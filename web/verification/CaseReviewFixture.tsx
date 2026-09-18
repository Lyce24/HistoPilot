/** Browser interaction fixture, invented data only; starts no Python or HTTP server. */
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CaseReviewWorkspace from '../src/components/CaseReviewWorkspace';
import type { ModelEvaluation } from '../src/api/predictors';
import '../src/styles.css';
import '../src/scientific.css';
import '../src/local-workspace.css';

const calls: { path: string; body: unknown }[] = [];
const supportsAttention = new URLSearchParams(location.search).get('mode') !== 'slide';
Object.assign(window, { __morphologyCalls: calls, __caseFixture: { conflictNext: false } });
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const points = ['a', 'b'].map((slideId, index) => ({ slideId, patientId: `patient-${slideId}`, hasImage: true, x: index, y: index, patchCount: 4, sampledPatches: 2, attributes: { site: `Site ${index + 1}` } }));
const patches = points.flatMap((row) => [0, 3].map((patchIndex) => ({ slideId: row.slideId, patchIndex, x: patchIndex ? 200 : 0, y: patchIndex ? 150 : 0 })));
const reviews: Record<string, Record<string, unknown>> = {};
function review(slideId: string) { return reviews[slideId] ?? { schemaVersion: 1, datasetId: 'dataset', slideId, status: 'unreviewed', notes: '', reviewer: '', reasons: [], regions: [], evaluationId: null, revision: 0, updatedAt: null, history: [] }; }
const evaluation = { id: 'evaluation', manifest: { name: 'Primary model', cohortId: 'cohort', target: { classes: ['Negative', 'Positive'] } }, execution: { status: 'completed' } } as unknown as ModelEvaluation;
const comparisons = [{ id: 'comparison', manifest: { name: 'Comparison model', cohortId: 'cohort' }, execution: { status: 'completed' } }] as unknown as ModelEvaluation[];
function cases(body: Record<string, unknown>) {
  const slides = (names: string[]) => names.map(slideId => ({ datasetId: 'dataset', slideId, slidePath: slideId === 'missing' ? null : `/exact/${slideId}.png`, hasImage: slideId !== 'missing', attributes: { site: 'A' }, review: review(slideId) }));
  let items = [
    { id: 'patient-one', patientId: 'patient-one', slideIds: ['a', 'b'], label: 'Positive', labelIndex: 1, probabilities: [.8, .2], predictedIndex: 0, predictedLabel: 'Negative', confidence: .8, outcome: 'false_negative', attributes: { site: ['A'] }, slides: slides(['a', 'b']) },
    { id: 'patient-two', patientId: 'patient-two', slideIds: ['missing'], label: 'Negative', labelIndex: 0, probabilities: [.1, .9], predictedIndex: 1, predictedLabel: 'Positive', confidence: .9, outcome: 'false_positive', attributes: { site: ['B'] }, slides: slides(['missing']) },
  ].map(item => ({ ...item, comparison: body.comparisonId ? { probabilities: [.1, .9], predictedIndex: 1, predictedLabel: 'Positive', disagrees: item.id === 'patient-one' } : null }));
  if (body.outcome !== 'all') items = items.filter(item => body.outcome === 'disagreement' ? item.comparison?.disagrees : item.outcome === body.outcome);
  return { evaluationId: 'evaluation', name: 'Primary model', predictorId: 'predictor', supportsAttention, cohortId: 'cohort', featureBundleId: supportsAttention ? null : 'slide-bundle', classOrder: ['Negative', 'Positive'], positiveClass: 'Positive', decisionThreshold: .5, unit: 'patient', source: { predictionsSha256: 'hash', comparisonSha256: body.comparisonId ? 'comparison-hash' : null }, comparison: body.comparisonId ? { id: 'comparison', name: 'Comparison model', predictorId: 'comparison-predictor', supportsAttention, decisionThreshold: .5 } : null, attributes: [{ key: 'site', label: 'Site', values: ['A', 'B'], valuesLimited: false }], summary: { total: 2 }, items, total: items.length, offset: 0, hasMore: false };
}
window.fetch = async (input, init) => {
  const url = new URL(String(input), 'http://offline.invalid'), path = url.pathname;
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  calls.push({ path: url.pathname + url.search, body });
  if (path.endsWith('/cases/query')) return reply(cases(body));
  if (path.endsWith('/interpretations')) return reply({ items: [] });
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (path.endsWith('/feature-bundles')) return reply({ items: [{ id: 'bundle', current: true, versionLabel: { tag: 'Verified test features' }, manifest: { datasetId: 'dataset' } }] });
  if (path.endsWith('/morphology/slides')) return reply({ total: 2, matching: 2, offset: 0, items: points.map((row) => ({ ...row, reviewStatus: review(row.slideId).status })) });
  if (path.endsWith('/morphology/index')) return reply({ indexId: 'index', datasetId: 'dataset', featureBundleId: 'bundle', encoderId: 'test-encoder', candidateSlides: 2, indexedSlides: 2, indexedPatches: 4, explainedVariance: [1, 0], method: 'PCA of sampled feature means', sampling: 'Evenly spaced original indices', points, patches, warnings: ['Temporary sampled exploration index.'] });
  if (path.endsWith('/morphology/neighbors')) return reply({ candidateCount: 1, scope: body.mode === 'patch' ? 'sampled_patches' : 'indexed_slides', items: [{ ...points.find((point) => point.slideId !== body.slideId), similarity: .92, ...(body.mode === 'patch' ? { patchIndex: 3 } : {}) }] });
  if (path.endsWith('/morphology/quality')) return reply({ slideId: url.searchParams.get('slideId'), datasetId: 'dataset', width: 6000, height: 4000, featureKind: url.searchParams.get('featureBundleId') === 'slide-bundle' ? 'slide' : undefined, patchWidth: null, patchHeight: null, patchCount: null, patches: [], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] });
  if (path.endsWith('/morphology/patch-region')) return reply({ x: 200, y: 150, width: 150, height: 120 });
  if (path.endsWith('/morphology/image') || path.endsWith('/morphology/patch')) return new Response('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><rect width="600" height="400" fill="#eee0e7"/><ellipse cx="260" cy="210" rx="230" ry="160" fill="#d19bba"/><text x="120" y="210" font-size="20">Synthetic browser fixture</text></svg>', { headers: { 'Content-Type': 'image/svg+xml' } });
  if (path.includes('/slide-reviews/')) {
    const slideId = decodeURIComponent(path.split('/').at(-1)!);
    if (init?.method === 'PUT') {
      let existing = review(slideId);
      const state = (window as unknown as { __caseFixture: { conflictNext: boolean } }).__caseFixture;
      if (state.conflictNext) { state.conflictNext = false; reviews[slideId] = { ...existing, revision: Number(existing.revision) + 1, notes: 'Other reviewer note', status: 'accept' }; existing = review(slideId); }
      if (body.expectedRevision !== existing.revision) return reply({ code: 'SLIDE_REVIEW_CONFLICT', detail: 'Review changed' }, 409);
      reviews[slideId] = { ...existing, ...body, revision: Number(existing.revision) + 1, updatedAt: new Date().toISOString() };
    }
    return reply(review(slideId));
  }
  if (path.endsWith('/slide-reviews')) return reply({ items: Object.values(reviews), total: Object.keys(reviews).length, offset: 0, hasMore: false });
  throw new Error(`Unexpected offline request: ${path}`);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
createRoot(document.getElementById('app')!).render(<QueryClientProvider client={client}><main style={{ maxWidth: 1280, margin: '20px auto', padding: 16 }}><p>Offline case-review verification · invented data · no server</p><CaseReviewWorkspace project="offline" evaluation={evaluation} comparisons={comparisons} /></main></QueryClientProvider>);
