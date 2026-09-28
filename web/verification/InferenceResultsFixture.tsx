/** Offline browser fixture for Run inference: invented BD-like data, mocked API, no server. */
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalInference from '../src/pages/LocalInference';
import type { Workspace } from '../src/api/types';
import '../src/styles.css';
import '../src/scientific.css';
import '../src/local-workspace.css';
import '../src/clinical-workspace.css';

const classes = ['ND', 'IND', 'LG', 'HG'];
const members = 5;
let seed = 20260925;
const random = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
function normalize(values: number[]) { const total = values.reduce((sum, value) => sum + value, 0); return values.map((value) => value / total); }
function vector(center: number, sharpness: number) { return normalize(classes.map((_, index) => Math.exp(-Math.abs(index - center) * sharpness) * (0.5 + random()))); }
const argmax = (values: number[]) => values.indexOf(Math.max(...values));
interface Row { slideId: string; patientId: string; probabilities: number[]; members: number[][]; shared: boolean; status: string; part: string }
const rows: Row[] = Array.from({ length: 383 }, (_, index) => {
  const center = [0, 0, 0, 1, 2, 3, 3, 2][Math.floor(random() * 8)] + (random() - 0.5) * 1.2;
  const sharpness = 0.8 + random() * 2.2;
  const memberVectors = Array.from({ length: members }, () => vector(center + (random() - 0.5) * 0.9, sharpness));
  const probabilities = normalize(classes.map((_, column) => memberVectors.reduce((sum, row) => sum + row[column], 0)));
  const patient = Math.floor(index / 1.5);
  return { slideId: `Slide ${300 + index}${'ABC'[index % 3]}`, patientId: String(patient), probabilities, members: memberVectors, shared: patient % 7 < 4, status: index % 4 ? '-1' : 'N/A', part: 'ABCD'[index % 4] };
});
function described(row: Row) {
  const index = argmax(row.probabilities);
  const ordered = [...row.probabilities].sort((a, b) => b - a);
  const votes = row.members.filter((values) => argmax(values) === index).length;
  const chosen = row.members.map((values) => values[index]);
  const mean = chosen.reduce((sum, value) => sum + value, 0) / chosen.length;
  return { index, label: classes[index], confidence: row.probabilities[index], margin: ordered[0] - ordered[1], agree: votes, spread: Math.sqrt(chosen.reduce((sum, value) => sum + (value - mean) ** 2, 0) / chosen.length) };
}
const edges = Array.from({ length: 21 }, (_, index) => index / 20);
const bin = (value: number) => Math.min(19, Math.floor(value * 20));
function histogram(values: { label: string; value: number }[]) {
  const counts = Object.fromEntries(classes.map((label) => [label, Array(20).fill(0) as number[]]));
  for (const item of values) counts[item.label][bin(item.value)] += 1;
  const sorted = values.map((item) => item.value).sort((a, b) => a - b);
  const q = (fraction: number) => sorted[Math.min(sorted.length - 1, Math.round((sorted.length - 1) * fraction))];
  return { mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length, quantiles: { p10: q(0.1), p25: q(0.25), median: q(0.5), p75: q(0.75), p90: q(0.9) }, edges, counts };
}
const counts = (items: ReturnType<typeof described>[]) => Object.fromEntries(classes.map((label) => [label, items.filter((item) => item.label === label).length]));
function summary(id: string, body: { attribute?: string | null; comparisonId?: string | null }) {
  const items = rows.map(described);
  const flags = rows.map((row) => row.shared);
  const agreement = Array.from({ length: members + 1 }, (_, offset) => members - offset).map((agree) => ({ agree, count: items.filter((item) => item.agree === agree).length }));
  const breakdown = body.attribute ? (() => {
    const key = body.attribute === 'part' ? 'part' : 'status';
    const values = [...new Set(rows.map((row) => row[key]))];
    return { attribute: body.attribute, label: key === 'part' ? 'Part' : 'Reader consensus', otherValues: 0, rows: values.map((value) => { const group = items.filter((_, index) => rows[index][key] === value); return { value, count: group.length, counts: counts(group), meanConfidence: group.reduce((sum, item) => sum + item.confidence, 0) / group.length }; }).sort((a, b) => b.count - a.count) };
  })() : undefined;
  const comparison = body.comparisonId ? (() => {
    const right = rows.map((row) => argmax(row.members[0]));
    const matrix = classes.map((_, left) => classes.map((__, column) => items.filter((item, index) => item.index === left && right[index] === column).length));
    const same = matrix.reduce((sum, row, index) => sum + row[index], 0);
    const expected = classes.reduce((sum, _, index) => sum + (matrix[index].reduce((a, b) => a + b, 0) / rows.length) * (matrix.reduce((a, row) => a + row[index], 0) / rows.length), 0);
    return { evaluationId: 'refit', name: 'BD refit', predictorId: 'predictor-refit', decisionThreshold: 0.5, predictionsSha256: 'b'.repeat(64), count: rows.length, agreement: same / rows.length, kappa: (same / rows.length - expected) / (1 - expected), disagreements: rows.length - same, matrix };
  })() : undefined;
  return {
    evaluationId: id, name: 'BD ensemble · fold 5×', purpose: 'inference', predictorId: 'predictor', cohortId: 'cohort', unit: 'slide', task: 'multiclass_classification', classOrder: classes, positiveClass: null,
    decisionThreshold: 0.5, patientAggregation: 'mean', patients: new Set(rows.map((row) => row.patientId)).size, source: { predictionsSha256: 'a'.repeat(64) }, count: rows.length,
    predicted: classes.map((label) => { const group = items.filter((item) => item.label === label); return { label, count: group.length, fraction: group.length / rows.length, meanConfidence: group.length ? group.reduce((sum, item) => sum + item.confidence, 0) / group.length : null }; }),
    confidence: histogram(items.map((item) => ({ label: item.label, value: item.confidence }))),
    margin: histogram(items.map((item) => ({ label: item.label, value: item.margin }))),
    ensemble: { memberCount: members, records: rows.length, unanimous: agreement[0].count, disagreements: rows.length - agreement[0].count, meanSpread: items.reduce((sum, item) => sum + item.spread, 0) / rows.length, agreement },
    development: { comparable: true, patients: new Set(rows.filter((row) => row.shared).map((row) => row.patientId)).size, records: flags.filter(Boolean).length, shared: counts(items.filter((_, index) => flags[index])), new: counts(items.filter((_, index) => !flags[index])) },
    attributes: [{ key: 'consensus', label: 'Reader consensus' }, { key: 'part', label: 'Part' }],
    ...(breakdown ? { breakdown } : {}), ...(comparison ? { comparison } : {}),
  };
}
const calls: { path: string; body: unknown }[] = [];
const unexpected: string[] = [];
let attentionQueued = false;
Object.assign(window, { __inferenceCalls: calls, __unexpected: unexpected, __summaryFailure: false, __summaryDelay: 0 });
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
function cases(body: Record<string, unknown>) {
  let items = rows.map((row) => ({ row, item: described(row) }));
  if (typeof body.minConfidence === 'number') items = items.filter(({ item }) => item.confidence >= (body.minConfidence as number));
  if (typeof body.maxMargin === 'number') items = items.filter(({ item }) => item.margin < (body.maxMargin as number));
  if (typeof body.maxConfidence === 'number') items = items.filter(({ item }) => item.confidence <= (body.maxConfidence as number));
  if (body.developmentPatients === 'shared') items = items.filter(({ row }) => row.shared);
  if (body.developmentPatients === 'new') items = items.filter(({ row }) => !row.shared);
  if (body.memberDisagreement) items = items.filter(({ item }) => item.agree < members);
  const sort = body.sort ?? 'confidence_desc';
  items.sort((a, b) => sort === 'confidence_asc' ? a.item.confidence - b.item.confidence : sort === 'margin_asc' ? a.item.margin - b.item.margin : sort === 'agreement_asc' ? a.item.agree - b.item.agree || b.item.spread - a.item.spread : b.item.confidence - a.item.confidence);
  const offset = Number(body.offset ?? 0), limit = Number(body.limit ?? 30);
  return {
    evaluationId: 'ensemble', name: 'BD ensemble · fold 5×', predictorId: 'predictor', purpose: 'inference', memberCount: members, developmentComparable: true, supportsAttention: true, cohortId: 'cohort', featureBundleId: 'bundle',
    classOrder: classes, positiveClass: null, decisionThreshold: 0.5, unit: 'slide', source: { predictionsSha256: 'a'.repeat(64), comparisonSha256: null }, comparison: null,
    attributes: [{ key: 'consensus', label: 'Reader consensus', values: ['-1', 'N/A'], valuesLimited: false }], summary: { total: rows.length, unlabeled: rows.length },
    total: items.length, offset, hasMore: offset + limit < items.length,
    items: items.slice(offset, offset + limit).map(({ row, item }) => ({ id: row.slideId, patientId: row.patientId, slideIds: [row.slideId], label: null, labelIndex: null, probabilities: row.probabilities, predictedIndex: item.index, predictedLabel: item.label, confidence: item.confidence, margin: item.margin,
      memberAgreement: { agree: item.agree, total: members, spread: item.spread }, developmentPatient: row.shared, outcome: 'unlabeled', comparison: null, attributes: { consensus: [row.status] },
      slides: [{ datasetId: 'dataset', slideId: row.slideId, slidePath: `/slides/${row.slideId}.sdpc`, hasImage: true, attributes: { consensus: row.status }, review: { schemaVersion: 1, datasetId: 'dataset', slideId: row.slideId, status: 'unreviewed', notes: '', reviewer: '', reasons: [], regions: [], evaluationId: null, revision: 0, updatedAt: null } }] })),
  };
}
const run = (id: string, name: string, predictorId: string) => ({ id, createdAt: '2026-09-25T12:00:00Z', contentHash: id, lifecycleState: 'active', manifest: { kind: 'model-evaluation', purpose: 'inference', name, status: 'planned', predictorId, cohortId: 'cohort', experimentId: 'experiment', overlap: { slideIds: [], patientIds: Array.from({ length: 146 }, (_, index) => String(index)), patientsComparable: true } },
  execution: { status: 'completed', result: { purpose: 'inference', slideCount: rows.length, summary: { purpose: 'inference', unit: 'slide', classOrder: classes, decisionThreshold: 0.5, patientAggregation: 'mean_probabilities', memberCount: members, slide: summary(id, {}), patient: { available: false, count: 0, reason: 'fixture' }, selected: summary(id, {}) } } } });
const predictor = (id: string, method: 'ensemble' | 'refit') => ({ id, createdAt: '', contentHash: id, lifecycleState: 'active', manifest: { kind: 'frozen-predictor', name: `ABMIL ${method}`, method, experimentId: 'experiment', batchId: 'batch', candidateId: 'candidate', trainingSeed: 11, splitSeed: 42, checkpoints: [], target: { field: 'Reader consensus Grades', task: 'multiclass_classification', unit: 'slide', classes, labels: {}, missing: 'block', unmapped: 'block' }, recipe: { model: 'abmil' }, experiment: { id: 'experiment', name: 'BD grading ABMIL' } } });
window.fetch = async (input, init) => {
  const url = new URL(String(input), 'http://offline.invalid'), path = url.pathname;
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  calls.push({ path: path + url.search, body });
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (path.endsWith('/evaluation-runs/bulk')) return reply({ items: [] });
  if (path.endsWith('/predictors')) return reply({ items: [predictor('predictor', 'ensemble'), predictor('predictor-refit', 'refit')] });
  if (path.endsWith('/model-experiments')) return reply({ items: [] });
  if (path.endsWith('/evaluation-cohorts')) return reply({ items: [{ id: 'cohort', projectId: 'offline', contentHash: 'c', createdAt: '', current: true, versionLabel: { tag: 'Unlabeled slides' }, manifest: { kind: 'evaluation-cohort', datasetId: 'dataset', spec: { purpose: 'review', datasetId: 'dataset', target: null, eligibility: [], patientIdentifiers: 'shared' }, target: null, summary: { includedSlides: rows.length, includedPatients: 256, excludedSlides: 728, labeledSlides: 0, classCounts: {}, developmentSlideOverlap: 0, developmentPatientOverlap: 0 }, findings: [] } }] });
  if (path.endsWith('/evaluation-runs')) return reply({ items: [run('ensemble', 'BD ensemble · fold 5×', 'predictor'), run('refit', 'BD refit', 'predictor-refit')], executionEnabled: true });
  if (path.endsWith('/execution')) return reply(run(path.split('/').at(-2)!, '', '').execution);
  if (path.endsWith('/inference/summary')) {
    const controls = window as unknown as { __summaryDelay: number; __summaryFailure: boolean };
    if (controls.__summaryDelay) await new Promise((resolve) => setTimeout(resolve, controls.__summaryDelay));
    if (controls.__summaryFailure) return reply({ code: 'PREDICTIONS_CHANGED', detail: 'Saved predictions failed verification.' }, 409);
    return reply(summary(path.split('/').at(-3)!, body));
  }
  if (path.endsWith('/cases/query')) return reply(cases(body));
  if (path.endsWith('/attention')) { attentionQueued = true; return reply({ evaluationId: 'ensemble', items: body.slideIds.map((slideId: string) => ({ slidePath: `/slides/${slideId}.sdpc`, slideId, interpretationId: `study-${slideId}`, status: 'queued', reused: false })), interpretations: [] }, 202); }
  if (path.endsWith('/interpretations')) return reply({ items: attentionQueued ? rows.slice(0, 400).map((row) => ({ id: `study-${row.slideId}`, createdAt: '', contentHash: '', lifecycleState: 'active', manifest: { kind: 'model-interpretation', predictorId: 'predictor', evaluationId: 'ensemble', featureBundleId: 'bundle', packArtifactId: null, experimentId: 'experiment', slides: [{ slideId: row.slideId, slidePath: `/slides/${row.slideId}.sdpc` }] }, execution: { status: 'queued' } })) : [] });
  if (path.endsWith('/morphology/quality')) return reply({ slideId: url.searchParams.get('slideId'), datasetId: 'dataset', width: 6000, height: 4000, patchWidth: null, patchHeight: null, patchCount: null, patches: [], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] });
  if (path.endsWith('/morphology/image')) return new Response('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><rect width="600" height="400" fill="#eee0e7"/><ellipse cx="260" cy="210" rx="230" ry="160" fill="#d19bba"/><text x="120" y="210" font-size="20">Synthetic browser fixture</text></svg>', { headers: { 'Content-Type': 'image/svg+xml' } });
  if (path.includes('/slide-reviews/')) { const slideId = decodeURIComponent(path.split('/').at(-1)!); return reply({ schemaVersion: 1, datasetId: 'dataset', slideId, status: 'unreviewed', notes: '', reviewer: '', reasons: [], regions: [], evaluationId: null, revision: 0, updatedAt: null, history: [] }); }
  unexpected.push(path + url.search);
  return reply({ items: [] });
};
const view = new URLSearchParams(location.search).get('view');
window.location.hash = view === 'library' ? '#inference' : '#inference?evaluation=ensemble';
const workspace = { mode: 'local', project: { id: 'offline', name: 'BD offline fixture' } } as unknown as Workspace;
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
createRoot(document.getElementById('app')!).render(<QueryClientProvider client={client}><main className="content module-content stage-workspace" style={{ maxWidth: 1280, margin: '0 auto', padding: 16 }}><LocalInference workspace={workspace} /></main></QueryClientProvider>);
