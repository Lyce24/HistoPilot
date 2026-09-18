/** Browser interaction fixture, invented data only; starts no Python or HTTP server. */
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import VisualQualityExplorer from '../src/components/VisualQualityExplorer';
import '../src/styles.css';
import '../src/scientific.css';
import '../src/local-workspace.css';

const calls: { path: string; body: unknown }[] = [];
const mode = new URLSearchParams(location.search).get('mode') ?? 'patch';
const initialBundleId = mode === 'patch' ? 'bundle' : `${mode}-bundle`;
Object.assign(window, { __morphologyCalls: calls });
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const points = ['a', 'b'].map((slideId, index) => ({ slideId, patientId: `patient-${slideId}`, hasImage: true, x: index, y: index, patchCount: 4, sampledPatches: 2, attributes: { site: `Site ${index + 1}` } }));
const patches = points.flatMap((row) => [0, 3].map((patchIndex) => ({ slideId: row.slideId, patchIndex, x: patchIndex ? 200 : 0, y: patchIndex ? 150 : 0 })));
const reviews: Record<string, Record<string, unknown>> = {};
function review(slideId: string) { return reviews[slideId] ?? { schemaVersion: 1, datasetId: 'dataset', slideId, status: 'unreviewed', notes: '', reviewer: '', reasons: [], regions: [], evaluationId: null, revision: 0, updatedAt: null, history: [] }; }
window.fetch = async (input, init) => {
  const url = new URL(String(input), 'http://offline.invalid'), path = url.pathname;
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  calls.push({ path: url.pathname + url.search, body });
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (path.endsWith('/feature-bundles')) return reply({ items: ['patch', 'slide', 'broken'].map(kind => ({ id: kind === 'patch' ? 'bundle' : `${kind}-bundle`, current: true, versionLabel: { tag: `${kind} test features` }, manifest: { datasetId: 'dataset', feature: { id: `${kind}-features` } } })) });
  if (path.includes('/configurations/')) return reply({ id: path.split('/').at(-1), manifest: { kind: 'feature', spec: { featureKind: path.endsWith('/slide-features') ? 'slide' : 'patch' } } });
  if (path.endsWith('/morphology/slides')) return reply({ total: 2, matching: 2, offset: 0, items: points.map((row) => ({ ...row, reviewStatus: review(row.slideId).status })) });
  if (path.endsWith('/morphology/index')) return reply({ indexId: 'index', datasetId: 'dataset', featureBundleId: 'bundle', encoderId: 'test-encoder', candidateSlides: 2, indexedSlides: 2, indexedPatches: 4, explainedVariance: [1, 0], method: 'PCA of sampled feature means', sampling: 'Evenly spaced original indices', points, patches, warnings: ['Temporary sampled exploration index.'] });
  if (path.endsWith('/morphology/neighbors')) return reply({ candidateCount: 1, scope: body.mode === 'patch' ? 'sampled_patches' : 'indexed_slides', items: [{ ...points.find((point) => point.slideId !== body.slideId), similarity: .92, ...(body.mode === 'patch' ? { patchIndex: 3 } : {}) }] });
  if (path.endsWith('/morphology/quality')) {
    const bundle = url.searchParams.get('featureBundleId');
    if (bundle === 'broken-bundle') return reply({ code: 'FEATURE_SOURCE_CHANGED', detail: 'The optional feature source is unavailable.' }, 422);
    const coverage = bundle === 'bundle';
    return reply({ slideId: url.searchParams.get('slideId'), datasetId: 'dataset', width: 6000, height: 4000, featureKind: bundle === 'slide-bundle' ? 'slide' : coverage ? 'patch' : undefined, patchWidth: coverage ? 150 : null, patchHeight: coverage ? 120 : null, patchCount: coverage ? 4 : null, patches: coverage ? [{ patchIndex: 0, x: 0, y: 0 }, { patchIndex: 3, x: 200, y: 150 }] : [], coordinateBounds: coverage ? { x: 0, y: 0, width: 350, height: 270 } : null, tissueContours: [], artifactRemoval: null, warnings: bundle === 'slide-bundle' ? ['Slide embeddings have no patch coordinates.'] : ['No recorded tissue mask; coverage is not a tumor score.'] });
  }
  if (path.endsWith('/morphology/patch-region')) return reply({ x: 200, y: 150, width: 150, height: 120 });
  if (path.endsWith('/morphology/image') || path.endsWith('/morphology/patch')) return new Response('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><rect width="600" height="400" fill="#eee0e7"/><ellipse cx="260" cy="210" rx="230" ry="160" fill="#d19bba"/><text x="120" y="210" font-size="20">Synthetic browser fixture</text></svg>', { headers: { 'Content-Type': 'image/svg+xml' } });
  if (path.includes('/slide-reviews/')) {
    const slideId = decodeURIComponent(path.split('/').at(-1)!);
    if (init?.method === 'PUT') {
      const existing = review(slideId);
      if (body.expectedRevision !== existing.revision) return reply({ code: 'SLIDE_REVIEW_CONFLICT', detail: 'Review changed' }, 409);
      reviews[slideId] = { ...existing, ...body, revision: Number(existing.revision) + 1, updatedAt: new Date().toISOString() };
    }
    return reply(review(slideId));
  }
  if (path.endsWith('/slide-reviews')) return reply({ items: Object.values(reviews), total: Object.keys(reviews).length, offset: 0, hasMore: false });
  throw new Error(`Unexpected offline request: ${path}`);
};
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
createRoot(document.getElementById('app')!).render(<QueryClientProvider client={client}><main style={{ maxWidth: 1280, margin: '20px auto', padding: 16 }}><p>Offline morphology verification · {mode} · invented data · no server</p><VisualQualityExplorer project={`offline-${mode}`} datasetId="dataset" initialBundleId={initialBundleId} /></main></QueryClientProvider>);
