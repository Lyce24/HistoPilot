/** Browser interaction fixture, invented data only; starts no Python or HTTP server. */
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import VisualQualityExplorer from '../src/components/VisualQualityExplorer';
import AttentionSlideViewer from '../src/components/AttentionSlideViewer';
import type { Interpretation, InterpretationSlide, InterpretationSlideResult } from '../src/api/interpretation';
import '../src/styles.css';
import '../src/scientific.css';
import '../src/local-workspace.css';

const calls: { path: string; body: unknown; startedAt: number; finishedAt?: number; completed?: boolean; aborted?: boolean; failed?: boolean; expectedFingerprint?: string; rejectedFingerprint?: boolean }[] = [];
const sourceFingerprints: Record<string, string> = { a: 'a'.repeat(64), b: 'b'.repeat(64) };
const imageControl = { holdRegions: new URLSearchParams(location.search).has('hold'), delayMs: 0, active: 0, maxActive: 0, failures: Number(new URLSearchParams(location.search).get('fail') ?? 0), pending: [] as (() => void)[] };
const releaseImages = () => { imageControl.holdRegions = false; imageControl.pending.splice(0).forEach((release) => release()); };
const attentionControl = { hold: false, pending: [] as (() => void)[] };
Object.assign(window, { __attentionControl: attentionControl, __releaseAttention: () => { attentionControl.hold = false; attentionControl.pending.splice(0).forEach((release) => release()); } });
const rasterControl = { hold: false, pending: [] as (() => void)[] };
const originalToBlob = HTMLCanvasElement.prototype.toBlob;
HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
  const release = () => originalToBlob.call(this, callback, type, quality);
  if (rasterControl.hold) rasterControl.pending.push(release); else release();
};
Object.assign(window, { __rasterControl: rasterControl, __releaseRasters: () => { rasterControl.hold = false; rasterControl.pending.splice(0).forEach((release) => release()); } });
const mode = new URLSearchParams(location.search).get('mode') ?? 'patch';
const dense = new URLSearchParams(location.search).has('dense');
const densePatches = Array.from({ length: 4096 }, (_, patchIndex) => ({ patchIndex, x: 60 + patchIndex % 64 * 90, y: 40 + Math.floor(patchIndex / 64) * 60 }));
const denseContours = Array.from({ length: 50 }, (_, ringIndex) => [Array.from({ length: 1000 }, (_, index) => { const angle = index / 999 * Math.PI * 2; return [500 + ringIndex % 10 * 540 + 210 * Math.cos(angle), 450 + Math.floor(ringIndex / 10) * 700 + 240 * Math.sin(angle)]; })]);
Object.assign(window, { __denseQuality: { patches: densePatches, tissueContours: denseContours, patchWidth: 60, patchHeight: 40 } });
const initialBundleId = mode === 'patch' ? 'bundle' : `${mode}-bundle`;
const largeSlide = new URLSearchParams(location.search).has('tiles') || mode === 'attention';
const slideScale = largeSlide ? 10 : 1;
const slideWidth = 6000 * slideScale, slideHeight = 4000 * slideScale;
const sourceControl = { blocked: false, geometry: { a: { width: slideWidth, height: slideHeight }, b: { width: slideWidth, height: slideHeight } } as Record<string, { width: number; height: number }> };
Object.assign(window, { __morphologySource: sourceControl });
Object.assign(window, { __morphologyGeometry: { width: slideWidth, height: slideHeight }, __morphologySlideScale: slideScale });
Object.assign(window, { __morphologyCalls: calls, __morphologyImages: imageControl, __releaseMorphologyImages: releaseImages, __morphologyFingerprints: sourceFingerprints });
const reply = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const points = ['a', 'b'].map((slideId, index) => ({ slideId, patientId: `patient-${slideId}`, hasImage: true, x: index, y: index, patchCount: 4, sampledPatches: 2, attributes: { site: `Site ${index + 1}` } }));
const patches = points.flatMap((row) => [0, 3].map((patchIndex) => ({ slideId: row.slideId, patchIndex, x: patchIndex ? 200 : 0, y: patchIndex ? 150 : 0 })));
const reviews: Record<string, Record<string, unknown>> = {};
function review(slideId: string) { return reviews[slideId] ?? { schemaVersion: 1, datasetId: 'dataset', slideId, status: 'unreviewed', notes: '', reviewer: '', reasons: [], regions: [], evaluationId: null, revision: 0, updatedAt: null, history: [] }; }
window.fetch = async (input, init) => {
  const url = new URL(String(input), 'http://offline.invalid'), path = url.pathname;
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  const call: (typeof calls)[number] = { path: url.pathname + url.search, body, startedAt: performance.now() };
  calls.push(call);
  if (path === '/api/v1/session') return reply({ token: 'offline' });
  if (path.endsWith('/feature-bundles')) return reply({ items: ['patch', 'slide', 'broken'].map(kind => ({ id: kind === 'patch' ? 'bundle' : `${kind}-bundle`, current: true, versionLabel: { tag: `${kind} test features` }, manifest: { datasetId: 'dataset', feature: { id: `${kind}-features` } } })) });
  if (path.includes('/configurations/')) return reply({ id: path.split('/').at(-1), manifest: { kind: 'feature', spec: { featureKind: path.endsWith('/slide-features') ? 'slide' : 'patch' } } });
  if (path.endsWith('/morphology/slides')) return reply({ total: 2, matching: 2, offset: 0, items: points.map((row) => ({ ...row, reviewStatus: review(row.slideId).status })) });
  if (path.endsWith('/morphology/index')) return reply({ indexId: 'index', datasetId: 'dataset', featureBundleId: 'bundle', encoderId: 'test-encoder', candidateSlides: 2, indexedSlides: 2, indexedPatches: 4, explainedVariance: [1, 0], method: 'PCA of sampled feature means', sampling: 'Evenly spaced original indices', points, patches, warnings: ['Temporary sampled exploration index.'] });
  if (path.endsWith('/morphology/neighbors')) return reply({ candidateCount: 1, scope: body.mode === 'patch' ? 'sampled_patches' : 'indexed_slides', items: [{ ...points.find((point) => point.slideId !== body.slideId), similarity: .92, ...(body.mode === 'patch' ? { patchIndex: 3 } : {}) }] });
  if (path.endsWith('/morphology/quality')) {
    if (sourceControl.blocked) return reply({ code: 'MORPHOLOGY_SLIDE_CHANGED', detail: 'This frozen slide has changed on disk. Restore its original file before reloading.' }, 409);
    const bundle = url.searchParams.get('featureBundleId');
    if (bundle === 'broken-bundle') return reply({ code: 'FEATURE_SOURCE_CHANGED', detail: 'The optional feature source is unavailable.' }, 422);
    const coverage = bundle === 'bundle';
    return reply({ slideId: url.searchParams.get('slideId'), sourceFingerprint: sourceFingerprints[url.searchParams.get('slideId')!], datasetId: 'dataset', ...sourceControl.geometry[url.searchParams.get('slideId')!], featureKind: bundle === 'slide-bundle' ? 'slide' : coverage ? 'patch' : undefined, patchWidth: coverage ? dense ? 60 : 150 : null, patchHeight: coverage ? dense ? 40 : 120 : null, patchCount: coverage ? dense ? 4096 : 4 : null, patches: coverage ? dense ? densePatches : [{ patchIndex: 0, x: 0, y: 0 }, { patchIndex: 3, x: 200, y: 150 }] : [], coordinateBounds: coverage ? dense ? { x: 60, y: 40, width: 5730, height: 3820 } : { x: 0, y: 0, width: 350, height: 270 } : null, tissueContours: coverage && dense ? denseContours : [], artifactRemoval: null, warnings: bundle === 'slide-bundle' ? ['Slide embeddings have no patch coordinates.'] : ['No recorded tissue mask; coverage is not a tumor score.'] });
  }
  if (path.includes('/interpretations/') && (path.endsWith('/attention/top') || path.endsWith('/attention'))) {
    const top = path.endsWith('/attention/top');
    if (!top && attentionControl.hold) await new Promise<void>((resolve, reject) => {
      const signal = init?.signal;
      const cleanup = () => { signal?.removeEventListener('abort', abort); attentionControl.pending = attentionControl.pending.filter((value) => value !== release); };
      const release = () => { cleanup(); resolve(); };
      const abort = () => { cleanup(); reject(new DOMException('Aborted attention request', 'AbortError')); };
      if (signal?.aborted) { abort(); return; }
      signal?.addEventListener('abort', abort, { once: true }); attentionControl.pending.push(release);
    });
    const patches = [{ index: 1700, x: 2000 * slideScale, y: 1600 * slideScale, weight: .025, percentile: .995, rank: 1 }, { index: 2500, x: 4000 * slideScale, y: 3000 * slideScale, weight: .02, percentile: .95, rank: 2 }];
    const visible = top ? patches : patches.filter((patch) => patch.x < Number(url.searchParams.get('x')) + Number(url.searchParams.get('width')) && patch.x + 150 * slideScale > Number(url.searchParams.get('x')) && patch.y < Number(url.searchParams.get('y')) + Number(url.searchParams.get('height')) && patch.y + 120 * slideScale > Number(url.searchParams.get('y')));
    return reply({ slideId: 'WSI', total: visible.length, offset: 0, limit: top ? 10 : 10000, scope: 'whole_slide', returned: visible.length, coordinateSpace: 'level0', attentionKind: 'class_independent_pooling', member: 'mean', patchWidthLevel0: 150 * slideScale, patchHeightLevel0: 120 * slideScale, probabilities: [.6, .4], classOrder: ['yes', 'no'], patches: visible });
  }
  if (['/morphology/image', '/morphology/patch', '/morphology/patch-region'].some((suffix) => path.endsWith(suffix))) {
    call.expectedFingerprint = sourceFingerprints[url.searchParams.get('slideId')!];
    if (sourceControl.blocked || !call.expectedFingerprint || url.searchParams.get('sourceFingerprint') !== call.expectedFingerprint) {
      call.failed = true; call.completed = true; call.rejectedFingerprint = true;
      return reply({ code: 'MORPHOLOGY_SLIDE_CHANGED', detail: 'The linked slide identity changed. Reload the slide geometry.' }, 409);
    }
  }
  if (path.endsWith('/morphology/patch-region')) {
    const chosen = densePatches[Number(url.searchParams.get('patchIndex'))];
    return reply(dense && chosen ? { x: chosen.x, y: chosen.y, width: 60, height: 40 } : { x: 200, y: 150, width: 150, height: 120 });
  }
  if (path.endsWith('/morphology/image') || path.endsWith('/morphology/patch') || (path.includes('/interpretations/') && (path.endsWith('/thumbnail') || path.endsWith('/region') || path.endsWith('/image')))) {
    const region = (path.endsWith('/morphology/image') || path.endsWith('/region')) && url.searchParams.has('x');
    if (region) {
      imageControl.active += 1;
      imageControl.maxActive = Math.max(imageControl.maxActive, imageControl.active);
      try {
        if (imageControl.holdRegions || imageControl.delayMs > 0) await new Promise<void>((resolve, reject) => {
          const signal = init?.signal;
          let timer: ReturnType<typeof setTimeout> | undefined;
          const cleanup = () => {
            signal?.removeEventListener('abort', abort);
            if (timer !== undefined) clearTimeout(timer);
            imageControl.pending = imageControl.pending.filter((pending) => pending !== release);
          };
          const release = () => { cleanup(); resolve(); };
          const abort = () => { call.aborted = true; cleanup(); reject(new DOMException('Aborted image request', 'AbortError')); };
          if (signal?.aborted) { abort(); return; }
          signal?.addEventListener('abort', abort, { once: true });
          if (imageControl.holdRegions) imageControl.pending.push(release);
          else timer = setTimeout(release, imageControl.delayMs);
        });
      } finally {
        imageControl.active -= 1;
        call.finishedAt = performance.now();
      }
    }
    call.completed = true;
    if (region && imageControl.failures > 0) {
      imageControl.failures -= 1;
      call.failed = true;
      return reply({ code: 'SLIDE_READ_FAILED', detail: 'The high-resolution crop is temporarily unavailable.' }, 503);
    }
    const imageRegion = region ? ['x', 'y', 'width', 'height'].map((key) => Number(url.searchParams.get(key))) : [0, 0, slideWidth, slideHeight];
    const outputScale = Math.min(1, Math.min(600, Number(url.searchParams.get('max_size') ?? 600)) / Math.max(imageRegion[2], imageRegion[3]));
    const imageWidth = Math.max(1, Math.round(imageRegion[2] * outputScale));
    const imageHeight = Math.max(1, Math.round(imageRegion[3] * outputScale));
    return new Response(`<svg xmlns="http://www.w3.org/2000/svg" width="${imageWidth}" height="${imageHeight}" viewBox="${imageRegion.join(' ')}"><rect width="${slideWidth}" height="${slideHeight}" fill="#eee0e7"/><ellipse cx="${2600 * slideScale}" cy="${2100 * slideScale}" rx="${2300 * slideScale}" ry="${1600 * slideScale}" fill="#d19bba"/><text x="${1200 * slideScale}" y="${2100 * slideScale}" font-size="${200 * slideScale}">Synthetic browser fixture</text></svg>`, { headers: { 'Content-Type': 'image/svg+xml' } });
  }
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
const attentionSlide: InterpretationSlide = { slideId: 'WSI', slidePath: '/invented/WSI.svs', featurePath: '/invented/WSI.h5', featureKey: 'features', coordinatesKey: 'coords', coordinateSpace: 'level0', confirmRowAlignment: true, width: slideWidth, height: slideHeight, backend: 'openslide', levelDownsamples: [1, 4, 16], patchCount: 2, dimensions: 1024, dtype: 'float32', patchWidthLevel0: 150 * slideScale, patchHeightLevel0: 120 * slideScale, alignment: 'embedded_verified' };
const attentionRecord: Interpretation = { id: 'study', createdAt: '', contentHash: 'invented-frozen', manifest: { kind: 'model-interpretation', name: 'Attention', predictorId: 'predictor', encoderId: 'encoder', experimentId: 'experiment', slides: [attentionSlide], method: 'ensemble', memberCount: 2 } };
const attentionResult: InterpretationSlideResult = { slideId: 'WSI', patchCount: 2, probabilities: [.6, .4], attentionArtifact: 'slide-0.json', members: [{ index: 0, checkpointSha256: 'a', probabilities: [.5, .5] }, { index: 1, checkpointSha256: 'b', probabilities: [.7, .3] }] };
const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
Object.assign(window, { __refreshMorphologyQuality: () => client.invalidateQueries({ queryKey: ['morphology-quality'] }) });
createRoot(document.getElementById('app')!).render(<QueryClientProvider client={client}><main style={{ maxWidth: 1280, margin: '20px auto', padding: 16 }}><p>Offline morphology verification · {mode} · invented data · no server</p>{mode === 'attention' ? <AttentionSlideViewer project="offline-attention" record={attentionRecord} slide={attentionSlide} result={attentionResult} /> : <VisualQualityExplorer project={`offline-${mode}`} datasetId="dataset" initialBundleId={initialBundleId} />}</main></QueryClientProvider>);
