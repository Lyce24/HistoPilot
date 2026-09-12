import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AttentionSlideViewer from './AttentionSlideViewer';
import type { Interpretation, InterpretationSlide, InterpretationSlideResult, TopAttentionMap } from '../api/interpretation';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
const slide: InterpretationSlide = { slideId: 'WSI', slidePath: '/data/WSI.svs', featurePath: '/data/WSI.h5', featureKey: 'features', coordinatesKey: 'coords', coordinateSpace: 'level0', confirmRowAlignment: true, width: 116736, height: 101376, backend: 'openslide', levelDownsamples: [1, 4, 16], patchCount: 3217, dimensions: 1024, dtype: 'float32', patchWidthLevel0: 512, patchHeightLevel0: 512, alignment: 'embedded_verified' };
const record: Interpretation = { id: 'study', createdAt: '', contentHash: 'frozen', manifest: { kind: 'model-interpretation', name: 'Attention', predictorId: 'predictor', encoderId: 'encoder', experimentId: 'experiment', slides: [slide], method: 'ensemble', memberCount: 2 } };
const result: InterpretationSlideResult = { slideId: 'WSI', patchCount: 3217, probabilities: [.6, .4], attentionArtifact: 'slide-0.json', members: [{ index: 0, checkpointSha256: 'a', probabilities: [.5, .5] }, { index: 1, checkpointSha256: 'b', probabilities: [.7, .3] }] };
function render(coverage = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); clients.push(client);
  const top: TopAttentionMap = { slideId: slide.slideId, total: 3217, offset: 0, limit: 10, scope: 'whole_slide', returned: 1, coordinateSpace: 'level0', attentionKind: 'class_independent_pooling', member: 'mean', patchWidthLevel0: 512, patchHeightLevel0: 512, probabilities: [.6, .4], classOrder: ['yes', 'no'], patches: [{ index: 1700, x: 40000, y: 30000, weight: .025, percentile: .995, rank: 1 }], ...(coverage ? { coordinateBounds: { x: 35000, y: 25000, width: 20000, height: 30000 } } : {}) };
  client.setQueryData(['interpretation-top-attention', 'p', record.id, slide.slideId, 'mean', 10], top);
  return renderToStaticMarkup(<QueryClientProvider client={client}><AttentionSlideViewer project="p" record={record} slide={slide} result={result} /></QueryClientProvider>);
}
describe('focused attention workspace', () => {
  it('starts with the strongest original crop and compact controls without shrinking SVG to the slide aspect', () => {
    const html = render();
    expect(html).toContain('Rank 1 · Patch 1700');
    expect(html).toContain('Loading selected patch…');
    expect(html).toContain('Fit patch coverage');
    expect(html).toContain('Expand view');
    expect(html).toContain('Hide patches');
    expect(html).toContain('aria-label="Slide zoom controls"');
    expect(html).toContain('class="ranked-patch-inspector is-compact"');
    expect(html).toContain('>Center selected</button>');
    expect(html).not.toContain('aspect-ratio:');
  });
  it('keeps provenance collapsed and exposes distinct frozen ensemble members', () => {
    const html = render();
    expect(html).toContain('<details class="attention-view-details"><summary>');
    expect(html).toContain('<option value="0">Member 1</option>');
    expect(html).toContain('<option value="1">Member 2</option>');
    expect(html).toContain('3,217 slide patches');
    expect(html).toContain('class-independent');
  });
  it('supports older top responses with no coverage metadata and retains whole-slide fit', () => {
    const html = render(false);
    expect(html).toContain('disabled="">Fit patch coverage');
    expect(html).toContain('>Fit slide</button>');
    expect(html).toContain('Rank 1 · Patch 1700');
  });
});
