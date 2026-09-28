import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { MorphologyIndex, QualityEvidence } from '../api/morphology';
import VisualQualityExplorer, { MorphologyPlot, QualitySlide, projectionGroups } from './VisualQualityExplorer';

vi.mock('./SlideGalleryCard', () => ({ useImageBlob: (blob?: Blob) => blob ? 'blob:original-slide' : undefined }));

const index: MorphologyIndex = {
  indexId: 'index', datasetId: 'dataset', featureBundleId: 'bundle', encoderId: 'test', candidateSlides: 2, indexedSlides: 2, indexedPatches: 4,
  method: 'PCA of L2-normalized means of sampled raw patch features', sampling: 'Evenly spaced', explainedVariance: [.8, .2], warnings: [], patches: [],
  points: [
    { slideId: 'a', patientId: 'p-a', hasImage: true, x: 0, y: 0, patchCount: 100, sampledPatches: 2, attributes: { site: 'Hospital A' } },
    { slideId: 'b', patientId: 'p-b', hasImage: true, x: 0, y: 0, patchCount: 100, sampledPatches: 2, attributes: { site: null } },
  ],
};

describe('visual QC and morphology explorer', () => {
  it('requires opening expensive feature and image inspection explicitly', () => {
    const html = renderToStaticMarkup(<VisualQualityExplorer project="project" datasetId="dataset" />);
    expect(html).toContain('Open visual review &amp; feature explorer');
    expect(html).not.toContain('<svg');
    expect(html).not.toContain('Building');
  });
  it('keeps degenerate PCA projections finite and offers keyboard-operable slide selection', () => {
    const html = renderToStaticMarkup(<MorphologyPlot index={index} selected="a" onSelect={() => {}} />);
    expect(html).not.toMatch(/NaN|Infinity/);
    expect(html).toContain('aria-label="Inspect a, All slides"');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('80.0% variance');
    expect(html).toContain('2/100 sampled patches');
    expect(html).toContain('full feature space');
  });
  it('preserves missing metadata as its own color group without changing scientific values', () => {
    const groups = projectionGroups(index.points, 'site');
    expect(groups.groups).toEqual(['Hospital A', 'Missing']);
    expect(groups.color(index.points[0])).not.toBe(groups.color(index.points[1]));
    expect(index.points[1].attributes.site).toBeNull();
  });
  it('does not reuse an unpinned thumbnail or coverage from another slide fingerprint', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const fingerprint = 'a'.repeat(64);
    const geometry: QualityEvidence = { sourceFingerprint: fingerprint, slideId: 'slide', datasetId: 'dataset', width: 90000, height: 60000, patchWidth: null, patchHeight: null, patchCount: null, patches: [], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] };
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', undefined], geometry);
    client.setQueryData(['morphology-image', 'project', 'dataset', 'slide', undefined], new Blob(['old-image']));
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', 'bundle', fingerprint], { ...geometry, sourceFingerprint: 'b'.repeat(64), patchCount: 1, patchWidth: 512, patchHeight: 512, patches: [{ patchIndex: 0, x: 1000, y: 1000 }] });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><QualitySlide project="project" datasetId="dataset" slideId="slide" featureBundleId="bundle" showReview={false} /></QueryClientProvider>);
    expect(html).not.toContain('href="blob:original-slide"');
    expect(html).not.toContain('>Patch coverage<');
    expect(html).toContain('Feature coverage does not match the current slide image');
    expect(html).toContain('viewBox="0 0 90000 60000"');
    client.clear();
  });
  it('uses the thumbnail and overlays only from the geometry fingerprint', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const fingerprint = 'a'.repeat(64);
    const geometry: QualityEvidence = { sourceFingerprint: fingerprint, slideId: 'slide', datasetId: 'dataset', width: 90000, height: 60000, patchWidth: null, patchHeight: null, patchCount: null, patches: [], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] };
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', undefined], geometry);
    client.setQueryData(['morphology-image', 'project', 'dataset', 'slide', undefined, fingerprint], new Blob(['current-image']));
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', 'bundle', fingerprint], { ...geometry, patchCount: 1, patchWidth: 512, patchHeight: 512, patches: [{ patchIndex: 0, x: 1000, y: 1000 }] });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><QualitySlide project="project" datasetId="dataset" slideId="slide" featureBundleId="bundle" showReview={false} /></QueryClientProvider>);
    expect(html).toContain('href="blob:original-slide"'); expect(html).toContain('>Patch coverage<');
    expect(html).not.toContain('Feature coverage does not match'); client.clear();
  });
  it('keeps original image and level-0 ROI tools when optional feature coordinates fail', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
    const geometry: QualityEvidence = { slideId: 'slide', datasetId: 'dataset', width: 90000, height: 60000, patchWidth: null, patchHeight: null, patchCount: null, patches: [], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] };
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', undefined], geometry);
    client.setQueryData(['morphology-image', 'project', 'dataset', 'slide', undefined], new Blob(['image']));
    client.getQueryCache().build(client, { queryKey: ['morphology-quality', 'project', 'dataset', 'slide', 'bundle'] }).setState({ status: 'error', error: new Error('No patch coordinates') });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><QualitySlide project="project" datasetId="dataset" slideId="slide" featureBundleId="bundle" showReview={false} /></QueryClientProvider>);
    expect(html).toContain('Optional feature coverage is unavailable');
    expect(html).toContain('href="blob:original-slide"');
    expect(html).toContain('viewBox="0 0 90000 60000"');
    expect(html).toContain('Draw review region');
    expect(html).not.toContain('>Patch coverage<');
    client.clear();
  });
  it('hides stale pixels and review controls when an exact patch detects a changed source', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
    const fingerprint = 'a'.repeat(64);
    const geometry: QualityEvidence = { sourceFingerprint: fingerprint, slideId: 'slide', datasetId: 'dataset', width: 6000, height: 4000, patchWidth: 512, patchHeight: 512, patchCount: 1, patches: [{ patchIndex: 0, x: 0, y: 0 }], coordinateBounds: null, tissueContours: [], artifactRemoval: null, warnings: [] };
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', undefined], geometry);
    client.setQueryData(['morphology-quality', 'project', 'dataset', 'slide', 'bundle', fingerprint], geometry);
    client.setQueryData(['morphology-image', 'project', 'dataset', 'slide', undefined, fingerprint], new Blob(['old']));
    client.getQueryCache().build(client, { queryKey: ['morphology-patch', 'project', 'dataset', 'slide', 'bundle', 0, fingerprint] }).setState({ status: 'error', error: Object.assign(new Error('Slide changed.'), { status: 409, code: 'MORPHOLOGY_SLIDE_CHANGED' }) });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><QualitySlide project="project" datasetId="dataset" slideId="slide" featureBundleId="bundle" patchIndex={0} /></QueryClientProvider>);
    expect(html).toContain('Reload slide');
    expect(html).not.toContain('href="blob:original-slide"');
    expect(html).not.toContain('Draw review region');
    expect(html).not.toContain('Original patch');
    expect(html).not.toContain('Save review');
    client.clear();
  });

});
