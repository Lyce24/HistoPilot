import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Configuration } from '../api/scientific';
import type { FeaturePackArtifact, FeatureValidationReport } from '../api/packing';
import { ApiError } from '../api/client';
import FeatureBundlePreparation, { bundleReviewInvalidated } from './FeatureBundlePreparation';

const configuration: Configuration = {
  id: 'features', projectId: 'project', contentHash: 'configuration-content', createdAt: '2026-09-11T12:00:00Z',
  manifest: {
    kind: 'feature', datasetId: 'dataset', spec: { datasetId: 'dataset', path: '/features', fileSuffix: '.h5', idSuffix: '', recursive: false },
    summary: { slideCount: 2, matchedSlides: 2, missingSlides: 0, orphanFiles: 0, dimensions: 8, patchCount: 12 },
  },
};
const validation: FeatureValidationReport = { valid: true, tensorValidationComplete: true, provenanceComplete: true, featureSetId: 'features', sourceContentHash: 'source', slideCount: 2, totalPatches: 12, dimensions: 8, sourceDtype: 'float32', current: true };
const pack: FeaturePackArtifact = { id: 'pack', materializationId: 'materialization', outputPath: '/packs/verified', featureSetId: 'features', sourceContentHash: 'source', slideCount: 2, totalPatches: 12, dimensions: 8, outputDtype: 'float32', dtypePolicy: 'preserve', verification: 'full', validation, current: true };

function render(initialPackIds: string[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['feature-packs', 'project'], { jobs: [], artifacts: [pack], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
  client.setQueryData(['feature-packs', 'project', 'validation', 'features'], validation);
  try {
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeatureBundlePreparation project="project" configuration={configuration} configurations={[configuration]} initialPackIds={initialPackIds} onSelectVersion={() => {}} onFrozen={() => {}} /></QueryClientProvider>);
    expect(client.getQueryCache().findAll().some((query) => query.queryKey.includes('selection'))).toBe(false);
    return html;
  } finally { client.clear(); }
}

describe('feature bundle preparation', () => {
  it('presents optional materialization choices followed by an explicit review and freeze step', () => {
    const html = render();
    expect(html).toContain('Features only — skip packing');
    expect(html).toContain('Features + existing pack');
    expect(html).toContain('Features + new pack');
    expect(html).toContain('Continue to bundle review');
    expect(html).toContain('data-stage-page="packing"');
    expect(html).not.toContain('aria-label="Bundle review"');
    expect(html).toContain('as one immutable input');
    expect(html).toContain('A features-only bundle includes no pack.');
    expect(html).toContain('Review bundle');
    expect(html).not.toContain('Name &amp; freeze bundle');
    expect(html).not.toContain('mil-loading-policy');
    expect(html).not.toContain('loading preference');
  });

  it('restores explicit bundle draft inclusions without selecting a training loading source', () => {
    const html = render(['pack']);
    expect(html).toContain('Features + 1 pack');
    expect(html).toContain('Remove from bundle');
    expect(html).toContain('/packs/verified');
    expect(html).not.toContain('Use pack');
    expect(html).not.toContain('Original per-slide HDF5 files');
    expect(html).not.toContain('Verified pack selected');
  });

  it('requires re-review for confirmed stale inputs while retaining retryable label and network errors', () => {
    expect(bundleReviewInvalidated(new ApiError('Inputs changed.', 409, 'PREVIEW_STALE'))).toBe(true);
    expect(bundleReviewInvalidated(new ApiError('Verification failed.', 422, 'FEATURE_BUNDLE_INVALID'))).toBe(true);
    for (const error of [
      new ApiError('Tag already exists.', 409, 'VERSION_TAG_CONFLICT'),
      new ApiError('Retry conflict.', 409, 'OPERATION_CONFLICT'),
      new ApiError('Server unavailable.', 503),
      new ApiError('Unknown conflict.', 409),
      new Error('Network request failed.'),
    ]) expect(bundleReviewInvalidated(error)).toBe(false);
  });
});
