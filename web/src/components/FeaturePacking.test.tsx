import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Configuration } from '../api/scientific';
import type { FeaturePackArtifact, FeaturePackJob, FeaturePackPreview, FeatureValidationReport } from '../api/packing';
import FeaturePacking, { ExistingPackComparison, FeaturePackCoverage, FeaturePackProgress, FeatureValidationSummary, SavedPackChoice, canIncludeFeaturePack, nextBundlePackIds, formatPackBytes } from './FeaturePacking';
import PackFolderExamples from './PackFolderExamples';

const report: FeatureValidationReport = {
  valid: true, tensorValidationComplete: true, provenanceComplete: false,
  featureSetId: 'features', sourceContentHash: 'content', slideCount: 2, totalPatches: 1024,
  dimensions: 768, sourceDtype: 'float32', current: true,
};
const configuration: Configuration = {
  id: 'features', projectId: 'project', contentHash: 'frozen', createdAt: '2026-09-10T12:00:00Z',
  manifest: {
    kind: 'feature', datasetId: 'dataset',
    spec: { datasetId: 'dataset', path: '/features', fileSuffix: '.h5', idSuffix: '', recursive: false },
    summary: { slideCount: 3, matchedSlides: 2, missingSlides: 1, orphanFiles: 0, dimensions: 768, patchCount: 1024 },
  },
};
function job(overrides: Partial<FeaturePackJob> = {}): FeaturePackJob {
  return {
    id: 'packing-job', state: 'running', featureSetId: 'features',
    spec: { featureSetId: 'features', action: 'pack', dtype: 'preserve', outputPath: '/pack' },
    outputPath: '/pack', sessionName: 'hp-pack-example', logPath: '/logs/packing.log',
    createdAt: '2026-09-10T12:00:00Z', updatedAt: '2026-09-10T12:01:00Z',
    progress: { stage: 'packing', completed: 1, total: 2, percent: 50, currentSlide: 'B', unit: 'slides' },
    ...overrides,
  };
}

describe('feature preparation presentation', () => {
  it('explains partial coverage without claiming the whole dataset was packed', () => {
    const html = renderToStaticMarkup(<FeaturePackCoverage summary={{ slideCount: 3, matchedSlides: 2, missingSlides: 1, orphanFiles: 0, dimensions: 768, patchCount: 1024 }} />);
    expect(html).toContain('2 of 3 dataset slides');
    expect(html).toContain('cover these attached slides only');
    expect(html).toContain('1 slide still lacks features');
  });

  it('distinguishes header inspection, tensor validation and incomplete provenance', () => {
    const unvalidated = renderToStaticMarkup(<FeatureValidationSummary report={null} />);
    expect(unvalidated).toContain('Headers inspected only');
    expect(unvalidated).not.toContain('Contents validated');
    const validated = renderToStaticMarkup(<FeatureValidationSummary report={report} />);
    expect(validated).toContain('Contents validated');
    expect(validated).toContain('checkpoint provenance remains incomplete');
    expect(validated).not.toContain('training-ready');
    const stale = renderToStaticMarkup(<FeatureValidationSummary report={{ ...report, current: false }} />);
    expect(stale).toContain('Source files changed');
    expect(stale).not.toContain('Contents validated');
  });

  it('does not interpret 100 percent stage progress as a completed pack', () => {
    const run = job();
    run.progress = { ...run.progress!, completed: 2, percent: 100 };
    const html = renderToStaticMarkup(<FeaturePackProgress job={run} />);
    expect(html).toContain('100%');
    expect(html).toContain('verification and publishing may follow');
    expect(html).not.toContain('Training pack created');
  });

  it('explains unavailable progress without hiding the live worker state', () => {
    const html = renderToStaticMarkup(<FeaturePackProgress job={job({ progress: null, progressWarning: 'Progress file could not be read. Worker state is still available.' })} />);
    expect(html).toContain('Progress file could not be read');
    expect(html).toContain('Running');
    expect(html).not.toContain('Training pack created');
  });

  it.each(['cancelling', 'cancelled', 'failed', 'interrupted'] as const)('shows the last reported progress without activity after %s', (state) => {
    const html = renderToStaticMarkup(<FeaturePackProgress job={job({ state, progress: null })} />);
    expect(html).not.toContain('is-indeterminate');
    expect(html).not.toContain('aria-valuenow');
    expect(html).toContain('Last reported stage progress');
  });

  it('only reports completion from worker success and explains explicit bundle inclusion', () => {
    const html = renderToStaticMarkup(<FeaturePackProgress job={job({ state: 'succeeded', result: { validation: report, artifact: null } })} />);
    expect(html).toContain('Training pack created');
    expect(html).toContain('Add it to this bundle draft');
    expect(html).not.toContain('role="progressbar"');
  });

  it('recovers an active saved job on mount with cancellation and collapsed operational details', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['feature-packs', 'project'], { jobs: [job()], artifacts: [], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
    client.setQueryData(['feature-packs', 'project', 'validation', 'features'], null);
    client.setQueryData(['feature-packs', 'project', 'job', 'packing-job'], job());
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeaturePacking project="project" configuration={configuration} configurations={[configuration]} onSelectVersion={() => {}} selectedPackIds={[]} onSelectedPackIdsChange={() => {}} /></QueryClientProvider>);
    expect(html).not.toContain('Features only — skip packing');
    expect(html).not.toContain('Features + new pack');
    expect(html).toContain('Optional packing');
    expect(html).not.toContain('Features + existing pack');
    expect(html).not.toContain('Validate feature contents');
    expect(html).not.toContain('Existing pack folder');
    expect(html).not.toContain('Destination folder');
    expect(html).toContain('data-stage-page="activity"');
    expect(html).toContain('Cancel job');
    expect(html).toContain('tmux attach -t hp-pack-example');
    expect(html).toContain('<details class="feature-pack-run-details">');
    expect(html).not.toContain('<details class="feature-pack-run-details" open');
    client.clear();
  });

  it('shows draft pack inclusions and keeps stale artifacts unavailable for addition', () => {
    const artifact: FeaturePackArtifact = { id: 'registered', materializationId: 'contents', outputPath: '/mmap/blca', featureSetId: 'features', sourceContentHash: 'content', slideCount: 2, totalPatches: 1024, dimensions: 768, outputDtype: 'float32', sourceDtype: 'float32', dtypePolicy: 'preserve', validation: report, current: true, origin: 'existing' };
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['feature-packs', 'project'], { jobs: [], artifacts: [artifact, { ...artifact, id: 'stale', current: false, outputPath: '/changed' }], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
    client.setQueryData(['feature-packs', 'project', 'validation', 'features'], report);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeaturePacking project="project" configuration={configuration} configurations={[configuration]} onSelectVersion={() => {}} selectedPackIds={['registered']} onSelectedPackIdsChange={() => {}} /></QueryClientProvider>);
    expect(html).toContain('Features + 1 pack');
    expect(html).toContain('Contents verified');
    expect(html).toContain('Packs verified for these features');
    expect(html).toContain('Remove from bundle');
    expect(html).toContain('aria-pressed="false" disabled="">Add pack to bundle');
    expect(html).toContain('<details class="feature-pack-connect">');
    expect(html).not.toContain('<details class="feature-pack-connect" open');
    expect(html).not.toContain('Validate feature contents');
    expect(client.getQueryCache().findAll().some((query) => query.queryKey.includes('selection'))).toBe(false);
    expect(html).not.toContain('loading choice');
    expect(html).not.toContain('loading preference');
    client.clear();
  });

  it('keeps historical jobs on the runs page without presenting an old job as the current task', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['feature-packs', 'project'], { jobs: [job({ state: 'succeeded' })], artifacts: [], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
    client.setQueryData(['feature-packs', 'project', 'validation', 'features'], report);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeaturePacking project="project" configuration={configuration} configurations={[configuration]} onSelectVersion={() => {}} selectedPackIds={[]} onSelectedPackIdsChange={() => {}} /></QueryClientProvider>);
    expect(html).toContain('Runs &amp; results');
    expect(html).toContain('1 jobs');
    expect(html).toContain('data-stage-page="settings"');
    expect(html).not.toContain('feature-pack-history');
    expect(html).not.toContain('aria-label="Current feature job"');
    expect(html).not.toContain('Training pack created');
    client.clear();
  });

  it('requires current full tensor verification before offering bundle inclusion', () => {
    const artifact: FeaturePackArtifact = { id: 'registered', materializationId: 'contents', outputPath: '/mmap/blca', featureSetId: 'features', sourceContentHash: 'content', slideCount: 2, totalPatches: 1024, dimensions: 768, outputDtype: 'float32', sourceDtype: 'float32', dtypePolicy: 'preserve', validation: report, current: true };
    expect(canIncludeFeaturePack(artifact)).toBe(true);
    for (const unchecked of [
      { ...artifact, current: undefined },
      { ...artifact, current: false },
      { ...artifact, validation: { ...report, valid: false } },
      { ...artifact, validation: { ...report, tensorValidationComplete: false } },
      { ...artifact, validation: { ...report, current: false } },
    ]) {
      expect(canIncludeFeaturePack(unchecked)).toBe(false);
      const html = renderToStaticMarkup(<SavedPackChoice artifact={unchecked} included={false} busy={false} onToggle={() => {}} />);
      expect(html).toContain('Verification required');
      expect(html).toContain('aria-pressed="false" disabled="">Add pack to bundle');
      expect(html).not.toContain('Contents verified');
    }
  });

  it('makes precision loss explicit when adding a pack, including legacy precision receipts', () => {
    const artifact: FeaturePackArtifact = { id: 'half', materializationId: 'contents', outputPath: '/mmap/half', featureSetId: 'features', sourceContentHash: 'content', slideCount: 2, totalPatches: 1024, dimensions: 768, outputDtype: 'float16', sourceDtype: 'float32', dtypePolicy: 'float16', validation: report, current: true };
    const html = renderToStaticMarkup(<SavedPackChoice artifact={artifact} included={false} busy={false} onToggle={() => {}} />);
    expect(html).toContain('Add float16 pack to bundle');
    expect(html).toContain('changes source feature precision');
    expect(html).not.toContain('Source precision preserved');
  });

  it('supports multiple included packs and allows removing a pack that became stale', () => {
    const artifact: FeaturePackArtifact = { id: 'second', materializationId: 'contents', outputPath: '/mmap/second', featureSetId: 'features', sourceContentHash: 'content', slideCount: 2, totalPatches: 1024, dimensions: 768, outputDtype: 'float32', sourceDtype: 'float32', dtypePolicy: 'preserve', validation: report, current: true };
    expect(nextBundlePackIds(['first'], artifact)).toEqual(['first', 'second']);
    expect(nextBundlePackIds(['first', 'second'], artifact)).toEqual(['first']);
    expect(nextBundlePackIds(['first'], { ...artifact, current: false })).toEqual(['first']);
    expect(nextBundlePackIds(['first', 'second'], { ...artifact, current: false })).toEqual(['first']);
    const html = renderToStaticMarkup(<SavedPackChoice artifact={{ ...artifact, current: false }} included busy={false} onToggle={() => {}} />);
    expect(html).toContain('Verification required');
    expect(html).toContain('aria-pressed="true">Remove from bundle');
    expect(html).not.toContain('disabled=""');
  });

  it('shows unavailable draft packs explicitly and leaves them removable', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['feature-packs', 'project'], { jobs: [], artifacts: [], tmuxAvailable: true, formatAvailable: true, defaultOutputRoot: '/packs' });
    client.setQueryData(['feature-packs', 'project', 'validation', 'features'], report);
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeaturePacking project="project" configuration={configuration} configurations={[configuration]} onSelectVersion={() => {}} selectedPackIds={['missing']} onSelectedPackIdsChange={() => {}} /></QueryClientProvider>);
    expect(html).toContain('Features + 1 pack');
    expect(html).toContain('Needs verification');
    expect(html).toContain('Remove missing from bundle');
    expect(html).toContain('Existing pack folder');
    expect(html).not.toContain('loading preference');
    client.clear();
  });

  it('distinguishes matching structure from verified contents and explains unequal container sizes', () => {
    const preview: FeaturePackPreview = {
      spec: { featureSetId: 'features', action: 'attach', dtype: 'preserve', existingPath: '/mmap/blca' }, previewHash: 'hash', canRun: true, findings: [], slideCount: 2, patchCount: 1024, dimensions: 768, sourceDtype: 'float32', outputDtype: 'float32', estimatedBytes: null, availableBytes: null, outputPath: null, tmuxAvailable: true, matchesFeatures: true,
      packInspection: { format: 'oceanpath', formatVariant: 'legacy', slideCount: 2, totalPatches: 1024, dimensions: 768, outputDtype: 'float32', featureBytes: 3145728, coordinateBytes: 8192, totalBytes: 3155000, expectedFeatureBytes: 3145728, expectedCoordinateBytes: 8192, sourceContainerBytes: 3210000, sourcePatchCount: 1024, missingSlideCount: 0, extraSlideCount: 0, mismatchedSlideCount: 0, missingSlides: [], extraSlides: [], mismatchedSlides: [] },
    };
    const html = renderToStaticMarkup(<ExistingPackComparison preview={preview} />);
    expect(html).toContain('Structure matches · full verification pending');
    expect(html).toContain('HDF5 metadata and compression affect file size');
    expect(html).toContain('every feature row and coordinate');
    expect(html).not.toContain('role="alert"');
    const mismatch = renderToStaticMarkup(<ExistingPackComparison preview={{ ...preview, matchesFeatures: false, packInspection: { ...preview.packInspection!, mismatchedSlideCount: 1 } }} />);
    expect(mismatch).toContain('Pack differs from source features');
    expect(mismatch).toContain('1 slides with different patch counts');
    expect(mismatch).toContain('role="alert"');
  });

  it('explains both pack folder layouts and marks full attachment verification clearly', () => {
    const example = renderToStaticMarkup(<PackFolderExamples />);
    for (const name of ['HistoPilot pack', 'OceanPath v1 pack', 'features.bin', 'coords.bin', 'index.parquet', 'meta.json', 'manifest.json', 'checksums.json']) expect(example).toContain(name);
    const html = renderToStaticMarkup(<FeaturePackProgress job={job({ spec: { featureSetId: 'features', action: 'attach', dtype: 'preserve', existingPath: '/mmap/blca' }, state: 'succeeded' })} />);
    expect(html).toContain('Existing pack verified');
    expect(html).toContain('Every feature value and coordinate matches');
    expect(html).not.toContain('Training pack created');
  });

  it('formats binary file sizes and does not claim unknown estimates are zero', () => {
    expect(formatPackBytes(0)).toBe('0 B');
    expect(formatPackBytes(1024 ** 3)).toBe('1 GiB');
    expect(formatPackBytes(null)).toBe('Unavailable');
    expect(formatPackBytes(NaN)).toBe('Unavailable');
  });
});
