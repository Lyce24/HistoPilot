import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { Configuration } from '../api/scientific';
import type { FeatureBundle } from '../api/bundles';
import type { ExtractionJob } from '../api/trident';
import LocalFeatures from './LocalFeatures';

const workspace = {
  project: { id: 'project', storagePath: '/project' },
  dataset: { id: 'dataset' },
  sources: [{ role: 'features', path: '/features' }],
} as Workspace;
const version = {
  id: 'feature-version', createdAt: '2026-09-10T12:00:00Z',
  versionLabel: { tag: 'UNI baseline', note: 'Reviewed extraction' },
  manifest: {
    kind: 'feature', datasetId: 'dataset',
    spec: { datasetId: 'dataset', path: '/features', encoderId: 'uni_v1', fileSuffix: '.h5', idSuffix: '', recursive: false },
    summary: { slideCount: 2, matchedSlides: 2, missingSlides: 0, orphanFiles: 0, dimensions: 8, patchCount: 12 },
    files: [],
  },
} as unknown as Configuration;
afterEach(() => vi.unstubAllGlobals());

function render(versions?: Configuration[], frozenBundles: FeatureBundle[] = [], jobs: ExtractionJob[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  if (versions) client.setQueryData(['scientific', 'project', 'configurations', 'feature'], { configurations: versions });
  if (versions) client.setQueryData(['feature-bundles', 'project'], { items: frozenBundles });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
  client.setQueryData(['extractions', 'project', 'jobs'], { jobs });
  for (const job of jobs) client.setQueryData(['extractions', 'project', 'job', job.id], job);
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}><LocalFeatures workspace={workspace} /></QueryClientProvider>);
  } finally { client.clear(); }
}

function frozen(datasetId = 'dataset', tag = 'UNI features only'): FeatureBundle {
  return {
    id: `bundle-${datasetId}`, createdAt: '2026-09-10T12:00:00Z', current: true, findings: [],
    versionLabel: { tag }, manifest: {
      kind: 'feature-bundle', datasetId, spec: { featureSetId: version.id, packArtifactIds: [] },
      summary: { slideCount: 2, patchCount: 12, dimensions: 8, dtype: 'float32', packCount: 0 },
      feature: { id: version.id }, packs: [],
    },
  } as unknown as FeatureBundle;
}

describe('feature bundle stage entry', () => {
  it('keeps every bundle available when a dataset is linked', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old&saved=protocol' } });
    const html = render([version], [frozen(), frozen('older', 'Older dataset bundle')]);
    expect(html).toContain('Older dataset bundle');
    expect(html).toContain('UNI features only');
    expect(html).toContain('Development protocol saved. Create or open an experiment to select features, check compatibility and configure training.');
    expect(html).toContain('All project bundles are available.');
    expect(html).toContain('Search feature bundles');
    expect(html).not.toContain('Stage 0 · Saved records');
  });

  it('shows reusable bundles prepared with another dataset', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old' } });
    const html = render([version], [frozen()]);
    expect(html).not.toContain('No feature bundles yet');
    expect(html).toContain('Create feature bundle');
    expect(html).not.toContain('UNI baseline');
    expect(html).toContain('UNI features only');
  });

  it.each([{ sources: [] }, { sources: [version] }])('always opens the library, even with no bundles and available sources $sources.length', ({ sources }) => {
    const html = render(sources);
    expect(html).toContain('class="stage-library card"');
    expect(html).not.toContain('Search feature bundles');
    expect(html).toContain('No feature bundles yet');
    expect(html).toContain('Create feature bundle');
    expect(html).toContain('>Slide features</h1>');
    expect(html.match(/data-stage-action="create"/g)).toHaveLength(1);
    expect(html).not.toContain('Stage 0 · Saved records');
    expect(html).not.toContain('aria-label="Add features"');
    expect(html).not.toContain('aria-label="Prepare feature bundle"');
    expect(html).not.toContain('Existing pack folder');
    expect(html).not.toContain('TRIDENT output directory');
    expect(html).not.toContain('Inspect &amp; review features');
  });

  it('lists saved bundles with an explicit Open action without auto-opening a record', () => {
    const html = render([version], [frozen()]);
    expect(html).toContain('aria-label="Saved feature bundles"');
    expect(html).toContain('aria-label="Open UNI features only"');
    expect(html).toContain('Create feature bundle');
    expect(html).not.toContain('No pack is included in this bundle.');
    expect(html).not.toContain('Edit bundle name &amp; note');
  });

  it('waits for the library response before showing empty or saved records', () => {
    const html = render();
    expect(html).toContain('Loading feature sources and bundles…');
    expect(html).not.toContain('No feature bundles yet');
    expect(html).not.toContain('Inspect &amp; review features');
  });
});

function extraction(id = 'running-extraction', state: ExtractionJob['state'] = 'running'): ExtractionJob {
  return {
    id, state, createdAt: '2026-09-25T10:00:00Z', updatedAt: '2026-09-25T10:05:00Z',
    spec: { datasetId: null, outputPath: '/extracted', options: { task: 'all', patch_encoder: 'uni_v1' } },
    outputPath: '/extracted', logPath: '/extracted/run.log', sessionName: 'extraction-session', logs: `Log for ${id}`,
    progress: {
      stage: 'patch_features', stages: [{ id: 'patch_features', label: 'Extract features', status: 'active' }],
      label: 'Extracting patch features', detail: 'Encoding tissue patches.', completed: 12, total: 40,
      unit: 'slides', percent: 30, currentSlide: 'slide-13', elapsedSeconds: 300, etaSeconds: 700,
      ratePerSecond: 0.04, scope: 'stage', warnings: [],
    },
  };
}

describe('extraction visibility and navigation', () => {
  it('shows a running extraction before any feature source or bundle exists', () => {
    const html = render([], [], [extraction()]);
    expect(html).toContain('Extraction runs');
    expect(html).toContain('1 active');
    expect(html).toContain('UNI v1');
    expect(html).toContain('Extracting patch features');
    expect(html).toContain('12 of 40 slides processed');
    expect(html).toContain('href="#features?extraction=running-extraction"');
    expect(html).toContain('View progress');
    expect(html).toContain('Extraction in progress');
    expect(html).not.toContain('No feature bundles yet');
  });

  it('keeps extraction runs visible alongside saved bundles and dataset filters', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=other' } });
    const html = render([version], [frozen()], [extraction()]);
    expect(html).toContain('UNI features only');
    expect(html).toContain('12 of 40 slides processed');
    expect(html).toContain('extraction=running-extraction&amp;dataset=other');
  });

  it.each(['queued', 'starting', 'cancelling'] as const)('includes %s runs in preparation activity', (state) => {
    const html = render([], [], [extraction('queued-extraction', state)]);
    expect(html).toContain('1 active');
    expect(html).toContain('View progress');
    expect(html).not.toContain('No feature bundles yet');
  });

  it('opens the exact extraction from a deep link even when it is an earlier run', () => {
    vi.stubGlobal('window', { location: { hash: '#features?extraction=older-extraction' } });
    const html = render([], [], [extraction(), extraction('latest-finished', 'succeeded'), extraction('older-extraction', 'failed')]);
    expect(html).toContain('Extraction activity');
    expect(html).toContain('aria-current="step"><span class="stage-step-number" aria-hidden="true">3');
    expect(html).toContain('Log for older-extraction');
    expect(html).not.toContain('Log for running-extraction');
    expect(html).toContain('class="trident-job-history" open=""');
  });

  it('retains completed and failed runs in the library with links to their outputs', () => {
    const html = render([], [], [extraction('finished', 'succeeded'), extraction('failed', 'failed')]);
    expect(html).toContain('Previous extraction runs (2)');
    expect(html).toContain('href="#features?extraction=finished"');
    expect(html).toContain('href="#features?extraction=failed"');
    expect(html).toContain('View run &amp; outputs');
    expect(html).not.toContain('Extraction in progress');
  });
});
