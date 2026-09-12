import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import type { Configuration } from '../api/scientific';
import type { FeatureBundle } from '../api/bundles';
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

function render(versions?: Configuration[], frozenBundles: FeatureBundle[] = []) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  if (versions) client.setQueryData(['scientific', 'project', 'configurations', 'feature'], { configurations: versions });
  if (versions) client.setQueryData(['feature-bundles', 'project'], { items: frozenBundles });
  client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
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
  it('scopes saved bundle records to the linked protocol dataset', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old&saved=protocol' } });
    const html = render([version], [frozen(), frozen('older', 'Older dataset bundle')]);
    expect(html).toContain('Older dataset bundle');
    expect(html).not.toContain('UNI features only');
    expect(html).toContain('Development protocol saved. Prepare or reuse a feature bundle for its dataset.');
    expect(html).toContain('Show all project features');
    expect(html).toContain('Search feature bundles');
    expect(html).not.toContain('Stage 0 · Saved records');
  });

  it('keeps the linked dataset at an empty library without using unrelated sources', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old' } });
    const html = render([version], [frozen()]);
    expect(html).toContain('No feature bundles yet');
    expect(html).toContain('Create feature bundle');
    expect(html).not.toContain('UNI baseline');
    expect(html).not.toContain('UNI features only');
  });

  it.each([{ sources: [] }, { sources: [version] }])('always opens the library, even with no bundles and available sources $sources.length', ({ sources }) => {
    const html = render(sources);
    expect(html).toContain('Search feature bundles');
    expect(html).toContain('No feature bundles yet');
    expect(html).toContain('Create feature bundle');
    expect(html).toContain('>Slide features</h1>');
    expect(html.match(/> Create feature bundle</g)).toHaveLength(1);
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
