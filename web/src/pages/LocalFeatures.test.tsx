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

describe('PFM feature sections', () => {
  it('scopes feature preparation to the saved protocol dataset rather than the newest dataset', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old&saved=protocol' } });
    const older = { ...version, id: 'older-source', versionLabel: { ...version.versionLabel!, tag: 'Older dataset features' }, manifest: { ...version.manifest, datasetId: 'older', spec: { ...version.manifest.spec, datasetId: 'older' } } } as Configuration;
    const html = render([version, older]);
    expect(html).toContain('Older dataset features');
    expect(html).not.toContain('UNI baseline');
    expect(html).toContain('Development protocol saved. Prepare or reuse a feature bundle for its dataset.');
    expect(html).toContain('Show all project features');
  });

  it('opens acquisition when the linked dataset has no sources instead of using an unrelated source', () => {
    vi.stubGlobal('window', { location: { hash: '#features?dataset=older&protocol=protocol-old' } });
    const html = render([version]);
    expect(html).toMatch(/<section class="pfm-content" aria-label="Add features">/);
    expect(html).not.toContain('UNI baseline');
    expect(html).toContain('<option value="older" disabled="" selected="">Selected dataset unavailable');
  });
  it('starts an empty library at feature acquisition and hides pack management', () => {
    const html = render([]);
    expect(html).toMatch(/<section class="pfm-content" aria-label="Add features">/);
    expect(html).toMatch(/<section hidden="" class="pfm-content" aria-label="Prepare feature bundle">/);
    expect(html).toMatch(/<section hidden="" class="pfm-content" aria-label="Frozen feature bundles">/);
    expect(html).toContain('Use existing features');
    expect(html).toContain('Extract with a PFM');
    expect(html).not.toContain('Existing pack folder');
    expect(html).toContain('Prepare slide features');
    expect(html).toContain('<details class="pfm-source-details"><summary>Advanced file matching and encoder settings');
    expect(html).not.toContain('aria-label="Module workflow"');
  });

  it('opens unbundled sources at packing and bundle preparation without loading choices', () => {
    const html = render([version]);
    expect(html).toMatch(/<section hidden="" class="pfm-content" aria-label="Add features">/);
    expect(html).toMatch(/<section class="pfm-content" aria-label="Prepare feature bundle">/);
    expect(html).toContain('UNI baseline');
    expect(html).toContain('uni_v1');
    expect(html).toContain('Decide which packs, if any, will be frozen with these features.');
    expect(html).toContain('Features only — skip packing');
    expect(html).toContain('Freeze this bundle');
    expect(html).toContain('Source details &amp; inspection');
    expect(html).not.toContain('Choose how to load');
    expect(html).not.toContain('loading preference');
    expect(html).toContain('<li aria-current="step" class="is-current"><span>3</span>');
    expect(html).not.toContain('aria-label="Choose a feature source"');
    expect(html).toContain('class="label pfm-version-select"');
  });

  it('opens existing frozen bundles as immutable library entries', () => {
    const frozen = {
      id: 'bundle', createdAt: '2026-09-10T12:00:00Z', current: true, findings: [],
      versionLabel: { tag: 'UNI features only' },
      manifest: {
        kind: 'feature-bundle', datasetId: 'dataset',
        spec: { featureSetId: version.id, packArtifactIds: [] },
        summary: { slideCount: 2, patchCount: 12, dimensions: 8, dtype: 'float32', packCount: 0 },
        feature: { id: version.id }, packs: [],
      },
    } as unknown as FeatureBundle;
    const html = render([version], [frozen]);
    expect(html).toMatch(/<section class="pfm-content" aria-label="Frozen feature bundles">/);
    expect(html).toMatch(/<section hidden="" class="pfm-content" aria-label="Prepare feature bundle">/);
    expect(html).toContain('UNI features only');
    expect(html).toContain('No pack is included in this bundle.');
    expect(html).toContain('Create another bundle to change the included packs.');
  });

  it('waits for the library response before choosing an empty or saved-version flow', () => {
    const html = render();
    expect(html).toContain('Loading feature sources and bundles…');
    expect(html).not.toContain('No frozen bundles yet');
    expect(html).not.toContain('Inspect &amp; review features');
  });
});
