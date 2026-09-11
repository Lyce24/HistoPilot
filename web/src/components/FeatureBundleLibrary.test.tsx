import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { FeatureBundle } from '../api/bundles';
import type { Configuration } from '../api/scientific';
import FeatureBundleLibrary from './FeatureBundleLibrary';

const source = {
  id: 'feature', versionLabel: { tag: 'UNI features', note: '' },
  manifest: { kind: 'feature', spec: { encoderId: 'uni_v1' } },
} as Configuration;
function bundle(withPack = true): FeatureBundle {
  const validation = { jobId: 'validation', sourceContentHash: 'source-content', tensorValidationComplete: true, provenanceComplete: true };
  return {
    id: 'bundle', contentHash: 'bundle-content', createdAt: '2026-09-11T12:00:00Z',
    versionLabel: { tag: 'Reviewed BLCA', note: 'Float32 baseline', revision: 1, createdAt: '', updatedAt: '' },
    current: true, findings: [],
    manifest: {
      kind: 'feature-bundle', datasetId: 'dataset', spec: { featureSetId: 'feature', packArtifactIds: withPack ? ['pack'] : [] },
      summary: { slideCount: 138, patchCount: 1191065, dimensions: 1024, dtype: 'float32', packCount: withPack ? 1 : 0 },
      feature: { id: 'feature', contentHash: 'feature-content', datasetId: 'dataset', sourceContentHash: 'source-content', validation },
      packs: withPack ? [{ id: 'pack', materializationId: 'materialization', featureSetId: 'feature', outputPath: '/mmap/blca-verified', outputDtype: 'float32', sourceContentHash: 'source-content', verification: 'full', jobId: 'pack-job', validation }] : [],
    },
  };
}
function render(items: FeatureBundle[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  try {
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><FeatureBundleLibrary project="project" items={items} features={[source]} selectedId={items[0]?.id ?? ''} onSelect={() => {}} onPrepare={() => {}} /></QueryClientProvider>);
    expect(client.getQueryCache().findAll().some((query) => query.queryKey.includes('selection'))).toBe(false);
    return html;
  } finally { client.clear(); }
}

describe('frozen feature bundle library', () => {
  it('shows frozen feature and pack contents with MIL handoff, without a mutable loading choice', () => {
    const html = render([bundle()]);
    expect(html).toContain('UNI features');
    expect(html).toContain('Reviewed BLCA');
    expect(html).toContain('Float32 baseline');
    expect(html).toContain('Features + 1 verified pack(s)');
    expect(html).toContain('/mmap/blca-verified');
    expect(html).toContain('float32');
    expect(html).toContain('1,191,065 patches');
    expect(html).toContain('href="#experiments"');
    expect(html).toContain('Create another bundle to change the included packs.');
    expect(html).not.toContain('Use pack');
    expect(html).not.toContain('Load this feature version from');
    expect(html).not.toContain('mil-loading-policy');
    expect(html).not.toContain('type="checkbox"');
  });

  it('represents a features-only bundle without implying that a pack exists', () => {
    const html = render([bundle(false)]);
    expect(html).toContain('Features only');
    expect(html).toContain('Features alone');
    expect(html).toContain('No pack is included in this bundle.');
    expect(html).not.toContain('Included packs');
    expect(html).not.toContain('/mmap/');
  });

  it('keeps stale frozen contents visible and presents their verification warning', () => {
    const changed = bundle();
    changed.current = false;
    changed.findings = [{ severity: 'error', code: 'PACK_CHANGED', message: 'The included pack changed after verification.' }];
    const html = render([changed]);
    expect(html).toContain('Inputs need attention');
    expect(html).toContain('The included pack changed after verification.');
    expect(html).toContain('Blocking');
    expect(html).toContain('/mmap/blca-verified');
    expect(html).not.toContain('Verified inputs');
  });

  it('directs an empty library to preparation rather than presenting loading controls', () => {
    const html = render([]);
    expect(html).toContain('No frozen bundles yet');
    expect(html).toContain('Prepare a bundle');
    expect(html).not.toContain('Frozen bundle<select');
  });
});
