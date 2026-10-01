import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import HistoricalProtocols from './HistoricalProtocols';

describe('retired combined protocol workflow', () => {
  it('preserves historical drafts and frozen records without exposing creation or training controls', () => {
    const cache = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const spec = { datasetId: 'dataset', target: { field: 'grade' }, split: { mode: 'kfold' } };
    const derived = { id: 'derived', versionLabel: { tag: 'Setup-derived design' }, manifest: { kind: 'protocol', sourceTargetSplit: { id: 'configuration-target' }, spec: { ...spec, sourceTargetSplitId: 'configuration-target' } } };
    cache.setQueryData(['scientific', 'p', 'configurations', 'protocol'], { configurations: [{ id: 'old', versionLabel: { tag: 'Earlier frozen design' }, manifest: { spec } }, derived] });
    cache.setQueryData(['scientific', 'p', 'drafts'], { drafts: [{ id: 'unfinished', name: 'Earlier draft', status: 'editable', payload: { type: 'analysis-protocol', spec } }] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={cache}><HistoricalProtocols workspace={{ project: { id: 'p' } } as Workspace} /></QueryClientProvider>);
      expect(html).toContain('Earlier frozen design');
      expect(html).toContain('Earlier draft');
      expect(html).not.toContain('Setup-derived design');
      expect(html).toContain('href="#experiments"');
      expect(html).not.toContain('Create protocol');
      expect(html).not.toContain('<input');
      expect(html).not.toContain('<select');
    } finally { cache.clear(); }
  });
});
