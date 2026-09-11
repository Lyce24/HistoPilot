import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import LocalExperiments, { LoadingOptions } from './LocalExperiments';

describe('MIL experiment loading ownership', () => {
  it('keeps pack access optional and does not offer unimplemented memory residency', () => {
    const html = renderToStaticMarkup(<LoadingOptions value="auto" hasPacks={false} onChange={() => {}} />);
    expect(html).toContain('Original files');
    expect(html).toMatch(/<input[^>]*disabled=""[^>]*value="mmap"/);
    expect(html).toContain('whole pack need not fit in RAM');
    expect(html).not.toContain('value="ram"');
    expect(html).not.toContain('value="cuda"');
  });

  it('exposes frozen inputs and persisted MIL drafts while explaining execution availability', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} }, encoders: [], milModels: [] } as unknown as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['scientific', 'project', 'drafts'], { drafts: [
      { id: 'mil-draft', name: 'Saved MIL plan', revision: 2, status: 'editable', payload: { type: 'mil-experiment', spec: {} } },
      { id: 'protocol-draft', name: 'Do not show protocol draft', revision: 1, status: 'editable', payload: { type: 'analysis-protocol', spec: {} } },
    ] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalExperiments workspace={workspace} /></QueryClientProvider>);
      expect(html).toContain('Frozen protocol');
      expect(html).toContain('Frozen feature bundle');
      expect(html).toContain('Saved MIL plan');
      expect(html).not.toContain('Do not show protocol draft');
      expect(html).toContain('MIL training is not implemented');
      expect(html).toContain('Initial project preferences');
      expect(html).toMatch(/<button[^>]*disabled=""[^>]*>[^<]*Review inputs/);
      expect(html).not.toContain('Start training');
    } finally { client.clear(); }
  });
});
