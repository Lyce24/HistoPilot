import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { CreateExperiment, ExperimentMetadata } from '../components/ExperimentRegistry';
import type { ModelExperiment } from '../api/experiments';
import { isCreateExperimentDraft, isExperimentMetadataDraft } from './experimentEditorDraft';
import { sessionDraftKey } from './sessionDraft';

const pending = { name: 'Question', notes: 'Keep note', tags: ['baseline'], operationId: 'original-operation', sourceExperimentId: 'source' };
const creation = { version: 1, sourceId: 'source', name: 'Question', notes: 'Keep note', tags: 'baseline, ', pending };
const metadata = { version: 1, name: 'Edited title', notes: 'Unfinished text', tags: 'baseline, ', baseline: { revision: 1, name: 'Saved title', notes: '', tags: [] } };
afterEach(() => vi.unstubAllGlobals());

describe('experiment form recovery', () => {
  it('retains the exact creation operation and raw fields', () => {
    expect(isCreateExperimentDraft(creation)).toBe(true);
    expect(isCreateExperimentDraft({ ...creation, pending: { ...pending, operationId: null } })).toBe(false);
    expect(isCreateExperimentDraft({ ...creation, pending: { ...pending, tags: null } })).toBe(false);
    expect(isCreateExperimentDraft({ ...creation, version: 2 })).toBe(false);
  });
  it('requires a valid metadata baseline instead of adopting an arbitrary current revision', () => {
    expect(isExperimentMetadataDraft(metadata)).toBe(true);
    expect(isExperimentMetadataDraft({ ...metadata, baseline: { ...metadata.baseline, revision: '1' } })).toBe(false);
    expect(isExperimentMetadataDraft({ ...metadata, tags: null })).toBe(false);
  });
  it('offers the original creation retry after reopening with an unavailable template', () => {
    const key = sessionDraftKey('project', 'new', 'create');
    vi.stubGlobal('window', { sessionStorage: { getItem: (requested: string) => requested === key ? JSON.stringify({ version: 1, value: creation }) : null } });
    const html = renderToStaticMarkup(<CreateExperiment project="project" onCreated={() => {}} onClose={() => {}} />);
    expect(html).toContain('Retry creation');
    expect(html).toContain('Selected template unavailable');
    expect(html).toContain('value="baseline, "');
    expect(html).toContain('Keep note');
    expect(html).toMatch(/<fieldset[^>]*disabled=""/);
    expect(renderToStaticMarkup(<CreateExperiment project="other" onCreated={() => {}} onClose={() => {}} />)).not.toContain('Retry creation');
  });
  it('retains metadata text while exposing a stale saved record', () => {
    vi.stubGlobal('window', { sessionStorage: { getItem: () => JSON.stringify({ version: 1, value: metadata }) } });
    const client = new QueryClient();
    const record = { id: 'record', name: 'A newer saved title', notes: '', tags: [], revision: 2, state: 'active' } as unknown as ModelExperiment;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentMetadata project="project" record={record} /></QueryClientProvider>);
      expect(html).toContain('value="Edited title"');
      expect(html).toContain('Unfinished text');
      expect(html).toContain('Reload saved details');
      expect(html).toContain('Your edits are retained');
    } finally { client.clear(); }
  });
});
