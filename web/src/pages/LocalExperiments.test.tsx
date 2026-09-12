import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import LocalExperiments, { ExperimentDetail, LoadingOptions, suggestedExperimentInputs, experimentRoute } from './LocalExperiments';
import type { Configuration } from '../api/scientific';
import type { FeatureBundle } from '../api/bundles';
import type { ModelExperiment } from '../api/experiments';

describe('MIL experiment loading ownership', () => {
  it('keeps pack access optional and does not offer unimplemented memory residency', () => {
    const html = renderToStaticMarkup(<LoadingOptions value="auto" hasPacks={false} onChange={() => {}} />);
    expect(html).toContain('Original files');
    expect(html).toMatch(/<input[^>]*disabled=""[^>]*value="mmap"/);
    expect(html).toContain('whole pack need not fit in RAM');
    expect(html).not.toContain('value="ram"');
    expect(html).not.toContain('value="cuda"');
  });

  it('opens a record list before selecting any inputs or batches', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['model-experiments', 'project', 'summary'], { items: [] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalExperiments workspace={workspace} /></QueryClientProvider>);
      expect(html).toContain('Create experiment');
      expect(html).toContain('Create your first experiment');
      expect(html).toContain('All records');
      expect(html).toContain('Trash');
      expect(html).not.toContain('Development protocol');
      expect(html).not.toContain('Configure a training batch');
      expect(html).not.toContain('Launch batch');
    } finally { client.clear(); }
  });

  it('scopes editable inputs and exact history to the selected stable experiment', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['development-batches', 'project'], { items: [], executions: [], executionImplemented: true });
    const record: ModelExperiment = { id: 'experiment-one', key: 'draft:experiment-one', name: 'Question one', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '', updatedAt: '', inputs: null, batches: [], drafts: [], predictorId: null };
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={record} onBack={() => {}} onOpen={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Question one');
      expect(html).toContain('Development protocol');
      expect(html).toContain('Feature bundle');
      expect(html).toContain('Save the experiment inputs before creating a batch');
      expect(html).toContain('Exact input history');
      expect(html).toContain('#post-development?experiment=experiment-one');
      expect(html).not.toContain('Saved experiment inputs (');
      expect(html).not.toContain('Saving them separately is optional');
      expect(html).not.toContain('Initial project preferences');
    } finally { client.clear(); }
  });

  it('reads explicit experiment and tab identity without falling back to the latest experiment', () => {
    expect(experimentRoute('#experiments')).toEqual({ id: '', tab: 'setup' });
    expect(experimentRoute('#experiments?experiment=one%2Ftwo&tab=runs')).toEqual({ id: 'one/two', tab: 'runs' });
    expect(experimentRoute('#experiments?experiment=one&tab=inputs')).toEqual({ id: 'one', tab: 'setup' });
    expect(experimentRoute('#source-cv', 'results')).toEqual({ id: '', tab: 'results' });
  });

  it('preselects only one verified compatible pair and never guesses between versions', () => {
    const protocol = { id: 'protocol', manifest: { datasetId: 'data', spec: {} } } as Configuration;
    const feature = { id: 'features', current: true, findings: [], manifest: { datasetId: 'data', spec: { featureSetId: 'source', packArtifactIds: [] } } } as unknown as FeatureBundle;
    expect(suggestedExperimentInputs([protocol], [feature])).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features', loadingPolicy: 'auto' });
    expect(suggestedExperimentInputs([protocol], [feature, { ...feature, id: 'other' }]).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, current: false }]).featureBundleId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, findings: [{ severity: 'error', code: 'STALE', message: 'Stale features' }] }]).featureBundleId).toBe('');
    const pinned = { ...protocol, manifest: { ...protocol.manifest, spec: { featurePackId: 'required-pack' } } } as Configuration;
    expect(suggestedExperimentInputs([pinned], [feature]).protocolId).toBe('');
    const context = { datasetId: 'data', protocolId: 'protocol', bundleId: 'features' };
    expect(suggestedExperimentInputs([protocol, { ...protocol, id: 'other-protocol' }], [feature, { ...feature, id: 'other-features' }], context)).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features' });
    expect(suggestedExperimentInputs([protocol], [feature], { ...context, bundleId: 'missing' }).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [feature], { ...context, datasetId: 'different-dataset' }).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, current: false }], context).featureBundleId).toBe('');
    expect(suggestedExperimentInputs([protocol, { ...protocol, id: 'other-protocol' }], [feature], { bundleId: 'features' })).toMatchObject({ protocolId: '', featureBundleId: 'features' });
  });
});
