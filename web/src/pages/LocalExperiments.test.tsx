import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import LocalExperiments, { ExperimentDetail, ExperimentSubmissionControl, LoadingOptions, suggestedExperimentInputs, experimentRoute, availableExperimentTab } from './LocalExperiments';
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
      expect(html).toContain('Verified inputs are shared by every batch');
      expect(html).toContain('Check &amp; continue to batches');
      expect(html).toMatch(/id="development-tab-batches"[^>]*disabled=""/);
      expect(html).not.toContain('Save predictor choices');
      expect(html).toContain('Exact input history');
      expect(html).not.toContain('#post-development');
      expect(html).not.toContain('Create from these inputs');
      expect(html).toContain('Manage experiment');
      expect(html).not.toContain('Saved experiment inputs (');
      expect(html).not.toContain('Saving them separately is optional');
      expect(html).not.toContain('Initial project preferences');
    } finally { client.clear(); }
  });

  it('locks future stages and protects deep links while keeping saved inputs readable', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    const record: ModelExperiment = { id: 'one', key: 'draft:one', name: 'Stage test', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '', updatedAt: '', inputs: { protocolId: 'retained-protocol', featureBundleId: 'retained-bundle', loadingPolicy: 'native', packArtifactId: null }, batches: [], drafts: [], predictorId: null };
    const render = (changes: Partial<ModelExperiment>, tab?: 'setup' | 'runs' | 'results') => renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={{ ...record, ...changes }} initialTab={tab} onBack={() => {}} onOpen={() => {}} /></QueryClientProvider>);
    try {
      const planning = render({ stage: 'planning' }, 'runs');
      expect(planning).toMatch(/id="development-tab-runs"[^>]*disabled=""/);
      expect(planning).toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(planning).toMatch(/id="development-tab-batches"[^>]*aria-selected="true"/);
      expect(planning).toContain('Add at least one batch before submitting');
      const running = render({ stage: 'running', configurationLocked: true, status: 'running' });
      expect(running).toMatch(/id="development-tab-runs"[^>]*aria-selected="true"/);
      expect(running).toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(running).toMatch(/<fieldset class="mil-plan-fields" disabled=""/);
      expect(running).toContain('retained-protocol');
      expect(running).not.toContain('Check &amp; continue');
      const finished = render({ stage: 'finished', configurationLocked: true, status: 'completed' });
      expect(finished).toMatch(/id="development-tab-results"[^>]*aria-selected="true"/);
      expect(finished).not.toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(finished).toContain('Inputs, batches and runs are read-only');
      expect(finished).not.toContain('Review &amp; submit');
    } finally { client.clear(); }
    expect(availableExperimentTab('planning', 'results')).toBe('batches');
    expect(availableExperimentTab('running', 'results')).toBe('runs');
    expect(availableExperimentTab('finished', 'setup')).toBe('setup');
  });

  it('provides a controlled retry for partially submitted experiments without reopening configuration', () => {
    const client = new QueryClient();
    const record = { id: 'one', key: 'draft:one', notes: '', tags: [], inputs: null, drafts: [], predictorId: null, createdAt: '', updatedAt: '', name: 'Recover', stage: 'running', state: 'active', status: 'running', legacy: false, revision: 4, batches: [], batchPlans: [], submission: { operationId: 'original', expectedRevision: 3, status: 'attention', submittedAt: '', batchIds: ['frozen'], error: { code: 'LAUNCH_FAILED', message: 'Training environment needs repair.' }, retryable: true } } as ModelExperiment;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentSubmissionControl project="p" record={record} disabledReason="Saved configuration is locked" onSubmitted={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Retry submission'); expect(html).toContain('Training environment needs repair.');
      expect(html).toContain('saved plan remains locked');
      expect(html).not.toContain('Review &amp; submit');
      expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Retry submission/);
    } finally { client.clear(); }
  });

  it('reads explicit experiment and tab identity without falling back to the latest experiment', () => {
    expect(experimentRoute('#experiments')).toEqual({ id: '', tab: 'setup' });
    expect(experimentRoute('#experiments?experiment=one%2Ftwo&tab=runs')).toEqual({ id: 'one/two', tab: 'runs' });
    expect(experimentRoute('#experiments?experiment=one&tab=inputs')).toEqual({ id: 'one', tab: 'setup' });
    expect(experimentRoute('#experiments?experiment=one&tab=predictors')).toEqual({ id: 'one', tab: 'results' });
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
