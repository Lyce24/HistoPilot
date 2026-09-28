import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import LocalExperiments, { ExperimentDetail, ExperimentSubmissionControl, LoadingOptions, suggestedExperimentInputs, experimentRoute, availableExperimentTab, isInputDraft } from './LocalExperiments';
import type { Configuration } from '../api/scientific';
import type { FeatureBundle } from '../api/bundles';
import type { ModelExperiment } from '../api/experiments';
import { fixtureRollup } from '../testFixtures/taskCenter';
import { taskCenterKeys } from '../api/taskCenter';

describe('MIL experiment loading ownership', () => {
  it('validates recovered input drafts with their baseline before rendering', () => {
    const spec = { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'auto', packArtifactId: null };
    expect(isInputDraft({ spec, baseInputs: null })).toBe(true);
    expect(isInputDraft({ spec, baseInputs: spec })).toBe(true);
    expect(isInputDraft({ spec })).toBe(false);
    expect(isInputDraft({ spec: { ...spec, loadingPolicy: 'cuda' }, baseInputs: null })).toBe(false);
    expect(isInputDraft({ spec, baseInputs: { protocolId: 'other' } })).toBe(false);
  });

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
      // State and stage filters return with the first saved experiment.
      expect(html).not.toContain('All records');
      expect(html).not.toContain('Trash');
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

  it('selects experiment features independently for dataset-only targets and splits', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const protocol = { id: 'protocol', manifest: { datasetId: 'development-dataset', spec: {
      target: { field: 'label', task: 'binary_classification', unit: 'patient' },
      split: { mode: 'kfold', seeds: [42] },
    } } } as Configuration;
    const feature = { id: 'bundle', current: true, findings: [], manifest: {
      datasetId: 'other-dataset', spec: { featureSetId: 'source', packArtifactIds: [] },
      summary: { slideCount: 12, patchCount: 120, dimensions: 4, dtype: 'float32', packCount: 0 }, packs: [],
    } } as unknown as FeatureBundle;
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [protocol] });
    client.setQueryData(['feature-bundles', 'project'], { items: [feature] });
    client.setQueryData(['development-batches', 'project'], { items: [], executions: [], executionImplemented: true });
    const record: ModelExperiment = { id: 'dataset-only', key: 'draft:dataset-only', name: 'Dataset-only inputs', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '', updatedAt: '', inputs: null, batches: [], drafts: [], predictorId: null };
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={record} onBack={() => {}} onOpen={() => {}} /></QueryClientProvider>);
      expect(html).toMatch(/<option value="bundle" selected=""/);
      expect(html).not.toMatch(/<option value="bundle"[^>]*disabled/);
      expect(html).toContain('This experiment selects its feature bundle');
      expect(html).toContain('complete development-slide coverage');
      expect(html).toContain('saved targets and split memberships stay fixed');
      expect(html).not.toContain('feature bundle is selected automatically');
      expect(html).not.toContain('Feature bundle pinned by older protocol');
      expect(html).toMatch(/id="development-tab-batches"[^>]*disabled=""/);
    } finally { client.clear(); }
  });

  it('locks future stages and protects deep links while keeping saved inputs readable', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    const record: ModelExperiment = { id: 'one', key: 'draft:one', name: 'Stage test', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '', updatedAt: '', inputs: { protocolId: 'retained-protocol', featureBundleId: 'retained-bundle', loadingPolicy: 'native', packArtifactId: null }, batches: [], drafts: [], predictorId: null };
    const render = (changes: Partial<ModelExperiment>, tab?: 'setup' | 'runs' | 'results' | 'review') => renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={{ ...record, ...changes }} initialTab={tab} onBack={() => {}} onOpen={() => {}} /></QueryClientProvider>);
    try {
      const planning = render({ stage: 'planning' }, 'runs');
      expect(planning).toMatch(/id="development-tab-runs"[^>]*disabled=""/);
      expect(planning).toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(planning).toMatch(/id="development-tab-batches"[^>]*aria-current="step"/);
      expect(planning).toContain('Continue to review &amp; submit');
      expect(planning).not.toContain('Freeze &amp; submit experiment');
      const review = render({ stage: 'planning' }, 'review');
      expect(review).toContain('Add at least one batch before submitting');
      expect(review).toContain('data-stage-page="review"');
      expect(review).toMatch(/<button[^>]*disabled=""[^>]*>Freeze &amp; submit experiment/);
      const running = render({ stage: 'running', configurationLocked: true, status: 'running' });
      expect(running).toMatch(/id="development-tab-runs"[^>]*aria-current="page"/);
      expect(running).toContain('Submitted plan');
      expect(running).toContain('aria-label="Experiment views"');
      expect(running).not.toContain('aria-label="Experiment setup"');
      // Results fill in seed by seed while the experiment runs; predictors have their own view.
      expect(running).not.toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(running).toContain('Partial results: training seeds appear as their folds finish.');
      expect(running).toContain('Results fill in as test folds and training seeds finish.');
      expect(running).toMatch(/id="development-tab-predictors"/);
      expect(running).toMatch(/<fieldset class="mil-plan-fields" disabled=""/);
      expect(running).toContain('retained-protocol');
      expect(running).not.toContain('Check &amp; continue');
      const finished = render({ stage: 'finished', configurationLocked: true, status: 'completed' });
      expect(finished).toMatch(/id="development-tab-results"[^>]*aria-current="page"/);
      expect(finished).not.toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(finished).toContain('Inputs, batches and runs are read-only');
      expect(finished).toContain('ready predictors are under Predictors');
      expect(finished).toContain('Continue to predictors');
      expect(finished).not.toContain('Review &amp; submit');
    } finally { client.clear(); }
    expect(availableExperimentTab('planning', 'results')).toBe('batches');
    expect(availableExperimentTab('running', 'results')).toBe('results');
    expect(availableExperimentTab('finished', 'setup')).toBe('setup');
    expect(availableExperimentTab('planning', 'predictors')).toBe('batches');
    expect(availableExperimentTab('finished', 'predictors')).toBe('predictors');
  });

  it('opens partial results while running and shows the experiment queue bar in Runs only', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['predictors', 'project'], { items: [] });
    client.setQueryData(taskCenterKeys.rollup({ ownerKind: 'experiment', ownerId: 'one', project: 'project' }), fixtureRollup({ state: 'queued', active: 0, queuePosition: 1, waitingReason: null, eta: null }));
    const record = { id: 'one', key: 'draft:one', name: 'Queued study', notes: '', tags: [], revision: 2, state: 'active', status: 'running', stage: 'running', configurationLocked: true, frozenSetupId: 'setup', legacy: false, createdAt: '', updatedAt: '', inputs: { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native', packArtifactId: null }, batches: [], drafts: [], predictorId: null, submission: { operationId: 'op', expectedRevision: 1, submittedAt: '', status: 'submitted', batchIds: [], error: null, retryable: false } } as ModelExperiment;
    const render = (tab: 'runs' | 'results' | 'predictors') => renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail mode="execution" workspace={workspace} record={record} initialTab={tab} onBack={() => {}} onOpen={() => {}} /></QueryClientProvider>);
    try {
      const runs = render('runs');
      expect(runs).toContain('#1 in line');
      expect(runs).toContain('href="#task-center?owner=owner-1&amp;project=project"');
      const results = render('results');
      expect(results).toMatch(/id="development-tab-results"[^>]*aria-current="page"/);
      expect(results).not.toContain('#1 in line');
      // The results view replaces the old per-batch picker and predictor library.
      expect(results).toContain('Loading results…');
      expect(results).not.toContain('Choose a batch in this experiment');
      expect(results).not.toContain('ready to evaluate');
      const predictors = render('predictors');
      expect(predictors).toMatch(/id="development-tab-predictors"[^>]*aria-current="page"/);
      expect(predictors).toContain('ready to evaluate');
    } finally { client.clear(); }
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
    expect(experimentRoute('#experiments?experiment=one&tab=predictors')).toEqual({ id: 'one', tab: 'predictors' });
    expect(experimentRoute('#experiments?experiment=one&tab=review')).toEqual({ id: 'one', tab: 'review' });
    expect(experimentRoute('#source-cv', 'results')).toEqual({ id: '', tab: 'results' });
  });

  it('retains an explicit dataset-only protocol while its experiment features are still missing', () => {
    const protocol = { id: 'protocol', manifest: { datasetId: 'data', spec: {} } } as Configuration;
    expect(suggestedExperimentInputs([protocol], [], { datasetId: 'data', protocolId: 'protocol' })).toMatchObject({ protocolId: 'protocol', featureBundleId: '' });
    expect(suggestedExperimentInputs([protocol], [], { protocolId: 'missing' }).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [], { datasetId: 'other', protocolId: 'protocol' }).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], []).protocolId).toBe('');
  });

  it('preselects only one verified compatible pair and never guesses between versions', () => {
    const protocol = { id: 'protocol', manifest: { datasetId: 'data', spec: {} } } as Configuration;
    const feature = { id: 'features', current: true, findings: [], manifest: { datasetId: 'data', spec: { featureSetId: 'source', packArtifactIds: [] } } } as unknown as FeatureBundle;
    expect(suggestedExperimentInputs([protocol], [feature])).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features', loadingPolicy: 'auto' });
    expect(suggestedExperimentInputs([protocol], [feature, { ...feature, id: 'other' }]).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, current: false }]).featureBundleId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, findings: [{ severity: 'error', code: 'STALE', message: 'Stale features' }] }]).featureBundleId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, manifest: { ...feature.manifest, datasetId: 'another-dataset' } }])).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features' });
    const named = { ...protocol, manifest: { ...protocol.manifest, spec: { featureBundleId: 'features' } } } as Configuration;
    expect(suggestedExperimentInputs([named], [{ ...feature, manifest: { ...feature.manifest, datasetId: 'another-dataset' } }, { ...feature, id: 'other' }])).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features' });
    const pinned = { ...protocol, manifest: { ...protocol.manifest, spec: { featurePackId: 'required-pack' } } } as Configuration;
    expect(suggestedExperimentInputs([pinned], [feature]).protocolId).toBe('');
    const context = { datasetId: 'data', protocolId: 'protocol', bundleId: 'features' };
    expect(suggestedExperimentInputs([protocol, { ...protocol, id: 'other-protocol' }], [feature, { ...feature, id: 'other-features' }], context)).toMatchObject({ protocolId: 'protocol', featureBundleId: 'features' });
    expect(suggestedExperimentInputs([protocol], [feature], { ...context, bundleId: 'missing' })).toMatchObject({ protocolId: 'protocol', featureBundleId: '' });
    expect(suggestedExperimentInputs([protocol], [feature], { ...context, datasetId: 'different-dataset' }).protocolId).toBe('');
    expect(suggestedExperimentInputs([protocol], [{ ...feature, current: false }], context).featureBundleId).toBe('');
    expect(suggestedExperimentInputs([protocol, { ...protocol, id: 'other-protocol' }], [feature], { bundleId: 'features' })).toMatchObject({ protocolId: '', featureBundleId: 'features' });
  });
});
