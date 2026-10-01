import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import LocalExperiments, { ExperimentDetail, ExperimentSubmissionControl, experimentRoute } from './LocalExperiments';
import type { ModelExperiment } from '../api/experiments';
import { fixtureRollup } from '../testFixtures/taskCenter';
import { taskCenterKeys } from '../api/taskCenter';

describe('Experiments from design to results', () => {
  it('opens a record list before selecting any inputs or batches', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['model-experiments', 'project', 'summary'], { items: [] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><LocalExperiments workspace={workspace} /></QueryClientProvider>);
      expect(html).toContain('Create experiment');
      expect(html).toContain('Create your first experiment');
      expect(html).toContain('03 Experiments');
      // State and stage filters return with the first saved experiment.
      expect(html).not.toContain('All records');
      expect(html).not.toContain('Trash');
      expect(html).not.toContain('Development protocol');
      expect(html).not.toContain('Configure a training batch');
      expect(html).not.toContain('Launch batch');
    } finally { client.clear(); }
  });

  it('keeps started records read-only with their design, runs, partial results and predictors views', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['scientific', 'project', 'configurations', 'target-split'], { configurations: [] });
    client.setQueryData(['scientific', 'project', 'datasets'], { datasets: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    const record: ModelExperiment = { id: 'one', key: 'draft:one', name: 'Stage test', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '', updatedAt: '', inputs: { protocolId: 'retained-protocol', featureBundleId: 'retained-bundle', loadingPolicy: 'native', packArtifactId: null }, batches: [], drafts: [], predictorId: null };
    const render = (changes: Partial<ModelExperiment>, tab?: 'setup' | 'runs' | 'results' | 'review') => renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={{ ...record, ...changes }} initialTab={tab} onBack={() => {}} /></QueryClientProvider>);
    try {
      const running = render({ stage: 'running', configurationLocked: true, status: 'running' });
      expect(running).toMatch(/id="development-tab-runs"[^>]*aria-current="page"/);
      expect(running).toContain('Started experiment');
      expect(running).toContain('aria-label="Experiment views"');
      expect(running).not.toContain('aria-label="Experiment design"');
      // Results fill in seed by seed while the experiment runs; predictors have their own view.
      expect(running).not.toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(running).toContain('Partial results: training seeds appear as their folds finish.');
      expect(running).toContain('Results fill in as test folds and training seeds finish.');
      expect(running).toMatch(/id="development-tab-predictors"/);
      expect(running).toContain('retained-protocol');
      expect(running).not.toContain('Check &amp; continue');
      // The detail header gives the service's reason for queued and waiting work.
      expect(render({ stage: 'running', configurationLocked: true, status: 'queued', statusReason: 'Waiting for a free GPU.' })).toContain('<p class="experiment-detail-reason" role="status">Waiting for a free GPU.</p>');
      expect(render({ stage: 'running', configurationLocked: true, status: 'running', statusReason: 'stale' })).not.toContain('experiment-detail-reason');
      const finished = render({ stage: 'finished', configurationLocked: true, status: 'completed' });
      expect(finished).toMatch(/id="development-tab-results"[^>]*aria-current="page"/);
      expect(finished).not.toMatch(/id="development-tab-results"[^>]*disabled=""/);
      expect(finished).toContain('The design and runs are read-only');
      expect(finished).toContain('ready predictors are under Predictors');
      expect(finished).toContain('Continue to predictors');
      expect(finished).not.toContain('Review &amp; submit');
      // Its design stays one view away, read-only.
      const design = render({ stage: 'finished', configurationLocked: true, status: 'completed' }, 'setup');
      expect(design).toMatch(/id="development-tab-setup"[^>]*aria-current="page"/);
      expect(design).toContain('class="science-fieldset" disabled=""');
      expect(design).toContain('Continue to hyperparameters');
      // A review link to a started experiment opens its runs, never a submission step.
      expect(render({ stage: 'running', configurationLocked: true, status: 'running' }, 'review')).toMatch(/id="development-tab-runs"[^>]*aria-current="page"/);
      // Before it starts, a design has steps and no runs; a frozen design opens on starting it.
      const draft = render({ stage: 'planning', setupVersion: 1 }, 'runs');
      expect(draft).toContain('aria-label="Experiment design"');
      expect(draft).not.toContain('id="development-tab-runs"');
      const ready = render({ stage: 'planning', status: 'ready', configurationLocked: true, frozenSetupId: 'setup', setupVersion: 1, setupDesign: { datasetId: 'dataset', targetSplitId: 'targets', trainingSplit: { version: 4, mode: 'kfold', folds: 5, seeds: [42], validationFraction: 0.15, stratify: true } as never }, batchPlans: [] });
      expect(ready).toContain('Frozen design');
      expect(ready).toContain('Start experiment');
      expect(ready).not.toContain('Freeze design<');
    } finally { client.clear(); }
  });

  it('opens partial results while running and shows the experiment queue bar in Runs only', () => {
    const workspace = { project: { id: 'project', name: 'BLCA', config: {} } } as Workspace;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['scientific', 'project', 'configurations', 'protocol'], { configurations: [] });
    client.setQueryData(['feature-bundles', 'project'], { items: [] });
    client.setQueryData(['predictors', 'project'], { items: [] });
    client.setQueryData(taskCenterKeys.rollup({ ownerKind: 'experiment', ownerId: 'one', project: 'project' }), fixtureRollup({ state: 'queued', active: 0, queuePosition: 1, waitingReason: null, eta: null }));
    const record = { id: 'one', key: 'draft:one', name: 'Queued study', notes: '', tags: [], revision: 2, state: 'active', status: 'running', stage: 'running', configurationLocked: true, frozenSetupId: 'setup', legacy: false, createdAt: '', updatedAt: '', inputs: { protocolId: 'protocol', featureBundleId: 'bundle', loadingPolicy: 'native', packArtifactId: null }, batches: [], drafts: [], predictorId: null, submission: { operationId: 'op', expectedRevision: 1, submittedAt: '', status: 'submitted', batchIds: [], error: null, retryable: false } } as ModelExperiment;
    const render = (tab: 'runs' | 'results' | 'predictors') => renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentDetail workspace={workspace} record={record} initialTab={tab} onBack={() => {}} /></QueryClientProvider>);
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
      expect(results).not.toContain('ready to apply');
      const predictors = render('predictors');
      expect(predictors).toMatch(/id="development-tab-predictors"[^>]*aria-current="page"/);
      expect(predictors).toContain('ready to apply');
    } finally { client.clear(); }
  });

  it('provides a controlled retry for partially submitted experiments without reopening configuration', () => {
    const client = new QueryClient();
    const record = { id: 'one', key: 'draft:one', notes: '', tags: [], inputs: null, drafts: [], predictorId: null, createdAt: '', updatedAt: '', name: 'Recover', stage: 'running', state: 'active', status: 'running', legacy: false, revision: 4, batches: [], batchPlans: [], submission: { operationId: 'original', expectedRevision: 3, status: 'attention', submittedAt: '', batchIds: ['frozen'], error: { code: 'LAUNCH_FAILED', message: 'Training environment needs repair.' }, retryable: true } } as ModelExperiment;
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentSubmissionControl project="p" record={record} disabledReason="Saved configuration is locked" onSubmitted={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Retry start'); expect(html).toContain('Training environment needs repair.');
      expect(html).toContain('The experiment did not start:'); expect(html).toContain('frozen design remains locked');
      expect(html).toContain('aria-label="Start experiment"');
      // Starting is the one verb: no submission wording survives in what the user reads.
      expect(html.replace(/<[^>]+>/g, ' ')).not.toMatch(/submi/i);
      expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Retry start/);
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
});
