import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ExperimentalSetupInputs, FreezeSetupControl, setupDesign } from './ExperimentalSetupInputs';
import ExperimentRegistry, { stageExperiments, newExperimentLibraryFilters } from './ExperimentRegistry';
import ExperimentNavigation from './ExperimentNavigation';
import { type ModelExperiment } from '../api/experiments';
import { defaultRecipe, defaultResources } from '../api/development';
import { newDevelopmentSplit } from '../lib/protocol';

const record = (id: string, changes: Partial<ModelExperiment> = {}): ModelExperiment => ({
  id, key: `draft:${id}`, name: id, notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false,
  createdAt: '2026-09-25T12:00:00Z', updatedAt: '2026-09-25T12:00:00Z', inputs: null, batches: [], drafts: [], predictorId: null,
  setupVersion: 1, stage: 'planning', ...changes,
});
const ready = record('Frozen baseline', { frozenSetupId: 'setup-1', status: 'ready', configurationLocked: true,
  inputs: { protocolId: 'derived-training-only', featureBundleId: 'features', loadingPolicy: 'auto', packArtifactId: null },
  setupDesign: { datasetId: 'dataset', targetSplitId: 'targets', trainingSplit: newDevelopmentSplit() },
});
const items = [record('Draft design'), ready, record('Active experiment', { stage: 'running', status: 'running', frozenSetupId: 'setup-2' }), record('Failed experiment', { stage: 'running', status: 'failed', frozenSetupId: 'setup-3' }), record('Queued experiment', { stage: 'running', status: 'queued', frozenSetupId: 'setup-4' })];
function client() { return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }); }

describe('independent experiment preparation and execution', () => {
  it('keeps unfinished designs in setup and frozen designs in the execution registry', () => {
    expect(stageExperiments(items, 'setup').map((item) => item.id)).toEqual(items.map((item) => item.id));
    expect(stageExperiments(items, 'execution').map((item) => item.id)).toEqual(items.slice(1).map((item) => item.id));
    expect(stageExperiments([record('historical', { setupVersion: null, legacy: true, stage: 'finished', status: 'completed' })], 'execution')).toHaveLength(1);
  });

  it('filters execution by actual status rather than grouping failed and queued records under running', () => {
    const cache = client(); cache.setQueryData(['model-experiments', 'p', 'summary'], { items });
    try {
      const render = (status: string) => renderToStaticMarkup(<QueryClientProvider client={cache}><ExperimentRegistry project="p" mode="execution" onOpen={() => {}} filters={{ ...newExperimentLibraryFilters(), status }} /></QueryClientProvider>);
      const active = render('running');
      expect(active).toContain('Open Active experiment');
      for (const name of ['Draft design', 'Frozen baseline', 'Failed experiment', 'Queued experiment']) expect(active).not.toContain(`Open ${name}`);
      expect(render('ready')).toContain('Open Frozen baseline');
      // An older service's "failed" is one of the runs needing attention.
      expect(render('needs-attention')).toContain('Open Failed experiment');
      expect(render('queued')).toContain('Open Queued experiment');
      expect(active).toContain('Prepare a setup');
      expect(active).not.toContain('Create experiment');
    } finally { cache.clear(); }
  });

  it('shows one execution status, and a setup badge that never encodes run outcome', () => {
    const attention = record('gej3', { stage: 'running', status: 'needs-attention', frozenSetupId: 'setup-5', statusReason: 'Another operation is changing this workspace.' });
    const cache = client(); cache.setQueryData(['model-experiments', 'p', 'summary'], { items: [...items, attention] });
    try {
      const render = (mode: 'setup' | 'execution') => renderToStaticMarkup(<QueryClientProvider client={cache}><ExperimentRegistry project="p" mode={mode} onOpen={() => {}} /></QueryClientProvider>);
      const execution = render('execution');
      expect(execution).toContain('<span class="badge badge-orange">Needs attention</span>');
      expect(execution).toContain('Another operation is changing this workspace.');
      expect(execution).toContain('<span class="badge badge-green">Running</span>');
      expect(execution).not.toContain('>Failed<');
      const setup = render('setup');
      expect(setup.match(/badge badge-green">Frozen</g)).toHaveLength(5);
      expect(setup).not.toContain('badge-orange');
    } finally { cache.clear(); }
  });

  it('keeps saved setup identity ahead of navigation suggestions', () => {
    expect(setupDesign(ready, { datasetId: 'other', targetSplitId: 'other-target', bundleId: 'other-bundle' })).toMatchObject({ datasetId: 'dataset', targetSplitId: 'targets', featureBundleId: 'features', trainingSplit: { version: 4, mode: 'kfold' } });
    expect(setupDesign(record('new'), { datasetId: 'chosen', targetSplitId: 'chosen-target', bundleId: 'chosen-features' })).toMatchObject({ datasetId: 'chosen', targetSplitId: 'chosen-target', featureBundleId: 'chosen-features' });
  });

  it('puts folds and validation in setup and locks them when frozen', () => {
    const cache = client();
    cache.setQueryData(['scientific', 'p', 'datasets'], { datasets: [] });
    cache.setQueryData(['scientific', 'p', 'configurations', 'target-split'], { configurations: [] });
    cache.setQueryData(['feature-bundles', 'p'], { items: [] });
    try {
      const render = (readOnly: boolean) => renderToStaticMarkup(<QueryClientProvider client={cache}><ExperimentalSetupInputs project="p" record={ready} context={{}} readOnly={readOnly} onVerified={() => {}} onDirtyChange={() => {}} onBusyChange={() => {}} /></QueryClientProvider>);
      const editable = render(false);
      expect(editable).toContain('Number of folds');
      expect(editable).toContain('Early-stop validation (% of fitting data)');
      expect(editable).toContain('Check &amp; continue to hyperparameters');
      expect(editable).toContain('Testing slides are excluded from all folds and validation sets');
      for (const value of ['monte_carlo', 'nested_kfold', 'held_out', 'leave_one_domain_out']) expect(editable).toMatch(new RegExp(`<input(?=[^>]*value="${value}")(?=[^>]*disabled="")[^>]*>`));
      const frozen = render(true);
      expect(frozen).toContain('class="science-fieldset" disabled=""');
      expect(frozen).not.toContain('Check &amp; continue');
      expect(frozen).not.toContain('Start experiment');
    } finally { cache.clear(); }
  });

  it('identifies inference-only testing without changing the training design', () => {
    const cache = client();
    cache.setQueryData(['scientific', 'p', 'datasets'], { datasets: [] });
    cache.setQueryData(['feature-bundles', 'p'], { items: [] });
    cache.setQueryData(['scientific', 'p', 'configurations', 'target-split'], { configurations: [{
      id: 'targets', versionLabel: { tag: 'Inference test cohort' }, manifest: {
        spec: { datasetId: 'dataset', target: { field: 'grade', classes: ['low', 'high'] }, testTarget: null },
        summary: { trainingSlides: 16, testingSlides: 4 },
      },
    }] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={cache}><ExperimentalSetupInputs project="p" record={ready} context={{}} readOnly onVerified={() => {}} onDirtyChange={() => {}} onBusyChange={() => {}} /></QueryClientProvider>);
      expect(html).toContain('16 training slides');
      expect(html).toContain('4 inference-only testing slides (no labels or evaluation metrics)');
      expect(html).toContain('Folds, validation and model selection use training records only');
    } finally { cache.clear(); }
  });

  it('uses slide wording throughout setup and frozen review for a slide experiment', () => {
    const cache = client();
    const slide = { ...ready, setupDesign: { ...ready.setupDesign!, splitUnit: 'slide' as const } };
    cache.setQueryData(['scientific', 'p', 'datasets'], { datasets: [] });
    cache.setQueryData(['feature-bundles', 'p'], { items: [] });
    cache.setQueryData(['scientific', 'p', 'configurations', 'target-split'], { configurations: [{ id: 'targets', manifest: { spec: { datasetId: 'dataset', splitUnit: 'slide', target: { field: 'grade', classes: ['low', 'high'] } }, summary: { trainingSlides: 16, testingSlides: 4 } } }] });
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={cache}><ExperimentalSetupInputs project="p" record={slide} context={{}} readOnly onVerified={() => {}} onDirtyChange={() => {}} onBusyChange={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Slide split');
      expect(html).toContain('Every training slide rotates through one assessment fold');
      expect(html).not.toMatch(/patient|group stays intact|development group/i);
      const freeze = renderToStaticMarkup(<FreezeSetupControl project="p" record={{ ...slide, frozenSetupId: null }} disabledReason={null} onFrozen={() => {}} />);
      expect(freeze).toContain('<dt>Split unit</dt><dd>Slide</dd>');
    } finally { cache.clear(); }
  });

  it('reviews actual explicit model configurations and freezes without offering a training action', () => {
    const spec = { version: 1 as const, experimentName: 'Frozen baseline', batchName: 'Explicit models', inputs: ready.inputs!, recipe: defaultRecipe(), mode: 'explicit' as const,
      grid: { learningRates: [0.001], weightDecays: [0], maxEpochs: [10] }, configurations: [{ ...defaultRecipe(), model: 'nnmil' as const }], trainingSeeds: [42], resources: defaultResources(), notes: '' };
    const html = renderToStaticMarkup(<FreezeSetupControl project="p" record={{ ...ready, frozenSetupId: null, batchPlans: [{ id: 'batch', spec }] }} disabledReason={null} onFrozen={() => {}} />);
    expect(html).toContain('Freeze experimental setup');
    expect(html).toContain('nnMIL');
    expect(html).toContain('Training starts separately in Experiments');
    expect(html).not.toContain('Start experiment');
    const frozen = renderToStaticMarkup(<FreezeSetupControl project="p" record={ready} disabledReason={null} onFrozen={() => {}} />);
    expect(frozen).toContain('href="#experiments?experiment=Frozen+baseline"');
    expect(frozen).not.toContain('Freeze experimental setup');
  });

  it('keeps execution navigation separate from design and hyperparameters', () => {
    const render = (mode: 'setup' | 'execution', stage: 'planning' | 'running') => renderToStaticMarkup(<ExperimentNavigation mode={mode} stage={stage} current="runs" disabled={false} inputsReady hasBatches onChange={() => {}} />);
    expect(render('setup', 'planning')).toContain('Hyperparameters');
    expect(render('setup', 'planning')).toContain('Review &amp; freeze');
    expect(render('execution', 'planning')).toContain('Ready to run');
    expect(render('execution', 'running')).toContain('>Runs</button>');
    expect(render('execution', 'running')).not.toContain('>Inputs</button>');
    expect(render('execution', 'running')).not.toContain('Hyperparameters');
  });
});
