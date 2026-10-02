import { describe, expect, it } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import type { ExperimentBatch, ModelExperiment } from '../api/experiments';
import { defaultRecipe, defaultResources } from '../api/development';
import ExperimentRegistry, { CreateExperiment, ExperimentComparison, ExperimentFilingEditor, ObjectivesManager, comparisonFieldLabel, comparisonSnapshot, experimentDifferences, filterExperiments, newExperimentLibraryFilters, statusSummary, type ExperimentLibraryFilters } from './ExperimentRegistry';
import type { Objective } from '../api/objectives';
import DevelopmentBatches from './DevelopmentBatches';

const inputs = { protocolId: 'protocol-one', featureBundleId: 'bundle-one', loadingPolicy: 'native' as const, packArtifactId: null };
const experiment = (id: string, changes: Partial<ModelExperiment> = {}): ModelExperiment => ({ id, key: `draft:${id}`, name: 'Same name', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '2026-09-01T10:00:00Z', updatedAt: '2026-09-01T10:00:00Z', inputs: null, batches: [], drafts: [], predictorId: null, ...changes });
const batch = (id: string, owner: string): ExperimentBatch => ({ id, key: `configuration:${id}`, name: id, state: 'active', status: 'planned', createdAt: '', manifest: {
  kind: 'mil-batch', version: 1, datasetId: 'data', spec: { version: 1, experimentId: owner, experimentRevision: 1, experimentName: 'Same name', batchName: id, inputs, recipe: defaultRecipe(), mode: 'single', grid: { learningRates: [0.001], weightDecays: [0], maxEpochs: [10] }, configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' },
  configurations: [{ id: 'config', number: 1, recipe: defaultRecipe() }], splitPlans: [], runs: [], summary: { configurationCount: 1, trainingSeedCount: 1, splitPlanCount: 3, runCount: 3 }, executionImplemented: true, previewHash: 'preview', resolvedInputs: { canPlan: true, findings: [], featureKind: 'patch', resolvedLoadingPolicy: 'native', packArtifactId: null, featureSetId: 'features', bundleId: 'bundle-one', executionImplemented: false },
} });

describe('experiment registry and exact comparison', () => {
  it('keeps same-name experiment identities separate and filters archived and deleted records explicitly', () => {
    const items = [experiment('one', { tags: ['baseline'] }), experiment('two', { state: 'archived', status: 'completed' }), experiment('three', { state: 'trashed', notes: 'discarded learning rate', status: 'failed' })];
    expect(filterExperiments(items, 'active', '', '', 'name').map((item) => item.id)).toEqual(['one']);
    expect(filterExperiments(items, 'archived', 'completed', 'same', 'name').map((item) => item.id)).toEqual(['two']);
    expect(filterExperiments(items, 'all', '', 'learning rate', 'name').map((item) => item.id)).toEqual(['three']);
    expect(filterExperiments(items, 'all', '', 'baseline', 'name').map((item) => item.id)).toEqual(['one']);
    expect(filterExperiments(items, 'all', 'planning', '', 'name').map((item) => item.id)).toEqual(['one']);
    expect(filterExperiments(items, 'all', 'finished', '', 'name').map((item) => item.id)).toEqual(['two']);
    expect(items.map((item) => item.id)).toEqual(['one', 'two', 'three']);
  });

  it('compares deeply nested recipe, data, protocol, feature, seed and resource fields without guessing absent history', () => {
    const left = { dataset: { contentHash: 'data-a' }, protocol: { spec: { target: 'tumor' } }, features: { id: 'features-a' }, configurations: [{ recipe: { learningRate: 0.001, bagSize: null } }], seeds: [42], resources: { threads: 2 }, optional: null };
    const right = { dataset: { contentHash: 'data-b' }, protocol: { spec: { target: 'tumor' } }, features: { id: 'features-a' }, configurations: [{ recipe: { learningRate: 0.002, bagSize: null } }], seeds: [43], resources: { threads: 4 } };
    const rows = experimentDifferences([left, right], true);
    expect(rows.map((row) => row.path)).toEqual(['configurations[0].recipe.learningRate', 'dataset.contentHash', 'optional', 'resources.threads', 'seeds']);
    expect(rows.find((row) => row.path === 'optional')?.values).toEqual([null, undefined]);
    expect(experimentDifferences([left, right], false).some((row) => row.path === 'protocol.spec.target')).toBe(true);
    expect(comparisonFieldLabel('batch.configurations[0].learningRate')).toBe('Batch / Configuration 1 / Learning rate');
  });

  it('uses an explicitly chosen frozen batch snapshot even after experiment defaults change', () => {
    const saved = batch('frozen', 'one');
    saved.inputSnapshot = { dataset: { id: 'original-data', contentHash: 'original-hash' } };
    const item = experiment('one', { inputs: { ...inputs, protocolId: 'later-protocol' }, batches: [saved] });
    expect(comparisonSnapshot(item, 'frozen')).toMatchObject({ dataset: { contentHash: 'original-hash' }, inputs: { protocolId: 'protocol-one' }, batch: { trainingSeeds: [42], resources: defaultResources() } });
    expect(comparisonSnapshot(item, '')).toEqual({ inputs: item.inputs });
    expect(comparisonSnapshot(item, 'other-experiment-batch')).not.toHaveProperty('batch');
  });

  it('compares editable recipes before submission without duplicating retained plans afterward', () => {
    const spec = batch('saved-plan', 'one').manifest.spec;
    const item = experiment('one', { inputs, stage: 'planning', batchPlans: [{ id: 'editable', spec }] });
    expect(comparisonSnapshot(item, 'plan:editable')).toMatchObject({ inputs, batch: { recipe: defaultRecipe(), trainingSeeds: [42], resources: defaultResources() } });
    expect(comparisonSnapshot(item, 'plan:editable')).not.toHaveProperty('batch.splitPlans');
    const planning = renderToStaticMarkup(<ExperimentComparison items={[item, experiment('two')]} />);
    expect(planning).toContain('value="plan:editable"');
    expect(planning).toContain('Editable recipe');
    const finished = renderToStaticMarkup(<ExperimentComparison items={[{ ...item, stage: 'finished', batches: [batch('submitted', 'one')] }, experiment('two')]} />);
    expect(finished).not.toContain('value="plan:editable"');
    expect(finished).toContain('value="submitted"');
  });

  it('shows create-first fields and comparison baseline controls with inspectable missing values', () => {
    const create = renderToStaticMarkup(<CreateExperiment project="p" onCreated={() => {}} onClose={() => {}} />);
    expect(create).toContain('Experiment name'); expect(create).toContain('Tags'); expect(create).toContain('Notes');
    expect(create).toContain('Start from template'); expect(create).toContain('Blank experiment');
    expect(create).not.toContain('Development protocol'); expect(create).not.toContain('Launch batch');
    const compare = renderToStaticMarkup(<ExperimentComparison items={[experiment('one', { inputSnapshot: { dataset: { id: 'd' } } }), experiment('two')]} />);
    expect(compare).toContain('Baseline'); expect(compare).toContain('Differences only'); expect(compare).toContain('Not recorded');
    expect(compare).toContain('dataset.id'); expect(compare).toContain('No current project defaults are substituted');
  });

  it('offers existing experiments as complete editable templates while excluding trash', () => {
    const source = experiment('source-one', { name: 'Baseline', inputs, state: 'archived', stage: 'finished', batches: [batch('batch-one', 'source-one')] });
    const html = renderToStaticMarkup(<CreateExperiment project="p" copy={source} templates={[source, experiment('trash-only', { state: 'trashed' })]} onCreated={() => {}} onClose={() => {}} />);
    expect(html).toContain('Baseline copy');
    expect(html).toContain('saved inputs and batch recipes');
    expect(html).toContain('editable plan');
    expect(html).toContain('Runs and results stay with the source experiment');
    expect(html).toContain('Create &amp; open inputs');
    expect(html).not.toContain('trash-only');
    expect(html).not.toContain('Only saved inputs are copied');
  });

  it('renders all lifecycle views without mixing another experiment’s same-name batches into details', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const owned = batch('owned-only', 'one');
    client.setQueryData(['development-batches', 'p'], { items: [owned, batch('UNRELATED-BATCH', 'two')], executions: [], executionImplemented: true });
    client.setQueryData(['model-experiments', 'p', 'summary'], { items: [experiment('one', { batches: [owned] }), experiment('trash', { state: 'trashed' })] });
    try {
      const registry = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentRegistry project="p" onOpen={() => {}} /></QueryClientProvider>);
      expect(registry).toContain('Archived'); expect(registry).toContain('Trash'); expect(registry).toContain('All records');
      expect(registry).toContain('All statuses'); expect(registry).toContain('Draft'); expect(registry).toContain('Ready to run');
      expect(registry).toContain('aria-label="Search experiments"');
      expect(registry).toContain('aria-label="Experiment state"');
      expect(registry).toContain('aria-label="Saved experiments"');
      expect(registry).toContain('1 batch');
      expect(registry).not.toContain('1 batches');
      expect(registry).not.toContain('Your experiments');
      expect(registry).not.toContain('Stage 0 · Saved records');
      expect(registry).not.toContain('Compare selected experiments');
      expect(registry).not.toContain('Archive keeps');
      expect(registry).not.toContain('Predictors'); expect(registry).not.toContain('#post-development'); expect(registry).not.toContain('#apply');
      const detail = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned]} ownedDrafts={[]} tab="runs" onOpenSetup={() => {}} /></QueryClientProvider>);
      expect(detail).toContain('owned-only'); expect(detail).not.toContain('UNRELATED-BATCH');
      const multiple = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned, batch('second-owned', 'one')]} ownedDrafts={[]} tab="runs" onOpenSetup={() => {}} /></QueryClientProvider>);
      expect(multiple).toContain('Choose a batch in this experiment'); expect(multiple).not.toContain('Launch batch');
    } finally { client.clear(); }
  });

  it('uses current service execution capability even when the immutable historical plan flag is false', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const owned = batch('historical-plan', 'one'); owned.manifest.executionImplemented = false;
    client.setQueryData(['training-runtime', 'p'], { available: true, python: '/training/python', versions: {}, cudaAvailable: false, gpuCount: 0, findings: [] });
    client.setQueryData(['training-execution', 'p', owned.id], null);
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned]} ownedDrafts={[]} executionImplemented tab="runs" onOpenSetup={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Launch batch');
      expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Launch batch/);
      expect(html).not.toContain('Training execution is unavailable');
      expect(owned.manifest.executionImplemented).toBe(false);
    } finally { client.clear(); }
  });
});

describe('experiments filed under objectives', () => {
  const aims: Objective[] = [
    { id: 'objective-grade', name: 'Dysplasia grade', rank: 'primary', description: 'ND, IND, LG and HG on consensus labels.', revision: 1, createdAt: '', updatedAt: '' },
    { id: 'objective-p53', name: 'p53 IHC pattern', rank: 'exploratory', description: '', revision: 1, createdAt: '', updatedAt: '' },
  ];
  const under = (id: string) => ({ objectiveId: id, objective: { id, name: aims.find((aim) => aim.id === id)!.name, rank: aims.find((aim) => aim.id === id)!.rank } });
  const items = [
    experiment('consented', { name: 'gej_consented', ...under('objective-grade'), role: 'primary', status: 'completed', stage: 'finished' }),
    experiment('gej3', { name: 'gej3', ...under('objective-grade'), role: 'variant', status: 'completed', stage: 'finished' }),
    experiment('v1', { name: 'gej_v1', ...under('objective-grade'), role: 'tuning', status: 'completed', stage: 'finished' }),
    experiment('loco-a', { name: 'LOCO site a', ...under('objective-grade'), role: 'generalization', series: 'LOCO', seriesMember: 'site a', status: 'queued', stage: 'running' }),
    experiment('loco-b', { name: 'LOCO site b', ...under('objective-grade'), role: 'generalization', series: 'LOCO', seriesMember: 'site b', status: 'running', stage: 'running' }),
    experiment('p53', { name: 'gej_p53', ...under('objective-p53'), role: 'primary', status: 'completed', stage: 'finished' }),
    experiment('loose', { name: 'Loose idea' }),
  ];
  function render(filters?: ExperimentLibraryFilters) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    client.setQueryData(['model-experiments', 'p', 'summary'], { items });
    client.setQueryData(['objectives', 'p'], { revision: 3, items: aims });
    client.setQueryData(['experiment-headlines', 'p'], { items: [] });
    try { return renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentRegistry project="p" onOpen={() => {}} filters={filters} /></QueryClientProvider>); }
    finally { client.clear(); }
  }

  it('lists objectives by rank with roles in order, a series as one row and tuning folded away', () => {
    const html = render();
    expect(html.indexOf('Dysplasia grade')).toBeLessThan(html.indexOf('p53 IHC pattern'));
    expect(html.indexOf('p53 IHC pattern')).toBeLessThan(html.indexOf('No objective'));
    expect(html).toContain('Primary objective'); expect(html).toContain('Exploratory objective');
    expect(html).toContain('ND, IND, LG and HG on consensus labels.');
    expect(html).toContain('5 experiments · 1 running · 1 queued · 3 completed');
    const roles = ['>Primary<', '>Variant<', '>Generalization<'].map((label) => html.indexOf(label));
    expect(roles).toEqual([...roles].sort((a, b) => a - b));
    // The series is one row until opened.
    expect(html).toContain('2 experiments · site a, site b');
    expect(html).not.toContain('title="Open LOCO site a"');
    // Tuning sits behind one row.
    expect(html).toContain('Pilots, hyperparameter rounds and earlier versions: gej_v1');
    expect(html).not.toContain('title="Open gej_v1"');
    expect(html).toContain('Group by objective'); expect(html).toContain('>Objectives</button>');
    expect(html).toContain('title="Open Loose idea"');
  });

  it('opens what a search finds, and lists one table when not grouped', () => {
    const searched = render({ ...newExperimentLibraryFilters(), search: 'site' });
    expect(searched).toContain('title="Open LOCO site a"');
    expect(searched).not.toContain('p53 IHC pattern');
    const tuning = render({ ...newExperimentLibraryFilters(), search: 'gej_v1' });
    expect(tuning).toContain('title="Open gej_v1"');
    const flat = render({ ...newExperimentLibraryFilters(), grouped: false });
    expect(flat).not.toContain('No objective');
    expect(flat).toContain('title="Open LOCO site a"');
    expect(flat).toContain('class="experiment-filed-under">Dysplasia grade');
    expect(filterExperiments(items, 'all', '', 'p53 ihc', 'name').map((item) => item.id)).toEqual(['p53']);
    expect(filterExperiments(items, 'all', '', 'generalization', 'name').map((item) => item.id)).toEqual(['loco-a', 'loco-b']);
  });

  it('summarizes statuses with live work first', () => {
    expect(statusSummary([experiment('a', { status: 'completed', stage: 'finished' }), experiment('b', { status: 'queued', stage: 'running' }), experiment('c', { status: 'created', stage: 'planning' })])).toBe('1 queued · 1 draft · 1 completed');
  });

  it('files a new experiment, and a copy under its source objective as tuning', () => {
    const blank = renderToStaticMarkup(<CreateExperiment project="p" objectives={aims} onCreated={() => {}} onClose={() => {}} />);
    expect(blank).toContain('>Objective<'); expect(blank).toContain('>Role<');
    expect(blank).toContain('Dysplasia grade · Primary objective');
    expect(blank).toMatch(/<option value="" selected="">No objective<\/option>/);
    const copy = experiment('consented', { name: 'gej_consented', ...under('objective-grade'), role: 'primary', inputs });
    const copied = renderToStaticMarkup(<CreateExperiment project="p" copy={copy} objectives={aims} onCreated={() => {}} onClose={() => {}} />);
    expect(copied).toMatch(/<option value="objective-grade" selected="">Dysplasia grade/);
    expect(copied).toMatch(/<option value="tuning" selected="">Tuning<\/option>/);
    expect(copied).toContain(experimentRoleHelpText);
  });

  it('files an experiment from its page and manages the objectives', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    try {
      const record = experiment('loco-a', { name: 'LOCO site a', ...under('objective-grade'), role: 'generalization', series: 'LOCO', seriesMember: 'site a', copiedFrom: { id: 'consented', name: 'gej_consented' } });
      const editor = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentFilingEditor project="p" record={record} objectives={aims} seriesNames={['LOCO']} /></QueryClientProvider>);
      expect(editor).toMatch(/<option value="generalization" selected="">Generalization<\/option>/);
      expect(editor).toContain('value="LOCO"'); expect(editor).toContain('value="site a"');
      expect(editor).toContain('Copied from gej_consented.');
      expect(editor).toMatch(/<button[^>]*disabled=""[^>]*>Save filing<\/button>/);
      const manager = renderToStaticMarkup(<QueryClientProvider client={client}><ObjectivesManager project="p" items={aims} experiments={items} /></QueryClientProvider>);
      expect(manager).toContain('aria-label="Objectives"');
      expect(manager).toMatch(/Dysplasia grade<\/strong><small>ND, IND, LG and HG on consensus labels.<\/small><\/th><td><span class="badge badge-purple">Primary objective<\/span><\/td><td>5<\/td>/);
      expect(manager).toContain('Add objective');
      // With a primary objective saved, a new one starts as secondary.
      expect(manager).toMatch(/<option value="secondary" selected="">Secondary objective<\/option>/);
      for (const role of ['Primary', 'Comparison', 'Variant', 'Generalization', 'Control', 'Tuning']) expect(manager).toContain(`<dt>${role}</dt>`);
    } finally { client.clear(); }
  });
});
const experimentRoleHelpText = 'Pilots, hyperparameter rounds and earlier versions. Folded away in the library.';
