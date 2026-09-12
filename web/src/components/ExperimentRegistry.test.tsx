import { describe, expect, it } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import type { ExperimentBatch, ModelExperiment } from '../api/experiments';
import { defaultRecipe, defaultResources } from '../api/development';
import ExperimentRegistry, { CreateExperiment, ExperimentComparison, comparisonFieldLabel, comparisonSnapshot, experimentDifferences, filterExperiments } from './ExperimentRegistry';
import DevelopmentBatches from './DevelopmentBatches';

const inputs = { protocolId: 'protocol-one', featureBundleId: 'bundle-one', loadingPolicy: 'native' as const, packArtifactId: null };
const experiment = (id: string, changes: Partial<ModelExperiment> = {}): ModelExperiment => ({ id, key: `draft:${id}`, name: 'Same name', notes: '', tags: [], revision: 1, state: 'active', status: 'created', legacy: false, createdAt: '2026-09-01T10:00:00Z', updatedAt: '2026-09-01T10:00:00Z', inputs: null, batches: [], drafts: [], predictorId: null, ...changes });
const batch = (id: string, owner: string): ExperimentBatch => ({ id, key: `configuration:${id}`, name: id, state: 'active', status: 'planned', createdAt: '', manifest: {
  kind: 'mil-batch', version: 1, datasetId: 'data', spec: { version: 1, experimentId: owner, experimentRevision: 1, experimentName: 'Same name', batchName: id, inputs, recipe: defaultRecipe(), mode: 'single', grid: { learningRates: [0.001], weightDecays: [0], maxEpochs: [10] }, configurations: [], trainingSeeds: [42], resources: defaultResources(), notes: '' },
  configurations: [{ id: 'config', number: 1, recipe: defaultRecipe() }], splitPlans: [], runs: [], summary: { configurationCount: 1, trainingSeedCount: 1, splitPlanCount: 3, runCount: 3 }, executionImplemented: true, previewHash: 'preview', resolvedInputs: { canPlan: true, findings: [], resolvedLoadingPolicy: 'native', packArtifactId: null, featureSetId: 'features', bundleId: 'bundle-one', executionImplemented: false },
} });

describe('experiment registry and exact comparison', () => {
  it('keeps same-name experiment identities separate and filters archived and deleted records explicitly', () => {
    const items = [experiment('one', { tags: ['baseline'] }), experiment('two', { state: 'archived', status: 'completed' }), experiment('three', { state: 'trashed', notes: 'discarded learning rate', status: 'failed' })];
    expect(filterExperiments(items, 'active', '', '', 'name').map((item) => item.id)).toEqual(['one']);
    expect(filterExperiments(items, 'archived', 'completed', 'same', 'name').map((item) => item.id)).toEqual(['two']);
    expect(filterExperiments(items, 'all', '', 'learning rate', 'name').map((item) => item.id)).toEqual(['three']);
    expect(filterExperiments(items, 'all', '', 'baseline', 'name').map((item) => item.id)).toEqual(['one']);
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

  it('shows create-first fields and comparison baseline controls with inspectable missing values', () => {
    const create = renderToStaticMarkup(<CreateExperiment project="p" onCreated={() => {}} onClose={() => {}} />);
    expect(create).toContain('Experiment name'); expect(create).toContain('Tags'); expect(create).toContain('Notes');
    expect(create).not.toContain('Development protocol'); expect(create).not.toContain('Launch batch');
    const compare = renderToStaticMarkup(<ExperimentComparison items={[experiment('one', { inputSnapshot: { dataset: { id: 'd' } } }), experiment('two')]} />);
    expect(compare).toContain('Baseline'); expect(compare).toContain('Differences only'); expect(compare).toContain('Not recorded');
    expect(compare).toContain('dataset.id'); expect(compare).toContain('No current project defaults are substituted');
  });

  it('renders all lifecycle views without mixing another experiment’s same-name batches into details', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const owned = batch('owned-only', 'one');
    client.setQueryData(['development-batches', 'p'], { items: [owned, batch('UNRELATED-BATCH', 'two')], executions: [], executionImplemented: true });
    client.setQueryData(['model-experiments', 'p', 'summary'], { items: [experiment('one', { batches: [owned] }), experiment('trash', { state: 'trashed' })] });
    try {
      const registry = renderToStaticMarkup(<QueryClientProvider client={client}><ExperimentRegistry project="p" onOpen={() => {}} /></QueryClientProvider>);
      expect(registry).toContain('Archived'); expect(registry).toContain('Trash'); expect(registry).toContain('All records');
      const detail = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned]} ownedDrafts={[]} tab="runs" onOpenSetup={() => {}} onRestoreInputs={() => {}} /></QueryClientProvider>);
      expect(detail).toContain('owned-only'); expect(detail).not.toContain('UNRELATED-BATCH');
      const multiple = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned, batch('second-owned', 'one')]} ownedDrafts={[]} tab="runs" onOpenSetup={() => {}} onRestoreInputs={() => {}} /></QueryClientProvider>);
      expect(multiple).toContain('Choose a batch in this experiment'); expect(multiple).not.toContain('Launch batch');
    } finally { client.clear(); }
  });

  it('uses current service execution capability even when the immutable historical plan flag is false', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const owned = batch('historical-plan', 'one'); owned.manifest.executionImplemented = false;
    client.setQueryData(['training-runtime', 'p'], { available: true, python: '/training/python', versions: {}, cudaAvailable: false, gpuCount: 0, findings: [] });
    client.setQueryData(['training-execution', 'p', owned.id], null);
    try {
      const html = renderToStaticMarkup(<QueryClientProvider client={client}><DevelopmentBatches project="p" inputs={inputs} experimentName="Same name" experimentId="one" experimentRevision={1} ownedBatches={[owned]} ownedDrafts={[]} executionImplemented tab="runs" onOpenSetup={() => {}} onRestoreInputs={() => {}} /></QueryClientProvider>);
      expect(html).toContain('Launch batch');
      expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Launch batch/);
      expect(html).not.toContain('Training execution is unavailable');
      expect(owned.manifest.executionImplemented).toBe(false);
    } finally { client.clear(); }
  });
});
