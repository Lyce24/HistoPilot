import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { targetSplits, type TargetSplit } from '../api/targetSplits';
import { bundles, type FeatureBundle } from '../api/bundles';
import { scientific, type DatasetVersion } from '../api/scientific';
import type { Workspace } from '../api/types';
import { useRoadmap } from './useRoadmap';

const keys = {
  drafts: ['scientific', 'project', 'drafts'],
  datasets: ['scientific', 'project', 'datasets'],
  targetSplits: ['scientific', 'project', 'configurations', 'target-split'],
  setups: ['scientific', 'project', 'configurations', 'experiment-setup'],
  features: ['scientific', 'project', 'configurations', 'feature'],
  bundles: ['feature-bundles', 'project'],
  extractions: ['extractions', 'project', 'jobs'],
  batches: ['development-batches', 'project'],
  evaluation: ['evaluation-cohorts', 'project'],
  predictors: ['predictors', 'project'],
  evaluations: ['model-evaluations', 'project'],
  clinical: ['clinical-analyses', 'project'],
  interpretations: ['interpretations', 'project'],
} as const;

const dataset: DatasetVersion = {
  id: 'dataset', projectId: 'project', contentHash: 'dataset-hash', createdAt: '', manifest: {}, artifacts: {},
};
const bundle: FeatureBundle = {
  id: 'bundle', contentHash: 'bundle-hash', createdAt: '', current: true, findings: [],
  manifest: {
    kind: 'feature-bundle', datasetId: 'dataset', spec: { featureSetId: 'feature', packArtifactIds: [] },
    summary: { slideCount: 20, patchCount: 200, dimensions: 128, dtype: 'float32', packCount: 0 },
    feature: {
      id: 'feature', datasetId: 'dataset', contentHash: 'feature-hash', sourceContentHash: 'source-hash',
      validation: { jobId: 'validation', sourceContentHash: 'source-hash', tensorValidationComplete: true, provenanceComplete: true },
    },
    packs: [],
  },
};

function workspace(mode: Workspace['mode'] = 'local'): Workspace {
  return {
    project: { id: 'project', name: 'Project', description: '', storagePath: '', mode, createdAt: '', updatedAt: '', config: {}, sources: [], available: true },
    mode, executionEnabled: false,
    dataset: { id: 'dataset', slideCount: 20, patientCount: 10, specimenCount: 20 }, sources: [],
  };
}

const clients: QueryClient[] = [];
function client() {
  // Keep a failed initial request settled while probing the SSR result.
  const value = new QueryClient({ defaultOptions: { queries: { retry: false, retryOnMount: false, staleTime: Infinity } } });
  clients.push(value);
  return value;
}

afterEach(() => {
  for (const value of clients.splice(0)) value.clear();
  vi.restoreAllMocks();
});

function seed(value: QueryClient) {
  value.setQueryData(keys.drafts, { drafts: [] });
  value.setQueryData(keys.datasets, { datasets: [dataset] });
  value.setQueryData(keys.targetSplits, { configurations: [{ id: 'targets', manifest: { kind: 'target-split', datasetId: 'dataset' } } as TargetSplit] });
  value.setQueryData(keys.setups, { configurations: [] });
  value.setQueryData(keys.features, { configurations: [] });
  value.setQueryData(keys.bundles, { items: [bundle] });
  value.setQueryData(keys.extractions, { jobs: [] });
  value.setQueryData(keys.batches, { items: [], executionImplemented: false });
  value.setQueryData(keys.evaluation, { items: [] });
  value.setQueryData(keys.predictors, { items: [] });
  value.setQueryData(keys.evaluations, { items: [] });
  value.setQueryData(keys.clinical, { items: [] });
  value.setQueryData(keys.interpretations, { items: [] });
}

function fail(value: QueryClient, queryKey: readonly string[], error: Error) {
  // Query errors retain any previously cached data, as a failed refetch does.
  value.getQueryCache().build(value, { queryKey }).setState({ status: 'error', error, fetchStatus: 'idle' });
}

function probe(value: QueryClient, source = workspace()) {
  let result: ReturnType<typeof useRoadmap> | undefined;
  function Probe() {
    result = useRoadmap(source);
    return null;
  }
  renderToStaticMarkup(<QueryClientProvider client={value}><Probe /></QueryClientProvider>);
  if (!result) throw new Error('The roadmap probe did not render.');
  return result;
}

describe('roadmap prerequisite query isolation', () => {
  it('uses the extraction jobs cache for live progress before a feature bundle exists', () => {
    const value = client();
    seed(value);
    value.setQueryData(keys.bundles, { items: [] });
    value.setQueryData(keys.extractions, { jobs: [{ id: 'extraction-live', state: 'running', progress: {
      label: 'Tissue segmentation', completed: 128, total: 1111, unit: 'slides', scope: 'stage',
    } }] });
    const roadmap = probe(value);
    expect(roadmap.byId.features.status).toBe('draft');
    expect(roadmap.byId.features.evidence).toBe('1 extraction in progress · Tissue segmentation · 128/1111 slides in stage');
    expect(roadmap.byId['experimental-setup'].blockers).toContain('features');
    value.setQueryData(keys.extractions, { jobs: [{ id: 'extraction-live', state: 'succeeded' }] });
    const updated = probe(value);
    expect(updated.byId.features.status).toBe('draft');
    expect(updated.byId.features.evidence).toContain('latest completed · review outputs and freeze a bundle');
  });

  it('retains visible extraction progress when its background refresh fails', () => {
    const value = client();
    seed(value);
    value.setQueryData(keys.bundles, { items: [] });
    value.setQueryData(keys.extractions, { jobs: [{ id: 'extraction-live', state: 'running' }] });
    const failure = new Error('Extraction refresh unavailable');
    fail(value, keys.extractions, failure);
    const roadmap = probe(value);
    expect(roadmap.byId.features.evidence).toContain('1 extraction in progress');
    expect(roadmap.error).toBe(failure);
    expect(roadmap.hasData).toBe(true);
    expect(roadmap.checksById.features).toEqual({ isLoading: false, error: null, hasData: true });
  });

  it('opens registries while execution evidence is loading without claiming completion', () => {
    const value = client();
    seed(value);
    value.removeQueries({ queryKey: keys.batches, exact: true });
    const roadmap = probe(value);
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.checksById.evaluation.hasData).toBe(true);
    expect(roadmap.checksById['experimental-setup']).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId['experimental-setup'].unlocked).toBe(true);
    expect(roadmap.byId.experiments.unlocked).toBe(true);
    expect(roadmap.byId.experiments.status).toBe('not-started');
  });

  it('keeps experiment readiness separate from the test-cohort listing', () => {
    const value = client();
    seed(value);
    value.removeQueries({ queryKey: keys.evaluation, exact: true });
    const failure = new Error('Test cohorts unavailable');
    fail(value, keys.evaluation, failure);
    const roadmap = probe(value);
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.checksById.evaluation).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.error).toBe(failure);
  });
  it('keeps data, later-test, target and feature editors available when an unrelated bundle request fails', () => {
    const value = client();
    seed(value);
    value.removeQueries({ queryKey: keys.bundles, exact: true });
    const failure = new Error('Feature bundles unavailable');
    fail(value, keys.bundles, failure);

    const roadmap = probe(value);
    expect(roadmap.error).toBe(failure);
    expect(roadmap.hasData).toBe(false);
    for (const id of ['dataset', 'experimental-setup', 'cohort', 'features'] as const) {
      expect(roadmap.checksById[id]).toEqual({ isLoading: false, error: null, hasData: true });
      expect(roadmap.byId[id].unlocked).toBe(true);
    }
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
  });

  it.each(['datasets', 'targetSplits', 'bundles'] as const)('allows creating experiments before %s are prepared', (missing) => {
    const value = client();
    seed(value);
    value.removeQueries({ queryKey: keys[missing], exact: true });

    const roadmap = probe(value);
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId.experiments.unlocked).toBe(true);
    expect(roadmap.checksById.dataset.hasData).toBe(true);
    expect(roadmap.checksById['experimental-setup']).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId['experimental-setup'].unlocked).toBe(true);
  });

  it.each(['datasets', 'targetSplits', 'bundles'] as const)('retains cached prerequisites and unlock state after a %s refresh fails', (failed) => {
    const value = client();
    seed(value);
    const failure = new Error('Background refresh failed');
    fail(value, keys[failed], failure);

    const roadmap = probe(value);
    expect(roadmap.error).toBe(failure);
    expect(roadmap.hasData).toBe(true);
    expect(roadmap.isLoading).toBe(false);
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId.experiments.unlocked).toBe(true);
    expect(roadmap.byId.dataset.status).toBe('complete');
    expect(roadmap.byId.cohort.status).toBe('complete');
    expect(roadmap.byId.features.status).toBe('complete');
  });

  it('keeps a retained target split accessible when its dataset is absent from active input choices', () => {
    const value = client();
    seed(value);
    value.setQueryData(keys.datasets, { datasets: [] });

    const roadmap = probe(value);
    expect(roadmap.checksById.cohort).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId.cohort.unlocked).toBe(true);
    expect(roadmap.byId.cohort.status).not.toBe('not-started');
    expect(roadmap.byId.cohort.blockers).toEqual([]);
  });

  it('keeps retained model work inspectable during an input-query failure without suppressing its warning', () => {
    const value = client();
    seed(value);
    value.setQueryData(keys.batches, { items: [{ id: 'retained-batch' }], executions: [], executionImplemented: true });
    value.removeQueries({ queryKey: keys.bundles, exact: true });
    const error = new Error('Active input choices unavailable');
    fail(value, keys.bundles, error);
    const roadmap = probe(value);
    expect(roadmap.byId.experiments.unlocked).toBe(true);
    expect(roadmap.byId.experiments.status).toBe('draft');
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, hasData: true, error: null });
    expect(roadmap.error).toBe(error);
    expect(roadmap.byId.experiments.unlocked).toBe(true);
  });

  it('does not require draft or feature-source listings to establish model input prerequisites', () => {
    const value = client();
    seed(value);
    for (const key of [keys.drafts, keys.features]) {
      value.removeQueries({ queryKey: key, exact: true });
      fail(value, key, new Error('Unrelated listing unavailable'));
    }

    const roadmap = probe(value);
    expect(roadmap.hasData).toBe(false);
    expect(roadmap.checksById.experiments).toEqual({ isLoading: false, error: null, hasData: true });
    expect(roadmap.byId.experiments.unlocked).toBe(true);
  });

  it('disables remote prerequisite checks and refreshes for the synthetic workspace', async () => {
    const value = client();
    for (const key of Object.values(keys)) fail(value, key, new Error('Local service unavailable'));
    const reads = [
      vi.spyOn(scientific, 'drafts').mockRejectedValue(new Error('Unexpected request')),
      vi.spyOn(scientific, 'datasets').mockRejectedValue(new Error('Unexpected request')),
      vi.spyOn(scientific, 'configurations').mockRejectedValue(new Error('Unexpected request')),
      vi.spyOn(targetSplits, 'list').mockRejectedValue(new Error('Unexpected request')),
      vi.spyOn(bundles, 'list').mockRejectedValue(new Error('Unexpected request')),
    ];

    const roadmap = probe(value, workspace('synthetic-demo'));
    expect(roadmap.hasData).toBe(true);
    expect(roadmap.error).toBeNull();
    expect(roadmap.isLoading).toBe(false);
    for (const check of Object.values(roadmap.checksById)) {
      expect(check).toEqual({ isLoading: false, error: null, hasData: true });
    }
    for (const query of value.getQueryCache().getAll()) expect(query.options).toMatchObject({ enabled: false });
    await roadmap.refetch();
    for (const read of reads) expect(read).not.toHaveBeenCalled();
  });
});
