import { describe, expect, it } from 'vitest';
import type { FeatureBundle } from '../api/bundles';
import type { Configuration, DatasetVersion, ProtocolSpec, ScientificDraft } from '../api/scientific';
import type { Workspace } from '../api/types';
import type { FrozenBatch, TrainingExecution } from '../api/development';
import { buildRoadmap, protocolBundleCompatible, completedDevelopmentBatches, suggestedRoadmapModule, ROADMAP_CONNECTIONS, ROADMAP_MODULES, type RoadmapEvidence, type RoadmapModuleId } from './roadmap';

function workspace(mode: Workspace['mode'] = 'local'): Workspace {
  return {
    project: { id: 'project', name: 'Project', description: '', storagePath: '', mode, createdAt: '', updatedAt: '', config: {}, sources: [], available: true }, mode, executionEnabled: false,
    dataset: { id: 'dataset', slideCount: 20, patientCount: 10, specimenCount: 20 },
    patients: [], slides: [], encoders: [], milModels: [], featureSets: [],
    split: { id: 'split', seed: 42, groupBy: 'patient' }, results: [],
    cohortSnapshots: [], drafts: [], sources: [], exampleManifests: [],
  } as Workspace;
}

function dataset(id = 'dataset'): DatasetVersion {
  return { id, projectId: 'project', contentHash: `hash-${id}`, createdAt: '', manifest: {}, artifacts: {} };
}

function protocol(datasetId = 'dataset', bindings: Partial<Pick<ProtocolSpec, 'featureBundleId' | 'featureSetId' | 'featurePackId'>> = {}): Configuration {
  return {
    id: `protocol-${datasetId}`, projectId: 'project', contentHash: 'protocol-hash', createdAt: '',
    manifest: { kind: 'protocol', datasetId, spec: { datasetId, ...bindings }, summary: {} },
  } as Configuration;
}

function bundle(datasetId = 'dataset'): FeatureBundle {
  return {
    id: `bundle-${datasetId}`, contentHash: 'bundle-hash', createdAt: '', current: true, findings: [],
    manifest: {
      kind: 'feature-bundle', datasetId, spec: { featureSetId: 'feature', packArtifactIds: [] },
      summary: { slideCount: 20, patchCount: 200, dimensions: 128, dtype: 'float32', packCount: 0 },
      feature: {
        id: 'feature', datasetId, contentHash: 'feature-hash', sourceContentHash: 'source-hash',
        validation: { jobId: 'validation', sourceContentHash: 'source-hash', tensorValidationComplete: true, provenanceComplete: true },
      },
      packs: [],
    },
  };
}

function draft(type: ScientificDraft['payload']['type'], status: ScientificDraft['status'] = 'editable'): ScientificDraft<unknown> {
  return {
    id: `draft-${type}`, projectId: 'project', kind: type === 'dataset-import' ? 'import' : 'experiment',
    name: 'Saved draft', payload: { type, spec: {} }, revision: 1, status, createdAt: '', updatedAt: '',
  };
}

function modules(evidence: Partial<RoadmapEvidence> = {}, source = workspace()) {
  return Object.fromEntries(buildRoadmap(source, evidence).map((module) => [module.id, module])) as Record<RoadmapModuleId, ReturnType<typeof buildRoadmap>[number]>;
}

function completedBatch() {
  const runs = [0, 1, 2].map((fold) => ({ id: `run-${fold}`, candidateId: 'candidate', trainingSeed: 42, splitPlanId: `fold-${fold}`, status: 'planned' as const }));
  const batch = { id: 'batch', manifest: { runs, summary: { runCount: runs.length } } } as FrozenBatch;
  const execution = {
    batchId: batch.id, status: 'completed', runCounts: { completed: 3, total: 3, queued: 0, running: 0, failed: 0, cancelled: 0 },
    runs: runs.map((run) => ({ ...run, status: 'completed' })), findings: [],
  } as unknown as TrainingExecution;
  return { batch, execution };
}

describe('project roadmap progress', () => {
  it('unlocks the self-contained BLCA walkthrough without claiming real completed artifacts', () => {
    const source = workspace('synthetic-demo');
    source.demoPipeline = { version: 1, synthetic: true, readOnly: true, seed: 20260912, sourceBasis: [], records: [{ id: 'demo-dataset', module: 'dataset', name: 'Synthetic BLCA slides', description: '', tags: [], steps: [] }] };
    const roadmap = buildRoadmap(source);
    expect(roadmap.every((module) => module.unlocked && module.blockers.length === 0)).toBe(true);
    expect(roadmap.every((module) => module.status !== 'complete')).toBe(true);
    expect(roadmap[0].evidence).toBe('1 illustrative record · synthetic walkthrough');
    expect(roadmap.find((module) => module.id === 'evaluation')?.evidence).toBe('Illustrative workflow explanation');
  });
  it('counts ready predictors as experiment outputs and never treats evaluation plans as completed results', () => {
    const predictor = { id: 'predictor-a', lifecycleState: 'active' } as RoadmapEvidence['predictors'][number];
    const record = { id: 'evaluation-a', lifecycleState: 'active', manifest: { status: 'planned' } } as RoadmapEvidence['modelEvaluations'][number];
    const roadmap = modules({ predictors: [predictor], modelEvaluations: [record] });
    expect(roadmap.experiments.status).toBe('complete');
    expect(roadmap.experiments.evidence).toBe('1 ready predictor');
    expect(roadmap.evaluation.status).toBe('draft');
    expect(roadmap.evaluation.evidence).toBe('1 saved evaluation plan');
    expect(roadmap.evaluation.unlocked).toBe(true);
    const trashed = modules({ predictors: [{ ...predictor, lifecycleState: 'trashed' }], modelEvaluations: [{ ...record, lifecycleState: 'trashed' }] });
    expect(trashed.experiments.status).toBe('not-started');
    expect(trashed.evaluation.status).toBe('not-started');
  });
  it('keeps completed training distinct from predictor readiness inside Experiments', () => {
    const { batch, execution } = completedBatch();
    const live = modules({ batches: [batch], executions: [{ ...execution, status: 'running', runCounts: { ...execution.runCounts, completed: 2 } }] });
    expect(live.experiments.evidence).toBe('2/3 training runs completed · 1 active batch');
    expect(live.experiments.status).toBe('draft');
    const complete = modules({ batches: [batch], executions: [execution] });
    expect(complete.experiments.evidence).toBe('1 completed development batch · 3/3 training runs completed');
    expect(complete.experiments.status).toBe('complete');
    expect(complete.evaluation.blockers).toContain('experiments');
  });

  it('requires a known complete batch with exact completed run membership, not partial candidates or inconsistent counts', () => {
    const { batch, execution } = completedBatch();
    expect(completedDevelopmentBatches([batch], [execution])).toEqual([batch]);
    for (const invalid of [
      { ...execution, batchId: 'unknown' },
      { ...execution, status: 'failed' as const },
      { ...execution, runCounts: { ...execution.runCounts, completed: 2 } },
      { ...execution, runs: execution.runs.slice(1) },
      { ...execution, runs: execution.runs.map((run) => ({ ...run, id: 'duplicate' })) },
      { ...execution, runs: execution.runs.map((run) => ({ ...run, id: `${run.id}-other` })) },
      { ...execution, findings: [{ severity: 'error' as const, code: 'OOF_FAILED', message: 'Incomplete OOF evidence' }] },
    ]) expect(completedDevelopmentBatches([batch], [invalid])).toEqual([]);
    expect(completedDevelopmentBatches([], [execution])).toEqual([]);
  });

  it('keeps development complete when another batch starts or fails', () => {
    const { batch, execution } = completedBatch();
    const later = { ...execution, batchId: 'second-batch', status: 'running' as const, runCounts: { ...execution.runCounts, completed: 0 } };
    const roadmap = modules({ batches: [batch], executions: [execution, later], drafts: [draft('development-batch')] });
    expect(roadmap.experiments.status).toBe('complete');
    expect(roadmap.evaluation.unlocked).toBe(true);
  });

  it('links eight modules with independent optional clinical utility and interpretation branches', () => {
    expect(ROADMAP_MODULES.map((module) => [module.id, module.phase])).toEqual([
      ['dataset', 'prepare'], ['features', 'prepare'], ['cohort', 'prepare'],
      ['experiments', 'develop'],
      ['test-data', 'evaluate'], ['evaluation', 'evaluate'],
      ['clinical-utility', 'insights'], ['interpretation', 'insights'],
    ]);
    expect(ROADMAP_CONNECTIONS).toEqual([
      { from: 'dataset', to: 'cohort' }, { from: 'features', to: 'cohort' },
      { from: 'cohort', to: 'experiments' }, { from: 'features', to: 'experiments' },
      { from: 'experiments', to: 'evaluation' }, { from: 'test-data', to: 'evaluation' },
      { from: 'evaluation', to: 'clinical-utility' }, { from: 'experiments', to: 'interpretation' },
    ]);
    expect(ROADMAP_MODULES.find((module) => module.id === 'test-data')?.prerequisites).toEqual(['dataset']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'evaluation')?.prerequisites).toEqual(['experiments', 'test-data']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'clinical-utility')?.prerequisites).toEqual(['evaluation']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'interpretation')?.prerequisites).toEqual(['experiments', 'features']);
    expect(ROADMAP_MODULES.filter((module) => module.optional).map((module) => module.id)).toEqual(['clinical-utility', 'interpretation']);
  });
  it('opens slide features and test-cohort planning independently of development', () => {
    const roadmap = buildRoadmap(workspace());
    expect(roadmap.filter((module) => module.unlocked).map((module) => module.id)).toEqual(['dataset', 'features', 'cohort', 'experiments', 'test-data', 'evaluation', 'clinical-utility', 'interpretation']);
    expect(roadmap.every((module) => module.status === 'not-started')).toBe(true);
    expect(modules().evaluation.blockers).toEqual(['experiments', 'test-data']);
    expect(roadmap).toHaveLength(8);
  });

  it('counts saved clinical analyses and completed attention maps separately from plans', () => {
    const clinical = { id: 'clinical', lifecycleState: 'active' };
    const planned = { id: 'attention', lifecycleState: 'active', execution: { status: 'queued' } };
    const roadmap = modules({ clinicalAnalyses: [clinical], interpretations: [planned] });
    expect(roadmap['clinical-utility'].status).toBe('complete');
    expect(roadmap.interpretation.status).toBe('draft');
    expect(roadmap.interpretation.blockers).toEqual(['experiments', 'features']);
    expect(modules({ interpretations: [{ ...planned, execution: { status: 'completed' } }] }).interpretation.status).toBe('complete');
    const trashed = modules({ clinicalAnalyses: [{ ...clinical, lifecycleState: 'trashed' }], interpretations: [{ ...planned, lifecycleState: 'trashed', execution: { status: 'completed' } }] });
    expect(trashed['clinical-utility'].status).toBe('not-started');
    expect(trashed.interpretation.status).toBe('not-started');
    expect(trashed.interpretation.unlocked).toBe(true);
  });

  it('recognizes saved drafts without treating them as frozen prerequisites', () => {
    const roadmap = modules({ drafts: [draft('dataset-import'), draft('analysis-protocol'), draft('mil-experiment')] });
    expect(roadmap.dataset.status).toBe('draft');
    expect(roadmap.cohort.status).toBe('draft');
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap['test-data'].unlocked).toBe(true);
  });

  it('requires a persisted artifact even when an import draft says frozen', () => {
    const roadmap = modules({ drafts: [draft('dataset-import', 'frozen')] });
    expect(roadmap.dataset.status).toBe('draft');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.cohort.blockers).toContain('dataset');
  });

  it('keeps retained batch results accessible after archiving inputs without claiming they are ready for new work', () => {
    const { batch, execution } = completedBatch();
    const roadmap = modules({ batches: [batch], executions: [execution] });
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.retainedWork).toBe(true);
    expect(roadmap.experiments.status).toBe('complete');
    expect(roadmap.dataset.status).toBe('not-started');
    expect(roadmap.features.status).toBe('not-started');
    expect(roadmap.evaluation.unlocked).toBe(true);
    expect(modules().experiments.unlocked).toBe(true);
  });

  it('unlocks targets after a frozen dataset and preserves existing completed work when another draft is saved', () => {
    const roadmap = modules({ datasets: [dataset()], drafts: [draft('dataset-import')] });
    expect(roadmap.dataset.status).toBe('complete');
    expect(roadmap.dataset.evidence).toBe('1 frozen dataset');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('keeps saved feature headers in progress until a current fully validated bundle exists', () => {
    const source = { ...protocol(), manifest: { ...protocol().manifest, kind: 'feature' } } as Configuration;
    const roadmap = modules({ datasets: [dataset()], protocols: [protocol()], features: [source] });
    expect(roadmap.cohort.status).toBe('complete');
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.experiments.blockers).toEqual(['features']);
    const ready = modules({ datasets: [dataset()], protocols: [protocol()], features: [source], bundles: [bundle()] });
    expect(ready.features.status).toBe('complete');
    expect(ready.experiments.unlocked).toBe(true);
  });

  it.each(['current', 'tensor', 'findings'] as const)('keeps the experiment registry open while feature %s evidence blocks new training', (failure) => {
    const features = bundle();
    if (failure === 'current') features.current = false;
    if (failure === 'tensor') features.manifest.feature.validation.tensorValidationComplete = false;
    if (failure === 'findings') features.findings.push({ severity: 'error', code: 'CHANGED', message: 'Features changed' });
    const roadmap = modules({ datasets: [dataset()], protocols: [protocol()], bundles: [features] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.blockers).toContain('features');
  });

  it('respects the service freeze decision for validated external features with incomplete extraction provenance', () => {
    const features = bundle();
    features.manifest.feature.validation.provenanceComplete = false;
    const roadmap = modules({ datasets: [dataset()], protocols: [protocol()], bundles: [features] });
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('reuses bundles across datasets and leaves slide coverage to the service', () => {
    const roadmap = modules({ datasets: [dataset('a'), dataset('b')], protocols: [protocol('a')], bundles: [bundle('b')] });
    expect(roadmap.cohort.status).toBe('complete');
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.compatibilityIssue).toBeUndefined();
    expect(roadmap['test-data'].status).toBe('not-started');
    // A store-scoped bundle names no cohort; the server checks slide coverage when planning.
    const store = bundle('b');
    store.manifest.datasetId = null;
    const shared = modules({ datasets: [dataset('a')], protocols: [protocol('a')], bundles: [store] });
    expect(shared.features.status).toBe('complete');
    expect(shared.experiments.compatibilityIssue).toBeUndefined();
  });

  it('starts slide features before any dataset exists', () => {
    const store = bundle('ignored');
    store.manifest.datasetId = null;
    const roadmap = modules({ bundles: [store] });
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap.features.blockers).toEqual([]);
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.dataset.status).toBe('not-started');
    // Training still needs a cohort, so the dataset remains the next required step.
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.cohort.blockers).toContain('dataset');
  });

  it('honors the exact named bundle pinned by a protocol', () => {
    const required = bundle('other-dataset');
    const bound = protocol('dataset', { featureBundleId: required.id });
    expect(protocolBundleCompatible(bound, required)).toBe(true);
    expect(protocolBundleCompatible(bound, { ...required, id: 'same-features-different-bundle' })).toBe(false);
    const roadmap = modules({ datasets: [dataset()], protocols: [bound], bundles: [required] });
    expect(roadmap.experiments.compatibilityIssue).toBeUndefined();
  });

  it('honors an existing protocol feature and pack binding before unlocking model development', () => {
    const features = bundle();
    const bound = protocol('dataset', { featureSetId: 'different-feature' });
    expect(protocolBundleCompatible(bound, features)).toBe(false);
    bound.manifest.spec = { datasetId: 'dataset', featureSetId: 'feature', featurePackId: 'required-pack' } as ProtocolSpec;
    expect(protocolBundleCompatible(bound, features)).toBe(false);
    features.manifest.spec.packArtifactIds = ['required-pack'];
    expect(protocolBundleCompatible(bound, features)).toBe(true);
  });

  it('does not complete downstream stages from a saved model plan or illustrative metrics', () => {
    const source = workspace();
    source.results = [{ id: 'example-result' }] as Workspace['results'];
    const roadmap = modules({
      datasets: [dataset()], protocols: [protocol()], bundles: [bundle()], drafts: [draft('mil-experiment', 'frozen')],
    }, source);
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.evaluation.status).toBe('not-started');
    expect(roadmap.evaluation.unlocked).toBe(true);
    expect(roadmap['test-data'].unlocked).toBe(true);
  });

  it('uses sample workspace artifacts in demo mode without claiming a completed predictor', () => {
    const demo = workspace('synthetic-demo');
    demo.featureSets = [{ id: 'sample-feature' }] as Workspace['featureSets'];
    demo.cohortSnapshots = [{ id: 'sample-cohort' }] as Workspace['cohortSnapshots'];
    demo.drafts = [{ id: 'sample-plan' }] as Workspace['drafts'];
    demo.results = [{ id: 'sample-result' }] as Workspace['results'];
    const roadmap = modules({}, demo);
    expect(roadmap.dataset.status).toBe('complete');
    expect(roadmap.features.evidence).toBe('1 sample feature set');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.evaluation.unlocked).toBe(false);
  });

  it('accepts a separately imported verified test cohort while development is only planned', () => {
    const cohort = { id: 'test-cohort', current: true, findings: [], manifest: { datasetId: 'other' } } as unknown as RoadmapEvidence['evaluationCohorts'][number];
    const roadmap = modules({ datasets: [dataset(), dataset('other')], protocols: [protocol()], evaluationCohorts: [cohort], drafts: [draft('development-batch')] });
    expect(roadmap['test-data'].status).toBe('complete');
    expect(roadmap['test-data'].unlocked).toBe(true);
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.evaluation.blockers).toEqual(['experiments']);
  });

  it('does not complete test preparation from a stale or unverified cohort', () => {
    for (const current of [false, undefined]) {
      const cohort = { id: 'test-cohort', current, findings: [] } as unknown as RoadmapEvidence['evaluationCohorts'][number];
      expect(modules({ evaluationCohorts: [cohort] })['test-data'].status).toBe('draft');
    }
  });

  it.each(['ABMIL', 'nnMIL'])('supports interpretation before evaluation or clinical utility when %s and feature inputs exist', (model) => {
    const predictor = { id: 'predictor', lifecycleState: 'active', manifest: { recipe: { model } } } as RoadmapEvidence['predictors'][number];
    const evidence = { datasets: [dataset()], bundles: [bundle()], predictors: [predictor] };
    expect(modules(evidence).interpretation.blockers).toEqual([]);
    expect(modules(evidence)['clinical-utility'].blockers).toEqual(['evaluation']);
    const otherModel = { ...predictor, manifest: { ...predictor.manifest, recipe: { ...predictor.manifest.recipe, model: 'other' } } };
    expect(modules({ ...evidence, predictors: [otherModel] }).interpretation.blockers).toEqual(['experiments']);
    expect(modules({ ...evidence, predictors: [{ ...predictor, lifecycleState: 'trashed' }] }).interpretation.blockers).toEqual(['experiments']);
  });

  it('suggests a reachable input task instead of an open registry with missing prerequisites', () => {
    const { batch, execution } = completedBatch();
    const roadmap = buildRoadmap(workspace(), { datasets: [dataset()], protocols: [protocol()], bundles: [bundle()], batches: [batch], executions: [execution] });
    expect(roadmap.find((module) => module.id === 'evaluation')?.unlocked).toBe(true);
    expect(suggestedRoadmapModule(roadmap)?.id).toBe('test-data');
    expect(suggestedRoadmapModule(buildRoadmap(workspace()))?.id).toBe('dataset');
  });

  it('does not present optional analyses as remaining required work', () => {
    const roadmap = buildRoadmap(workspace()).map((module) => ({ ...module, unlocked: true, blockers: [], status: module.optional ? 'draft' as const : 'complete' as const }));
    expect(suggestedRoadmapModule(roadmap)).toBeUndefined();
  });
});
