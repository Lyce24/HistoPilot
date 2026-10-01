import { describe, expect, it } from 'vitest';
import { newTargetSplitSpec, type TargetSplit } from '../api/targetSplits';
import type { FeatureBundle } from '../api/bundles';
import type { Configuration, DatasetVersion, ProtocolSpec, ScientificDraft } from '../api/scientific';
import type { Workspace } from '../api/types';
import type { FrozenBatch, TrainingExecution } from '../api/development';
import type { ExtractionJob } from '../api/trident';
import { buildRoadmap, completedDevelopmentBatches, suggestedRoadmapModule, stageEyebrow, stageStep, ROADMAP_MODULES, ROADMAP_STEPS, type RoadmapEvidence, type RoadmapModuleId } from './roadmap';

function workspace(mode: Workspace['mode'] = 'local'): Workspace {
  return {
    project: { id: 'project', name: 'Project', description: '', storagePath: '', mode, createdAt: '', updatedAt: '', config: {}, sources: [], available: true }, mode, executionEnabled: false,
    dataset: { id: 'dataset', slideCount: 20, patientCount: 10, specimenCount: 20 }, sources: [],
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

function targetSplit(datasetId = 'dataset'): TargetSplit {
  return { id: `targets-${datasetId}`, projectId: 'project', createdAt: '', contentHash: 'targets-hash',
    manifest: { kind: 'target-split', datasetId, spec: newTargetSplitSpec(datasetId), summary: {}, memberships: [], findings: [], previewHash: 'targets-preview' },
  } as unknown as TargetSplit;
}

function setup(): Configuration {
  return { id: 'setup', projectId: 'project', createdAt: '', contentHash: 'setup-hash', manifest: { kind: 'experiment-setup', datasetId: 'dataset', spec: {}, summary: {} } } as unknown as Configuration;
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

function extraction(state: ExtractionJob['state'] = 'running'): ExtractionJob {
  return {
    id: 'extraction-live', state, createdAt: '2026-09-25T02:00:00Z', updatedAt: '2026-09-25T03:00:00Z',
    spec: { datasetId: null, outputPath: '/output', options: {} }, outputPath: '/output', logPath: '/job/worker.log', sessionName: 'extraction-live',
    progress: {
      stage: 'segmentation', stages: [], label: 'Tissue segmentation', detail: '', completed: 128, total: 1000,
      unit: 'slides', percent: 11.52, currentSlide: null, elapsedSeconds: 4400, etaSeconds: null,
      ratePerSecond: null, scope: 'stage', warnings: [],
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
    expect(roadmap.find((module) => module.id === 'apply')?.evidence).toBe('Illustrative workflow explanation');
  });
  it('counts ready predictors as experiment outputs and never treats unfinished runs as results', () => {
    const predictor = { id: 'predictor-a', lifecycleState: 'active' } as RoadmapEvidence['predictors'][number];
    const record = { id: 'evaluation-a', lifecycleState: 'active', manifest: { status: 'planned' } } as RoadmapEvidence['modelEvaluations'][number];
    const roadmap = modules({ predictors: [predictor], modelEvaluations: [record] });
    expect(roadmap.experiments.status).toBe('complete');
    expect(roadmap.experiments.evidence).toBe('1 ready predictor');
    expect(roadmap.apply.status).toBe('draft');
    expect(roadmap.apply.evidence).toBe('1 run not completed yet');
    expect(roadmap.apply.unlocked).toBe(true);
    const trashed = modules({ predictors: [{ ...predictor, lifecycleState: 'trashed' }], modelEvaluations: [{ ...record, lifecycleState: 'trashed' }] });
    expect(trashed.experiments.status).toBe('not-started');
    expect(trashed.apply.status).toBe('not-started');
  });
  it('completes Apply models with a run on either kind of cohort and says which were scored', () => {
    const predictor = { id: 'predictor-a', lifecycleState: 'active' } as RoadmapEvidence['predictors'][number];
    const inference = { id: 'inference-a', lifecycleState: 'active', manifest: { status: 'planned', purpose: 'inference' }, execution: { status: 'completed' } } as RoadmapEvidence['modelEvaluations'][number];
    const review = { id: 'review-a', lifecycleState: 'active', manifest: { status: 'planned', purpose: 'review' } } as RoadmapEvidence['modelEvaluations'][number];
    const scored = { id: 'scored-a', lifecycleState: 'active', manifest: { status: 'planned' }, execution: { status: 'completed' } } as RoadmapEvidence['modelEvaluations'][number];
    const unlabeled = modules({ predictors: [predictor], modelEvaluations: [inference, review] });
    expect(unlabeled.apply.status).toBe('complete');
    expect(unlabeled.apply.evidence).toBe('1 completed run · 1 predictions only');
    expect(unlabeled.apply.blockers).toEqual([]);
    expect(unlabeled.apply.optional).toBeUndefined();
    const both = modules({ predictors: [predictor], modelEvaluations: [inference, scored], clinicalAnalyses: [{ id: 'clinical', lifecycleState: 'active' }] });
    expect(both.apply.evidence).toBe('2 completed runs · 1 scored, 1 predictions only · 1 clinical analysis');
  });
  it('keeps completed training distinct from predictor readiness inside Experiments', () => {
    const { batch, execution } = completedBatch();
    const live = modules({ batches: [batch], executions: [{ ...execution, status: 'running', runCounts: { ...execution.runCounts, completed: 2 } }] });
    expect(live.experiments.evidence).toBe('2/3 training runs completed · 1 active batch');
    expect(live.experiments.status).toBe('draft');
    const complete = modules({ batches: [batch], executions: [execution] });
    expect(complete.experiments.evidence).toBe('1 completed development batch · 3/3 training runs completed');
    expect(complete.experiments.status).toBe('complete');
    expect(complete.apply.blockers).toContain('experiments');
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
    expect(roadmap.apply.unlocked).toBe(true);
  });

  it('links six modules with an optional interpretation branch', () => {
    expect(ROADMAP_MODULES.map((module) => [module.id, module.phase])).toEqual([
      ['dataset', 'prepare'], ['features', 'prepare'], ['cohort', 'prepare'],
      ['experiments', 'develop'],
      ['apply', 'evaluate'], ['interpretation', 'insights'],
    ]);
    // Cohorts are prepared inside Apply models, so ready predictors are its only prerequisite.
    expect(ROADMAP_MODULES.find((module) => module.id === 'apply')?.prerequisites).toEqual(['experiments']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'cohort')?.prerequisites).toEqual(['dataset']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'features')?.prerequisites).toEqual(['dataset']);
    // An experiment's design is its first steps, so Experiments needs targets, splits and features.
    expect(ROADMAP_MODULES.find((module) => module.id === 'experiments')?.prerequisites).toEqual(['cohort', 'features']);
    for (const id of ['test-data', 'evaluation', 'inference', 'clinical-utility', 'experimental-setup']) expect(ROADMAP_MODULES.some((module) => module.id === id)).toBe(false);
    expect(ROADMAP_MODULES.find((module) => module.id === 'interpretation')?.prerequisites).toEqual(['dataset', 'features', 'experiments']);
    expect(ROADMAP_MODULES.filter((module) => module.optional).map((module) => module.id)).toEqual(['interpretation']);
  });
  it('opens each registry while keeping training and application prerequisites distinct', () => {
    const roadmap = buildRoadmap(workspace());
    expect(roadmap.filter((module) => module.unlocked).map((module) => module.id)).toEqual(['dataset', 'features', 'cohort', 'experiments', 'apply', 'interpretation']);
    expect(roadmap.every((module) => module.status === 'not-started')).toBe(true);
    expect(modules().apply.blockers).toEqual(['experiments']);
    expect(roadmap).toHaveLength(6);
  });

  it('counts completed attention maps separately from plans, and clinical analyses only within runs', () => {
    const clinical = { id: 'clinical', lifecycleState: 'active' };
    const planned = { id: 'attention', lifecycleState: 'active', execution: { status: 'queued' } };
    const roadmap = modules({ clinicalAnalyses: [clinical], interpretations: [planned] });
    expect(roadmap.apply.status).toBe('not-started');
    expect(roadmap.interpretation.status).toBe('draft');
    // Interpretation needs a dataset, its features and trained weights; never evaluation or clinical results.
    expect(roadmap.interpretation.blockers).toEqual(['dataset', 'features', 'experiments']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'interpretation')?.prerequisites).toEqual(['dataset', 'features', 'experiments']);
    expect(modules({ interpretations: [{ ...planned, execution: { status: 'completed' } }] }).interpretation.status).toBe('complete');
    const trashed = modules({ interpretations: [{ ...planned, lifecycleState: 'trashed', execution: { status: 'completed' } }] });
    expect(trashed.interpretation.status).toBe('not-started');
    expect(trashed.interpretation.unlocked).toBe(true);
  });

  it('recognizes saved drafts without treating them as frozen prerequisites', () => {
    const roadmap = modules({ drafts: [draft('dataset-import'), draft('target-split'), draft('mil-experiment')] });
    expect(roadmap.dataset.status).toBe('draft');
    expect(roadmap.cohort.status).toBe('draft');
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.experiments.evidence).toBe('1 saved experiment design');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.features.unlocked).toBe(true);
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
    expect(roadmap.apply.unlocked).toBe(true);
    expect(modules().experiments.unlocked).toBe(true);
  });

  it('unlocks targets after a frozen dataset and preserves existing completed work when another draft is saved', () => {
    const roadmap = modules({ datasets: [dataset()], drafts: [draft('dataset-import')] });
    expect(roadmap.dataset.status).toBe('complete');
    expect(roadmap.dataset.evidence).toBe('1 frozen dataset');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.cohort.blockers).toEqual([]);
    expect(roadmap.features.status).toBe('not-started');
    expect(roadmap.experiments.blockers).toEqual(['cohort', 'features']);
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('prepares features and targets independently, then keeps a frozen design as saved work until training finishes', () => {
    const prepared = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()] });
    expect(prepared.cohort.status).toBe('complete');
    expect(prepared.features.status).toBe('complete');
    expect(prepared.experiments.blockers).toEqual([]);
    expect(prepared.experiments.status).toBe('not-started');
    const frozen = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()], setups: [setup()] });
    expect(frozen.experiments.blockers).toEqual([]);
    expect(frozen.experiments.status).toBe('draft');
    expect(frozen.experiments.evidence).toBe('1 saved experiment design');
  });

  it('keeps saved feature headers in progress until a current fully validated bundle exists', () => {
    const source = { ...protocol(), manifest: { ...protocol().manifest, kind: 'feature' } } as Configuration;
    const roadmap = modules({ datasets: [dataset()], targetSplits: [targetSplit()], features: [source] });
    expect(roadmap.cohort.status).toBe('complete');
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.experiments.blockers).toEqual(['features']);
    const ready = modules({ datasets: [dataset()], targetSplits: [targetSplit()], features: [source], bundles: [bundle()] });
    expect(ready.features.status).toBe('complete');
    expect(ready.experiments.unlocked).toBe(true);
  });

  it('shows a live extraction before any feature source or bundle exists without satisfying feature prerequisites', () => {
    const roadmap = modules({ datasets: [dataset()], extractions: [extraction()] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.features.evidence).toBe('1 extraction in progress · Tissue segmentation · 128/1000 slides in stage');
    expect(roadmap.features.artifactCount).toBe(1);
    expect(roadmap.features.retainedWork).toBe(true);
    expect(roadmap.cohort.blockers).toEqual([]);
    expect(roadmap.experiments.blockers).toContain('features');
  });

  it('names the latest worker batch without aggregating separate active extraction counters', () => {
    const first = extraction();
    const second = { ...extraction(), id: 'extraction-other' };
    const roadmap = modules({ extractions: [{ ...first, progress: { ...first.progress!, scope: 'batch' } }, second] });
    expect(roadmap.features.evidence).toBe('2 extractions in progress · Tissue segmentation · 128/1000 slides in latest batch');
    expect(roadmap.features.artifactCount).toBe(2);
  });

  it('shows queued extraction without claiming stale stage counters are running', () => {
    const roadmap = modules({ extractions: [extraction('queued')] });
    expect(roadmap.features.evidence).toBe('1 extraction in progress · Queued');
    expect(roadmap.features.status).toBe('draft');
  });

  it.each(['failed', 'interrupted', 'cancelled'] as const)('keeps a %s extraction visible for review', (state) => {
    const roadmap = modules({ extractions: [extraction(state)] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.features.evidence).toBe(`1 extraction run · latest ${state} · review run`);
    expect(roadmap.experiments.blockers).toContain('features');
  });

  it('requires a frozen verified bundle after extraction succeeds and preserves readiness while another extraction runs', () => {
    const finished = modules({ extractions: [extraction('succeeded')] });
    expect(finished.features.status).toBe('draft');
    expect(finished.features.evidence).toContain('latest completed · review outputs and freeze a bundle');
    expect(finished.experiments.blockers).toContain('features');
    const ready = modules({ bundles: [bundle()], extractions: [extraction()] });
    expect(ready.features.status).toBe('complete');
    expect(ready.features.artifactCount).toBe(1);
    expect(ready.features.evidence).toBe('1 verified frozen bundle · 1 extraction in progress · Tissue segmentation · 128/1000 slides in stage');
  });

  it.each(['current', 'tensor', 'findings'] as const)('keeps the experiment registry open while feature %s evidence blocks new training', (failure) => {
    const features = bundle();
    if (failure === 'current') features.current = false;
    if (failure === 'tensor') features.manifest.feature.validation.tensorValidationComplete = false;
    if (failure === 'findings') features.findings.push({ severity: 'error', code: 'CHANGED', message: 'Features changed' });
    const roadmap = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [features] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.blockers).toContain('features');
  });

  it('respects the service freeze decision for validated external features with incomplete extraction provenance', () => {
    const features = bundle();
    features.manifest.feature.validation.provenanceComplete = false;
    const roadmap = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [features] });
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('reuses bundles across datasets and leaves slide coverage to the service', () => {
    const roadmap = modules({ datasets: [dataset('a'), dataset('b')], targetSplits: [targetSplit('a')], bundles: [bundle('b')] });
    expect(roadmap.cohort.status).toBe('complete');
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.compatibilityIssue).toBeUndefined();
    expect(roadmap.apply.blockers).toEqual(['experiments']);
    // A store-scoped bundle names no cohort; the server checks slide coverage when planning.
    const store = bundle('b');
    store.manifest.datasetId = null;
    const shared = modules({ datasets: [dataset('a')], targetSplits: [targetSplit('a')], bundles: [store] });
    expect(shared.features.status).toBe('complete');
    expect(shared.experiments.compatibilityIssue).toBeUndefined();
  });

  it('preserves an independently prepared feature store while dataset preparation remains outstanding', () => {
    const store = bundle('ignored');
    store.manifest.datasetId = null;
    const roadmap = modules({ bundles: [store] });
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap.features.retainedWork).toBe(true);
    expect(roadmap.features.status).toBe('complete');
    expect(roadmap.dataset.status).toBe('not-started');
    // Training still needs a cohort, so the dataset remains the next required step.
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.cohort.blockers).toContain('dataset');
  });

  it('does not complete downstream stages from a saved model plan', () => {
    const roadmap = modules({
      datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()], drafts: [draft('mil-experiment', 'frozen')],
    });
    expect(roadmap.experiments.status).toBe('not-started');
    expect(roadmap.apply.status).toBe('not-started');
    expect(roadmap.apply.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('keeps a separately prepared cohort as saved work while development is only planned', () => {
    const cohort = { id: 'test-cohort', current: true, findings: [], manifest: { datasetId: 'other' } } as unknown as RoadmapEvidence['evaluationCohorts'][number];
    const roadmap = modules({ datasets: [dataset(), dataset('other')], targetSplits: [targetSplit()], evaluationCohorts: [cohort], drafts: [draft('development-batch')] });
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.experiments.status).toBe('draft');
    expect(roadmap.apply.status).toBe('draft');
    expect(roadmap.apply.unlocked).toBe(true);
    expect(roadmap.apply.blockers).toEqual(['experiments']);
  });

  it('treats prepared cohorts, current or not, as saved work and never as results', () => {
    for (const current of [true, false, undefined]) {
      const cohort = { id: 'test-cohort', current, findings: [] } as unknown as RoadmapEvidence['evaluationCohorts'][number];
      const roadmap = modules({ evaluationCohorts: [cohort], drafts: [draft('evaluation-cohort')] });
      expect(roadmap.apply.status).toBe('draft');
      expect(roadmap.apply.evidence).toBe('2 prepared cohorts · no runs yet');
      expect(roadmap.apply.blockers).toEqual(['experiments']);
    }
  });

  it.each(['ABMIL', 'nnMIL'])('supports interpretation before any run in Apply models when %s and feature inputs exist', (model) => {
    const predictor = { id: 'predictor', lifecycleState: 'active', manifest: { recipe: { model } } } as RoadmapEvidence['predictors'][number];
    const evidence = { datasets: [dataset()], bundles: [bundle()], predictors: [predictor] };
    expect(modules(evidence).interpretation.blockers).toEqual([]);
    expect(modules(evidence).apply.blockers).toEqual([]);
    expect(modules(evidence).apply.status).toBe('not-started');
    const otherModel = { ...predictor, manifest: { ...predictor.manifest, recipe: { ...predictor.manifest.recipe, model: 'other' } } };
    expect(modules({ ...evidence, predictors: [otherModel] }).interpretation.blockers).toEqual(['experiments']);
    expect(modules({ ...evidence, predictors: [{ ...predictor, lifecycleState: 'trashed' }] }).interpretation.blockers).toEqual(['experiments']);
  });

  it('suggests a reachable input task instead of an open registry with missing prerequisites', () => {
    const roadmap = buildRoadmap(workspace(), { datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()] });
    expect(roadmap.find((module) => module.id === 'apply')?.unlocked).toBe(true);
    expect(roadmap.find((module) => module.id === 'apply')?.blockers).toEqual(['experiments']);
    expect(suggestedRoadmapModule(roadmap)?.id).toBe('experiments');
    expect(suggestedRoadmapModule(buildRoadmap(workspace()))?.id).toBe('dataset');
  });

  it('does not present optional analyses as remaining required work', () => {
    const roadmap = buildRoadmap(workspace()).map((module) => ({ ...module, unlocked: true, blockers: [], status: module.optional ? 'draft' as const : 'complete' as const }));
    expect(suggestedRoadmapModule(roadmap)).toBeUndefined();
  });
});

describe('stage numbers', () => {
  it('gives every stage one roadmap step, shared only by stages worked on side by side', () => {
    expect(ROADMAP_STEPS.map((item) => item.step)).toEqual(['01', '02', '03', '04', '05']);
    const stages: RoadmapModuleId[] = ROADMAP_MODULES.map((item) => item.id);
    for (const id of stages) expect(ROADMAP_STEPS.filter((item) => item.modules.includes(id))).toHaveLength(1);
    expect(stages.map(stageEyebrow)).toEqual(['01 Datasets', '02 Slide features', '02 Targets & splits', '03 Experiments', '04 Apply models', '05 Model interpretation']);
    expect(stageStep('features')).toBe(stageStep('cohort'));
    expect(stageStep('apply')).not.toBe(stageStep('experiments'));
  });
});
