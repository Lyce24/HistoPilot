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
      stage: 'segmentation', stages: [], label: 'Tissue segmentation', detail: '', completed: 128, total: 1111,
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
  it('counts inference runs in Run inference and never as labeled evaluations', () => {
    const predictor = { id: 'predictor-a', lifecycleState: 'active' } as RoadmapEvidence['predictors'][number];
    const inference = { id: 'inference-a', lifecycleState: 'active', manifest: { status: 'planned', purpose: 'inference' }, execution: { status: 'completed' } } as RoadmapEvidence['modelEvaluations'][number];
    const review = { id: 'review-a', lifecycleState: 'active', manifest: { status: 'planned', purpose: 'review' } } as RoadmapEvidence['modelEvaluations'][number];
    const roadmap = modules({ predictors: [predictor], modelEvaluations: [inference, review] });
    expect(roadmap.inference.status).toBe('complete');
    expect(roadmap.inference.evidence).toBe('1 completed inference run');
    expect(roadmap.inference.blockers).toEqual([]);
    expect(roadmap.evaluation.status).toBe('not-started');
    expect(roadmap.inference.optional).toBe(true);
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

  it('links nine modules with optional inference, clinical utility and interpretation branches', () => {
    expect(ROADMAP_MODULES.map((module) => [module.id, module.phase])).toEqual([
      ['dataset', 'prepare'], ['features', 'prepare'], ['cohort', 'prepare'],
      ['experimental-setup', 'develop'], ['experiments', 'develop'],
      ['evaluation', 'evaluate'], ['inference', 'evaluate'],
      ['clinical-utility', 'insights'], ['interpretation', 'insights'],
    ]);
    expect(ROADMAP_MODULES.find((module) => module.id === 'inference')?.prerequisites).toEqual(['experiments']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'cohort')?.prerequisites).toEqual(['dataset']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'features')?.prerequisites).toEqual(['dataset']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'experimental-setup')?.prerequisites).toEqual(['cohort', 'features']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'experiments')?.prerequisites).toEqual(['experimental-setup']);
    expect(ROADMAP_MODULES.some((module) => module.id === 'test-data')).toBe(false);
    expect(ROADMAP_MODULES.find((module) => module.id === 'evaluation')?.prerequisites).toEqual(['experiments', 'cohort']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'clinical-utility')?.prerequisites).toEqual(['evaluation']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'interpretation')?.prerequisites).toEqual(['dataset', 'features', 'experiments']);
    expect(ROADMAP_MODULES.filter((module) => module.optional).map((module) => module.id)).toEqual(['inference', 'clinical-utility', 'interpretation']);
  });
  it('opens each registry while keeping setup and execution prerequisites distinct', () => {
    const roadmap = buildRoadmap(workspace());
    expect(roadmap.filter((module) => module.unlocked).map((module) => module.id)).toEqual(['dataset', 'features', 'cohort', 'experimental-setup', 'experiments', 'evaluation', 'inference', 'clinical-utility', 'interpretation']);
    expect(roadmap.every((module) => module.status === 'not-started')).toBe(true);
    expect(modules().evaluation.blockers).toEqual(['experiments', 'cohort']);
    expect(modules().inference.blockers).toEqual(['experiments']);
    expect(roadmap).toHaveLength(9);
  });

  it('counts saved clinical analyses and completed attention maps separately from plans', () => {
    const clinical = { id: 'clinical', lifecycleState: 'active' };
    const planned = { id: 'attention', lifecycleState: 'active', execution: { status: 'queued' } };
    const roadmap = modules({ clinicalAnalyses: [clinical], interpretations: [planned] });
    expect(roadmap['clinical-utility'].status).toBe('complete');
    expect(roadmap.interpretation.status).toBe('draft');
    // Interpretation needs a dataset, its features and trained weights; never evaluation or clinical results.
    expect(roadmap.interpretation.blockers).toEqual(['dataset', 'features', 'experiments']);
    expect(ROADMAP_MODULES.find((module) => module.id === 'interpretation')?.prerequisites).toEqual(['dataset', 'features', 'experiments']);
    expect(modules({ interpretations: [{ ...planned, execution: { status: 'completed' } }] }).interpretation.status).toBe('complete');
    const trashed = modules({ clinicalAnalyses: [{ ...clinical, lifecycleState: 'trashed' }], interpretations: [{ ...planned, lifecycleState: 'trashed', execution: { status: 'completed' } }] });
    expect(trashed['clinical-utility'].status).toBe('not-started');
    expect(trashed.interpretation.status).toBe('not-started');
    expect(trashed.interpretation.unlocked).toBe(true);
  });

  it('recognizes saved drafts without treating them as frozen prerequisites', () => {
    const roadmap = modules({ drafts: [draft('dataset-import'), draft('target-split'), draft('mil-experiment')] });
    expect(roadmap.dataset.status).toBe('draft');
    expect(roadmap.cohort.status).toBe('draft');
    expect(roadmap['experimental-setup'].status).toBe('draft');
    expect(roadmap.experiments.status).toBe('not-started');
    expect(roadmap.cohort.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap['experimental-setup'].unlocked).toBe(true);
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
    expect(roadmap.cohort.blockers).toEqual([]);
    expect(roadmap.features.status).toBe('not-started');
    expect(roadmap['experimental-setup'].blockers).toEqual(['cohort', 'features']);
    expect(roadmap.experiments.blockers).toEqual(['experimental-setup']);
    expect(roadmap.features.unlocked).toBe(true);
    expect(roadmap.experiments.unlocked).toBe(true);
  });

  it('prepares features and targets independently, then requires a frozen setup before execution', () => {
    const prepared = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()] });
    expect(prepared.cohort.status).toBe('complete');
    expect(prepared.features.status).toBe('complete');
    expect(prepared['experimental-setup'].blockers).toEqual([]);
    expect(prepared['experimental-setup'].status).toBe('not-started');
    expect(prepared.experiments.blockers).toEqual(['experimental-setup']);
    const frozen = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()], setups: [setup()] });
    expect(frozen['experimental-setup'].status).toBe('complete');
    expect(frozen.experiments.blockers).toEqual([]);
    expect(frozen.experiments.status).toBe('draft');
    expect(frozen.experiments.evidence).toBe('1 setup ready to run');
  });

  it('keeps saved feature headers in progress until a current fully validated bundle exists', () => {
    const source = { ...protocol(), manifest: { ...protocol().manifest, kind: 'feature' } } as Configuration;
    const roadmap = modules({ datasets: [dataset()], targetSplits: [targetSplit()], features: [source] });
    expect(roadmap.cohort.status).toBe('complete');
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap['experimental-setup'].blockers).toEqual(['features']);
    const ready = modules({ datasets: [dataset()], targetSplits: [targetSplit()], features: [source], bundles: [bundle()] });
    expect(ready.features.status).toBe('complete');
    expect(ready.experiments.unlocked).toBe(true);
  });

  it('shows a live extraction before any feature source or bundle exists without satisfying feature prerequisites', () => {
    const roadmap = modules({ datasets: [dataset()], extractions: [extraction()] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.features.evidence).toBe('1 extraction in progress · Tissue segmentation · 128/1111 slides in stage');
    expect(roadmap.features.artifactCount).toBe(1);
    expect(roadmap.features.retainedWork).toBe(true);
    expect(roadmap.cohort.blockers).toEqual([]);
    expect(roadmap['experimental-setup'].blockers).toContain('features');
  });

  it('names the latest worker batch without aggregating separate active extraction counters', () => {
    const first = extraction();
    const second = { ...extraction(), id: 'extraction-other' };
    const roadmap = modules({ extractions: [{ ...first, progress: { ...first.progress!, scope: 'batch' } }, second] });
    expect(roadmap.features.evidence).toBe('2 extractions in progress · Tissue segmentation · 128/1111 slides in latest batch');
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
    expect(roadmap['experimental-setup'].blockers).toContain('features');
  });

  it('requires a frozen verified bundle after extraction succeeds and preserves readiness while another extraction runs', () => {
    const finished = modules({ extractions: [extraction('succeeded')] });
    expect(finished.features.status).toBe('draft');
    expect(finished.features.evidence).toContain('latest completed · review outputs and freeze a bundle');
    expect(finished['experimental-setup'].blockers).toContain('features');
    const ready = modules({ bundles: [bundle()], extractions: [extraction()] });
    expect(ready.features.status).toBe('complete');
    expect(ready.features.artifactCount).toBe(1);
    expect(ready.features.evidence).toBe('1 verified frozen bundle · 1 extraction in progress · Tissue segmentation · 128/1111 slides in stage');
  });

  it.each(['current', 'tensor', 'findings'] as const)('keeps the experiment registry open while feature %s evidence blocks new training', (failure) => {
    const features = bundle();
    if (failure === 'current') features.current = false;
    if (failure === 'tensor') features.manifest.feature.validation.tensorValidationComplete = false;
    if (failure === 'findings') features.findings.push({ severity: 'error', code: 'CHANGED', message: 'Features changed' });
    const roadmap = modules({ datasets: [dataset()], targetSplits: [targetSplit()], bundles: [features] });
    expect(roadmap.features.status).toBe('draft');
    expect(roadmap.experiments.unlocked).toBe(true);
    expect(roadmap['experimental-setup'].blockers).toContain('features');
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
    expect(roadmap.evaluation.blockers).toContain('cohort');
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
    expect(roadmap['experimental-setup'].status).toBe('not-started');
    expect(roadmap.evaluation.status).toBe('not-started');
    expect(roadmap.evaluation.unlocked).toBe(true);
    expect(roadmap['experimental-setup'].unlocked).toBe(true);
  });

  it('accepts a separately imported verified test cohort while development is only planned', () => {
    const cohort = { id: 'test-cohort', current: true, findings: [], manifest: { datasetId: 'other' } } as unknown as RoadmapEvidence['evaluationCohorts'][number];
    const roadmap = modules({ datasets: [dataset(), dataset('other')], targetSplits: [targetSplit()], evaluationCohorts: [cohort], drafts: [draft('development-batch')] });
    expect(roadmap.evaluation.blockers).not.toContain('cohort');
    expect(roadmap['experimental-setup'].unlocked).toBe(true);
    expect(roadmap['experimental-setup'].status).toBe('draft');
    expect(roadmap.experiments.status).toBe('not-started');
    expect(roadmap.evaluation.blockers).toEqual(['experiments']);
  });

  it('does not complete test preparation from a stale or unverified cohort', () => {
    for (const current of [false, undefined]) {
      const cohort = { id: 'test-cohort', current, findings: [] } as unknown as RoadmapEvidence['evaluationCohorts'][number];
      expect(modules({ evaluationCohorts: [cohort] }).evaluation.blockers).toContain('cohort');
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
    const roadmap = buildRoadmap(workspace(), { datasets: [dataset()], targetSplits: [targetSplit()], bundles: [bundle()], batches: [batch], executions: [execution] });
    expect(roadmap.find((module) => module.id === 'evaluation')?.unlocked).toBe(true);
    expect(suggestedRoadmapModule(roadmap)?.id).toBe('experimental-setup');
    expect(suggestedRoadmapModule(buildRoadmap(workspace()))?.id).toBe('dataset');
  });

  it('does not present optional analyses as remaining required work', () => {
    const roadmap = buildRoadmap(workspace()).map((module) => ({ ...module, unlocked: true, blockers: [], status: module.optional ? 'draft' as const : 'complete' as const }));
    expect(suggestedRoadmapModule(roadmap)).toBeUndefined();
  });
});

describe('stage numbers', () => {
  it('gives every stage one roadmap step, shared only by stages worked on side by side', () => {
    expect(ROADMAP_STEPS.map((item) => item.step)).toEqual(['01', '02', '03', '04', '05', '06', '07']);
    const stages: RoadmapModuleId[] = [...ROADMAP_MODULES.map((item) => item.id), 'test-data'];
    for (const id of stages) expect(ROADMAP_STEPS.filter((item) => item.modules.includes(id))).toHaveLength(1);
    expect(stages.map(stageEyebrow)).toEqual(['01 Datasets', '02 Slide features', '02 Targets & splits', '03 Experimental Setup', '04 Experiments', '05 Evaluate models', '05 Run inference', '06 Clinical utility', '07 Model interpretation', '05 Test cohorts']);
    // Test cohorts belong with evaluation, not with Experimental Setup's step.
    expect(stageStep('test-data')).toBe(stageStep('evaluation'));
    expect(stageStep('test-data')).not.toBe(stageStep('experimental-setup'));
  });
});
