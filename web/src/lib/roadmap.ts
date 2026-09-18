import type { FeatureBundle } from '../api/bundles';
import type { Configuration, DatasetVersion, ProtocolSpec, ScientificDraft } from '../api/scientific';
import type { Workspace } from '../api/types';
import { trainingActive, type FrozenBatch, type TrainingExecution } from '../api/development';
import type { EvaluationCohort } from '../api/evaluation';
import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { supportsAttention } from './modelCapabilities';

export type RoadmapModuleId =
  | 'dataset'
  | 'cohort'
  | 'features'
  | 'experiments'
  | 'test-data'
  | 'evaluation'
  | 'clinical-utility'
  | 'interpretation';

export type RoadmapStatus = 'not-started' | 'draft' | 'complete';

export interface RoadmapModuleDefinition {
  id: RoadmapModuleId;
  title: string;
  shortTitle: string;
  description: string;
  phase: 'prepare' | 'develop' | 'evaluate' | 'insights';
  prerequisites: readonly RoadmapModuleId[];
  optional?: boolean;
}

export const ROADMAP_MODULES: readonly RoadmapModuleDefinition[] = [
  {
    id: 'dataset', title: 'Datasets', shortTitle: 'Datasets', phase: 'prepare',
    description: 'Import slide and patient information, map annotations, and save a reusable dataset.',
    prerequisites: [],
  },
  {
    id: 'features', title: 'Slide features', shortTitle: 'Slide features', phase: 'prepare',
    description: 'Register or extract slide features and save a verified bundle. Encoding depends on slide files, not on a dataset, so this can start before any table is imported.',
    prerequisites: [],
  },
  {
    id: 'cohort', title: 'Targets & splits', shortTitle: 'Targets & splits', phase: 'prepare',
    description: 'Combine a dataset and named feature bundle, filter their shared slides, then define targets and development splits.',
    prerequisites: ['dataset', 'features'],
  },
  {
    id: 'experiments', title: 'Experiments', shortTitle: 'Experiments', phase: 'develop',
    description: 'Plan training, track runs, and generate ensemble or refit predictors for every configuration and seed.',
    prerequisites: ['cohort', 'features'],
  },
  {
    id: 'test-data', title: 'Test cohorts', shortTitle: 'Test cohorts', phase: 'evaluate',
    description: 'Select test datasets, filter slides, define prediction targets, and freeze reusable test cohorts.',
    prerequisites: ['dataset'],
  },
  {
    id: 'evaluation', title: 'Evaluate models', shortTitle: 'Evaluate models', phase: 'evaluate',
    description: 'Select development models and test cohorts, check matching targets and extracted or packed features, then evaluate.',
    prerequisites: ['experiments', 'test-data'],
  },
  {
    id: 'clinical-utility', title: 'Clinical utility', shortTitle: 'Clinical utility', phase: 'insights',
    description: 'Assess calibration, operating thresholds and net benefit using saved evaluation predictions.',
    prerequisites: ['evaluation'],
    optional: true,
  },
  {
    id: 'interpretation', title: 'Model interpretation', shortTitle: 'Model interpretation', phase: 'insights',
    description: 'Inspect ABMIL attention on slide images using a compatible model and feature bundle.',
    prerequisites: ['experiments', 'features'],
    optional: true,
  },
];

/** Main workflow branches. Cards describe additional inputs checked before execution. */
export const ROADMAP_CONNECTIONS: readonly { from: RoadmapModuleId; to: RoadmapModuleId }[] = [
  { from: 'dataset', to: 'cohort' }, { from: 'features', to: 'cohort' },
  { from: 'cohort', to: 'experiments' }, { from: 'features', to: 'experiments' },
  { from: 'experiments', to: 'evaluation' }, { from: 'test-data', to: 'evaluation' },
  { from: 'evaluation', to: 'clinical-utility' }, { from: 'experiments', to: 'interpretation' },
];

export interface RoadmapModule extends RoadmapModuleDefinition {
  status: RoadmapStatus;
  unlocked: boolean;
  blockers: RoadmapModuleId[];
  evidence: string;
  artifactCount: number;
  compatibilityIssue?: string;
  retainedWork?: boolean;
}

export interface RoadmapEvidence {
  drafts: readonly ScientificDraft<unknown>[];
  datasets: readonly DatasetVersion[];
  protocols: readonly Configuration[];
  features: readonly Configuration[];
  bundles: readonly FeatureBundle[];
  batches: readonly FrozenBatch[];
  executions: readonly TrainingExecution[];
  evaluationCohorts: readonly EvaluationCohort[];
  predictors: readonly FrozenPredictor[];
  modelEvaluations: readonly ModelEvaluation[];
  clinicalAnalyses: readonly { id: string; lifecycleState?: string }[];
  interpretations: readonly { id: string; lifecycleState?: string; execution?: { status: string } | null }[];
}

/** Project-level guidance, not a claim about the last active draft or a selected experiment. */
export function suggestedRoadmapModule(modules: readonly RoadmapModule[]): RoadmapModule | undefined {
  return modules.find((module) => !module.optional && module.status !== 'complete'
    && module.unlocked && module.blockers.length === 0);
}

const EMPTY_EVIDENCE: RoadmapEvidence = { drafts: [], datasets: [], protocols: [], features: [], bundles: [], batches: [], executions: [], evaluationCohorts: [], predictors: [], modelEvaluations: [], clinicalAnalyses: [], interpretations: [] };

interface ModuleProgress {
  status: RoadmapStatus;
  evidence: string;
  artifactCount: number;
}

function progress(complete: number, draft: number, completedLabel: string, draftLabel: string, emptyLabel: string): ModuleProgress {
  if (complete > 0) return { status: 'complete', artifactCount: complete, evidence: `${complete} ${completedLabel}${complete === 1 ? '' : 's'}` };
  if (draft > 0) return { status: 'draft', artifactCount: draft, evidence: `${draft} ${draftLabel}${draft === 1 ? '' : 's'}` };
  return { status: 'not-started', artifactCount: 0, evidence: emptyLabel };
}

/** A compatible input pair still requires the existing MIL review before a plan can be saved. */
export function protocolBundleCompatible(protocol: Configuration, bundle: FeatureBundle): boolean {
  const spec = protocol.manifest.spec as ProtocolSpec;
  return (!spec.featureBundleId || spec.featureBundleId === bundle.id)
    && (!spec.featureSetId || spec.featureSetId === bundle.manifest.spec.featureSetId)
    && (!spec.featurePackId || bundle.manifest.spec.packArtifactIds.includes(spec.featurePackId));
}

function bundleReady(bundle: FeatureBundle): boolean {
  return bundle.current
    && !bundle.findings.some((finding) => finding.severity === 'error')
    && bundle.manifest.feature.validation.tensorValidationComplete
    && bundle.manifest.packs.every((pack) => pack.validation.tensorValidationComplete);
}

/** Only a complete execution of a known immutable batch establishes development completion. */
export function completedDevelopmentBatches(batches: readonly FrozenBatch[], executions: readonly TrainingExecution[]): FrozenBatch[] {
  const byBatch = new Map(executions.map((execution) => [execution.batchId, execution]));
  return batches.filter((batch) => {
    const execution = byBatch.get(batch.id);
    const planned = batch.manifest.runs;
    const expected = batch.manifest.summary.runCount;
    if (!execution || execution.status !== 'completed' || expected < 1 || planned.length !== expected
      || execution.runCounts.total !== expected || execution.runCounts.completed !== expected
      || execution.runs.length !== expected || execution.findings.some((finding) => finding.severity === 'error')) return false;
    const completed = new Set(execution.runs.filter((run) => run.status === 'completed').map((run) => run.id));
    return completed.size === expected && planned.every((run) => completed.has(run.id));
  });
}

/**
 * Progress represents persisted project artifacts, rather than the currently open form.
 * A new draft or running batch does not undo completed development evidence. Predictor
 * freezing requires a published predictor; planned evaluations are not results.
 */
export function buildRoadmap(workspace: Workspace, evidence: Partial<RoadmapEvidence> = {}): RoadmapModule[] {
  if (workspace.mode === 'synthetic-demo' && workspace.demoPipeline) {
    return ROADMAP_MODULES.map((module) => {
      const count = workspace.demoPipeline!.records.filter((record) => record.module === module.id).length;
      return { ...module, status: 'draft', unlocked: true, blockers: [], artifactCount: count,
        evidence: count ? `${count} illustrative ${count === 1 ? 'record' : 'records'} · synthetic walkthrough` : 'Illustrative workflow explanation' };
    });
  }
  const saved = { ...EMPTY_EVIDENCE, ...evidence };
  const demo = workspace.mode === 'synthetic-demo';
  const datasetIds = new Set(saved.datasets.map((dataset) => dataset.id));
  const protocols = saved.protocols.filter((item) => item.manifest.kind === 'protocol' && datasetIds.has(item.manifest.datasetId));
  const readyBundles = saved.bundles.filter((bundle) => bundleReady(bundle));
  const importDrafts = saved.drafts.filter((draft) => draft.payload.type === 'dataset-import');
  const protocolDrafts = saved.drafts.filter((draft) => draft.payload.type === 'analysis-protocol');
  const modelDrafts = saved.drafts.filter((draft) => ['model-experiment', 'mil-experiment', 'development-batch'].includes(draft.payload.type));
  const states = Object.fromEntries(ROADMAP_MODULES.map((module) => [module.id, {
    status: 'not-started', artifactCount: 0, evidence: 'No completed artifact yet',
  } satisfies ModuleProgress])) as Record<RoadmapModuleId, ModuleProgress>;

  if (demo) {
    states.dataset = progress(workspace.dataset.slideCount > 0 ? 1 : 0, 0, 'synthetic dataset', '', 'No sample dataset');
    states.cohort = progress(workspace.cohortSnapshots.length, 0, 'saved sample cohort', '', 'Save a sample cohort');
    states.features = progress(workspace.featureSets.length, 0, 'sample feature set', '', 'No sample feature sets');
    states.experiments = progress(0, workspace.drafts.length, '', 'saved sample experiment draft', 'No saved sample experiment drafts');
  } else {
    states.dataset = progress(saved.datasets.length, importDrafts.length, 'frozen dataset', 'saved import draft', 'No frozen dataset or saved import');
    states.cohort = progress(protocols.length, protocolDrafts.length + saved.protocols.length - protocols.length, 'frozen protocol', 'saved protocol', 'No frozen protocol or saved target draft');
    states.features = progress(readyBundles.length, saved.features.length + saved.bundles.length - readyBundles.length, 'verified frozen bundle', 'saved feature artifact', 'No saved feature source or frozen bundle');
    if (states.features.status === 'draft') states.features.evidence += ' · complete bundle verification';
    states.experiments = progress(0, modelDrafts.length + saved.batches.length, '', 'saved development plan', 'No saved model development plan');
    if (saved.executions.length) {
      const finishedBatches = completedDevelopmentBatches(saved.batches, saved.executions);
      const completed = saved.executions.reduce((sum, execution) => sum + execution.runCounts.completed, 0);
      const total = saved.executions.reduce((sum, execution) => sum + execution.runCounts.total, 0);
      const active = saved.executions.filter(trainingActive).length;
      states.experiments = {
        status: finishedBatches.length ? 'complete' : 'draft', artifactCount: Math.max(saved.batches.length, saved.executions.length),
        evidence: `${finishedBatches.length ? `${finishedBatches.length} completed development batch${finishedBatches.length === 1 ? '' : 'es'} · ` : ''}${completed}/${total} training runs completed${active ? ` · ${active} active batch${active === 1 ? '' : 'es'}` : ''}`,
      };
    }
    const currentCohorts = saved.evaluationCohorts.filter((item) => item.current === true && !item.findings?.some((finding) => finding.severity === 'error'));
    states['test-data'] = progress(currentCohorts.length, saved.evaluationCohorts.length - currentCohorts.length + saved.drafts.filter((draft) => draft.payload.type === 'evaluation-cohort').length, 'frozen test cohort', 'saved test cohort', 'No prepared test cohort');
  }

  const retainedPredictors = demo ? [] : saved.predictors.filter((item) => item.lifecycleState !== 'trashed');
  const retainedEvaluations = demo ? [] : saved.modelEvaluations.filter((item) => item.lifecycleState !== 'trashed');
  if (retainedPredictors.length) {
    const published = `${retainedPredictors.length} ready predictor${retainedPredictors.length === 1 ? '' : 's'}`;
    states.experiments = { status: 'complete', artifactCount: Math.max(states.experiments.artifactCount, retainedPredictors.length), evidence: states.experiments.artifactCount ? `${states.experiments.evidence} · ${published}` : published };
  }
  const completedEvaluations = retainedEvaluations.filter((item) => item.execution?.status === 'completed').length;
  states.evaluation = progress(completedEvaluations, retainedEvaluations.length - completedEvaluations, 'completed evaluation', 'saved evaluation plan', 'No evaluation of a predictor');
  const clinicalAnalyses = demo ? [] : saved.clinicalAnalyses.filter((item) => item.lifecycleState !== 'trashed');
  const interpretations = demo ? [] : saved.interpretations.filter((item) => item.lifecycleState !== 'trashed');
  const completedInterpretations = interpretations.filter((item) => item.execution?.status === 'completed').length;
  states['clinical-utility'] = progress(clinicalAnalyses.length, 0, 'saved clinical analysis', '', 'No saved clinical utility analysis');
  states.interpretation = progress(completedInterpretations, interpretations.length - completedInterpretations, 'completed attention map', 'saved interpretation plan', 'No attention overlay generated');

  const compatibleInputs = demo || protocols.some((protocol) => readyBundles.some((bundle) => protocolBundleCompatible(protocol, bundle)));
  const retained: Partial<Record<RoadmapModuleId, boolean>> = demo ? {} : {
    cohort: saved.protocols.length > 0 || protocolDrafts.length > 0,
    features: saved.features.length > 0 || saved.bundles.length > 0,
    experiments: saved.batches.length > 0 || modelDrafts.length > 0 || retainedPredictors.length > 0,
    'test-data': saved.evaluationCohorts.length > 0 || saved.drafts.some((draft) => draft.payload.type === 'evaluation-cohort'),
  };
  return ROADMAP_MODULES.map((module) => {
    // Archived inputs disappear from new-input pickers. Retained records still
    // need an accessible editor/results view; opening it does not authorize a
    // new publication or run, which always passes the backend input checks.
    const retainedWork = retained[module.id] === true;
    const blockers = retainedWork ? [] : module.prerequisites.filter((id) => {
      if (id === 'experiments' && module.id === 'evaluation') return retainedPredictors.length === 0;
      if (id === 'experiments' && module.id === 'interpretation') return !retainedPredictors.some((item) => supportsAttention(item.manifest?.recipe?.model));
      return states[id].status !== 'complete';
    });
    const compatibilityIssue = module.id === 'experiments' && !retainedWork && blockers.length === 0 && !compatibleInputs
      ? 'Freeze a feature bundle that covers this protocol\u2019s slides, including any features or pack the protocol requires. Use the bundle named by the protocol when one is pinned.'
      : undefined;
    if (compatibilityIssue) blockers.push('features');
    // These pages are registries: users can create an experiment before inputs,
    // inspect historical chains and recover records without completing all other
    // experiments. Individual training/freeze/evaluation actions check readiness.
    const registry = !demo && ['dataset', 'features', 'cohort', 'experiments', 'test-data', 'evaluation', 'clinical-utility', 'interpretation'].includes(module.id);
    return { ...module, ...states[module.id], blockers, unlocked: registry || blockers.length === 0, compatibilityIssue, retainedWork };
  });
}
