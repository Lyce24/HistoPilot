import type { FeatureBundle } from '../api/bundles';
import type { Configuration, DatasetVersion, ScientificDraft } from '../api/scientific';
import type { TargetSplit } from '../api/targetSplits';
import type { Workspace } from '../api/types';
import { trainingActive, type FrozenBatch, type TrainingExecution } from '../api/development';
import type { EvaluationCohort } from '../api/evaluation';
import type { FrozenPredictor, ModelEvaluation } from '../api/predictors';
import { extractionActive, type ExtractionJob } from '../api/trident';
import { supportsAttention } from './modelCapabilities';
import { isInferenceRun } from './inference';

export type RoadmapModuleId =
  | 'dataset'
  | 'cohort'
  | 'features'
  | 'experimental-setup'
  | 'experiments'
  | 'test-data'
  | 'evaluation'
  | 'inference'
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
    description: 'Register or extract slide features and save a verified bundle. Prepare this alongside Targets & splits.',
    prerequisites: ['dataset'],
  },
  {
    id: 'cohort', title: 'Targets & splits', shortTitle: 'Targets & splits', phase: 'prepare',
    description: 'Choose the prediction target and split dataset records into fixed training and testing sets.',
    prerequisites: ['dataset'],
  },
  {
    id: 'experimental-setup', title: 'Experimental Setup', shortTitle: 'Experimental Setup', phase: 'develop',
    description: 'Combine a dataset, features and targets/splits. Design folds, validation and hyperparameters, then freeze the setup.',
    prerequisites: ['cohort', 'features'],
  },
  {
    id: 'experiments', title: 'Experiments', shortTitle: 'Experiments', phase: 'develop',
    description: 'Run frozen setups and track queued, active, completed and failed experiments.',
    prerequisites: ['experimental-setup'],
  },
  {
    id: 'evaluation', title: 'Evaluate models', shortTitle: 'Evaluate models', phase: 'evaluate',
    description: 'Select development models and test cohorts, check matching targets and extracted or packed features, then evaluate.',
    prerequisites: ['experiments', 'cohort'],
  },
  {
    id: 'inference', title: 'Run inference', shortTitle: 'Run inference', phase: 'evaluate',
    description: 'Apply ready predictors to unlabeled slides: predictions, label-free analysis, attention and exports. No labels or metrics.',
    prerequisites: ['experiments'],
    optional: true,
  },
  {
    id: 'clinical-utility', title: 'Clinical utility', shortTitle: 'Clinical utility', phase: 'insights',
    description: 'Assess calibration, operating thresholds and net benefit using saved evaluation predictions.',
    prerequisites: ['evaluation'],
    optional: true,
  },
  {
    id: 'interpretation', title: 'Model interpretation', shortTitle: 'Model interpretation', phase: 'insights',
    description: 'Load trained model weights with a dataset and its feature bundle, then review attention overlays, top patches and predicted labels. Needs no evaluation, inference or clinical results.',
    prerequisites: ['dataset', 'features', 'experiments'],
    optional: true,
  },
];

export interface RoadmapStep { id: string; step: string; title: string; modules: readonly RoadmapModuleId[] }

/**
 * The roadmap's numbered steps, the one source for stage numbers. Stages worked on side by side
 * share a step: slide features with targets & splits, and test cohorts with evaluation and
 * inference. The roadmap page and every stage page's eyebrow read their numbers from here.
 */
export const ROADMAP_STEPS: readonly RoadmapStep[] = [
  { id: 'datasets', step: '01', title: 'Datasets', modules: ['dataset'] },
  { id: 'prepare', step: '02', title: 'Prepare in parallel', modules: ['features', 'cohort'] },
  { id: 'setup', step: '03', title: 'Experimental Setup', modules: ['experimental-setup'] },
  { id: 'develop', step: '04', title: 'Experiments', modules: ['experiments'] },
  { id: 'evaluate', step: '05', title: 'Evaluate models & run inference', modules: ['test-data', 'evaluation', 'inference'] },
  { id: 'clinical', step: '06', title: 'Clinical utility', modules: ['clinical-utility'] },
  { id: 'interpret', step: '07', title: 'Interpretation', modules: ['interpretation'] },
];

/** A stage's roadmap step number, for example "02" for Slide features. */
export const stageStep = (id: RoadmapModuleId): string | undefined => ROADMAP_STEPS.find((item) => item.modules.includes(id))?.step;

/** A stage page's eyebrow: its roadmap step number and short name, for example "02 Slide features". */
export function stageEyebrow(id: RoadmapModuleId): string {
  const step = stageStep(id);
  const name = ROADMAP_MODULES.find((item) => item.id === id)?.shortTitle ?? (id === 'test-data' ? 'Test cohorts' : id);
  return step ? `${step} ${name}` : name;
}

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
  targetSplits: readonly TargetSplit[];
  setups: readonly Configuration[];
  features: readonly Configuration[];
  bundles: readonly FeatureBundle[];
  extractions: readonly ExtractionJob[];
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

const EMPTY_EVIDENCE: RoadmapEvidence = { drafts: [], datasets: [], targetSplits: [], setups: [], features: [], bundles: [], extractions: [], batches: [], executions: [], evaluationCohorts: [], predictors: [], modelEvaluations: [], clinicalAnalyses: [], interpretations: [] };

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

function extractionEvidence(jobs: readonly ExtractionJob[]): string | undefined {
  const active = jobs.filter(extractionActive);
  if (active.length) {
    const job = active[0];
    const current = job.progress;
    const label = job.state === 'queued' ? 'Queued' : current?.label ?? (job.state === 'cancelling' ? 'Stopping extraction' : 'Preparing');
    // Each worker/batch owns its counter. Do not present it as whole-job coverage.
    const count = current && current.completed !== null && current.total !== null && current.total > 0 && job.state !== 'queued'
      ? ` · ${current.completed}/${current.total} ${current.unit} in ${current.scope === 'batch' ? 'latest batch' : 'stage'}`
      : '';
    return `${active.length} extraction${active.length === 1 ? '' : 's'} in progress · ${label}${count}`;
  }
  const latest = jobs[0];
  if (!latest) return undefined;
  const outcome = latest.state === 'succeeded' ? 'completed' : latest.state;
  return `${jobs.length} extraction run${jobs.length === 1 ? '' : 's'} · latest ${outcome} · ${latest.state === 'succeeded' ? 'review outputs and freeze a bundle' : 'review run'}`;
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
  const datasetIds = new Set(saved.datasets.map((dataset) => dataset.id));
  const readyBundles = saved.bundles.filter((bundle) => bundleReady(bundle));
  const importDrafts = saved.drafts.filter((draft) => draft.payload.type === 'dataset-import');
  const protocolDrafts = saved.drafts.filter((draft) => draft.payload.type === 'target-split');
  const targetSplits = saved.targetSplits.filter((item) => datasetIds.has(item.manifest.datasetId));
  const modelDrafts = saved.drafts.filter((draft) => ['model-experiment', 'mil-experiment', 'development-batch'].includes(draft.payload.type));
  const states = Object.fromEntries(ROADMAP_MODULES.map((module) => [module.id, {
    status: 'not-started', artifactCount: 0, evidence: 'No completed artifact yet',
  } satisfies ModuleProgress])) as Record<RoadmapModuleId, ModuleProgress>;

  states.dataset = progress(saved.datasets.length, importDrafts.length, 'frozen dataset', 'saved import draft', 'No frozen dataset or saved import');
  states.cohort = progress(targetSplits.length, protocolDrafts.length + saved.targetSplits.length - targetSplits.length, 'frozen target and split', 'saved target draft', 'No frozen targets and splits');
  states['experimental-setup'] = progress(saved.setups.length, modelDrafts.filter((draft) => draft.status !== 'frozen').length, 'frozen setup', 'setup draft', 'No frozen experimental setup');
  states.features = progress(readyBundles.length, saved.features.length + saved.bundles.length - readyBundles.length, 'verified frozen bundle', 'saved feature artifact', 'No saved feature source or frozen bundle');
  if (states.features.status === 'draft') states.features.evidence += ' · complete bundle verification';
  const extraction = extractionEvidence(saved.extractions);
  if (extraction) {
    const existing = states.features;
    states.features = {
      status: existing.status === 'complete' ? 'complete' : 'draft',
      artifactCount: existing.status === 'complete' ? existing.artifactCount : existing.artifactCount + saved.extractions.length,
      evidence: existing.status === 'not-started' ? extraction : `${existing.evidence} · ${extraction}`,
    };
  }
  states.experiments = progress(0, saved.batches.length + saved.setups.length, '', 'setup ready to run', 'No experiments started');
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

  const retainedPredictors = saved.predictors.filter((item) => item.lifecycleState !== 'trashed');
  const retainedEvaluations = saved.modelEvaluations.filter((item) => item.lifecycleState !== 'trashed');
  if (retainedPredictors.length) {
    const published = `${retainedPredictors.length} ready predictor${retainedPredictors.length === 1 ? '' : 's'}`;
    states.experiments = { status: 'complete', artifactCount: Math.max(states.experiments.artifactCount, retainedPredictors.length), evidence: states.experiments.artifactCount ? `${states.experiments.evidence} · ${published}` : published };
  }
  // Inference runs share the evaluation record kind but never count as labeled evidence.
  const scored = retainedEvaluations.filter((item) => !isInferenceRun(item));
  const predicted = retainedEvaluations.filter(isInferenceRun);
  const completedEvaluations = scored.filter((item) => item.execution?.status === 'completed').length;
  states.evaluation = progress(completedEvaluations, scored.length - completedEvaluations, 'completed evaluation', 'saved evaluation plan', 'No evaluation of a predictor');
  const completedInference = predicted.filter((item) => item.execution?.status === 'completed').length;
  states.inference = progress(completedInference, predicted.length - completedInference, 'completed inference run', 'saved inference plan', 'No predictions for unlabeled slides');
  const clinicalAnalyses = saved.clinicalAnalyses.filter((item) => item.lifecycleState !== 'trashed');
  const interpretations = saved.interpretations.filter((item) => item.lifecycleState !== 'trashed');
  const completedInterpretations = interpretations.filter((item) => item.execution?.status === 'completed').length;
  states['clinical-utility'] = progress(clinicalAnalyses.length, 0, 'saved clinical analysis', '', 'No saved clinical utility analysis');
  states.interpretation = progress(completedInterpretations, interpretations.length - completedInterpretations, 'completed attention map', 'saved interpretation plan', 'No attention overlay generated');

  const compatibleInputs = targetSplits.length > 0 && readyBundles.length > 0;
  const retained: Partial<Record<RoadmapModuleId, boolean>> = {
    cohort: saved.targetSplits.length > 0 || protocolDrafts.length > 0,
    'experimental-setup': saved.setups.length > 0 || modelDrafts.length > 0,
    features: saved.features.length > 0 || saved.bundles.length > 0 || saved.extractions.length > 0,
    experiments: saved.batches.length > 0 || saved.setups.length > 0 || retainedPredictors.length > 0,
    'test-data': saved.evaluationCohorts.length > 0 || saved.drafts.some((draft) => draft.payload.type === 'evaluation-cohort'),
  };
  return ROADMAP_MODULES.map((module) => {
    // Archived inputs disappear from new-input pickers. Retained records still
    // need an accessible editor/results view; opening it does not authorize a
    // new publication or run, which always passes the backend input checks.
    const retainedWork = retained[module.id] === true;
    const blockers = retainedWork ? [] : module.prerequisites.filter((id) => {
      if (id === 'cohort' && module.id === 'evaluation') return !saved.evaluationCohorts.some((item) => item.current && !item.findings?.some((finding) => finding.severity === 'error'));
      if (id === 'experiments' && (module.id === 'evaluation' || module.id === 'inference')) return retainedPredictors.length === 0;
      if (id === 'experiments' && module.id === 'interpretation') return !retainedPredictors.some((item) => supportsAttention(item.manifest?.recipe?.model));
      return states[id].status !== 'complete';
    });
    const compatibilityIssue = module.id === 'experimental-setup' && !retainedWork && blockers.length === 0 && !compatibleInputs
      ? 'Freeze targets and splits and a current feature bundle, then check their training coverage in Experimental Setup.'
      : undefined;
    if (compatibilityIssue) blockers.push('features');
    // These pages are registries: users can create an experiment before inputs,
    // inspect historical chains and recover records without completing all other
    // experiments. Individual training/freeze/evaluation actions check readiness.
    const registry = ['dataset', 'features', 'cohort', 'experimental-setup', 'experiments', 'test-data', 'evaluation', 'inference', 'clinical-utility', 'interpretation'].includes(module.id);
    return { ...module, ...states[module.id], blockers, unlocked: registry || blockers.length === 0, compatibilityIssue, retainedWork };
  });
}
