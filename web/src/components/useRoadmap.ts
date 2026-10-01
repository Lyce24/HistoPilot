import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { bundles } from '../api/bundles';
import { targetSplits as targetSplitApi } from '../api/targetSplits';
import { scientific } from '../api/scientific';
import { development, developmentPollInterval, trainingActive } from '../api/development';
import { evaluation } from '../api/evaluation';
import { computePollInterval, modelEvaluations, predictors } from '../api/predictors';
import { clinicalAnalyses } from '../api/clinicalUtility';
import { interpretations } from '../api/interpretation';
import { extractionActive, trident } from '../api/trident';
import type { Workspace } from '../api/types';
import { buildRoadmap, type RoadmapModule, type RoadmapModuleId } from '../lib/roadmap';

export function useRoadmap(workspace: Workspace) {
  const project = workspace.project.id;
  const enabled = workspace.mode === 'local';
  // Share cache keys with the module editors so saving and freezing refresh progress.
  const drafts = useQuery({ queryKey: ['scientific', project, 'drafts'], queryFn: () => scientific.drafts(project), enabled });
  const datasets = useQuery({ queryKey: ['scientific', project, 'datasets'], queryFn: () => scientific.datasets(project), enabled });
  const targetSplits = useQuery({ queryKey: ['scientific', project, 'configurations', 'target-split'], queryFn: () => targetSplitApi.list(project), enabled });
  const setups = useQuery({ queryKey: ['scientific', project, 'configurations', 'experiment-setup'], queryFn: () => scientific.configurations(project, 'experiment-setup'), enabled });
  const features = useQuery({ queryKey: ['scientific', project, 'configurations', 'feature'], queryFn: () => scientific.configurations(project, 'feature'), enabled });
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project), enabled });
  const extractions = useQuery({ queryKey: ['extractions', project, 'jobs'], queryFn: () => trident.jobs(project), enabled, refetchIntervalInBackground: false, refetchInterval: (query) => query.state.data?.jobs.some(extractionActive) ? 3000 : false });
  const batches = useQuery({ queryKey: ['development-batches', project], queryFn: () => development.list(project), enabled, refetchIntervalInBackground: false, refetchInterval: (query) => developmentPollInterval(query.state.data) });
  const evaluationCohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project), enabled });
  // Predictors are published by finished training and by the predictor coordinator,
  // so they only need a fast refresh while a development batch is running.
  const frozenPredictors = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), enabled, refetchIntervalInBackground: false, refetchInterval: () => batches.data?.executions?.some(trainingActive) ? 10000 : 60000 });
  const evaluationRecords = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), enabled });
  const clinicalRecords = useQuery({ queryKey: ['clinical-analyses', project], queryFn: () => clinicalAnalyses.list(project), enabled });
  const interpretationRecords = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled, refetchIntervalInBackground: false, refetchInterval: (query) => computePollInterval(query.state.data?.items) });
  const modules = useMemo(() => buildRoadmap(workspace, {
    drafts: drafts.data?.drafts ?? [],
    datasets: datasets.data?.datasets ?? [],
    targetSplits: targetSplits.data?.configurations ?? [],
    setups: setups.data?.configurations ?? [],
    features: features.data?.configurations ?? [],
    bundles: featureBundles.data?.items ?? [],
    extractions: extractions.data?.jobs ?? [],
    batches: batches.data?.items ?? [],
    executions: batches.data?.executions ?? [],
    evaluationCohorts: evaluationCohorts.data?.items ?? [],
    predictors: frozenPredictors.data?.items ?? [],
    modelEvaluations: evaluationRecords.data?.items ?? [],
    clinicalAnalyses: clinicalRecords.data?.items ?? [],
    interpretations: interpretationRecords.data?.items ?? [],
  }), [workspace, drafts.data, datasets.data, targetSplits.data, setups.data, features.data, featureBundles.data, extractions.data, batches.data, evaluationCohorts.data, frozenPredictors.data, evaluationRecords.data, clinicalRecords.data, interpretationRecords.data]);
  const byId = useMemo(() => Object.fromEntries(modules.map((module) => [module.id, module])) as Record<RoadmapModuleId, RoadmapModule>, [modules]);
  const queries = [drafts, datasets, targetSplits, setups, features, featureBundles, extractions, batches, evaluationCohorts, frozenPredictors, evaluationRecords, clinicalRecords, interpretationRecords];
  const check = (required: typeof queries) => ({
    isLoading: enabled && required.some((query) => query.isPending),
    error: enabled ? required.find((query) => query.error)?.error ?? null : null,
    // A failed background refresh must not discard an editor's existing inputs.
    hasData: !enabled || required.every((query) => query.data !== undefined),
  });
  const { isLoading, error, hasData } = check(queries);
  const inputQueries = [datasets, featureBundles];
  const checksById = Object.fromEntries(modules.map(({ id, retainedWork }) => {
    const result = check(
    id === 'dataset' || (enabled && ['cohort', 'features', 'experiments', 'apply', 'interpretation'].includes(id))
      ? []
      : id === 'apply'
        ? [batches, evaluationCohorts]
      : id === 'interpretation'
        ? [frozenPredictors, featureBundles]
      : id === 'cohort' || id === 'features'
        ? [datasets]
        : inputQueries,
    );
    // Cached retained work remains inspectable even if fresh input choices are
    // unavailable. Keep the query warning and enforce readiness on new actions.
    return [id, retainedWork ? { ...result, hasData: true, isLoading: false } : result];
  })) as Record<RoadmapModuleId, ReturnType<typeof check>>;

  return {
    modules,
    byId,
    isLoading,
    loading: isLoading,
    error,
    hasData,
    checksById,
    async refetch() {
      if (enabled) await Promise.all(queries.map((query) => query.refetch()));
    },
  };
}
