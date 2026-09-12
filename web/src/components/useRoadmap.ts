import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { bundles } from '../api/bundles';
import { scientific } from '../api/scientific';
import { development, developmentPollInterval } from '../api/development';
import { evaluation } from '../api/evaluation';
import { modelEvaluations, predictors } from '../api/predictors';
import { clinicalAnalyses } from '../api/clinicalUtility';
import { interpretations } from '../api/interpretation';
import type { Workspace } from '../api/types';
import { buildRoadmap, type RoadmapModule, type RoadmapModuleId } from '../lib/roadmap';

export function useRoadmap(workspace: Workspace) {
  const project = workspace.project.id;
  const enabled = workspace.mode === 'local';
  // Share cache keys with the module editors so saving and freezing refresh progress.
  const drafts = useQuery({ queryKey: ['scientific', project, 'drafts'], queryFn: () => scientific.drafts(project), enabled });
  const datasets = useQuery({ queryKey: ['scientific', project, 'datasets'], queryFn: () => scientific.datasets(project), enabled });
  const protocols = useQuery({ queryKey: ['scientific', project, 'configurations', 'protocol'], queryFn: () => scientific.configurations(project, 'protocol'), enabled });
  const features = useQuery({ queryKey: ['scientific', project, 'configurations', 'feature'], queryFn: () => scientific.configurations(project, 'feature'), enabled });
  const featureBundles = useQuery({ queryKey: ['feature-bundles', project], queryFn: () => bundles.list(project), enabled });
  const batches = useQuery({ queryKey: ['development-batches', project], queryFn: () => development.list(project), enabled, refetchInterval: (query) => developmentPollInterval(query.state.data) });
  const evaluationCohorts = useQuery({ queryKey: ['evaluation-cohorts', project], queryFn: () => evaluation.list(project), enabled });
  const frozenPredictors = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project), enabled });
  const evaluationRecords = useQuery({ queryKey: ['model-evaluations', project], queryFn: () => modelEvaluations.list(project), enabled });
  const clinicalRecords = useQuery({ queryKey: ['clinical-analyses', project], queryFn: () => clinicalAnalyses.list(project), enabled });
  const interpretationRecords = useQuery({ queryKey: ['interpretations', project], queryFn: () => interpretations.list(project), enabled, refetchInterval: 10000 });
  const modules = useMemo(() => buildRoadmap(workspace, {
    drafts: drafts.data?.drafts ?? [],
    datasets: datasets.data?.datasets ?? [],
    protocols: protocols.data?.configurations ?? [],
    features: features.data?.configurations ?? [],
    bundles: featureBundles.data?.items ?? [],
    batches: batches.data?.items ?? [],
    executions: batches.data?.executions ?? [],
    evaluationCohorts: evaluationCohorts.data?.items ?? [],
    predictors: frozenPredictors.data?.items ?? [],
    modelEvaluations: evaluationRecords.data?.items ?? [],
    clinicalAnalyses: clinicalRecords.data?.items ?? [],
    interpretations: interpretationRecords.data?.items ?? [],
  }), [workspace, drafts.data, datasets.data, protocols.data, features.data, featureBundles.data, batches.data, evaluationCohorts.data, frozenPredictors.data, evaluationRecords.data, clinicalRecords.data, interpretationRecords.data]);
  const byId = useMemo(() => Object.fromEntries(modules.map((module) => [module.id, module])) as Record<RoadmapModuleId, RoadmapModule>, [modules]);
  const queries = [drafts, datasets, protocols, features, featureBundles, batches, evaluationCohorts, frozenPredictors, evaluationRecords, clinicalRecords, interpretationRecords];
  const check = (required: typeof queries) => ({
    isLoading: enabled && required.some((query) => query.isPending),
    error: enabled ? required.find((query) => query.error)?.error ?? null : null,
    // A failed background refresh must not discard an editor's existing inputs.
    hasData: !enabled || required.every((query) => query.data !== undefined),
  });
  const { isLoading, error, hasData } = check(queries);
  const inputQueries = [datasets, protocols, featureBundles];
  const checksById = Object.fromEntries(modules.map(({ id, retainedWork }) => {
    const result = check(
    id === 'dataset' || (enabled && ['experiments', 'post-development', 'evaluation', 'clinical-utility', 'interpretation'].includes(id))
      ? []
      : id === 'post-development'
        ? [batches]
      : id === 'evaluation'
        ? [batches, evaluationCohorts]
      : id === 'clinical-utility'
        ? [evaluationRecords]
      : id === 'interpretation'
        ? [clinicalRecords, frozenPredictors]
      : id === 'test-data'
        ? [datasets, protocols]
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
