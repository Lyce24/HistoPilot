import { useEffect, useState } from 'react';
import { queryOptions, useQuery } from '@tanstack/react-query';
import { development, type DevelopmentBatchSpec, type ResourcePolicy, type RuntimeRecommendation as Recommendation } from '../api/development';

export const runtimeRecommendationDelay = 600;

/** Cancel superseded drafts and recheck numeric input immediately before requesting. */
export function scheduleRuntimeRecommendation(isDraftValid: () => boolean, onReady: () => void, onInvalid: () => void = () => {}) {
  const timeout = setTimeout(() => { if (isDraftValid()) onReady(); else onInvalid(); }, runtimeRecommendationDelay);
  return () => clearTimeout(timeout);
}

/** Include unfinished editor text: its numeric model value may still be the old value. */
export function runtimeRecommendationSource(spec: DevelopmentBatchSpec | null, draftKey: string) {
  return spec ? JSON.stringify([spec, draftKey]) : null;
}

export function runtimeRecommendationOptions(project: string, sourceKey: string | null, spec: DevelopmentBatchSpec | null, enabled: boolean) {
  return queryOptions({
    queryKey: ['runtime-recommendation', project, sourceKey],
    queryFn: async ({ signal }) => {
      if (!spec || !sourceKey) throw new Error('Complete the batch settings before requesting a suggestion.');
      return { sourceKey, recommendation: await development.runtimeRecommendation(project, spec, signal) };
    },
    enabled: enabled && Boolean(spec && sourceKey),
    staleTime: 30_000, gcTime: 60_000, retry: false, refetchOnWindowFocus: false,
  });
}

export function canApplyRuntimeRecommendation(currentKey: string | null, resultKey: string | null, recommendation?: Recommendation, refreshing = false, error: unknown = null) {
  return Boolean(currentKey && currentKey === resultKey && recommendation?.applicable && recommendation.resources && !refreshing && !error);
}

const basisLabels: Record<Recommendation['basis'], string> = {
  measured: 'Uses matching run measurements', estimated: 'Estimated from this workload',
  hardware_only: 'Hardware-based starting point', unavailable: 'Suggestion unavailable',
};
const memory = (value: number | null) => value === null ? 'Not measured' : `${value.toFixed(1)} GiB`;

export function RuntimeRecommendationView({ recommendation, pending, error, canApply, canRefresh, onRefresh, onApply }: {
  recommendation?: Recommendation; pending: boolean; error: Error | null; canApply: boolean; canRefresh: boolean;
  onRefresh: () => void; onApply: () => void;
}) {
  return <aside className="development-capacity runtime-recommendation" aria-label="Suggested runtime settings">
    <div className="inline-actions"><strong>Suggested runtime settings</strong><button type="button" className="text-button" onClick={onRefresh} disabled={!canRefresh || pending}>Refresh suggestion</button></div>
    <p className="muted">Detection uses this system, the batch workload, and compatible run measurements when available. Suggestions are starting points; benchmark complete epoch time to find the fastest settings.</p>
    {pending ? <p role="status">Checking hardware and workload…</p> : null}
    {error ? <p className="callout" role="status">Could not get a suggestion: {error.message} You can continue configuring resources manually.</p> : null}
    {!pending && !error && !recommendation ? <p role="status">Complete valid batch settings to detect a suggestion automatically.</p> : null}
    {recommendation ? <>
      <p><strong>{basisLabels[recommendation.basis]}</strong> · {recommendation.summary}</p>
      {recommendation.resources ? <dl className="batch-settings-summary">
        <div><dt>Concurrent runs</dt><dd>{recommendation.resources.maxConcurrentRuns}</dd></div>
        <div><dt>Device</dt><dd>{recommendation.resources.gpuIds.length ? `GPU ${recommendation.resources.gpuIds.join(', ')} · ${recommendation.resources.runsPerGpu} runs per GPU` : 'CPU'}</dd></div>
        <div><dt>CPU threads per run</dt><dd>{recommendation.resources.cpuThreadsPerRun}</dd></div>
        <div><dt>Data workers per loader</dt><dd>{recommendation.resources.dataLoaderWorkers}</dd></div>
        <div><dt>RAM reservation per run</dt><dd>{memory(recommendation.resources.ramGbPerRun)}</dd></div>
        <div><dt>GPU memory allowance per run</dt><dd>{memory(recommendation.memory.perRunGpuGb)}</dd></div>
      </dl> : null}
      <p>{recommendation.memory.observedRuns} matching measured run{recommendation.memory.observedRuns === 1 ? '' : 's'}.
        {recommendation.limits.additionalRunsNow !== null ? <> Current capacity for additional runs: <strong>{recommendation.limits.additionalRunsNow}</strong>.</> : ' Current additional capacity is unknown.'}</p>
      {recommendation.hardware ? <p className="muted">Detected {recommendation.hardware.cpuCount} logical CPUs · {memory(recommendation.hardware.availableRamGb)} RAM available · {recommendation.hardware.gpus.length} GPU{recommendation.hardware.gpus.length === 1 ? '' : 's'}.</p> : null}
      {recommendation.findings.length ? <ul className="runtime-recommendation-findings">{recommendation.findings.map((finding, index) => <li key={`${finding.code}-${index}`}>{finding.message}</li>)}</ul> : null}
      {recommendation.evidence.length ? <details className="setup-details"><summary>Run measurements used</summary><ul>{recommendation.evidence.map((row) => <li key={`${row.batchId}-${row.runId}-${row.gpuUuid}-${row.key}-${row.stage}-${row.epoch}`}>Run {row.runId}, epoch {row.epoch} ({row.stage === 'fit' ? 'training and validation' : 'assessment'}): {memory(row.peakReservedGpuGb)} peak GPU reservation · {row.trainingPatches.toLocaleString()} training patches · {row.evaluationPatches.toLocaleString()} evaluation patches.</li>)}</ul></details> : null}
      <button type="button" className="btn btn-secondary" disabled={!canApply} onClick={onApply}>Apply suggested settings</button>
      <p className="muted">Applies compute settings to this editable batch. Save and submit the batch to use them. Active and frozen batches keep their saved settings.</p>
      <small>Detected {new Date(recommendation.generatedAt).toLocaleString()}. Refresh after other jobs start or finish.</small>
    </> : null}
  </aside>;
}

export interface RuntimeRecommendationProps {
  project: string; spec: DevelopmentBatchSpec | null; draftKey: string;
  isDraftValid: () => boolean; onApply: (resources: ResourcePolicy) => void;
}

export default function RuntimeRecommendation({ project, spec, draftKey, isDraftValid, onApply }: RuntimeRecommendationProps) {
  const sourceKey = runtimeRecommendationSource(spec, draftKey);
  const [ready, setReady] = useState<{ sourceKey: string; spec: DevelopmentBatchSpec } | null>(null);
  const [checked, setChecked] = useState<{ key: string | null; valid: boolean }>({ key: null, valid: false });
  useEffect(() => {
    const valid = Boolean(sourceKey && isDraftValid());
    setChecked({ key: sourceKey, valid });
    if (!valid || !sourceKey) { setReady(null); return; }
    // Capture the exact serialized draft without depending on the caller's object identity.
    const snapshot = (JSON.parse(sourceKey) as [DevelopmentBatchSpec, string])[0];
    return scheduleRuntimeRecommendation(isDraftValid, () => setReady({ sourceKey, spec: snapshot }), () => { setChecked({ key: sourceKey, valid: false }); setReady(null); });
  }, [sourceKey, isDraftValid]);
  const valid = Boolean(sourceKey && checked.key === sourceKey && checked.valid);
  const current = Boolean(valid && ready?.sourceKey === sourceKey);
  const query = useQuery(runtimeRecommendationOptions(project, ready?.sourceKey ?? null, ready?.spec ?? null, current));
  const recommendation = current && query.data?.sourceKey === sourceKey ? query.data.recommendation : undefined;
  const pending = Boolean(sourceKey && (checked.key !== sourceKey || valid && (!current || query.isFetching)));
  const error = current ? query.error : null;
  const canApply = canApplyRuntimeRecommendation(sourceKey, query.data?.sourceKey ?? null, recommendation, pending, error);
  function refresh() {
    if (!sourceKey || !spec || !isDraftValid()) return;
    if (current) void query.refetch();
    else setReady({ sourceKey, spec });
  }
  function apply() {
    if (canApply && recommendation?.resources && isDraftValid()) onApply({ ...recommendation.resources, gpuIds: [...recommendation.resources.gpuIds] });
  }
  return <RuntimeRecommendationView recommendation={recommendation} pending={pending} error={error} canApply={canApply}
    canRefresh={valid} onRefresh={refresh} onApply={apply} />;
}

/** Buttons do not emit a form change: preserve dirty-state and reviewed-plan invalidation. */
export function EditingRuntimeRecommendation({ onEdit, onApply, ...props }: RuntimeRecommendationProps & { onEdit: () => void }) {
  return <RuntimeRecommendation {...props} onApply={(resources) => { onApply(resources); onEdit(); }} />;
}
