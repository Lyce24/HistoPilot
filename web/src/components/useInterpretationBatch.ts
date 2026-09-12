import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react';
import { useQueries, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { interpretations, type GallerySlide, type Interpretation, type VisualizeSelection } from '../api/interpretation';
import { computeActive } from '../api/predictors';
import { mergeVisualizationItems, visualizationRequest } from '../lib/interpretationGallery';
import { persistWizardDraft, selectedBatchReady, type InterpretationWizardDraft, type SelectedStudyState } from '../lib/interpretationWizard';

export function useInterpretationBatch(project: string, draft: InterpretationWizardDraft, setDraft: Dispatch<SetStateAction<InterpretationWizardDraft>>) {
  const client = useQueryClient();
  const alive = useRef(true);
  const inFlight = useRef(new Set<string>());
  const currentDraft = useRef(draft);
  currentDraft.current = draft;
  function commit(next: InterpretationWizardDraft) {
    currentDraft.current = next;
    // Save the concrete operation before posting: a reload must retry the same request.
    persistWizardDraft(project, next);
    setDraft(next);
  }
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const batch = draft.batch;
  const items = batch?.items ?? [];
  const ids = [...new Set(items.map((item) => item.interpretationId).filter((id): id is string => Boolean(id)))];
  const documents = useQueries({ queries: ids.map((id) => ({ queryKey: ['interpretation', project, id], queryFn: () => interpretations.get(project, id), staleTime: 60000 })) });
  const jobs = useQueries({ queries: ids.map((id) => ({ queryKey: ['compute-job', project, 'interpretation', id], queryFn: () => interpretations.execution(project, id), refetchInterval: (query: { state: { data?: import('../api/interpretation').InterpretationExecution } }) => !query.state.data || computeActive(query.state.data) ? 2500 : false })) });
  const records = new Map<string, Interpretation>();
  const states = new Map<string, SelectedStudyState>();
  items.forEach((item) => {
    const index = item.interpretationId ? ids.indexOf(item.interpretationId) : -1;
    const record = index >= 0 ? documents[index].data : undefined;
    const execution = index >= 0 ? jobs[index].data ?? record?.execution : undefined;
    if (record) records.set(record.id, { ...record, execution });
    states.set(item.slidePath, { item, execution, hasRecord: Boolean(record), error: index >= 0 ? documents[index].error ?? jobs[index].error : item.error ? new Error(item.error.message) : null });
  });
  const ready = Boolean(batch && !batch.pending && selectedBatchReady(batch.selected, states));
  async function execute(batchId: string, request: ReturnType<typeof visualizationRequest>) {
    if (inFlight.current.has(request.operationId)) return;
    inFlight.current.add(request.operationId);
    const current = currentDraft.current;
    if (current.batch?.id !== batchId) { inFlight.current.delete(request.operationId); return; }
    commit({ ...current, batch: { ...current.batch, pending: request, uncertain: false, error: null } });
    try {
      const result = await interpretations.visualize(project, request.selection, request.operationId);
      for (const record of result.interpretations) { client.setQueryData(['interpretation', project, record.id], record); if (record.execution) client.setQueryData(['compute-job', project, 'interpretation', record.id], record.execution); }
      if (!alive.current) return;
      const latest = currentDraft.current;
      if (latest.batch?.id === batchId) commit({ ...latest, batch: { ...latest.batch, items: mergeVisualizationItems(latest.batch.items, result.items), pending: null, uncertain: false, error: null } });
      void client.invalidateQueries({ queryKey: ['interpretations', project] });
    } catch (reason) {
      if (!alive.current) return;
      const rejected = reason instanceof ApiError && reason.status < 500 && reason.status !== 408;
      const message = reason instanceof Error ? reason.message : 'The visualization response was lost.';
      const latest = currentDraft.current;
      if (latest.batch?.id === batchId) commit({ ...latest, batch: { ...latest.batch, pending: rejected ? null : request, uncertain: !rejected, error: message } });
    } finally { inFlight.current.delete(request.operationId); }
  }
  function start(selection: VisualizeSelection, selected: GallerySlide[]) {
    if (currentDraft.current.batch?.pending) return;
    const id = crypto.randomUUID(), pending = visualizationRequest(selection, crypto.randomUUID());
    commit({ ...currentDraft.current, batch: { id, selected: structuredClone(selected), request: structuredClone(selection), items: [], pending } });
    void execute(id, pending);
  }
  function retryExact() { const latest = currentDraft.current.batch; if (latest?.pending) void execute(latest.id, latest.pending); }
  function retrySlides(selection: VisualizeSelection) { const latest = currentDraft.current.batch; if (latest && !latest.pending) void execute(latest.id, visualizationRequest(selection, crypto.randomUUID())); }
  function refresh() { void Promise.all([...documents, ...jobs].map((query) => query.refetch())); }
  return { batch, records, states, ready, start, retryExact, retrySlides, refresh, busy: Boolean(batch?.pending && !batch.uncertain), locked: Boolean(batch?.pending), selectedCount: batch?.selected.length ?? 0 };
}
