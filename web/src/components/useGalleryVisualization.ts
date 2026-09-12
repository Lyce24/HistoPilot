import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { interpretations, type Interpretation, type VisualizeItem, type VisualizeSelection } from '../api/interpretation';
import { mergeVisualizationItems, visualizationRequest } from '../lib/interpretationGallery';

/** A lost response keeps the exact operation and request; source remounts discard stale UI state. */
export function useGalleryVisualization(project: string, onResult: (records: Interpretation[], items: VisualizeItem[]) => void) {
  const client = useQueryClient();
  const alive = useRef(true);
  const inFlight = useRef(false);
  const [pending, setPending] = useState<ReturnType<typeof visualizationRequest> | null>(null);
  const [busy, setBusy] = useState(false);
  const [uncertain, setUncertain] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [items, setItems] = useState<VisualizeItem[]>([]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  async function perform(value: ReturnType<typeof visualizationRequest>) {
    if (inFlight.current) return;
    inFlight.current = true; setBusy(true); setPending(value); setError(null);
    try {
      const result = await interpretations.visualize(project, value.selection, value.operationId);
      if (!alive.current) return;
      setUncertain(false); setPending(null); setItems((current) => mergeVisualizationItems(current, result.items));
      for (const record of result.interpretations) { client.setQueryData(['interpretation', project, record.id], record); if (record.execution) client.setQueryData(['compute-job', project, 'interpretation', record.id], record.execution); }
      onResult(result.interpretations, result.items);
      void client.invalidateQueries({ queryKey: ['interpretations', project] });
    } catch (reason) {
      if (!alive.current) return;
      setError(reason instanceof Error ? reason : new Error('The visualization response was lost.'));
      const rejected = reason instanceof ApiError && reason.status < 500 && reason.status !== 408;
      setUncertain(!rejected);
      if (rejected) setPending(null);
    } finally { inFlight.current = false; if (alive.current) setBusy(false); }
  }
  function start(selection: VisualizeSelection) { if (!uncertain && !inFlight.current) void perform(visualizationRequest(selection, crypto.randomUUID())); }
  function retry() { if (pending) void perform(pending); }
  return { start, retry, busy, uncertain, error, items, pending, locked: busy || uncertain };
}
