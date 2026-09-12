import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import { cleanupApplyRequest, cleanupPollInterval, cleanupReviewMatches, lifecycle } from '../api/lifecycle';
import type { CleanupAction, CleanupPreview, LifecycleState } from '../api/lifecycle';
import { CleanupReviewPanel } from '../pages/WorkspaceCleanup';
import { ErrorNotice } from './ui';

/** Reuses the workspace lifecycle review, including exact retries and dependency acknowledgement. */
export default function ExperimentLifecycle({ project, recordKey, state, name }: {
  project: string; recordKey: string; state: LifecycleState; name: string;
}) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const inventory = useQuery({ queryKey: ['cleanup', project], queryFn: () => lifecycle.inventory(project), enabled: open,
    refetchInterval: (query) => open ? cleanupPollInterval(query.state.data) : false });
  const [review, setReview] = useState<{ preview: CleanupPreview; operationId: string; uncertain: boolean } | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [notice, setNotice] = useState('');
  const [selected, setSelected] = useState([recordKey]);
  const current = Boolean(review && !inventory.isError && cleanupReviewMatches(review.preview, inventory.data, review.preview.action, selected));
  async function preview(action: CleanupAction, keys = [recordKey]) {
    setOpen(true); setBusy(true); setReview(null); setAcknowledged(false); setError(null); setNotice(''); setSelected(keys);
    try {
      const result = await lifecycle.preview(project, action, keys);
      setReview({ preview: result, operationId: `experiment-cleanup:${crypto.randomUUID()}`, uncertain: false });
      await inventory.refetch();
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('The change could not be reviewed.')); }
    finally { setBusy(false); }
  }
  async function apply() {
    if (!review || busy || (!review.uncertain && (!current || !acknowledged))) return;
    setBusy(true); setError(null);
    try {
      await lifecycle.apply(project, cleanupApplyRequest(review.preview, review.operationId));
      setReview(null); setAcknowledged(false);
      setNotice('Records updated. Files and completed work were retained.');
      await client.invalidateQueries({ predicate: (query) => query.queryKey.includes(project) || query.queryKey[0] === 'projects' });
    } catch (reason) {
      if (reason instanceof ApiError) { setReview(null); setAcknowledged(false); void inventory.refetch(); }
      else setReview({ ...review, uncertain: true });
      setError(reason instanceof Error ? reason : new Error('The confirmation response was lost.'));
    } finally { setBusy(false); }
  }
  return <section className="experiment-lifecycle" aria-label={`Manage ${name}`}>
    <div className="inline-actions">
      {state === 'active' ? <button className="btn btn-secondary btn-small" disabled={busy || Boolean(review?.uncertain)} onClick={() => void preview('archive')}>Archive</button> : <button className="btn btn-secondary btn-small" disabled={busy || Boolean(review?.uncertain)} onClick={() => void preview('restore')}>Restore to Active</button>}
      {state !== 'trashed' ? <button className="btn btn-secondary btn-small" disabled={busy || Boolean(review?.uncertain)} onClick={() => void preview('trash')}>Delete…</button> : null}
    </div>
    <ErrorNotice error={error ?? (open ? inventory.error : null)} />
    {notice ? <p role="status" className="muted">{notice}</p> : null}
    {review ? <CleanupReviewPanel preview={review.preview} items={inventory.data?.items ?? []} current={current} acknowledged={acknowledged} busy={busy} uncertain={review.uncertain}
      onAcknowledge={setAcknowledged} onApply={() => void apply()}
      onInclude={() => void preview(review.preview.action, [...new Set([...selected, ...review.preview.requiredKeys])])}
      onRefresh={() => { setReview(null); setAcknowledged(false); void inventory.refetch(); }} /> : null}
  </section>;
}
