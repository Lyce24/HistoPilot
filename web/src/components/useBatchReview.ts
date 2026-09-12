import { useState } from 'react';
import { ApiError } from '../api/client';

/** Keep a reviewed bulk request unchanged until every submitted item has a receipt. */
export function useBatchReview<S, P extends { previewHash: string | null }, R>(
  previewRequest: (selection: S) => Promise<P>, applyRequest: (selection: S, preview: P, operationId: string) => Promise<R>,
  allowed: (preview: P) => boolean, complete: (result: R) => boolean, onResult: (result: R) => Promise<void>,
) {
  const [review, setReview] = useState<{ selection: S; preview: P; operationId: string } | null>(null);
  const [result, setResult] = useState<R | null>(null);
  const [busy, setBusy] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  function reset() { if (busy) return; setReview(null); setResult(null); setSubmitted(false); setAcknowledged(false); setError(null); }
  async function preview(selection: S) {
    if (busy || submitted) return;
    reset(); setBusy(true);
    try { setReview({ selection: structuredClone(selection), preview: await previewRequest(selection), operationId: crypto.randomUUID() }); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Review failed.')); }
    finally { setBusy(false); }
  }
  async function apply() {
    if (busy || !review || !review.preview.previewHash || !allowed(review.preview) || (!acknowledged && !submitted)) return;
    setBusy(true); setSubmitted(true); setError(null);
    let saved: R;
    try { saved = await applyRequest(review.selection, review.preview, review.operationId); }
    catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The response was lost. Retry this same request.'));
      // A rejected initial request made no accepted changes; partial results retain their retry identity.
      if (reason instanceof ApiError && reason.status < 500 && reason.status !== 408 && !result) { setSubmitted(false); setReview(null); setAcknowledged(false); }
      setBusy(false); return;
    }
    setResult(saved);
    if (complete(saved)) { setReview(null); setSubmitted(false); setAcknowledged(false); }
    try { await onResult(saved); }
    catch { setError(new Error('The operation was saved. Refresh to load current records.')); }
    finally { setBusy(false); }
  }
  return { review, result, busy, submitted, acknowledged, setAcknowledged, error, reset, preview, apply, locked: busy || submitted };
}
