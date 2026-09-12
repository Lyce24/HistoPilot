import { useState } from 'react';
import { ApiError } from '../api/client';

/** Retain the exact reviewed request when a publication response is uncertain. */
export function useReviewedPublication<S, P extends { previewHash: string | null }, R>(
  previewRequest: (selection: S) => Promise<P>,
  publishRequest: (selection: S, hash: string, operation: string) => Promise<R>,
  allowed: (preview: P) => boolean,
  onSaved: (record: R) => Promise<void>,
) {
  const [review, setReview] = useState<{ selection: S; preview: P; operationId: string; uncertain: boolean } | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [saved, setSaved] = useState<R | null>(null);
  function reset() { setReview(null); setAcknowledged(false); setError(null); setSaved(null); }
  async function preview(selection: S) {
    if (busy || review?.uncertain) return;
    reset(); setBusy(true);
    try { setReview({ selection: structuredClone(selection), preview: await previewRequest(selection), operationId: crypto.randomUUID(), uncertain: false }); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Review failed.')); }
    finally { setBusy(false); }
  }
  async function publish() {
    if (busy || !review || !review.preview.previewHash || !allowed(review.preview) || (!acknowledged && !review.uncertain)) return;
    setBusy(true); setError(null);
    let result: R;
    try { result = await publishRequest(review.selection, review.preview.previewHash, review.operationId); }
    catch (reason) {
      setError(reason instanceof Error ? reason : new Error('The save response was lost.'));
      if (reason instanceof ApiError) { setReview(null); setAcknowledged(false); }
      else setReview({ ...review, uncertain: true });
      setBusy(false); return;
    }
    setSaved(result); setReview(null); setAcknowledged(false);
    // A refresh failure is not a failed publication and must never repeat it.
    try { await onSaved(result); }
    catch { setError(new Error('Saved successfully. Refresh the list to see the latest records.')); }
    finally { setBusy(false); }
  }
  return { review, acknowledged, setAcknowledged, busy, error, saved, reset, preview, publish, locked: busy || Boolean(review?.uncertain) };
}
