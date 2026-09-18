import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { slideReviews, reviewStatusLabels, type ReviewRegion, type ReviewStatus, type SlideReview, type SlideReviewValues } from '../api/slideReviews';
import { readSessionDraft, useSessionDraftBackup, writeSessionDraft } from '../lib/sessionDraft';
import { useWorkspaceNavigationGuard } from '../lib/workspaceNavigation';
import { ErrorNotice } from './ui';
import './CaseReview.css';

interface Props { project: string; datasetId: string; slideId: string; evaluationId?: string; selectedRegion?: Omit<ReviewRegion, 'id' | 'label'>; onSaved?: (review: SlideReview) => void }
interface Draft { expectedRevision: number; values: SlideReviewValues }
export function isReviewDraft(value: unknown): value is Draft {
  if (!value || typeof value !== 'object') return false;
  const draft = value as Partial<Draft>, fields = draft.values;
  return Number.isSafeInteger(draft.expectedRevision) && Number(draft.expectedRevision) >= 0 && Boolean(fields)
    && Object.hasOwn(reviewStatusLabels, fields?.status ?? '') && typeof fields?.notes === 'string' && fields.notes.length <= 8000
    && typeof fields?.reviewer === 'string' && fields.reviewer.length <= 120 && Array.isArray(fields?.reasons)
    && fields.reasons.length <= 16 && fields.reasons.every((item) => typeof item === 'string' && item.length <= 120)
    && Array.isArray(fields?.regions) && fields.regions.length <= 128 && fields.regions.every((item) => item && typeof item.id === 'string' && typeof item.label === 'string'
      && ['x', 'y', 'width', 'height'].every((key) => typeof item[key as keyof ReviewRegion] === 'number' && Number.isFinite(item[key as keyof ReviewRegion]))
      && item.x >= 0 && item.y >= 0 && item.width > 0 && item.height > 0)
    && (fields.evaluationId === null || typeof fields.evaluationId === 'string');
}
export function reviewValues(record: SlideReview, evaluationId?: string): SlideReviewValues {
  return { status: record.status, notes: record.notes, reviewer: record.reviewer, reasons: record.reasons, regions: record.regions, evaluationId: evaluationId ?? record.evaluationId };
}
const reasons = ['Artifact', 'Limited tumor', 'Label disagreement', 'Unusual morphology', 'Model disagreement', 'Poor tissue coverage'];

export default function SlideReviewEditor(props: Props) {
  return <ReviewLoader key={`${props.project}:${props.datasetId}:${props.slideId}`} {...props} />;
}
function ReviewLoader(props: Props) {
  const saved = useQuery({ queryKey: ['slide-review', props.project, props.datasetId, props.slideId], queryFn: ({ signal }) => slideReviews.get(props.project, props.datasetId, props.slideId, signal) });
  return <section className="slide-review-editor" aria-label={`Review ${props.slideId}`}>
    <h3>Slide review</h3>
    {saved.isPending ? <p role="status">Loading saved review…</p> : saved.isError ? <><ErrorNotice error={saved.error} /><button type="button" className="btn btn-secondary" onClick={() => void saved.refetch()}>Retry review</button></> : <ReviewForm {...props} initial={saved.data} />}
  </section>;
}
function ReviewForm({ project, datasetId, slideId, evaluationId, selectedRegion, onSaved, initial }: Props & { initial: SlideReview }) {
  const key = `histopilot:slide-review:v1:${project}:${datasetId}:${encodeURIComponent(slideId)}`;
  const client = useQueryClient();
  const [baseline, setBaseline] = useState(initial);
  const [draft, setDraft] = useState<Draft>(() => readSessionDraft(key, isReviewDraft) ?? { expectedRevision: initial.revision, values: reviewValues(initial, evaluationId) });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [latest, setLatest] = useState<SlideReview | null>(null);
  const dirty = JSON.stringify(draft.values) !== JSON.stringify(reviewValues(baseline, evaluationId));
  const backup = useSessionDraftBackup(key, dirty ? draft : null, isReviewDraft);
  useWorkspaceNavigationGuard(busy ? 'A slide review is being saved.' : backup.error && dirty ? backup.error : null);
  function update(values: Partial<SlideReviewValues>) { setDraft((current) => ({ ...current, values: { ...current.values, ...values } })); setMessage(''); }
  async function save() {
    if (busy) return;
    setBusy(true); setError(null); setMessage('');
    try {
      const record = await slideReviews.save(project, datasetId, slideId, { ...draft.values, expectedRevision: draft.expectedRevision });
      setBaseline(record); setDraft({ expectedRevision: record.revision, values: reviewValues(record, evaluationId) }); setLatest(null);
      writeSessionDraft(key, null);
      client.setQueryData(['slide-review', project, datasetId, slideId], record);
      void client.invalidateQueries({ queryKey: ['slide-reviews', project, datasetId] });
      void client.invalidateQueries({ queryKey: ['morphology-slides', project, datasetId] });
      void client.invalidateQueries({ queryKey: ['case-review', project] });
      setMessage(`Review saved · revision ${record.revision}`); onSaved?.(record);
    } catch (reason) { setError(reason instanceof Error ? reason : new Error('Review could not be saved.')); }
    finally { setBusy(false); }
  }
  async function loadLatest() {
    setBusy(true); setError(null);
    try { setLatest(await slideReviews.get(project, datasetId, slideId)); }
    catch (reason) { setError(reason instanceof Error ? reason : new Error('Saved review could not be loaded.')); }
    finally { setBusy(false); }
  }
  return <>
    <p className="muted">Notes and QC decisions are separate from frozen labels and cohort membership. Exclusion is a recommendation for a new dataset version.</p>
    <ErrorNotice error={error} />
    {backup.error ? <p className="callout" role="alert">{backup.error}</p> : null}
    {draft.expectedRevision !== initial.revision ? <p className="callout">Saved review has changed since these edits began. Inspect the latest review before saving.</p> : null}
    <fieldset disabled={busy} className="case-review-fields">
      <label className="label">Decision<select className="field" value={draft.values.status} onChange={(event) => update({ status: event.target.value as ReviewStatus })}>{Object.entries(reviewStatusLabels).map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label>
      <label className="label">Reviewer<input className="field" maxLength={120} value={draft.values.reviewer} onChange={(event) => update({ reviewer: event.target.value })} placeholder="Name or initials" /></label>
      <fieldset className="review-reasons"><legend>Findings</legend>{reasons.map((reason) => <label key={reason}><input type="checkbox" checked={draft.values.reasons.includes(reason)} onChange={(event) => update({ reasons: event.target.checked ? [...draft.values.reasons, reason] : draft.values.reasons.filter((item) => item !== reason) })} /> {reason}</label>)}</fieldset>
      <label className="label review-notes">Notes<textarea className="field" rows={4} maxLength={8000} value={draft.values.notes} onChange={(event) => update({ notes: event.target.value })} /></label>
      <div className="review-notes">
        {selectedRegion ? <button type="button" className="btn btn-secondary" disabled={draft.values.regions.length >= 128} onClick={() => update({ regions: [...draft.values.regions, { ...selectedRegion, id: crypto.randomUUID(), label: `Region ${draft.values.regions.length + 1}` }] })}>Save selected region with review</button> : null}
        {draft.values.regions.length ? <details><summary>Review regions ({draft.values.regions.length})</summary><p className="muted">Coordinates refer to the original full-resolution slide.</p>{draft.values.regions.map((region) => <div key={region.id} className="review-region"><label className="label">Region label<input className="field" maxLength={300} value={region.label} onChange={(event) => update({ regions: draft.values.regions.map((item) => item.id === region.id ? { ...item, label: event.target.value } : item) })} /></label><span>{region.x}, {region.y} · {region.width} × {region.height}</span><button type="button" className="text-button" onClick={() => update({ regions: draft.values.regions.filter((item) => item.id !== region.id) })}>Remove</button></div>)}</details> : null}
      </div>
    </fieldset>
    <div className="inline-actions"><button type="button" className="btn btn-primary" disabled={busy || !dirty} onClick={() => void save()}>{busy ? 'Working…' : 'Save review'}</button><button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void loadLatest()}>Inspect latest saved review</button>{message ? <span role="status">{message}</span> : dirty ? <span className="muted">Unsaved edits retained in this tab</span> : null}</div>
    {latest ? <aside className="callout"><strong>Saved revision {latest.revision}</strong><p>{reviewStatusLabels[latest.status]} · {latest.reviewer || 'Reviewer not specified'}</p><p className="review-note-text">{latest.notes || 'No saved notes.'}</p><div className="inline-actions"><button type="button" className="btn btn-secondary" disabled={busy} onClick={() => { setBaseline(latest); setDraft({ expectedRevision: latest.revision, values: reviewValues(latest, evaluationId) }); setLatest(null); setError(null); }}>Use saved review</button><button type="button" className="btn btn-secondary" disabled={busy} onClick={() => { setBaseline(latest); setDraft((current) => ({ ...current, expectedRevision: latest.revision })); setLatest(null); setError(null); }}>Keep my edits against this revision</button></div></aside> : null}
    {baseline.history.length ? <details><summary>Review history ({baseline.history.length})</summary><ol>{[...baseline.history].reverse().map((entry) => <li key={entry.revision}><strong>Revision {entry.revision} · {reviewStatusLabels[entry.status]}</strong> · {entry.reviewer || 'Reviewer not specified'} · {new Date(entry.updatedAt).toLocaleString()}<p className="review-note-text">{entry.notes}</p></li>)}</ol></details> : null}
  </>;
}
