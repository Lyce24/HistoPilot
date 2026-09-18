export default function PublicationConfirmation({ busy, uncertain, acknowledged, onAcknowledge, onConfirm, label }: {
  busy: boolean; uncertain: boolean; acknowledged: boolean; onAcknowledge: (value: boolean) => void;
  onConfirm: () => void; label: string;
}) {
  return uncertain ? <div className="callout callout-warning" role="alert">
    <p>The save response is not confirmed. This record may already exist. Retry uses the identical reviewed request. Resolve this save before changing its inputs.</p>
    <button type="button" className="btn btn-primary" disabled={busy} onClick={onConfirm}>Retry this save</button>
  </div> : <div className="stack">
    <label className="development-check"><input type="checkbox" checked={acknowledged} disabled={busy} onChange={(event) => onAcknowledge(event.target.checked)} />I reviewed the selected inputs and lineage.</label>
    <button type="button" className="btn btn-primary" disabled={busy || !acknowledged} onClick={onConfirm}>{busy ? 'Saving…' : label}</button>
  </div>;
}
