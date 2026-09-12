export default function PublicationConfirmation({ busy, uncertain, acknowledged, onAcknowledge, onConfirm, onReset, label }: {
  busy: boolean; uncertain: boolean; acknowledged: boolean; onAcknowledge: (value: boolean) => void;
  onConfirm: () => void; onReset: () => void; label: string;
}) {
  return uncertain ? <div className="callout callout-warning" role="alert">
    <p>The save response was lost. This record may already exist. Retry uses the identical reviewed request.</p>
    <div className="inline-actions"><button type="button" className="btn btn-primary" disabled={busy} onClick={onConfirm}>Retry this save</button><button type="button" className="btn btn-secondary" disabled={busy} onClick={onReset}>Return to review</button></div>
  </div> : <div className="stack">
    <label className="development-check"><input type="checkbox" checked={acknowledged} disabled={busy} onChange={(event) => onAcknowledge(event.target.checked)} />I reviewed the selected inputs and lineage.</label>
    <button type="button" className="btn btn-primary" disabled={busy || !acknowledged} onClick={onConfirm}>{busy ? 'Saving…' : label}</button>
  </div>;
}
