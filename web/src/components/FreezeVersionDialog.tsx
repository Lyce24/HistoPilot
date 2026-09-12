import { useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { ApiError } from '../api/client';
import type { VersionLabelInput } from '../api/scientific';
import { ErrorNotice, Icon } from './ui';
import './FreezeVersionDialog.css';

const kinds = {
  dataset: { name: 'dataset', placeholder: 'e.g. reviewed-slides-v1', saved: 'slide records and column mapping' },
  protocol: { name: 'development protocol', placeholder: 'e.g. grade-baseline-v1', saved: 'cohort, prediction target and split assignments' },
  feature: { name: 'feature', placeholder: 'e.g. uni-20x-v1', saved: 'feature configuration and file references' },
  bundle: { name: 'feature bundle', placeholder: 'e.g. uni-20x-training-v1', saved: 'feature source, included packs and their verification evidence' },
};

/** Mount for a reviewed freeze request; failed saves retain the user's naming draft. */
export default function FreezeVersionDialog({
  kind, children, onFreeze, onClose, initialLabel, onLabelChange,
}: {
  kind: keyof typeof kinds;
  children?: ReactNode;
  onFreeze: (label: VersionLabelInput) => Promise<void>;
  onClose: () => void;
  initialLabel?: VersionLabelInput;
  onLabelChange?: (label: VersionLabelInput) => void;
}) {
  const copy = kinds[kind];
  const tagId = useId();
  const noteId = useId();
  const tagHelpId = useId();
  const [tag, setTag] = useState(initialLabel?.tag ?? '');
  const [note, setNote] = useState(initialLabel?.note ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const saving = useRef(false);

  async function freeze() {
    if (saving.current || !tag.trim()) return;
    saving.current = true;
    setBusy(true);
    setError(null);
    try {
      await onFreeze({ tag: tag.trim(), note });
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('This version could not be frozen. Please try again.'));
    } finally {
      saving.current = false;
      setBusy(false);
    }
  }

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open && !saving.current) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="freeze-version-dialog" onEscapeKeyDown={(event) => { if (saving.current) event.preventDefault(); }} onInteractOutside={(event) => { if (saving.current) event.preventDefault(); }}>
          <div className="freeze-version-heading">
            <span className="freeze-version-icon"><Icon name="lock" size={23} /></span>
            <div><p className="freeze-version-eyebrow">FINAL SAVE STEP</p><Dialog.Title>Name your {copy.name} version</Dialog.Title></div>
          </div>
          <Dialog.Description className="freeze-version-description">
            Choose a clear personal tag so you can recognize and select this version later. The tag and optional note are saved together with the version.
          </Dialog.Description>
          {children ? <div className="freeze-version-review">{children}</div> : null}
          <form onSubmit={(event) => { event.preventDefault(); event.stopPropagation(); void freeze(); }} aria-busy={busy}>
            <fieldset disabled={busy} className="freeze-version-fields">
              <label className="label" htmlFor={tagId}>
                <span>Version tag <span className="freeze-version-required">Required</span></span>
                <input id={tagId} className="field" value={tag} required maxLength={80} autoComplete="off" placeholder={copy.placeholder} aria-describedby={tagHelpId} onChange={(event) => { setTag(event.target.value); onLabelChange?.({ tag: event.target.value, note }); }} />
                <small id={tagHelpId}>Use your own name for this version. Up to 80 characters; unique among saved {copy.name} versions in this project.</small>
              </label>
              <label className="label" htmlFor={noteId}>
                Commit note (optional)
                <textarea id={noteId} className="field freeze-version-note" value={note} rows={3} maxLength={2000} placeholder="What changed, or why are you keeping this version?" onChange={(event) => { setNote(event.target.value); onLabelChange?.({ tag, note: event.target.value }); }} />
                <small>{note.length.toLocaleString()} / 2,000 characters</small>
              </label>
              <div className="freeze-version-explanation"><Icon name="info" size={18} /><p>Freezing keeps the {copy.saved} fixed. You can edit the tag and note later. To change the saved data or settings, create another version.</p></div>
              <ErrorNotice error={error} />
              {error instanceof ApiError && error.status === 409 ? <p className="freeze-version-conflict">Your tag and note are still here. If this content is already saved, open its existing version; creating a different version requires changes to the data or settings.</p> : null}
              <div className="freeze-version-actions">
                <button type="button" className="btn btn-secondary" onClick={onClose}>Back to review</button>
                <button type="submit" className="btn btn-primary" disabled={!tag.trim()}><Icon name="lock" size={17} />{busy ? 'Freezing version…' : `Freeze ${copy.name} version`}</button>
              </div>
            </fieldset>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
