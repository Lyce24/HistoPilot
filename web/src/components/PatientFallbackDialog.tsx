import * as Dialog from '@radix-ui/react-dialog';
import { ErrorNotice } from './ui';

export default function PatientFallbackDialog({
  count,
  busy,
  error,
  onContinue,
  onBack,
}: {
  count: number | null;
  busy: boolean;
  error: Error | null;
  onContinue: () => void;
  onBack: () => void;
}) {
  return (
    <Dialog.Root
      open={count !== null}
      onOpenChange={(open) => {
        if (!open && !busy) onBack();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="patient-fallback-dialog">
          <Dialog.Title>Patient ID unresolved</Dialog.Title>
          <Dialog.Description>
            Patient ID unresolved, fallback to Slide ID. Continue or go back to change.
          </Dialog.Description>
          <p>
            <strong>{count?.toLocaleString()} slides</strong> have no Patient_ID. Continuing
            uses each slide's Slide_ID as its fallback Patient_ID and saves that choice with the
            dataset.
          </p>
          <p className="muted">
            Each fallback slide becomes a separate split group. These are not verified patients;
            slides from the same person could land in different sets. Existing patient IDs stay
            intact.
          </p>
          <ErrorNotice error={error} />
          <div className="inline-actions">
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={onBack}
            >
              Go back to change
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy}
              onClick={onContinue}
            >
              {busy ? 'Updating preview…' : 'Continue with Slide ID'}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
