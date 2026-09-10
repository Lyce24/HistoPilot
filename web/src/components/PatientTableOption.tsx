import { useEffect, useId, useRef, useState } from 'react';
import { Icon } from './ui';

export default function PatientTableOption({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const helpId = useId();
  function close() {
    setOpen(false);
    setPinned(false);
  }
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) close();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);
  return (
    <div
      className="patient-table-option"
      ref={container}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => {
        if (!pinned && !container.current?.contains(document.activeElement)) setOpen(false);
      }}
      onFocus={() => setOpen(true)}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) close();
      }}
    >
      <div className="patient-table-option-label">
        <label className="science-check">
          <input
            type="checkbox"
            checked={checked}
            aria-describedby={open ? helpId : undefined}
            onChange={(event) => onChange(event.target.checked)}
          />
          Add patient IDs or attributes from another file
        </label>
        <button
          type="button"
          className="text-button patient-table-help-button"
          aria-label="When should I add a patient table?"
          aria-expanded={open}
          aria-controls={open ? helpId : undefined}
          aria-describedby={open ? helpId : undefined}
          onClick={() => {
            setPinned(!pinned);
            setOpen(!pinned);
          }}
        >
          <Icon name="info" size={16} /> What is this?
        </button>
      </div>
      <p className="muted">
        Optional. Skip this if your main table already has the patient information you need.
      </p>
      {open ? (
        <div id={helpId} role="tooltip" className="patient-table-help">
          <strong>Two ways to use a separate CSV or XLSX</strong>
          <p>
            <b>Patient crosswalk:</b> links each Slide_ID to a verified Patient_ID, so slides
            from the same person stay together when splitting data.
          </p>
          <pre>
            {'Slide_ID       Patient_ID\nexample_A      patient_01\nexample_B      patient_01'}
          </pre>
          <p className="muted">Illustrative values: both slides belong to one patient.</p>
          <p>
            <b>Patient attribute table:</b> adds information such as age or sex by matching
            Patient_ID. Your main table must already contain Patient_ID for this option.
          </p>
          <p>
            You can also keep everything in one table. Use the same Patient_ID for all slides
            from one person.
          </p>
        </div>
      ) : null}
    </div>
  );
}
