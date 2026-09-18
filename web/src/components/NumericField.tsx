import { useEffect, useId, useRef, useState } from 'react';
import { useNumericDraft, type NumericDraft } from './NumericFieldDrafts';

export interface NumericConstraints {
  label: string;
  integer?: boolean;
  min?: number;
  max?: number;
  minExclusive?: boolean;
  maxExclusive?: boolean;
}

/** Empty and partially typed text must never become a scientific numeric value. */
export function parseNumericField(text: string, constraints: NumericConstraints): { value: number; error: '' } | { value: null; error: string } {
  const { label, integer = true, min, max, minExclusive = false, maxExclusive = false } = constraints;
  const trimmed = text.trim();
  if (!trimmed) return { value: null, error: `Enter ${label.toLowerCase()}.` };
  const pattern = integer ? /^[+-]?\d+$/ : /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;
  const value = Number(trimmed);
  if (!pattern.test(trimmed) || !Number.isFinite(value) || (integer && !Number.isSafeInteger(value))) {
    return { value: null, error: `${label} must be a ${integer ? 'whole ' : 'finite '}number.` };
  }
  if (min !== undefined && (minExclusive ? value <= min : value < min)) {
    return { value: null, error: `${label} must be ${minExclusive ? 'greater than' : 'at least'} ${min}.` };
  }
  if (max !== undefined && (maxExclusive ? value >= max : value > max)) {
    return { value: null, error: `${label} must be ${maxExclusive ? 'less than' : 'at most'} ${max}.` };
  }
  return { value, error: '' };
}

/** Keep editing text locally; the batch editor checks validity before every save. */
export default function NumericField({ value, onChange, disabled, ...constraints }: NumericConstraints & {
  value: number; onChange: (value: number) => void; disabled?: boolean;
}) {
  const [localDraft, setLocalDraft] = useState({ source: value, text: String(value) });
  const recovery = useNumericDraft(constraints.label);
  const draft = recovery.draft ?? localDraft;
  function setDraft(next: NumericDraft) { setLocalDraft(next); recovery.setDraft(next); }
  const [touched, setTouched] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const errorId = useId();
  const text = draft.source === value ? draft.text : String(value);
  const result = parseNumericField(text, constraints);
  useEffect(() => { input.current?.setCustomValidity(result.error); }, [result.error]);
  return <label className="label">{constraints.label}
    <input ref={input} className="field" type="text" inputMode={constraints.integer === false ? 'decimal' : 'numeric'}
      value={text} required disabled={disabled} autoComplete="off" spellCheck={false}
      aria-invalid={!disabled && touched && Boolean(result.error) || undefined}
      aria-describedby={!disabled && touched && result.error ? errorId : undefined}
      onChange={(event) => {
        const next = parseNumericField(event.target.value, constraints);
        event.target.setCustomValidity(next.error);
        setDraft({ source: next.value ?? value, text: event.target.value });
        if (next.value !== null) onChange(next.value);
      }}
      onBlur={() => {
        setTouched(true);
        if (result.value !== null) setDraft({ source: result.value, text: String(result.value) });
      }}
      onInvalid={() => setTouched(true)} />
    {!disabled && touched && result.error ? <small id={errorId} className="development-field-error">{result.error}</small> : null}
  </label>;
}

export function reportEditorValidity(editor: HTMLFieldSetElement | null): boolean {
  if (!editor) return true;
  for (const field of editor.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input, select, textarea')) {
    if (!field.checkValidity()) {
      let details = field.closest('details');
      while (details) { details.open = true; details = details.parentElement?.closest('details') ?? null; }
      field.reportValidity();
      return false;
    }
  }
  return true;
}
