import { useEffect, useId, useRef, useState } from 'react';
import { parseNumericField, type NumericConstraints } from './NumericField';

type ListConstraints = NumericConstraints & { maxItems?: number; allowEmpty?: boolean };

export function validateBatchNumberList(text: string, constraints: ListConstraints): string {
  if (constraints.allowEmpty && !text.trim()) return '';
  const entries = text.split(',');
  if (entries.length > (constraints.maxItems ?? 100)) return `${constraints.label}: enter at most ${constraints.maxItems ?? 100} values.`;
  const values: number[] = [];
  for (const entry of entries) {
    const parsed = parseNumericField(entry, constraints);
    if (parsed.error) return parsed.error;
    values.push(parsed.value!);
  }
  if (new Set(values).size !== values.length) return `${constraints.label}: values must be distinct.`;
  return '';
}

/** List text stays editable, but invalid or partial values cannot be saved. */
export default function BatchNumberList({ value, onChange, hint, ...constraints }: ListConstraints & {
  value: string; onChange: (value: string) => void; hint?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const errorId = useId();
  const hintId = useId();
  const [touched, setTouched] = useState(false);
  const error = validateBatchNumberList(value, constraints);
  useEffect(() => { input.current?.setCustomValidity(error); }, [error]);
  return <label className="label">{constraints.label}
    <input ref={input} className="field" value={value} required={!constraints.allowEmpty} autoComplete="off" spellCheck={false}
      aria-invalid={touched && Boolean(error) || undefined} aria-describedby={[hint ? hintId : '', touched && error ? errorId : ''].filter(Boolean).join(' ') || undefined}
      onChange={(event) => { event.target.setCustomValidity(validateBatchNumberList(event.target.value, constraints)); onChange(event.target.value); }}
      onBlur={() => setTouched(true)} onInvalid={() => setTouched(true)} />
    {hint ? <small id={hintId}>{hint}</small> : null}
    {touched && error ? <small id={errorId} className="development-field-error">{error}</small> : null}
  </label>;
}
