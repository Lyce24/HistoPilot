import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import NumericField, { parseNumericField, reportEditorValidity } from './NumericField';

describe('direct numeric entry', () => {
  it('allows replacement values and treats empty or incomplete text as unfinished input', () => {
    const epochs = { label: 'Maximum epochs', min: 1, max: 100000 };
    for (const text of ['', ' ', '-', '1e', '0', '-1', '1.5', 'Infinity', '0x20', '100001']) {
      expect(parseNumericField(text, epochs).value).toBeNull();
      expect(parseNumericField(text, epochs).error).not.toBe('');
    }
    expect(parseNumericField('020', epochs)).toEqual({ value: 20, error: '' });
    expect(parseNumericField('7', { label: 'Patience', min: 1 })).toEqual({ value: 7, error: '' });
    expect(parseNumericField('0', { label: 'Workers', min: 0 })).toEqual({ value: 0, error: '' });
  });

  it('preserves scientific notation and enforces strict dropout and learning-rate bounds', () => {
    const lr = { label: 'Learning rate', integer: false, min: 0, minExclusive: true };
    expect(parseNumericField('1e-4', lr)).toEqual({ value: 0.0001, error: '' });
    expect(parseNumericField('0', lr).value).toBeNull();
    expect(parseNumericField('1e-', lr).value).toBeNull();
    const dropout = { label: 'Dropout', integer: false, min: 0, max: 1, maxExclusive: true };
    expect(parseNumericField('0', dropout).value).toBe(0);
    expect(parseNumericField('0.999', dropout).value).toBe(0.999);
    expect(parseNumericField('1', dropout).value).toBeNull();
  });

  it('renders editable text fields with numeric keyboards and no spinner controls', () => {
    const html = renderToStaticMarkup(<NumericField label="Maximum epochs" value={100} min={1} onChange={() => {}} />);
    expect(html).toContain('type="text"');
    expect(html).toContain('inputMode="numeric"');
    expect(html).toContain('required=""');
    expect(html).not.toContain('type="number"');
  });

  it('blocks submission at an invalid editor field and opens its details for correction', () => {
    const details = { open: false, parentElement: null };
    const field = { checkValidity: () => false, reportValidity: vi.fn(), closest: () => details };
    const editor = { querySelectorAll: () => [field] } as unknown as HTMLFieldSetElement;
    expect(reportEditorValidity(editor)).toBe(false);
    expect(details.open).toBe(true);
    expect(field.reportValidity).toHaveBeenCalledOnce();
  });
});
