import { describe, expect, it } from 'vitest';
import { formatDate, formatDateTime } from './dates';

describe('saved times', () => {
  it('reads as a date or a date and time, and never as an ISO string or "Invalid Date"', () => {
    const date = formatDate('2026-10-02T13:15:00Z');
    expect(date).toMatch(/2026/); expect(date).not.toContain('T13');
    expect(formatDateTime('2026-10-02T13:15:00Z')).toMatch(/2026.*\d{1,2}:\d{2}/);
    for (const value of [undefined, null, '', 'not a time']) {
      expect(formatDate(value)).toBe('—'); expect(formatDateTime(value)).toBe('—');
    }
  });
});
