import { describe, expect, it } from 'vitest';
import { statusLabel } from './statusLabels';

describe('status words', () => {
  it('names known statuses and capitalizes others without underscores', () => {
    expect(statusLabel('running')).toBe('Running');
    expect(statusLabel('succeeded')).toBe('Completed');
    expect(statusLabel('not_started')).toBe('Ready to run');
    expect(statusLabel('awaiting_requeue')).toBe('Awaiting requeue');
    expect(statusLabel(undefined)).toBe('');
  });
});
