import { describe, expect, it } from 'vitest';
import { slideJobControlsShown } from './InterpretationReview';

describe('per-slide attention job controls', () => {
  it('offer Resume, a first launch, and a job created before the Task Center its read-only status', () => {
    for (const status of ['failed', 'cancelled', 'interrupted', 'not_started'] as const) expect(slideJobControlsShown({ status, executor: 'task-center' })).toBe(true);
    expect(slideJobControlsShown({ status: 'interrupted', executor: 'tmux' })).toBe(true);
  });

  it('leave jobs in progress, and finished slides, to the batch chip', () => {
    expect(slideJobControlsShown({ status: 'running', executor: 'task-center' })).toBe(false);
    expect(slideJobControlsShown({ status: 'queued', executor: 'task-center' })).toBe(false);
    expect(slideJobControlsShown({ status: 'completed', executor: 'task-center' })).toBe(false);
    expect(slideJobControlsShown({ status: 'completed', executor: 'tmux' })).toBe(false);
    expect(slideJobControlsShown(undefined)).toBe(false);
  });
});
