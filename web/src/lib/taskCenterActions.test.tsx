import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ApiError } from '../api/client';
import { TaskCenterActionNotice, runTaskCenterOperation } from './taskCenterActions';

describe('Task Center operation identity', () => {
  it('uses a fresh operation after an acknowledged request', async () => {
    const receipts = new Map<string, string>();
    const call = vi.fn().mockResolvedValue({ ok: true });
    await runTaskCenterOperation(receipts, 'owner:hold', call);
    await runTaskCenterOperation(receipts, 'owner:hold', call);
    expect(call.mock.calls[0][0]).not.toEqual(call.mock.calls[1][0]);
    expect(receipts.size).toBe(0);
  });

  it('retries a lost or server-side failure with the same operation ID', async () => {
    for (const failure of [new TypeError('Connection lost'), new ApiError('Gateway timeout', 503), new ApiError('Timeout', 408)]) {
      const receipts = new Map<string, string>();
      const call = vi.fn().mockRejectedValueOnce(failure).mockResolvedValueOnce({ ok: true });
      await expect(runTaskCenterOperation(receipts, 'task:cancel', call)).rejects.toBe(failure);
      expect(receipts.has('task:cancel')).toBe(true);
      await runTaskCenterOperation(receipts, 'task:cancel', call);
      expect(call.mock.calls[0][0]).toEqual(call.mock.calls[1][0]);
      expect(receipts.size).toBe(0);
    }
  });

  it('clears the receipt after a definite rejection and keeps different requests apart', async () => {
    const receipts = new Map<string, string>();
    const rejected = vi.fn().mockRejectedValue(new ApiError('Owner belongs to another workspace', 409, 'TASK_OWNER_OTHER_WORKSPACE'));
    await expect(runTaskCenterOperation(receipts, 'owner:cancel', rejected)).rejects.toThrow('another workspace');
    expect(receipts.size).toBe(0);
    const lost = vi.fn().mockRejectedValue(new TypeError('Connection lost'));
    await expect(runTaskCenterOperation(receipts, 'owner:move:up', lost)).rejects.toThrow();
    await expect(runTaskCenterOperation(receipts, 'owner:move:down', lost)).rejects.toThrow();
    expect(lost.mock.calls[0][0]).not.toEqual(lost.mock.calls[1][0]);
  });

  it('reuses a lost request identity only until a different request is sent', async () => {
    const receipts = new Map<string, string>();
    const hold = vi.fn().mockRejectedValueOnce(new TypeError('Connection lost')).mockResolvedValue({ held: true });
    await expect(runTaskCenterOperation(receipts, 'owner:hold', hold)).rejects.toThrow();
    await runTaskCenterOperation(receipts, 'owner:release', vi.fn().mockResolvedValue({ held: false }));
    await runTaskCenterOperation(receipts, 'owner:hold', hold);
    expect(hold.mock.calls[1][0]).not.toEqual(hold.mock.calls[0][0]);
    expect(receipts.size).toBe(0);
  });

  it('explains that repeating an uncertain request is safe', () => {
    const html = renderToStaticMarkup(<TaskCenterActionNotice actions={{ error: new Error('Could not reach HistoPilot.'), uncertain: 'owner:hold', pending: null }} />);
    expect(html).toContain('Could not reach HistoPilot.');
    expect(html).toContain('Repeating it is safe');
    expect(renderToStaticMarkup(<TaskCenterActionNotice actions={{ error: null, uncertain: null, pending: null }} />)).toBe('');
  });
});
