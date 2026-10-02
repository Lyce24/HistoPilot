import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Objective } from '../api/objectives';
import { effectiveScope, inObjectiveScope, readObjectiveScope, setObjectiveScope, UNFILED_SCOPE } from './objectiveScope';

const aims: Objective[] = ['aim-1', 'aim-2'].map((id) => ({ id, name: id, rank: 'primary', description: '', revision: 1, createdAt: '', updatedAt: '' }));
afterEach(() => vi.unstubAllGlobals());

describe('the project-wide objective focus', () => {
  it('lives beside the project in the address, not in a module\'s hash', () => {
    expect(readObjectiveScope('http://host/?project=p&objective=aim-1#apply?tab=runs')).toBe('aim-1');
    expect(readObjectiveScope('http://host/?project=p#apply?objective=aim-2')).toBe('');
    expect(readObjectiveScope('')).toBe('');
  });

  it('reads an objective the project does not have as every objective', () => {
    expect(effectiveScope('aim-1', aims)).toBe('aim-1');
    expect(effectiveScope('objective-of-another-project', aims)).toBe('');
    expect(effectiveScope('aim-1', undefined)).toBe('');
    expect(effectiveScope(UNFILED_SCOPE, [])).toBe(UNFILED_SCOPE);
    expect(effectiveScope('', aims)).toBe('');
  });

  it('keeps work by the objective of its experiment', () => {
    expect(inObjectiveScope('', 'aim-1')).toBe(true);
    expect(inObjectiveScope('', null)).toBe(true);
    expect(inObjectiveScope('aim-1', 'aim-1')).toBe(true);
    expect(inObjectiveScope('aim-1', 'aim-2')).toBe(false);
    expect(inObjectiveScope('aim-1', undefined)).toBe(false);
    expect(inObjectiveScope(UNFILED_SCOPE, null)).toBe(true);
    expect(inObjectiveScope(UNFILED_SCOPE, 'aim-1')).toBe(false);
  });

  it('changes the focus in place, keeping the page\'s history entry, and tells its readers', () => {
    const replaceState = vi.fn((_state: unknown, _title: string, url: URL) => { location.href = String(url); });
    const dispatchEvent = vi.fn();
    const location = { href: 'http://host/?project=p#apply?tab=runs' };
    vi.stubGlobal('window', { location, history: { state: { kept: true }, replaceState }, dispatchEvent });
    setObjectiveScope('aim-2');
    expect(location.href).toBe('http://host/?project=p&objective=aim-2#apply?tab=runs');
    expect(replaceState.mock.calls[0][0]).toEqual({ kept: true });
    expect(dispatchEvent).toHaveBeenCalledOnce();
    setObjectiveScope('');
    expect(location.href).toBe('http://host/?project=p#apply?tab=runs');
  });
});
