import { describe, expect, it } from 'vitest';
import { refitRollupScope } from './RefitJobs';

describe('refit jobs header status', () => {
  it('covers active refit plans only, so a trashed or archived failure does not ask for attention', () => {
    const builds = [{ id: 'b', lifecycleState: 'active' as const }, { id: 'old', lifecycleState: 'trashed' as const }, { id: 'a', lifecycleState: 'active' as const }, { id: 'kept', lifecycleState: 'archived' as const }];
    expect(refitRollupScope('p', builds)).toEqual({ recordKind: 'refit', recordIds: 'a,b', project: 'p' });
    expect(refitRollupScope('p', [{ id: 'old', lifecycleState: 'trashed' }])).toBeNull();
  });

  it('falls back to every refit of the project when the list would not fit in a URL', () => {
    const many = Array.from({ length: 400 }, (_, index) => ({ id: `refit-${String(index).padStart(32, '0')}`, lifecycleState: 'active' as const }));
    expect(refitRollupScope('p', many)).toEqual({ recordKind: 'refit', project: 'p' });
  });
});
