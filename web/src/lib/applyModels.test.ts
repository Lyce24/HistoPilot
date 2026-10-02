import { describe, expect, it } from 'vitest';
import type { EvaluationCohort } from '../api/evaluation';
import { cohortObjectives } from './applyModels';

const cohort = (id: string, sourceTargetSplitId?: string) => ({ id, manifest: { spec: { sourceTargetSplitId } } }) as unknown as EvaluationCohort;

describe('the objectives a cohort serves', () => {
  it('comes from its runs, the batches that name it and the experiments that reserved it', () => {
    const found = cohortObjectives({
      cohorts: [cohort('held-out', 'split-1'), cohort('external'), cohort('unused', 'split-9')],
      runs: [{ manifest: { cohortId: 'external', experimentId: 'loco' } }, { manifest: { cohortId: 'external', experimentId: 'unfiled' } }],
      batches: [{ cohortId: 'external', items: [{ experimentId: 'main' }, {}] }],
      experiments: [{ id: 'main', setupDesign: { targetSplitId: 'split-1' } }, { id: 'loco', setupDesign: null }] as never,
      filings: { main: { objectiveId: 'aim-1' }, loco: { objectiveId: 'aim-2' }, unfiled: { objectiveId: null } },
    });
    expect(Object.fromEntries([...found].map(([id, objectives]) => [id, [...objectives].sort()]))).toEqual({ 'held-out': ['aim-1'], external: ['aim-1', 'aim-2'] });
    expect(found.has('unused')).toBe(false);
  });
});
