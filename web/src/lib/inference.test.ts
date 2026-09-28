import { describe, expect, it } from 'vitest';
import type { InferenceUnitSummary } from '../api/inference';
import type { ModelEvaluation } from '../api/predictors';
import { cohortKind, cohortKindLabel, comparableRuns, countBelow, isInferenceBatch, isInferenceCohort, isInferenceRun, percent, predictedMix, unanimousShare } from './inference';

const run = (purpose?: 'inference' | 'review') => ({ manifest: { purpose } }) as unknown as ModelEvaluation;
const distribution = (counts: Record<string, number[]>): InferenceUnitSummary['confidence'] => ({
  mean: null, quantiles: { p10: null, p25: null, median: null, p75: null, p90: null },
  edges: Array.from({ length: 21 }, (_, index) => index / 20), counts,
});

describe('inference helpers', () => {
  it('treats inference and earlier review records as prediction-only', () => {
    expect(isInferenceRun(run('inference'))).toBe(true);
    expect(isInferenceRun(run('review'))).toBe(true);
    expect(isInferenceRun(run())).toBe(false);
    expect(isInferenceCohort({ manifest: { spec: { purpose: 'review' } } } as never)).toBe(true);
    expect(isInferenceCohort({ manifest: { spec: { purpose: 'independent' } } } as never)).toBe(false);
    expect(cohortKind({ purpose: 'inference', target: null })).toBe('inference');
    expect(cohortKind({ target: null })).toBe('unlabeled-evaluation');
    expect(cohortKind({ target: { field: 'grade' } as never })).toBe('evaluation');
    expect(cohortKindLabel.inference).toBe('Inference · unlabeled');
  });

  it('classifies batches by their runs and pairs only runs with the same scoring context', () => {
    const runs = new Set(['inference-run']), cohorts = new Set(['active-inference-cohort']);
    // An archived inference cohort is absent from the cohort list; its runs still decide.
    expect(isInferenceBatch({ cohortId: 'archived', items: [{ evaluationId: 'inference-run' }] }, runs, cohorts)).toBe(true);
    expect(isInferenceBatch({ cohortId: 'active-inference-cohort', items: [{ evaluationId: 'labeled-run' }] }, runs, cohorts)).toBe(false);
    expect(isInferenceBatch({ cohortId: 'active-inference-cohort', items: [{ evaluationId: null }] }, runs, cohorts)).toBe(true);
    const target = { classes: ['ND', 'HG'], field: 'grade' };
    const record = (id: string, extra: Record<string, unknown> = {}) => ({ id, lifecycleState: 'active', execution: { status: 'completed' }, manifest: { cohortId: 'cohort', target, inference: { patientAggregation: 'mean' }, ...extra } }) as unknown as ModelEvaluation;
    const self = record('self');
    const choices = comparableRuns(self, [self, record('same'), record('field', { target: { ...target, field: 'alias' } }), record('logits', { inference: { patientAggregation: 'mean_logits' } }), record('cohort', { cohortId: 'other' })]);
    expect(choices.map((item) => item.id)).toEqual(['same']);
  });

  it('summarizes predicted classes and low-confidence counts from frozen bins', () => {
    const summary = { predicted: [
      { label: 'ND', count: 212, fraction: 0.55, meanConfidence: 0.8 }, { label: 'IND', count: 0, fraction: 0, meanConfidence: null },
      { label: 'LG', count: 60, fraction: 0.16, meanConfidence: 0.6 }, { label: 'HG', count: 111, fraction: 0.29, meanConfidence: 0.7 },
    ] };
    expect(predictedMix(summary)).toBe('ND 212 · HG 111 · LG 60');
    expect(predictedMix(summary, 2)).toBe('ND 212 · HG 111 · +1');
    expect(predictedMix(null)).toBe('—');
    const counts = Array(20).fill(0);
    counts[9] = 3; counts[11] = 2; counts[12] = 5;
    // Bins 0-11 lie below 0.6; bin 12 starts at 0.6.
    expect(countBelow(distribution({ ND: counts, LG: [...counts] }), 0.6)).toBe(10);
    expect(countBelow(distribution({ ND: counts }), 1)).toBe(10);
    expect(unanimousShare({ ensemble: { memberCount: 5, records: 4, unanimous: 3, disagreements: 1, meanSpread: 0, agreement: [] } } as unknown as InferenceUnitSummary)).toBe(0.75);
    expect(unanimousShare(null)).toBeNull();
    expect(percent(0.1234)).toBe('12.3%');
    expect(percent(null)).toBe('—');
  });
});
