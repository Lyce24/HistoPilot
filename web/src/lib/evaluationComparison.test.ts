import { describe, expect, it } from 'vitest';
import { fixtureEvaluation } from '../testFixtures/evaluations';
import { fixturePredictor } from '../testFixtures/predictors';
import { compareEvaluationMethods, pairedEvaluationMean } from './evaluationComparison';

const ensemble = fixturePredictor(1, 11, 'ensemble'), refit = fixturePredictor(1, 11, 'refit');
describe('paired evaluation comparison', () => {
  it('compares only matching sources and excludes unmatched high scores from means', () => {
    const unpaired = fixturePredictor(2, 11, 'ensemble');
    const groups = compareEvaluationMethods([fixtureEvaluation(ensemble, 0.7, 0.6), fixtureEvaluation(refit, 0.8, 0.7), fixtureEvaluation(unpaired, 1, 1)], [ensemble, refit, unpaired]);
    expect(groups).toHaveLength(1);
    expect(groups[0].pairs).toHaveLength(1); expect(groups[0].ensembleOnly).toBe(1);
    expect(pairedEvaluationMean(groups[0].pairs, 'auroc')).toEqual({ count: 1, ensemble: 0.7, refit: 0.8 });
  });
  it('never pairs across experiment, batch, configuration, training seed or split seed', () => {
    for (const change of [{ experimentId: 'other' }, { batchId: 'other' }, { candidateId: 'other' }, { trainingSeed: 99 }, { splitSeed: 99 }]) {
      const other = { ...refit, id: 'other', manifest: { ...refit.manifest, ...change } };
      expect(compareEvaluationMethods([fixtureEvaluation(ensemble, 0.7, 0.6), fixtureEvaluation(other, 0.8, 0.7)], [ensemble, other])[0].pairs).toHaveLength(0);
    }
  });
  it('separates test cohorts, units, thresholds, class orders and labeled counts', () => {
    const base = fixtureEvaluation(ensemble, 0.7, 0.6);
    const changes = [fixtureEvaluation(refit, 0.8, 0.7, 'another-cohort'), fixtureEvaluation(refit, 0.8, 0.7, 'cohort', 'slide')];
    for (const field of ['decisionThreshold', 'classOrder', 'count'] as const) {
      const record = fixtureEvaluation(refit, 0.8, 0.7);
      if (field === 'count') record.execution!.result!.metrics!.selected.count = 99;
      else if (field === 'classOrder') record.execution!.result!.metrics!.classOrder = ['b', 'a'];
      else record.execution!.result!.metrics!.decisionThreshold = 0.7;
      changes.push(record);
    }
    for (const changed of changes) {
      const groups = compareEvaluationMethods([base, changed], [ensemble, refit]);
      expect(groups).toHaveLength(2); expect(groups.every((group) => group.pairs.length === 0)).toBe(true);
    }
  });
  it('uses latest completed evaluations rather than highest score, without mutating records', () => {
    const older = fixtureEvaluation(ensemble, 0.99, 0.99, 'cohort', 'patient', 'old');
    older.createdAt = '2026-09-11';
    const newer = fixtureEvaluation(ensemble, 0.6, 0.5, 'cohort', 'patient', 'new');
    const unfinished = { ...fixtureEvaluation(ensemble, 1, 1), execution: { status: 'running' as const } };
    const records = [older, newer, unfinished, fixtureEvaluation(refit, 0.7, 0.6)];
    const ids = records.map((item) => item.id);
    const group = compareEvaluationMethods(records, [ensemble, refit])[0];
    expect(group.pairs[0].ensemble?.id).toBe('new');
    expect(pairedEvaluationMean(group.pairs, 'auroc').ensemble).toBe(0.6);
    expect(records.map((item) => item.id)).toEqual(ids);
  });
  it('uses identical finite-score pairs for each metric and supports legacy ensemble manifests', () => {
    const legacy = { ...ensemble, manifest: { ...ensemble.manifest, method: undefined } };
    const second = fixturePredictor(2, 11, 'ensemble'), secondRefit = fixturePredictor(2, 11, 'refit');
    const group = compareEvaluationMethods([fixtureEvaluation(legacy, 0.7, 0.6), fixtureEvaluation(refit, 0.8, 0.7), fixtureEvaluation(second, null, 0.8), fixtureEvaluation(secondRefit, 0.99, 0.9)], [legacy, refit, second, secondRefit])[0];
    expect(group.pairs).toHaveLength(2);
    expect(pairedEvaluationMean(group.pairs, 'auroc')).toEqual({ count: 1, ensemble: 0.7, refit: 0.8 });
    expect(pairedEvaluationMean(group.pairs, 'accuracy').count).toBe(2);
  });
});
