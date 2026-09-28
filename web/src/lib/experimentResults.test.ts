import { describe, expect, it } from 'vitest';
import { csv, fixed, interval, leadingBatch, meanSd, metricLabel, metricPhrase, niceTicks, orient, orientedPairs, percent, reportedConfiguration, resultRows, shade, signed, signedInterval, tickDigits, verdict, verdictLabel } from './experimentResults';
import { batch, configuration, experimentResults } from '../testFixtures/experimentResults';

describe('experiment result formatting', () => {
  it('labels metrics for multiclass and binary targets, in tables and in sentences', () => {
    expect(metricLabel('auroc', 'multiclass_classification')).toBe('Macro AUROC');
    expect(metricLabel('auroc', 'binary_classification')).toBe('AUROC');
    expect(metricPhrase('auroc', 'multiclass_classification')).toBe('macro AUROC');
    expect(metricPhrase('auroc', 'binary_classification')).toBe('AUROC');
    expect(metricPhrase('balancedAccuracy')).toBe('balanced accuracy');
  });

  it('formats values, spreads, intervals and signed differences', () => {
    expect(fixed(0.12345)).toBe('0.123');
    expect(fixed(null)).toBe('—');
    expect(meanSd({ mean: 0.9, sd: 0.012, min: 0.88, max: 0.91, n: 3 })).toBe('0.900 ± 0.012');
    expect(meanSd({ mean: 0.9, sd: null, min: 0.9, max: 0.9, n: 1 })).toBe('0.900');
    expect(meanSd(null)).toBe('—');
    expect(interval({ lower: 0.1, upper: 0.2 })).toBe('0.100–0.200');
    expect(signed(0.0123)).toBe('+0.012');
    expect(signed(-0.0123)).toBe('−0.012');
    expect(signed(-0.0001)).toBe('0.000');
    expect(signedInterval({ lower: -0.01, upper: 0.02 })).toBe('−0.010 to +0.020');
    expect(percent(0.256)).toBe('26%');
  });

  it('calls a difference clear only when its interval excludes zero', () => {
    expect(verdict({ lower: 0.01, upper: 0.03 })).toBe('higher');
    expect(verdict({ lower: -0.03, upper: -0.01 })).toBe('lower');
    expect(verdict({ lower: -0.01, upper: 0.03 })).toBe('unclear');
    expect(verdict(null)).toBe('unavailable');
    expect(verdictLabel('higher', 'auroc')).toBe('Higher (better)');
    expect(verdictLabel('lower', 'loss')).toBe('Lower (better)');
    expect(verdictLabel('unclear', 'auroc')).toBe('No clear difference');
  });
});

describe('paired comparisons', () => {
  it('flips the service order so the reference is subtracted', () => {
    const [raw] = experimentResults().comparisons;
    const flipped = orient(raw, 'batch-abmil').metric('auroc');
    expect(flipped.difference).toBeCloseTo(0.04);
    expect(flipped.interval).toEqual({ lower: 0.02, upper: 0.06 });
    expect(flipped.folds).toMatchObject({ n: 2, better: 2, worse: 0 });
    expect(flipped.other).toBe(0.95);
    const kept = orient(raw, 'batch-nnmil').metric('auroc');
    expect(kept.difference).toBeCloseTo(-0.04);
    expect(kept.folds.better).toBe(0);
  });

  it('puts the batch with the better primary metric first, and the earlier batch otherwise', () => {
    const [pair] = orientedPairs(experimentResults());
    expect(pair.otherId).toBe('batch-nnmil');
    expect(pair.referenceId).toBe('batch-abmil');
    const tied = experimentResults();
    tied.batches[1] = batch('batch-nnmil', 'nnMIL', {}, { candidateId: 'candidate-n' });
    const [fallback] = orientedPairs(tied);
    expect(fallback.otherId).toBe('batch-nnmil');
    expect(fallback.referenceId).toBe('batch-abmil');
  });

  it('names a leader only when a batch is strictly best', () => {
    const results = experimentResults();
    expect(leadingBatch(results.batches, 'auroc')).toBe('batch-nnmil');
    expect(leadingBatch(results.batches, 'auprc')).toBeNull();
    expect(leadingBatch([results.batches[0]], 'auroc')).toBeNull();
  });
});

describe('result helpers', () => {
  it('reports the validation-selected configuration', () => {
    const two = batch('b', 'Grid', { selectedCandidateId: 'second', configurations: [configuration({ candidateId: 'first', number: 1 }), configuration({ candidateId: 'second', number: 2 })] });
    expect(reportedConfiguration(two)?.number).toBe(2);
    expect(reportedConfiguration({ ...two, selectedCandidateId: null })?.number).toBe(1);
  });

  it('shades within the readable ramp and picks clean ticks', () => {
    expect(shade(0.5, 0, 1)).toBe('#9ec5f4');
    expect(shade(2, 0, 1)).toBe('#6da7ec');
    expect(shade(null, 0, 1)).toBeUndefined();
    expect(shade(0.5, 1, 1)).toBeUndefined();
    expect(niceTicks(0.9, 0.97, 5)).toEqual([0.9, 0.92, 0.94, 0.96]);
    expect(niceTicks(0.9, 0.97, 3)).toEqual([0.9, 0.925, 0.95]);
    expect(tickDigits([0.9, 0.925, 0.95])).toBe(3);
    expect(tickDigits([0.9, 0.92])).toBe(2);
  });

  it('exports every OOF and fold row as CSV', () => {
    const rows = resultRows(experimentResults());
    expect(rows[0]).toEqual(['batch', 'configuration', 'selected', 'split_seed', 'training_seed', 'level', 'fold', 'count', 'auroc', 'auprc', 'balancedAccuracy', 'macroF1', 'accuracy', 'loss', 'best_epoch', 'epochs_completed']);
    expect(rows.filter((row) => row[5] === 'oof')).toHaveLength(4);
    const fold = rows.find((row) => row[5] === 'test_fold');
    expect(fold?.slice(0, 9)).toEqual(['ABMIL baseline', 1, 'true', 7, 42, 'test_fold', 1, 20, 0.93]);
    expect(csv([['a,b', 'say "hi"', null, 3]])).toBe('"a,b","say ""hi""",,3\n');
  });
});
