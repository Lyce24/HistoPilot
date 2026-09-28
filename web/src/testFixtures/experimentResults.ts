import type { BatchResult, ConfigurationResult, ExperimentResults, FoldResult, MetricBlock, MetricStats, PointResult, SeedResult } from '../api/experimentResults';

const stats = (mean: number, sd: number | null, min = mean, max = mean, n = 2): MetricStats => ({ mean, sd, min, max, n });
const block = (auroc: MetricStats | null, rest: Partial<MetricBlock> = {}): MetricBlock => ({
  auroc, auprc: stats(0.8, 0.01), balancedAccuracy: stats(0.7, 0.02), macroF1: stats(0.72, 0.02), accuracy: stats(0.9, 0.01), loss: stats(0.3, 0.02), ...rest,
});
const point = (auroc: number, extra: Partial<PointResult> = {}): PointResult => ({ auroc, auprc: 0.8, balancedAccuracy: 0.7, macroF1: 0.72, accuracy: 0.9, loss: 0.3, count: 40, ...extra });
const fold = (index: number, auroc: number | null, bestEpoch: number | null = 3, status = 'completed'): FoldResult => ({
  fold: index, splitPlanId: `split-${index}`, runId: `run-${index}`, status, testCount: 20,
  metrics: auroc === null ? null : point(auroc, { count: 20 }), bestEpoch, epochsCompleted: bestEpoch === null ? null : bestEpoch + 5,
  validationScore: 0.9, checkpointMetric: 'validation_auroc',
});
const seed = (trainingSeed: number, oof: number | null, folds: FoldResult[]): SeedResult => ({
  trainingSeed, splitSeed: 7, complete: oof !== null, completedRuns: folds.filter((row) => row.metrics).length, totalRuns: folds.length,
  oof: oof === null ? null : point(oof), folds, foldStats: block(stats(0.9, 0.02)),
});

export function configuration(changes: Partial<ConfigurationResult> = {}): ConfigurationResult {
  return {
    candidateId: 'candidate-a', number: 1, model: 'abmil', inputMode: 'image', selected: true, validationScore: 0.91,
    splitSeeds: [{ splitSeed: 7, folds: [{ fold: 0, splitPlanId: 'split-0', testCount: 20 }, { fold: 1, splitPlanId: 'split-1', testCount: 20 }],
      seeds: [seed(42, 0.9, [fold(0, 0.93, 12), fold(1, 0.89, 2)]), seed(43, 0.92, [fold(0, 0.95, 4), fold(1, 0.9, 6)])] }],
    seedCount: 2, plannedSeedCount: 2, seedAverage: block(stats(0.91, 0.014, 0.9, 0.92)), foldAverage: block(stats(0.9175, 0.027, 0.89, 0.95, 4)),
    foldCount: 4, plannedFoldCount: 4,
    perClass: [
      { label: 'HG', support: 10, recall: stats(0.9, 0.02), precision: stats(0.8, 0.03), f1: stats(0.85, 0.02), auroc: stats(0.97, 0.01), auprc: stats(0.9, 0.02) },
      { label: 'LG', support: 8, recall: stats(0.25, 0.05), precision: stats(0.5, 0.1), f1: stats(0.33, 0.05), auroc: stats(0.85, 0.02), auprc: stats(0.5, 0.05) },
      { label: 'ND', support: 22, recall: stats(0.97, 0.01), precision: stats(0.95, 0.01), f1: stats(0.96, 0.01), auroc: stats(0.96, 0.01), auprc: stats(0.98, 0.01) },
    ],
    confusion: { meanCounts: [[9, 0.5, 0.5], [3, 2, 3], [0.5, 0.5, 21]], rowRates: [[0.9, 0.05, 0.05], [0.375, 0.25, 0.375], [0.023, 0.023, 0.954]], seeds: 2 },
    ensemble: point(0.925),
    intervals: { unit: 'slide', resamples: 2000, seed: 42, note: 'note', available: true, units: 40,
      seedAverage: { available: true, validResamples: 2000, excludedResamples: 0, intervals: { auroc: { lower: 0.88, upper: 0.94 }, balancedAccuracy: { lower: 0.66, upper: 0.74 } } },
      ensemble: { available: true, intervals: { auroc: { lower: 0.89, upper: 0.95 } } } },
    complete: true,
    ...changes,
  };
}

export function batch(batchId: string, name: string, changes: Partial<BatchResult> = {}, config: Partial<ConfigurationResult> = {}): BatchResult {
  return {
    batchId, name, state: 'active', status: 'completed', progress: { completedRuns: 4, totalRuns: 4 },
    selection: { source: 'single', metric: 'validation_auroc', ready: true, scores: {} },
    selectedCandidateId: config.candidateId ?? 'candidate-a', configurations: [configuration(config)], findings: [], ...changes,
  };
}

/** Two batches on the same folds: nnMIL leads on AUROC with an interval clear of zero. */
export function experimentResults(changes: Partial<ExperimentResults> = {}): ExperimentResults {
  const abmil = batch('batch-abmil', 'ABMIL baseline');
  const nnmil = batch('batch-nnmil', 'nnMIL', {}, { candidateId: 'candidate-n', model: 'nnmil', seedAverage: block(stats(0.95, 0.005, 0.945, 0.955)) });
  return {
    experimentId: 'exp',
    target: { task: 'multiclass_classification', unit: 'slide', classes: ['HG', 'LG', 'ND'], positiveClass: null, field: 'grade' },
    design: { splitUnit: 'slide', groupByPatient: false, folds: 2, splitSeeds: [7], slideCount: 40, resamplingUnit: 'slide' },
    policy: { resamples: 2000, seed: 42, confidenceLevel: 0.95, method: 'unit_percentile_bootstrap_seed_mean_v1' },
    primaryMetric: 'auroc',
    batches: [abmil, nnmil],
    comparisons: [{
      leftBatchId: 'batch-abmil', rightBatchId: 'batch-nnmil', leftCandidateId: 'candidate-a', rightCandidateId: 'candidate-n',
      difference: 'left_minus_right', available: true,
      oof: {
        auroc: { left: 0.91, right: 0.95, difference: -0.04 }, auprc: { left: 0.8, right: 0.8, difference: 0 },
        balancedAccuracy: { left: 0.7, right: 0.7, difference: 0.01 }, macroF1: { left: 0.72, right: 0.72, difference: 0 },
        accuracy: { left: 0.9, right: 0.9, difference: 0 }, loss: { left: 0.3, right: 0.3, difference: 0 },
      },
      oofInterval: { available: true, unit: 'slide', units: 40, intervals: { auroc: { lower: -0.06, upper: -0.02 }, balancedAccuracy: { lower: -0.03, upper: 0.05 } } },
      folds: {
        auroc: { n: 2, mean: -0.03, sd: 0.01, min: -0.04, max: -0.02, better: 0, worse: 2, tied: 0, values: [-0.04, -0.02] },
        auprc: { n: 2 }, balancedAccuracy: { n: 2, mean: 0.01, sd: 0.02, better: 1, worse: 1, tied: 0 }, macroF1: { n: 0 }, accuracy: { n: 0 }, loss: { n: 0 },
      },
    }],
    findings: [{ severity: 'warning', code: 'CLASS_RECALL_LOW', message: 'ABMIL baseline: LG recall is 0.250 ± 0.050 (8 slides). The model seldom predicts this class correctly.', batchId: 'batch-abmil' }],
    ...changes,
  };
}

/**
 * One batch that declared a controlled comparison, shaped like the service's output:
 * ABMIL (reference) against nnMIL, a clinical-only arm, and a mean-pooling arm that has
 * not finished. Contrasts are reference − arm on AUROC with Holm-adjusted p-values.
 */
export function comparisonResults(): ExperimentResults {
  const arm = (candidateId: string, number: number, model: string, inputMode: string, auroc: number, changes: Partial<ConfigurationResult> = {}) => configuration({
    candidateId, number, model, inputMode, selected: number === 1, validationScore: null,
    seedAverage: block(stats(auroc, 0.01, auroc - 0.01, auroc + 0.01)),
    intervals: { unit: 'slide', resamples: 2000, seed: 42, note: 'note', available: true, units: 40,
      seedAverage: { available: true, validResamples: 2000, excludedResamples: 0, intervals: { auroc: { lower: auroc - 0.03, upper: auroc + 0.03 } } } },
    ...changes,
  });
  const contrast = (armId: string, armNumber: number, model: string, inputMode: string, difference: number, pValue: number, pValueHolm: number) => ({
    armId, armNumber, model, inputMode, difference: 'reference_minus_arm' as const, available: true,
    oof: { auroc: { left: 0.91, right: 0.91 - difference, difference }, auprc: null, balancedAccuracy: null, macroF1: null, accuracy: null, loss: null },
    oofInterval: { available: true, unit: 'slide' as const, units: 40, validResamples: 2000, excludedResamples: 0, intervals: { auroc: { lower: difference - 0.03, upper: difference + 0.03 } } },
    folds: { auroc: { n: 2, mean: difference, sd: 0.01, min: difference - 0.01, max: difference + 0.01, better: difference > 0.03 ? 2 : 1, worse: difference > 0.03 ? 0 : 1, tied: 0, values: [difference - 0.01, difference + 0.01] },
      auprc: { n: 0 }, balancedAccuracy: { n: 0 }, macroF1: { n: 0 }, accuracy: { n: 0 }, loss: { n: 0 } },
    pValue, pValueHolm,
  });
  const ablation = batch('batch-ablation', 'Model ablation', {
    selection: { source: 'reference', metric: 'validation_auroc', ready: true, scores: {} }, selectedCandidateId: 'cand-1',
    configurations: [
      arm('cand-1', 1, 'abmil', 'image', 0.91),
      arm('cand-2', 2, 'nnmil', 'image', 0.87),
      arm('cand-3', 3, 'abmil', 'clinical', 0.9),
      arm('cand-4', 4, 'mean_pool', 'image', 0.8, { complete: false, seedCount: 0, plannedSeedCount: 2, foldCount: 1,
        seedAverage: { auroc: null, auprc: null, balancedAccuracy: null, macroF1: null, accuracy: null, loss: null },
        intervals: { unit: 'slide', resamples: 2000, seed: 42, note: 'note', available: false, reason: 'No complete seed yet.' } }),
    ],
    comparison: {
      referenceId: 'cand-1', referenceNumber: 1, primaryMetric: 'auroc', adjustment: 'holm',
      contrasts: [
        contrast('cand-2', 2, 'nnmil', 'image', 0.04, 0.0004, 0.0008),
        contrast('cand-3', 3, 'abmil', 'clinical', 0.01, 0.25, 0.5),
        { armId: 'cand-4', armNumber: 4, model: 'mean_pool', inputMode: 'image', difference: 'reference_minus_arm', available: false,
          reason: 'Both batches need complete OOF results.', pValue: null, pValueHolm: null },
      ],
    },
  });
  return experimentResults({
    target: { task: 'binary_classification', unit: 'slide', classes: ['Negative', 'Positive'], positiveClass: 'Positive', field: 'status' },
    batches: [ablation], comparisons: [],
    findings: [{ severity: 'warning', code: 'SELECTION_UNAVAILABLE', message: 'Model ablation: no validation-based configuration choice is available; configuration 1 is shown.', batchId: 'batch-ablation' }],
  });
}
