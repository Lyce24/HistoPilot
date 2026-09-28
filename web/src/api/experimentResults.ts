import { request } from './client';
import type { Finding } from './scientific';
import type { ConfidenceInterval } from './statistics';

/** Metrics in display order. Loss is reported, never resampled or used to rank models. */
export type ResultMetric = 'auroc' | 'auprc' | 'balancedAccuracy' | 'macroF1' | 'accuracy' | 'loss';
export const lowerIsBetter = (metric: ResultMetric) => metric === 'loss';

/** Mean, sample SD (n − 1), range and count of one metric across seeds or folds. */
export interface MetricStats { mean: number; sd: number | null; min: number; max: number; n: number }
export type MetricBlock = Record<ResultMetric, MetricStats | null>;

export interface ClassRates {
  label: string; support: number; predicted: number; recall: number | null; precision: number | null; f1: number;
  /** One-versus-rest ranking; absent when only a saved confusion matrix is available. */
  auroc?: number | null; auprc?: number | null;
}
export type PointResult = Partial<Record<ResultMetric, number | null>> & {
  count?: number | null; classCounts?: Record<string, number> | null; confusionMatrix?: number[][] | null;
  perClass?: ClassRates[] | null;
};
export interface FoldResult {
  fold: number; splitPlanId: string; runId: string | null; status: string; testCount: number | null;
  metrics: PointResult | null; bestEpoch: number | null; epochsCompleted: number | null;
  validationScore: number | null; checkpointMetric: string | null;
}
export interface SeedResult {
  trainingSeed: number; splitSeed: number; complete: boolean; completedRuns: number; totalRuns: number;
  /** Out-of-fold metrics: every assessed unit scored once, by the fold model that did not train on it. */
  oof: PointResult | null; folds: FoldResult[]; foldStats: MetricBlock;
}
export interface SplitSeedResult {
  splitSeed: number; folds: { fold: number; splitPlanId: string; testCount: number | null }[]; seeds: SeedResult[];
}
export interface IntervalBlock {
  available: boolean; reason?: string; validResamples?: number; excludedResamples?: number;
  intervals?: Partial<Record<ResultMetric, ConfidenceInterval>>;
}
export interface ClassSummary {
  label: string; support: number | null;
  recall: MetricStats | null; precision: MetricStats | null; f1: MetricStats | null; auroc: MetricStats | null; auprc: MetricStats | null;
}
export interface ConfigurationResult {
  candidateId: string; number: number; model: string | null; inputMode: string; selected: boolean; validationScore: number | null;
  splitSeeds: SplitSeedResult[]; seedCount: number; plannedSeedCount: number;
  seedAverage: MetricBlock; foldAverage: MetricBlock; foldCount: number; plannedFoldCount: number;
  perClass: ClassSummary[]; confusion: { meanCounts: number[][]; rowRates: (number | null)[][]; seeds: number } | null;
  ensemble: PointResult | null;
  intervals: {
    unit: 'slide' | 'patient'; resamples: number; seed: number; note: string; available: boolean; reason?: string; units?: number;
    seedAverage?: IntervalBlock; ensemble?: IntervalBlock;
  };
  complete: boolean;
}
export interface BatchResult {
  batchId: string; name: string; state: string; status: string;
  progress: { completedRuns: number; totalRuns: number };
  /** A declared comparison reports its reference arm (`reference`), not a validation choice. */
  selection: { source: 'validation' | 'single' | 'first' | 'reference'; metric: string | null; ready: boolean; scores: Record<string, number | null> } | null;
  selectedCandidateId: string | null; configurations: ConfigurationResult[]; findings: (Finding & { batchId?: string })[];
  /** Present when the batch declared a controlled comparison and its reference arm has results. */
  comparison?: BatchComparison;
}
/** One arm against the reference: reference − arm on shared draws and folds, with bootstrap p-values. */
export interface ArmContrast {
  armId: string; armNumber: number; model: string | null; inputMode: string; difference: 'reference_minus_arm';
  available: boolean; reason?: string;
  oof?: PairedComparison['oof']; oofInterval?: PairedComparison['oofInterval']; folds?: PairedComparison['folds'];
  /** Two-sided bootstrap p-value on the primary metric; null without paired draws. */
  pValue: number | null;
  /** Holm-adjusted across the batch's planned contrasts; null when pValue is null. */
  pValueHolm: number | null;
}
export interface BatchComparison {
  referenceId: string; referenceNumber: number; primaryMetric: Exclude<ResultMetric, 'loss'>; adjustment: 'holm';
  contrasts: ArmContrast[];
}
export interface FoldDifference { n: number; mean?: number; sd?: number | null; min?: number; max?: number; better?: number; worse?: number; tied?: number; values?: number[] }
export interface PairedComparison {
  leftBatchId: string; rightBatchId: string; leftCandidateId: string | null; rightCandidateId: string | null;
  difference: 'left_minus_right'; available: boolean; reason?: string;
  oof?: Record<ResultMetric, { left: number; right: number; difference: number } | null>;
  oofInterval?: IntervalBlock & { unit?: 'slide' | 'patient'; units?: number };
  folds?: Record<ResultMetric, FoldDifference>;
}
export interface ExperimentResults {
  experimentId: string;
  target: { task: string; unit: 'slide' | 'patient'; classes: string[]; positiveClass: string | null; field?: string } | null;
  design: { splitUnit: 'slide' | 'patient'; groupByPatient: boolean; folds: number; splitSeeds: number[]; slideCount: number | null; resamplingUnit: 'slide' | 'patient' } | null;
  policy: { resamples: number; seed: number; confidenceLevel: number; method: string };
  primaryMetric: ResultMetric; batches: BatchResult[]; comparisons: PairedComparison[];
  findings: (Finding & { batchId?: string })[];
}

export const experimentResults = {
  get: (project: string, experiment: string) =>
    request<ExperimentResults>(`/projects/${encodeURIComponent(project)}/model-experiments/${encodeURIComponent(experiment)}/results`),
};

/** One seed-mean OOF headline per batch, for the experiment list. */
export interface BatchHeadline {
  batchId: string; name: string; status: string | null; seeds: number; plannedSeeds: number; configurations: number;
  task: string; metrics: Partial<Record<'auroc' | 'balancedAccuracy' | 'macroF1' | 'accuracy', MetricStats | null>>;
}
export interface ExperimentHeadlines { items: { experimentId: string; batches: BatchHeadline[] }[] }
export const experimentHeadlines = {
  list: (project: string) => request<ExperimentHeadlines>(`/projects/${encodeURIComponent(project)}/model-experiments/headlines`),
};
