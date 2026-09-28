import type { BatchResult, ConfigurationResult, ExperimentResults, MetricStats, PairedComparison, ResultMetric } from '../api/experimentResults';
import { lowerIsBetter } from '../api/experimentResults';
import type { ConfidenceInterval } from '../api/statistics';
import { modelLabel } from './modelCapabilities';

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const MINUS = '−';

export function metricLabel(metric: ResultMetric, task?: string | null) {
  const macro = task !== 'binary_classification';
  return ({
    auroc: macro ? 'Macro AUROC' : 'AUROC', auprc: macro ? 'Macro AUPRC' : 'AUPRC',
    balancedAccuracy: 'Balanced accuracy', macroF1: 'Macro F1', accuracy: 'Accuracy', loss: 'Log loss',
  } as const)[metric];
}
/** The label inside a sentence: "macro AUROC", "balanced accuracy", but "AUROC" stays. */
export function metricPhrase(metric: ResultMetric, task?: string | null) {
  const label = metricLabel(metric, task);
  const [first, ...rest] = label.split(' ');
  return first === first.toUpperCase() ? label : [first.toLowerCase(), ...rest].join(' ');
}
/** Short column headings for dense tables. */
export function metricShortLabel(metric: ResultMetric) {
  return ({ auroc: 'AUROC', auprc: 'AUPRC', balancedAccuracy: 'Bal. acc.', macroF1: 'Macro F1', accuracy: 'Accuracy', loss: 'Loss' } as const)[metric];
}

export const fixed = (value: unknown, digits = 3) => finite(value) ? value.toFixed(digits) : '—';
/** A signed difference with a true minus sign, so +0.004 and −0.004 align. */
export function signed(value: unknown, digits = 3) {
  if (!finite(value)) return '—';
  const text = Math.abs(value).toFixed(digits);
  return Number(text) === 0 ? text : `${value > 0 ? '+' : MINUS}${text}`;
}
export function meanSd(stats?: MetricStats | null, digits = 3) {
  if (!stats) return '—';
  return stats.sd === null || stats.n < 2 ? fixed(stats.mean, digits) : `${fixed(stats.mean, digits)} ± ${fixed(stats.sd, digits)}`;
}
export const range = (stats?: MetricStats | null, digits = 3) => stats ? `${fixed(stats.min, digits)}–${fixed(stats.max, digits)}` : '—';
export const interval = (value?: ConfidenceInterval | null, digits = 3) => value ? `${fixed(value.lower, digits)}–${fixed(value.upper, digits)}` : '—';
export const signedInterval = (value?: ConfidenceInterval | null, digits = 3) => value ? `${signed(value.lower, digits)} to ${signed(value.upper, digits)}` : '—';
export const percent = (value: unknown, digits = 0) => finite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
/** A p-value to three significant figures, or "< 0.001" below that. */
export function pValue(value: unknown) {
  if (!finite(value)) return '—';
  return value < 0.001 ? '< 0.001' : value.toPrecision(3);
}

/** The configuration a batch reports: chosen on validation, never on OOF results. */
export function reportedConfiguration(batch: BatchResult): ConfigurationResult | undefined {
  return batch.configurations.find((row) => row.candidateId === batch.selectedCandidateId) ?? batch.configurations[0];
}
export const hasResults = (batch: BatchResult) => Boolean(reportedConfiguration(batch)?.seedCount || reportedConfiguration(batch)?.foldCount);

export function batchModel(batch: BatchResult) {
  const configuration = reportedConfiguration(batch);
  if (!configuration) return '';
  return configuration.inputMode === 'clinical' ? 'Clinical baseline' : modelLabel(configuration.model ?? undefined) || configuration.model || '';
}

export type Verdict = 'higher' | 'lower' | 'unclear' | 'unavailable';
/** "Higher" only when the whole 95% interval of the difference sits above zero. */
export function verdict(value?: ConfidenceInterval | null): Verdict {
  if (!value || !finite(value.lower) || !finite(value.upper)) return 'unavailable';
  if (value.lower > 0) return 'higher';
  if (value.upper < 0) return 'lower';
  return 'unclear';
}
export function verdictLabel(result: Verdict, metric: ResultMetric) {
  if (result === 'unavailable') return 'Interval unavailable';
  if (result === 'unclear') return 'No clear difference';
  const better = (result === 'higher') !== lowerIsBetter(metric);
  return `${result === 'higher' ? 'Higher' : 'Lower'} (${better ? 'better' : 'worse'})`;
}

export interface OrientedComparison {
  otherId: string; referenceId: string; available: boolean; reason?: string;
  metric: (metric: ResultMetric) => {
    difference: number | null; other: number | null; reference: number | null; interval: ConfidenceInterval | null;
    folds: { n: number; mean: number | null; sd: number | null; better: number; worse: number; tied: number };
  };
  intervalReason?: string;
}
/**
 * The service reports left − right for its pair order. Present every pair as
 * other − reference, flipping the interval and the fold tallies when needed.
 */
export function orient(comparison: PairedComparison, referenceId: string): OrientedComparison {
  const flip = comparison.leftBatchId === referenceId;
  const sign = flip ? -1 : 1;
  return {
    referenceId, otherId: flip ? comparison.rightBatchId : comparison.leftBatchId,
    available: comparison.available, reason: comparison.reason,
    intervalReason: comparison.oofInterval?.available === false ? comparison.oofInterval.reason : undefined,
    metric(metric) {
      const point = comparison.oof?.[metric] ?? null;
      const ci = comparison.oofInterval?.available ? comparison.oofInterval.intervals?.[metric] ?? null : null;
      const folds = comparison.folds?.[metric];
      const better = folds?.better ?? 0, worse = folds?.worse ?? 0;
      return {
        difference: point ? sign * point.difference : null,
        other: point ? (flip ? point.right : point.left) : null,
        reference: point ? (flip ? point.left : point.right) : null,
        interval: ci ? (flip ? { lower: -ci.upper, upper: -ci.lower } : ci) : null,
        folds: {
          n: folds?.n ?? 0, mean: finite(folds?.mean) ? sign * folds.mean : null, sd: folds?.sd ?? null,
          better: flip ? worse : better, worse: flip ? better : worse, tied: folds?.tied ?? 0,
        },
      };
    },
  };
}
/**
 * Each pair reads "leader vs other": the batch with the better seed mean on the primary
 * metric comes first, so its column is a lead, not a deficit. The orientation is fixed per
 * row across metrics; without a leader, the later batch is compared with the earlier one.
 */
export function orientedPairs(results: ExperimentResults) {
  const order = new Map(results.batches.map((batch, index) => [batch.batchId, index]));
  const primary = new Map(results.batches.map((batch) => [batch.batchId, reportedConfiguration(batch)?.seedAverage[results.primaryMetric]?.mean]));
  return results.comparisons.map((comparison) => {
    const left = primary.get(comparison.leftBatchId), right = primary.get(comparison.rightBatchId);
    if (finite(left) && finite(right) && left !== right) {
      const leftLeads = lowerIsBetter(results.primaryMetric) ? left < right : left > right;
      return orient(comparison, leftLeads ? comparison.rightBatchId : comparison.leftBatchId);
    }
    const earlier = (order.get(comparison.leftBatchId) ?? 0) < (order.get(comparison.rightBatchId) ?? 0);
    return orient(comparison, earlier ? comparison.leftBatchId : comparison.rightBatchId);
  });
}

/** The best batch on a metric's seed mean; ties and missing values give no winner. */
export function leadingBatch(batches: BatchResult[], metric: ResultMetric) {
  const values = batches.map((batch) => ({ batch, value: reportedConfiguration(batch)?.seedAverage[metric]?.mean }))
    .filter((row): row is { batch: BatchResult; value: number } => finite(row.value));
  if (values.length < 2) return null;
  const sorted = [...values].sort((a, b) => lowerIsBetter(metric) ? a.value - b.value : b.value - a.value);
  return sorted[0].value === sorted[1].value ? null : sorted[0].batch.batchId;
}

/**
 * A light-to-dark single-hue ramp for table-cell shading: validated blue steps 100–300,
 * the range where dark ink stays above 6:1, so every shaded cell keeps its number readable.
 */
const ramp = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec'];
export function shade(value: number | null | undefined, low: number, high: number) {
  if (!finite(value) || !(high > low)) return undefined;
  const position = Math.min(1, Math.max(0, (value - low) / (high - low)));
  return ramp[Math.round(position * (ramp.length - 1))];
}

export function niceTicks(low: number, high: number, count = 5) {
  if (!(high > low)) return [low];
  const raw = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * power).find((candidate) => candidate >= raw) ?? raw;
  const ticks = [];
  for (let tick = Math.ceil(low / step) * step; tick <= high + step * 1e-9; tick += step) ticks.push(Number(tick.toFixed(10)));
  return ticks;
}

/** The fewest decimals (at least two) that print every tick exactly, e.g. 0.925 with three. */
export function tickDigits(ticks: number[]) {
  for (let digits = 2; digits <= 5; digits += 1) if (ticks.every((tick) => Math.abs(Number(tick.toFixed(digits)) - tick) < 1e-9)) return digits;
  return 5;
}

/** Plain CSV text; values are quoted only when they need it. */
export function csv(rows: (string | number | null | undefined)[][]) {
  const cell = (value: string | number | null | undefined) => {
    const text = value === null || value === undefined ? '' : String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return rows.map((row) => row.map(cell).join(',')).join('\n') + '\n';
}

/** Every completed fold and every seed's OOF result, one row each, for export. */
export function resultRows(results: ExperimentResults) {
  const metrics: ResultMetric[] = ['auroc', 'auprc', 'balancedAccuracy', 'macroF1', 'accuracy', 'loss'];
  const header = ['batch', 'configuration', 'selected', 'split_seed', 'training_seed', 'level', 'fold', 'count', ...metrics, 'best_epoch', 'epochs_completed'];
  const rows: (string | number | null)[][] = [header];
  for (const batch of results.batches) {
    for (const configuration of batch.configurations) {
      for (const split of configuration.splitSeeds) {
        for (const seed of split.seeds) {
          if (seed.oof) rows.push([batch.name, configuration.number, String(configuration.selected), split.splitSeed, seed.trainingSeed, 'oof', '', seed.oof.count ?? '', ...metrics.map((metric) => seed.oof?.[metric] ?? ''), '', '']);
          for (const fold of seed.folds) {
            if (fold.metrics) rows.push([batch.name, configuration.number, String(configuration.selected), split.splitSeed, seed.trainingSeed, 'test_fold', fold.fold + 1, fold.metrics.count ?? '', ...metrics.map((metric) => fold.metrics?.[metric] ?? ''), fold.bestEpoch ?? '', fold.epochsCompleted ?? '']);
          }
        }
      }
    }
  }
  return rows;
}
