import { templates } from './templates';
import type { ProtocolSpec } from '../api/scientific';

type Split = ProtocolSpec['split'];

export const DEFAULT_VALIDATION_FRACTION = templates.starters.trainingSplit.validationFraction;
/** The service's limit on split seeds per training design (`SplitSpec.seeds`). */
export const MAX_SPLIT_SEEDS = 10;
const MAX_SEED = 2 ** 32 - 1;

/** Why comma-separated split seeds cannot be saved, or '' when they can. */
export function splitSeedsError(text: string): string {
  const entries = text.split(',').map((value) => value.trim());
  if (entries.some((value) => !/^\d+$/.test(value) || Number(value) > MAX_SEED)) return `Enter whole-number seeds from 0 to ${MAX_SEED}, separated by commas.`;
  if (new Set(entries.map(Number)).size !== entries.length) return 'Each split seed must be different.';
  if (entries.length > MAX_SPLIT_SEEDS) return `Enter at most ${MAX_SPLIT_SEEDS} split seeds; you entered ${entries.length}.`;
  return '';
}

/**
 * Development designs that train (the service's `TRAINABLE_MODES`): each unit is assessed at
 * most once per split seed. Monte Carlo and nested designs stay for planning only.
 */
export const TRAINABLE_MODES: readonly Split['mode'][] = ['kfold', 'predefined_folds', 'leave_one_domain_out', 'held_out'];

/** Why a training design cannot be checked yet, or '' when it can. */
export function trainingDesignIssue(split: Split): string {
  if (!TRAINABLE_MODES.includes(split.mode)) return 'Choose a strategy that trains: k-fold, predefined folds, leave one site out or a held-out assessment.';
  if (split.mode === 'predefined_folds' && !split.foldField) return 'Choose the column that assigns each unit’s fold.';
  if (split.mode === 'leave_one_domain_out' && !split.domainField) return 'Choose the site or cohort column.';
  if (split.mode === 'leave_one_domain_out' && split.domainPolicy === 'selected' && !split.heldOutDomains?.length) return 'Select at least one site or cohort to hold out.';
  if (split.mode === 'held_out' && split.seeds.length !== 1) return 'A held-out assessment trains with one split seed; repeat it over training seeds instead.';
  return '';
}

const percentText = (value: number) => `${Number((value * 100).toFixed(6))}%`;

/** A training design in a few words: its assessment, split seeds and early-stop validation. */
export function trainingDesignText(split: Split): string {
  const design = split.mode === 'predefined_folds' ? `Predefined folds from ${split.foldField ?? 'a column'}`
    : split.mode === 'leave_one_domain_out' ? `Leave one site out by ${split.domainField ?? 'a column'}${split.domainPolicy === 'selected' ? ` (${(split.heldOutDomains ?? []).join(', ')})` : ''}`
      : split.mode === 'held_out' ? `Held-out assessment of ${percentText(split.testFraction ?? 0.2)}`
        : `${split.folds} folds`;
  return `${design} · ${split.seeds.length} split seed${split.seeds.length === 1 ? '' : 's'} · ${percentText(split.validationFraction ?? DEFAULT_VALIDATION_FRACTION)} early-stop validation`;
}

/** What one assessment plan of a design is called: a fold, a held-out site, or the held-out assessment. */
export function assessmentPlanNoun(split?: { mode?: string } | null): { one: string; many: string } {
  if (split?.mode === 'leave_one_domain_out') return { one: 'held-out site', many: 'held-out sites' };
  if (split?.mode === 'held_out') return { one: 'held-out assessment', many: 'held-out assessment' };
  return { one: 'fold', many: 'folds' };
}

const validFraction = (value: number) =>
  Number.isFinite(value) && value > 0 && value < 1;
const validInteger = (value: number, min: number, max: number) =>
  Number.isInteger(value) && value >= min && value <= max;

/**
 * Unused invalid controls must not block a development (version 4) strategy after their
 * inputs disappear.
 */
export function changeSplitStrategy(split: Split, mode: Split['mode']): Partial<Split> {
  const next: Partial<Split> = {
    mode,
    rules: { train: [], val: [], test: [] },
    imported: undefined,
    heldOutSource: 'fractions',
    domainField: undefined,
    foldField: undefined,
    domainPolicy: 'all',
    heldOutDomains: [],
  };
  if (mode !== 'kfold' && !validInteger(split.folds, 2, 10)) next.folds = 5;
  if (mode !== 'nested_kfold') {
    if (split.outerFolds !== undefined && !validInteger(split.outerFolds, 2, 10))
      next.outerFolds = 5;
    if (split.innerFolds !== undefined && !validInteger(split.innerFolds, 2, 10))
      next.innerFolds = 3;
  }
  if (
    mode !== 'monte_carlo' &&
    split.repeats !== undefined &&
    !validInteger(split.repeats, 1, 100)
  )
    next.repeats = 5;
  const showsTestFraction = mode === 'monte_carlo' || mode === 'held_out';
  if (
    !showsTestFraction &&
    split.testFraction !== undefined &&
    !validFraction(split.testFraction)
  )
    next.testFraction = 0.2;
  if (
    split.pools?.validationSource === 'fixed' &&
    split.validationFraction !== undefined &&
    !validFraction(split.validationFraction)
  )
    next.validationFraction = DEFAULT_VALIDATION_FRACTION;
  return next;
}
