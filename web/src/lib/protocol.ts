import type { Configuration, ProtocolSpec, ScientificDraft } from '../api/scientific';
import { fromTemplate, templates } from './templates';

export function newDevelopmentSplit(seeds: number[] = [42], folds = 5): ProtocolSpec['split'] {
  return { ...fromTemplate<ProtocolSpec['split']>(templates.starters.trainingSplit), folds, seeds };
}

/** Retain an existing positive-class choice only while it is still a class. */
export function preservePositiveClass(
  current: string | undefined,
  classes: string[],
): string | undefined {
  return current !== undefined && classes.includes(current) ? current : undefined;
}

/** Initial suggestions use complete, non-missing source values without changing their spelling. */
export function inferTargetSettings(values: (string | null)[], truncated = false) {
  const classes = truncated
    ? []
    : [
        ...new Set(
          values.filter((value): value is string => value !== null && value.trim() !== ''),
        ),
      ];
  const task: '' | 'binary_classification' | 'multiclass_classification' =
    classes.length === 2
      ? 'binary_classification'
      : classes.length > 2
        ? 'multiclass_classification'
        : '';
  return {
    task,
    classes,
    labels: Object.fromEntries(classes.map((value) => [value, value])),
    // Value ordering is not evidence of which clinical outcome is positive.
    positiveClass: undefined as string | undefined,
  };
}

/** Training designs derived for an experiment's inputs belong to that experiment, not to history. */
export function setupDerivedProtocol(configuration: Pick<Configuration, 'manifest'>): boolean {
  const { manifest } = configuration;
  return Boolean(manifest.sourceTargetSplit || (manifest.spec as { sourceTargetSplitId?: string | null } | undefined)?.sourceTargetSplitId);
}

/** Combined target/split protocols saved before Targets & Splits and experiment designs existed. */
export function historicalProtocols(configurations: readonly Configuration[], drafts: readonly ScientificDraft[]) {
  return {
    versions: configurations.filter((item) => !setupDerivedProtocol(item)),
    drafts: drafts.filter((item) => item.payload.type === 'analysis-protocol' && item.status === 'editable'),
  };
}
