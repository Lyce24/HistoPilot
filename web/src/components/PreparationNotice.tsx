import type { PreparationContext } from '../lib/preparationRoute';

export default function PreparationNotice({ context }: { context: PreparationContext }) {
  const message = context.saved === 'dataset' && context.datasetId
    ? 'Dataset saved. Prepare slide features and targets and splits independently.'
    : context.saved === 'target-split' && context.targetSplitId
      ? 'Targets and splits saved. Use them with a feature bundle in a new experiment.'
      : context.saved === 'protocol' && context.protocolId
        ? 'Historical development protocol saved. Its existing experiments remain available.'
        : context.saved === 'bundle' && context.bundleId
          ? 'Feature bundle saved. Select it with targets and splits in a new experiment to check compatibility and design training.'
          : null;
  return message ? <p className="callout science-success" role="status">{message}</p> : null;
}
