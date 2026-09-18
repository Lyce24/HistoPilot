import type { PreparationContext } from '../lib/preparationRoute';

export default function PreparationNotice({ context }: { context: PreparationContext }) {
  const message = context.saved === 'dataset' && context.datasetId
    ? 'Dataset saved. Choose a feature bundle, then define the target and development splits.'
    : context.saved === 'protocol' && context.protocolId && context.datasetId
      ? 'Development protocol saved with its dataset and feature bundle. Create or open an experiment to configure training.'
      : context.saved === 'bundle' && context.bundleId
        ? 'Feature bundle saved. Combine it with a dataset in Targets & splits, or use an existing protocol in Experiments.'
        : null;
  return message ? <p className="callout science-success" role="status">{message}</p> : null;
}
