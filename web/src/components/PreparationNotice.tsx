import type { PreparationContext } from '../lib/preparationRoute';

export default function PreparationNotice({ context }: { context: PreparationContext }) {
  const message = context.saved === 'dataset' && context.datasetId
    ? 'Dataset saved. Define the target and development splits.'
    : context.saved === 'protocol' && context.protocolId && context.datasetId
      ? 'Development protocol saved. Create or open an experiment to select features, check compatibility and configure training.'
      : context.saved === 'bundle' && context.bundleId
        ? 'Feature bundle saved. Select it with a development protocol in Experiments to check compatibility and configure training.'
        : null;
  return message ? <p className="callout science-success" role="status">{message}</p> : null;
}
