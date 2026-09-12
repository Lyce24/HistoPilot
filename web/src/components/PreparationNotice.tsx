import type { PreparationContext } from '../lib/preparationRoute';

export default function PreparationNotice({ context }: { context: PreparationContext }) {
  const message = context.saved === 'dataset' && context.datasetId
    ? 'Dataset saved. Define the target and development splits for this dataset below.'
    : context.saved === 'protocol' && context.protocolId && context.datasetId
      ? 'Development protocol saved. Prepare or reuse a feature bundle for its dataset.'
      : context.saved === 'bundle' && context.bundleId
        ? 'Feature bundle saved. Create or open an experiment to review these prepared inputs before training.'
        : null;
  return message ? <p className="callout science-success" role="status">{message}</p> : null;
}
