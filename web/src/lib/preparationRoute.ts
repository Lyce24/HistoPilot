import { useHashParameters } from './hashRoute';

export interface PreparationContext {
  datasetId?: string;
  protocolId?: string;
  bundleId?: string;
  saved?: 'dataset' | 'protocol' | 'bundle';
}

export function preparationContext(parameters: URLSearchParams): PreparationContext {
  const saved = parameters.get('saved');
  return {
    datasetId: parameters.get('dataset') || undefined,
    protocolId: parameters.get('protocol') || undefined,
    bundleId: parameters.get('bundle') || undefined,
    saved: saved === 'dataset' || saved === 'protocol' || saved === 'bundle' ? saved : undefined,
  };
}

export function usePreparationContext() {
  return preparationContext(useHashParameters());
}

/** Carry exact prepared inputs through the next module and experiment selection. */
export function preparationLink(page: string, context: PreparationContext, extra: Record<string, string> = {}) {
  const parameters = new URLSearchParams(extra);
  if (context.datasetId) parameters.set('dataset', context.datasetId);
  if (context.protocolId) parameters.set('protocol', context.protocolId);
  if (context.bundleId) parameters.set('bundle', context.bundleId);
  if (context.saved) parameters.set('saved', context.saved);
  const query = parameters.toString();
  return `#${page}${query ? `?${query}` : ''}`;
}
