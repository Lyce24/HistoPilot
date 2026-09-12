import type { Configuration, DatasetVersion, FeatureSpec, ProtocolSpec, VersionLabelResource } from '../api/scientific';

/** Personal tags name a version; canonical IDs remain its stable reference. */
export function versionLabelText(resource: VersionLabelResource, fallback: string): string {
  const tag = resource.versionLabel?.tag.trim();
  return tag || `${fallback.trim() || 'Saved version'} · ${resource.id.slice(-8)}`;
}

export function datasetVersionLabel(dataset: DatasetVersion): string {
  return versionLabelText(dataset, dataset.manifest.name?.trim() || 'Dataset');
}

export function configurationVersionLabel(configuration: Configuration): string {
  const { manifest } = configuration;
  if (manifest.kind === 'feature') {
    const encoder = manifest.layout?.encoderId?.trim() || (manifest.spec as FeatureSpec).encoderId?.trim();
    return versionLabelText(configuration, encoder ? `${encoder} features` : 'Features');
  }
  const target = (manifest.spec as ProtocolSpec).target?.field?.trim();
  return versionLabelText(configuration, target ? `${target} protocol` : 'Development protocol');
}
