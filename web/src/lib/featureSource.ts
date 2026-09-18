import type { FeatureSpec } from '../api/scientific';

/** Editing a source cannot retain an extraction claim for different files. */
export function editFeatureSource(current: FeatureSpec, update: Partial<FeatureSpec>): FeatureSpec {
  const keys: (keyof FeatureSpec)[] = ['datasetId', 'slideListPath', 'slideList', 'path', 'encoderId', 'coordinatesPath', 'layout', 'fileSuffix', 'idSuffix', 'recursive', 'featureKind'];
  const changed = keys.some((key) => key in update && JSON.stringify(current[key]) !== JSON.stringify(update[key]));
  const moved = 'path' in update && update.path !== current.path;
  const next = { ...current, ...update };
  if (changed && !('sourceExtractionJobId' in update)) next.sourceExtractionJobId = undefined;
  if (moved && !('coordinatesPath' in update)) next.coordinatesPath = undefined;
  if (moved && current.sourceExtractionJobId && !('encoderId' in update)) next.encoderId = undefined;
  if (next.featureKind === 'slide') next.coordinatesPath = undefined;
  return next;
}
