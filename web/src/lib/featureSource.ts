import type { FeatureSpec } from '../api/scientific';
import type { ExtractionJob } from '../api/trident';

/** The features an extraction wrote, as a features spec names them. */
export interface ExtractionFeatureInput {
  datasetId: string | null; path: string; encoderId?: string; featureKind?: 'patch' | 'slide'; sourceExtractionJobId?: string;
}

/** A succeeded extraction's outputs, encoder and job; null before it succeeds with outputs. */
export function extractionFeatureInput(job: ExtractionJob): ExtractionFeatureInput | null {
  const layout = job.result?.outputLayout ?? job.outputLayout;
  const featurePath = job.result?.featurePath ?? job.result?.featureDirectory;
  if (job.state !== 'succeeded' || !featurePath) return null;
  return { datasetId: job.spec.datasetId ?? null, path: featurePath, encoderId: String((layout?.featureKind === 'slide' ? job.spec.options.slide_encoder : job.spec.options.patch_encoder) || '') || undefined, featureKind: layout?.featureKind ?? 'patch', sourceExtractionJobId: job.id };
}

/** An extraction's outputs are read exactly as written: that folder only, with no slide list. */
export const extractionFeatureUpdate = (input: ExtractionFeatureInput): Partial<FeatureSpec> =>
  ({ ...input, slideList: null, slideListPath: null, layout: 'auto', fileSuffix: '.h5', recursive: false, idSuffix: '', coordinatesPath: undefined });

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
