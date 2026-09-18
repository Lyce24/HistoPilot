import { describe, expect, it } from 'vitest';
import type { FeatureSpec } from '../api/scientific';
import { editFeatureSource } from './featureSource';

const source: FeatureSpec = { datasetId: null, path: '/extraction/slide_features_titan', fileSuffix: '.h5', idSuffix: '', recursive: false, featureKind: 'slide', encoderId: 'titan', sourceExtractionJobId: 'extraction' };

describe('feature attachment identity', () => {
  it('keeps explicit slide kind after moving files, but clears inherited extraction and encoder claims', () => {
    expect(editFeatureSource(source, { path: '/other/features' })).toMatchObject({ featureKind: 'slide', path: '/other/features', sourceExtractionJobId: undefined, encoderId: undefined });
    expect(source.sourceExtractionJobId).toBe('extraction');
  });
  it('preserves a verified extraction handoff and its declared kind', () => {
    expect(editFeatureSource(source, { path: '/new/patches', featureKind: 'patch', encoderId: 'uni', sourceExtractionJobId: 'new-job' })).toMatchObject({ sourceExtractionJobId: 'new-job', featureKind: 'patch', encoderId: 'uni' });
  });
  it('invalidates provenance when changing feature kind and removes patch coordinates for slide vectors', () => {
    expect(editFeatureSource({ ...source, featureKind: 'patch', coordinatesPath: '/coords' }, { featureKind: 'slide' })).toMatchObject({ featureKind: 'slide', coordinatesPath: undefined, sourceExtractionJobId: undefined });
  });
  it('does not invalidate identical edits and never reuses coordinates from another directory', () => {
    expect(editFeatureSource(source, { path: source.path }).sourceExtractionJobId).toBe('extraction');
    expect(editFeatureSource({ ...source, featureKind: 'patch', coordinatesPath: '/old/coords' }, { path: '/new' }).coordinatesPath).toBeUndefined();
  });
});
