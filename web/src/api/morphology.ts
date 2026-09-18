import { fetchArtifactBlob, request } from './client';
import type { ReviewStatus } from './slideReviews';

export interface MorphologySlide {
  slideId: string; patientId: string | null; hasImage: boolean;
  reviewStatus?: ReviewStatus; reviewRevision?: number;
  attributes: Record<string, string | number | boolean | null>;
}
export interface MorphologyPoint extends MorphologySlide {
  x: number; y: number; patchCount: number; sampledPatches: number;
}
export interface MorphologyPatch { slideId: string; patchIndex: number; x: number; y: number }
export interface MorphologyRegion { x: number; y: number; width: number; height: number }
export interface MorphologyIndex {
  indexId: string; datasetId: string; featureBundleId: string; encoderId: string | null;
  candidateSlides: number; indexedSlides: number; indexedPatches: number;
  method: string; sampling: string; explainedVariance: [number, number];
  points: MorphologyPoint[]; patches: MorphologyPatch[]; warnings: string[];
}
export interface QualityEvidence {
  featureKind?: 'patch' | 'slide';
  slideId: string; datasetId: string; width: number; height: number;
  patchWidth: number | null; patchHeight: number | null; patchCount: number | null;
  patches: Omit<MorphologyPatch, 'slideId'>[]; coverageSampled?: boolean;
  coordinateBounds: { x: number; y: number; width: number; height: number } | null;
  tissueContours: number[][][][]; artifactRemoval: boolean | null; warnings: string[];
}
export interface MorphologyNeighbors {
  scope: 'indexed_slides' | 'sampled_patches'; metric: string; candidateCount: number;
  items: (MorphologyPoint | MorphologyPatch & { similarity: number })[];
}
export type Neighbor = { slideId: string; patchIndex?: number; similarity: number; x: number; y: number };
const base = (project: string) => `/projects/${encodeURIComponent(project)}/morphology`;
const params = (datasetId: string, slideId: string, featureBundleId?: string) => new URLSearchParams({ datasetId, slideId, ...(featureBundleId ? { featureBundleId } : {}) });
export const morphology = {
  slides: (project: string, datasetId: string, search: string, offset: number, signal?: AbortSignal) =>
    request<{ items: MorphologySlide[]; total: number; matching: number; offset: number }>(`${base(project)}/slides?${new URLSearchParams({ datasetId, search, offset: String(offset), limit: '100' })}`, { signal }),
  build: (project: string, value: { datasetId: string; featureBundleId: string; maxSlides: number; patchesPerSlide: number; slideIds?: string[] }) =>
    request<MorphologyIndex>(`${base(project)}/index`, { method: 'POST', body: JSON.stringify(value) }),
  neighbors: (project: string, value: { indexId: string; slideId: string; mode: 'slide' | 'patch'; patchIndex?: number; otherSlidesOnly?: boolean }, signal?: AbortSignal) =>
    request<{ items: Neighbor[]; candidateCount: number; scope: string }>(`${base(project)}/neighbors`, { method: 'POST', body: JSON.stringify(value), signal }),
  quality: (project: string, datasetId: string, slideId: string, featureBundleId?: string, signal?: AbortSignal) =>
    request<QualityEvidence>(`${base(project)}/quality?${params(datasetId, slideId, featureBundleId)}`, { signal }),
  image: (project: string, datasetId: string, slideId: string, signal?: AbortSignal, region?: MorphologyRegion) =>
    fetchArtifactBlob(`${base(project)}/image?${params(datasetId, slideId)}&max_size=1536${region ? `&${new URLSearchParams(Object.fromEntries(Object.entries(region).map(([key, value]) => [key, String(value)])))}` : ''}`, signal),
  patchRegion: (project: string, datasetId: string, slideId: string, featureBundleId: string, patchIndex: number, signal?: AbortSignal) =>
    request<MorphologyRegion>(`${base(project)}/patch-region?${params(datasetId, slideId, featureBundleId)}&patchIndex=${patchIndex}`, { signal }),
  patch: (project: string, datasetId: string, slideId: string, featureBundleId: string, patchIndex: number, signal?: AbortSignal) =>
    fetchArtifactBlob(`${base(project)}/patch?${params(datasetId, slideId, featureBundleId)}&patchIndex=${patchIndex}`, signal),
};
