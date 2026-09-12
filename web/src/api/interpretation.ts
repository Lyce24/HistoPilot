import { downloadArtifact, fetchArtifactBlob, request } from './client';
import type { ResourcePolicy } from './development';
import type { LifecycleState } from './lifecycle';
import type { ComputeExecution } from './predictors';
import type { Finding } from './scientific';
export interface InterpretationSlideInput {
  slideId: string; slidePath: string; featurePath: string; coordinatesPath?: string | null;
  featureKey: string; coordinatesKey: string; patchWidthLevel0?: number; patchHeightLevel0?: number;
  coordinateSpace: 'level0'; confirmRowAlignment: boolean;
}
export interface SlideGeometry { width: number; height: number; backend: string; levelDownsamples: number[] }
export interface InterpretationSlide extends InterpretationSlideInput, SlideGeometry {
  patchCount: number; dimensions: number; dtype: string; patchWidthLevel0: number; patchHeightLevel0: number;
  alignment: 'embedded_verified' | 'user_confirmed';
}
export interface InterpretationSelection {
  name: string; predictorId: string; evaluationId?: string | null; clinicalAnalysisId?: string | null;
  encoderId: string; slides: InterpretationSlideInput[]; resources?: ResourcePolicy;
}
export interface InterpretationSlideResult {
  slideId: string; patchCount: number; probabilities: number[]; attentionArtifact: string;
  members: { index: number; checkpointSha256: string; probabilities: number[] }[];
}
export interface InterpretationExecution extends ComputeExecution {
  result?: { slides?: InterpretationSlideResult[]; classOrder?: string[]; [key: string]: unknown } | null;
}
export interface Interpretation {
  id: string; createdAt: string; contentHash: string; lifecycleState?: LifecycleState;
  manifest: Omit<InterpretationSelection, 'slides'> & { kind: 'model-interpretation'; experimentId: string; slides: InterpretationSlide[]; method: 'ensemble' | 'refit'; memberCount: number; [key: string]: unknown };
  execution?: InterpretationExecution;
}
export interface InterpretationPreview { canSave: boolean; previewHash: string | null; findings: Finding[]; manifest: Interpretation['manifest'] | null }
export interface AttentionPatch { index: number; x: number; y: number; weight: number; percentile: number }
export interface RankedAttentionPatch extends AttentionPatch { rank: number }
export interface TopAttentionMap extends Omit<AttentionMap, 'patches'> { patches: RankedAttentionPatch[]; scope: 'whole_slide'; returned: number; coordinateBounds?: SlideRegion | null }
export interface AttentionMap {
  slideId: string; total: number; offset: number; limit: number; patchWidthLevel0: number; patchHeightLevel0: number;
  coordinateSpace: 'level0'; attentionKind: 'class_independent_pooling'; member: string | number;
  patches: AttentionPatch[]; probabilities: number[]; classOrder: string[];
}
export interface SlideRegion { x: number; y: number; width: number; height: number }
export interface InterpretationSource {
  id: string; name: string; current: boolean; findings: Finding[]; encoderId: string | null; dimensions: number | null;
  dtype: string | string[] | null; slideCount: number; featureSetId: string;
  datasetId?: string; datasetName?: string; slideFolder?: string | null; slideFolderSource?: 'dataset_import' | 'dataset_records' | null; slideFolderFinding?: Finding | null;
  packs: { id: string; name: string; outputDtype: string }[];
}
export interface GallerySource { slideFolder: string; featureBundleId: string; packArtifactId?: string | null; predictorId: string }
export interface GallerySlide { slideId: string; slidePath: string; relativePath: string; name: string; available: boolean; reason: string | null; patchCount: number | null }
export interface SlideGallery {
  items: GallerySlide[]; total: number; offset: number; limit: number; hasMore: boolean; folder: string;
  source: { featureBundleId: string; packArtifactId: string | null; encoderId: string; dimensions: number; dtype: string | string[] }; warnings: string[];
}
export interface VisualizeSelection extends GallerySource {
  slidePaths: string[]; evaluationId?: string | null; clinicalAnalysisId?: string | null; resources?: ResourcePolicy;
  patchWidthLevel0?: number; patchHeightLevel0?: number;
}
export interface VisualizeItem { slidePath: string; slideId: string; interpretationId?: string | null; status: string; reused: boolean; error?: { code: string; message: string } | null; interpretation?: Interpretation }
export interface VisualizeResult { items: VisualizeItem[]; interpretations: Interpretation[] }
const base = (project: string) => `/projects/${encodeURIComponent(project)}/interpretations`;
const record = (project: string, id: string) => `${base(project)}/${encodeURIComponent(id)}`;
const slide = (project: string, id: string, slideId: string) => `${record(project, id)}/slides/${encodeURIComponent(slideId)}`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const interpretations = {
  sources: (project: string) => request<{ items: InterpretationSource[] }>(`${base(project)}/sources`),
  gallery: (project: string, source: GallerySource, search: string, offset: number, signal?: AbortSignal) => request<SlideGallery>(`${base(project)}/gallery`, { ...post({ ...source, search, offset, limit: 24 }), signal }),
  galleryThumbnail: (project: string, path: string, signal?: AbortSignal) => fetchArtifactBlob(`${base(project)}/gallery/thumbnail?${new URLSearchParams({ path, max_size: '320' })}`, signal),
  visualize: (project: string, selection: VisualizeSelection, operationId: string) => request<VisualizeResult>(`${base(project)}/visualize`, post({ ...selection, operationId })),
  list: (project: string) => request<{ items: Interpretation[]; executionNote?: string }>(base(project)),
  get: (project: string, id: string) => request<Interpretation>(record(project, id)),
  inspectSlide: (project: string, path: string) => request<SlideGeometry>(`${base(project)}/slide-inspection`, post({ path })),
  preview: (project: string, selection: InterpretationSelection) => request<InterpretationPreview>(`${base(project)}/preview`, post(selection)),
  save: (project: string, selection: InterpretationSelection, previewHash: string, operationId: string) => request<Interpretation>(base(project), post({ ...selection, previewHash, operationId })),
  execution: (project: string, id: string) => request<InterpretationExecution>(`${record(project, id)}/execution`),
  job: (project: string, id: string, action: 'launch' | 'resume' | 'cancel', operationId: string) => request<InterpretationExecution>(`${record(project, id)}/${action}`, post({ operationId })),
  attention: (project: string, id: string, slideId: string, member: string, region: SlideRegion, offset = 0, limit = 10000, signal?: AbortSignal) => request<AttentionMap>(`${slide(project, id, slideId)}/attention?${new URLSearchParams({ member, offset: String(offset), limit: String(limit), ...Object.fromEntries(Object.entries(region).map(([key, value]) => [key, String(value)])) })}`, { signal }),
  topAttention: (project: string, id: string, slideId: string, member: string, limit: 10 | 20, signal?: AbortSignal) => request<TopAttentionMap>(`${slide(project, id, slideId)}/attention/top?${new URLSearchParams({ member, limit: String(limit) })}`, { signal }),
  patchImage: (project: string, id: string, slideId: string, member: string, patchIndex: number, signal?: AbortSignal) => fetchArtifactBlob(`${slide(project, id, slideId)}/patches/${encodeURIComponent(patchIndex)}/image?${new URLSearchParams({ member, max_size: '512' })}`, signal),
  thumbnail: (project: string, id: string, slideId: string, signal?: AbortSignal) => fetchArtifactBlob(`${slide(project, id, slideId)}/thumbnail?max_size=1536`, signal),
  region: (project: string, id: string, slideId: string, region: SlideRegion, signal?: AbortSignal) => fetchArtifactBlob(`${slide(project, id, slideId)}/region?${new URLSearchParams({ ...Object.fromEntries(Object.entries(region).map(([key, value]) => [key, String(value)])), max_size: '1536' })}`, signal),
  download: (project: string, id: string, filename: string) => downloadArtifact(`${record(project, id)}/artifacts/${encodeURIComponent(filename)}`, filename),
};
