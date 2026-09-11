import { request } from './client';
import type { Finding, VersionLabel, VersionLabelInput } from './scientific';

export interface FeatureBundleSpec {
  featureSetId: string;
  packArtifactIds: string[];
}
export interface BundleValidation {
  jobId: string;
  sourceContentHash: string;
  tensorValidationComplete: boolean;
  provenanceComplete: boolean;
  validatedAt?: string;
}
export interface BundleFeature {
  id: string;
  contentHash: string;
  datasetId: string;
  sourceContentHash: string;
  validation: BundleValidation;
}
export interface BundlePack {
  id: string;
  materializationId: string;
  featureSetId: string;
  outputPath: string;
  outputDtype: string;
  sourceContentHash: string;
  verification: string;
  jobId: string;
  validation: BundleValidation;
}
export interface BundleSummary {
  slideCount: number;
  patchCount: number;
  dimensions: number | null;
  dtype: string | string[] | null;
  packCount: number;
}
export interface FeatureBundlePreview {
  spec: FeatureBundleSpec;
  previewHash: string;
  canFreeze: boolean;
  summary: BundleSummary;
  feature: BundleFeature | null;
  packs: BundlePack[];
  findings: Finding[];
}
export interface FeatureBundle {
  id: string;
  contentHash: string;
  createdAt: string;
  versionLabel?: VersionLabel | null;
  manifest: {
    kind: 'feature-bundle';
    datasetId: string;
    spec: FeatureBundleSpec;
    summary: BundleSummary;
    feature: BundleFeature;
    packs: BundlePack[];
  };
  current: boolean;
  findings: Finding[];
}
const base = (project: string) => `/projects/${encodeURIComponent(project)}/feature-bundles`;
export const bundles = {
  list: (project: string) => request<{ items: FeatureBundle[] }>(base(project)),
  get: (project: string, id: string) => request<FeatureBundle>(`${base(project)}/${encodeURIComponent(id)}`),
  preview: (project: string, spec: FeatureBundleSpec) => request<FeatureBundlePreview>(`${base(project)}/preview`, { method: 'POST', body: JSON.stringify(spec) }),
  freeze: (project: string, spec: FeatureBundleSpec, previewHash: string, operationId: string, versionLabel: VersionLabelInput) => request<FeatureBundle>(`${base(project)}/freeze`, {
    method: 'POST', body: JSON.stringify({ ...spec, previewHash, operationId, versionLabel }),
  }),
};
