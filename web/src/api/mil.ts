import { request } from './client';
import type { Finding } from './scientific';

export type LoadingPolicy = 'auto' | 'native' | 'mmap';

export interface MILExperimentSpec {
  protocolId: string;
  featureBundleId: string;
  loadingPolicy: LoadingPolicy;
  packArtifactId: string | null;
}

export interface MILExperimentPreview {
  canPlan: boolean;
  findings: Finding[];
  resolvedLoadingPolicy: 'native' | 'mmap' | null;
  packArtifactId: string | null;
  featureSetId: string | null;
  bundleId: string;
  executionImplemented: false;
}

export const mil = {
  preview: (project: string, spec: MILExperimentSpec) =>
    request<MILExperimentPreview>(`/projects/${encodeURIComponent(project)}/mil-experiments/preview`, {
      method: 'POST', body: JSON.stringify(spec),
    }),
};
