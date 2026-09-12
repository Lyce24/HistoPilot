import { request } from './client';
import type { PredictorSelection, PredictorMethod, PredictorManifest } from './predictors';
import type { Finding } from './scientific';

export type PredictorSource = Pick<PredictorSelection, 'experimentId' | 'batchId' | 'candidateId' | 'trainingSeed' | 'splitSeed'>;
export type BuildMethod = PredictorMethod | 'both';
export const predictorSourceKey = (source: PredictorSource) => JSON.stringify([source.experimentId, source.batchId, source.candidateId, source.trainingSeed, source.splitSeed]);
export interface PredictorBuildSelection { selections: PredictorSource[]; method: BuildMethod; refitPercentile: number; namePrefix?: string }
export interface PredictorBuildPreview {
  canBuild: boolean; previewHash: string | null; findings: Finding[];
  items: { key: string; name?: string; selection: PredictorSelection; action: 'create' | 'reuse' | 'blocked'; recordId?: string; recordKind: 'frozen-predictor' | 'predictor-refit'; epochBudget?: PredictorManifest['epochBudget']; trainingSlideCount?: number; findings: Finding[] }[];
  counts: Record<string, number>;
}
export interface PredictorBuildResult {
  operationId: string; status: 'completed' | 'partial';
  items: { key: string; method: PredictorMethod; selection: PredictorSelection; status: 'created' | 'reused' | 'failed'; recordId?: string; recordKind: 'frozen-predictor' | 'predictor-refit'; error?: { code: string; message: string } }[];
}
const base = (project: string) => `/projects/${encodeURIComponent(project)}/predictors/builds`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const predictorBuilds = {
  preview: (project: string, selection: PredictorBuildSelection) => request<PredictorBuildPreview>(`${base(project)}/preview`, post(selection)),
  create: (project: string, selection: PredictorBuildSelection, previewHash: string, operationId: string) => request<PredictorBuildResult>(base(project), post({ ...selection, previewHash, operationId })),
};
