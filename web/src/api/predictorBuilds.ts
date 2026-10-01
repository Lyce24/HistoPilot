import { request } from './client';
import type { PredictorSelection, PredictorMethod, PredictorManifest } from './predictors';
import type { Finding } from './scientific';

export type PredictorSource = Pick<PredictorSelection, 'experimentId' | 'batchId' | 'candidateId' | 'trainingSeed' | 'splitSeed'>;
/** Builds create one seed group's predictors; seed ensembles are built on their own. */
export type BuildMethod = Exclude<PredictorMethod, 'seed_ensemble'> | 'both';
/** A seed ensemble has no single seed: its key keeps null seeds and never matches a seed group. */
export const predictorSourceKey = (source: Omit<PredictorSource, 'trainingSeed' | 'splitSeed'> & { trainingSeed?: number; splitSeed?: number }) => JSON.stringify([source.experimentId, source.batchId, source.candidateId, source.trainingSeed ?? null, source.splitSeed ?? null]);
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
