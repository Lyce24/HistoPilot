import { request } from './client';
import type { ComputeExecution, PredictorMethod } from './predictors';
import type { Finding } from './scientific';

export interface BulkEvaluationSelection { cohortId: string; scope: 'all' | 'selected'; predictorIds?: string[]; namePrefix?: string }
export interface BulkEvaluationPreview {
  canRun: boolean; previewHash: string | null; reviewedPredictorIds: string[]; eligibleCount: number; blockedCount: number;
  items: { predictorId: string; predictorName: string; method: PredictorMethod; eligible: boolean; findings: Finding[] }[];
}
export interface EvaluationBatch {
  id: string; status: string; cohortId: string; name?: string; createdAt?: string; lifecycleState?: string; cancelRequested?: boolean;
  items: { predictorId: string; predictorName?: string; method?: PredictorMethod; status: string; evaluationId?: string; execution?: ComputeExecution; findings?: Finding[]; error?: string }[];
}
const base = (project: string) => `/projects/${encodeURIComponent(project)}/evaluation-runs/bulk`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const bulkEvaluations = {
  preview: (project: string, selection: BulkEvaluationSelection) => request<BulkEvaluationPreview>(`${base(project)}/preview`, post(selection)),
  run: (project: string, selection: BulkEvaluationSelection, review: Pick<BulkEvaluationPreview, 'previewHash' | 'reviewedPredictorIds'>, operationId: string) => request<EvaluationBatch>(base(project), post({ ...selection, ...review, operationId })),
  list: (project: string) => request<{ items: EvaluationBatch[] }>(base(project)),
  get: (project: string, id: string) => request<EvaluationBatch>(`${base(project)}/${encodeURIComponent(id)}`),
  cancel: (project: string, id: string, operationId: string) => request<EvaluationBatch>(`${base(project)}/${encodeURIComponent(id)}/cancel`, post({ operationId })),
};
