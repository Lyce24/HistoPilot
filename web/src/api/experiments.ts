import { request } from './client';
import type { BatchManifest, DevelopmentBatchSpec, FrozenBatch, TrainingExecution } from './development';
import type { LifecycleState } from './lifecycle';
import type { MILExperimentSpec } from './mil';
import type { ScientificDraft } from './scientific';

export interface ExperimentBatch extends FrozenBatch {
  key: string; name: string; state: LifecycleState; status: string;
  inputSnapshot?: Record<string, unknown>; execution?: TrainingExecution | null;
}
export interface ModelExperiment {
  id: string; key: string; name: string; notes: string; tags: string[];
  revision: number; state: LifecycleState; status: string; legacy: boolean;
  createdAt: string; updatedAt: string; inputs: MILExperimentSpec | null; executionImplemented?: boolean;
  batches: ExperimentBatch[]; drafts: ScientificDraft[]; predictorId: string | null; inputSnapshot?: Record<string, unknown> | null;
  predictorIds?: Partial<Record<'ensemble' | 'refit', string>>;
  predictors?: { id: string; method: 'ensemble' | 'refit'; batchId: string; candidateId: string; trainingSeed: number; splitSeed: number; lifecycleState: LifecycleState }[];
}
export interface ExperimentBatchSummary extends Omit<ExperimentBatch, 'manifest' | 'inputSnapshot' | 'execution'> {
  manifest: Pick<BatchManifest, 'kind' | 'version' | 'summary'> & {
    spec: Pick<DevelopmentBatchSpec, 'experimentId' | 'experimentRevision' | 'experimentName' | 'batchName' | 'inputs'>;
  };
}
export interface ModelExperimentSummary extends Omit<ModelExperiment, 'batches' | 'drafts' | 'inputSnapshot'> {
  summary?: boolean; inputSnapshot?: null; batches: ExperimentBatchSummary[];
  drafts: Omit<ScientificDraft, 'payload'>[];
}
export interface ExperimentInput {
  name: string; notes?: string; tags?: string[]; inputs?: MILExperimentSpec | null;
}
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/model-experiments`;
export const experiments = {
  list: (project: string) => request<{ items: ModelExperiment[] }>(prefix(project)),
  summaries: (project: string) => request<{ items: ModelExperimentSummary[] }>(`${prefix(project)}?summary=true`),
  get: (project: string, id: string) => request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}`),
  create: (project: string, input: ExperimentInput & { operationId: string }) =>
    request<ModelExperiment>(prefix(project), { method: 'POST', body: JSON.stringify(input) }),
  update: (project: string, id: string, input: ExperimentInput & { expectedRevision: number }) =>
    request<ModelExperiment>(`${prefix(project)}/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(input) }),
};
export function experimentPollInterval(data?: { items: { status: string }[] }) {
  return data?.items.some((item) => ['running', 'queued', 'scheduled', 'cancelling'].includes(item.status)) ? 3000 : 15000;
}
export const experimentStatusLabel = (status: string) => ({ completed: 'Finished', queued: 'Scheduled (queued)', cancelling: 'Cancelling' }[status] ?? status.replace(/^./, (letter) => letter.toUpperCase()));
