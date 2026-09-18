import type { CreateExperimentInput } from '../api/experiments';

export interface CreateExperimentDraft {
  version: 1; sourceId: string; name: string; notes: string; tags: string;
  pending: CreateExperimentInput | null;
}
export interface ExperimentMetadataBaseline { revision: number; name: string; notes: string; tags: string[] }
export interface ExperimentMetadataDraft {
  version: 1; name: string; notes: string; tags: string; baseline: ExperimentMetadataBaseline;
}
const object = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === 'object' && !Array.isArray(value));
const text = (value: unknown, max: number): value is string => typeof value === 'string' && value.length <= max;
const tags = (value: unknown): value is string[] => Array.isArray(value) && value.length <= 128 && value.every((tag) => text(tag, 200));
function fields(value: Record<string, unknown>) { return text(value.name, 120) && text(value.notes, 10000) && text(value.tags, 20000); }
export function isCreateExperimentDraft(value: unknown): value is CreateExperimentDraft {
  if (!object(value) || value.version !== 1 || !fields(value) || !text(value.sourceId, 256)) return false;
  if (value.pending === null) return true;
  const pending = value.pending;
  return object(pending) && text(pending.name, 120) && Boolean(pending.name.trim())
    && text(pending.notes, 10000) && tags(pending.tags) && text(pending.operationId, 256) && Boolean(pending.operationId)
    && (pending.sourceExperimentId === undefined || text(pending.sourceExperimentId, 256));
}
export function isExperimentMetadataDraft(value: unknown): value is ExperimentMetadataDraft {
  return object(value) && value.version === 1 && fields(value) && object(value.baseline)
    && typeof value.baseline.revision === 'number' && Number.isSafeInteger(value.baseline.revision) && value.baseline.revision >= 0
    && text(value.baseline.name, 120) && text(value.baseline.notes, 10000) && tags(value.baseline.tags);
}
