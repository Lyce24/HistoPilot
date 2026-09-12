import { request } from './client';

export type LifecycleState = 'active' | 'archived' | 'trashed';
export type CleanupAction = 'archive' | 'trash' | 'restore';
export interface CleanupItem {
  key: string; type: 'project' | 'dataset' | 'configuration' | 'draft' | 'packing' | 'extraction';
  id: string; kind: string; name: string; state: LifecycleState; createdAt?: string;
  dependsOn: string[]; usedBy: string[]; job?: { status: string; cancellable: boolean };
}
export interface CleanupInventory {
  projectId: string; revision: number; projectState: LifecycleState;
  items: CleanupItem[]; audit: { at: string; action?: string; changes?: unknown; operationId?: string }[];
  note: string;
}
export interface CleanupPreview {
  action: CleanupAction; keys: string[]; revision: number; previewHash: string; canApply: boolean;
  blockers: { code: string; message: string; keys: string[] }[];
  requiredKeys: string[]; recordCount: number; note: string;
}
export interface CleanupApply {
  action: CleanupAction; keys: string[]; previewHash: string; operationId: string;
}
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/cleanup`;
const post = (value: unknown) => ({ method: 'POST', body: JSON.stringify(value) });
export const lifecycle = {
  inventory: (project: string) => request<CleanupInventory>(prefix(project)),
  preview: (project: string, action: CleanupAction, keys: string[]) =>
    request<CleanupPreview>(`${prefix(project)}/preview`, post({ action, keys })),
  apply: (project: string, input: CleanupApply) =>
    request<{ action: CleanupAction; revision: number; changed: string[] }>(`${prefix(project)}/apply`, post(input)),
  cancel: (project: string, key: string, operationId: string) =>
    request<{ key: string; job: Record<string, unknown> }>(`${prefix(project)}/cancel`, post({ key, operationId })),
};

export const lifecycleLabel = { active: 'Active', archived: 'Archived', trashed: 'Trash' } as const;
export function cleanupJobActive(item: CleanupItem) {
  return Boolean(item.job && (item.job.cancellable || ['queued', 'pending', 'active', 'running', 'starting', 'stopping', 'cancelling', 'cancel_requested'].includes(item.job.status)));
}
export function cleanupPollInterval(inventory?: CleanupInventory) {
  return inventory?.items.some(cleanupJobActive) ? 3000 : 15000;
}
export function cleanupReviewMatches(preview: CleanupPreview | null, inventory: CleanupInventory | undefined, action: CleanupAction, keys: string[]) {
  if (!preview || !inventory || preview.revision !== inventory.revision || preview.action !== action) return false;
  const selected = new Set(keys);
  return selected.size === keys.length && preview.keys.length === selected.size && new Set(preview.keys).size === selected.size && preview.keys.every((key) => selected.has(key));
}
export function cleanupApplyRequest(preview: CleanupPreview, operationId: string): CleanupApply {
  if (!preview.canApply || preview.blockers.length || preview.requiredKeys.some((key) => !preview.keys.includes(key)) || !preview.keys.length || !operationId) throw new Error('Review and resolve every blocker before confirming cleanup.');
  return { action: preview.action, keys: [...preview.keys], previewHash: preview.previewHash, operationId };
}
