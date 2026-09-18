import { request } from './client';

export type ArchiveAction = 'export' | 'verify' | 'restore';
export interface OperationJob {
  key: string; id: string; kind: string; name: string;
  job: { status: string; cancellable: boolean; busy?: boolean; waitingReason?: string };
}
export interface OperationsInventory {
  projectId: string; jobs: OperationJob[];
  reservations: { batchId: string; runId: string; kind?: string; cpus: number; ramGb: number; gpu: number | null; runsPerGpu: number }[];
  capacity: { cpus: number; availableRamGb: number }; note: string;
}
export interface SourceRegistration {
  id: string; path: string; name: string; role: string; available: boolean; permitted: boolean;
}
export interface SourceInventory {
  sources: SourceRegistration[]; referenceCount: number;
  missingReferences: { path: string; reason: string }[]; note: string;
}
export interface ArchiveJob {
  id: string; action: ArchiveAction; status: string; createdAt: string;
  sessionName: string; logPath: string; error: string | null;
  progress?: { stage: string; completed: number; total: number; file: string };
  result: { verified: boolean; fileCount: number; totalBytes: number; archivePath?: string;
    destinationPath?: string; originalPath?: string; relocated?: boolean; note?: string;
    externalSources?: SourceInventory } | null;
}
export interface ArchiveRequest {
  action: ArchiveAction; archivePath: string; destinationPath?: string; operationId: string;
}
const prefix = (project: string) => `/projects/${encodeURIComponent(project)}/operations`;
const post = (body: unknown) => ({ method: 'POST', body: JSON.stringify(body) });
export const operations = {
  inventory: (project: string) => request<OperationsInventory>(prefix(project)),
  sources: (project: string) => request<SourceInventory>(`${prefix(project)}/sources`),
  relink: (project: string, sourceId: string, expectedPath: string, replacementPath: string) =>
    request<{ note: string }>(`${prefix(project)}/sources/relink`, post({ sourceId, expectedPath, replacementPath })),
  archives: (project: string) => request<{ jobs: ArchiveJob[] }>(`${prefix(project)}/archives`),
  submit: (project: string, input: ArchiveRequest) => request<ArchiveJob>(`${prefix(project)}/archives`, post(input)),
  cancel: (project: string, job: string) => request<ArchiveJob>(`${prefix(project)}/archives/${encodeURIComponent(job)}/cancel`, post({})),
  retry: (project: string, job: string) => request<ArchiveJob>(`${prefix(project)}/archives/${encodeURIComponent(job)}/retry`, post({})),
};
