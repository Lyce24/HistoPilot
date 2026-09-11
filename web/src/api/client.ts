import type {
  Cohort,
  CohortInput,
  DirectoryListing,
  Experiment,
  ExperimentInput,
  FilesystemRoot,
  Jobs,
  InitialConfig,
  ProjectInput,
  ProjectSummary,
  Source,
  SystemStatus,
  Workspace,
} from './types';

const BASE = '/api/v1';
type ScientificCapability = 'versionLabels' | 'taggedFreeze';
interface Session {
  token: string;
  scientificCapabilities?: Partial<Record<ScientificCapability, boolean>>;
}
let session: Promise<Session> | null = null;
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
async function responseError(response: Response): Promise<ApiError> {
  let message = `The control service returned HTTP ${response.status}.`;
  let code: string | undefined;
  try {
    const body = await response.json();
    if (typeof body.code === 'string') code = body.code;
    if (typeof body.detail === 'string') message = body.detail;
    else if (Array.isArray(body.detail))
      message = body.detail
        .map((item: { msg?: string }) => item.msg ?? 'Invalid request')
        .join('; ');
  } catch {
    /* A proxy may return a non-JSON error page. */
  }
  return new ApiError(message, response.status, code);
}
function sessionDetails(): Promise<Session> {
  if (!session) {
    session = fetch(`${BASE}/session`, { credentials: 'same-origin', cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) throw await responseError(response);
        const body: Session = await response.json();
        if (!body.token) throw new ApiError('The service did not return a session token.', 401);
        return body;
      })
      .catch((error) => {
        session = null;
        throw error;
      });
  }
  return session;
}
/** Always use the service session; no scientific fallback data lives in the browser. */
export async function request<T>(
  path: string,
  init: RequestInit = {},
  retrySession = true,
  scientificCapability?: ScientificCapability,
): Promise<T> {
  const activeSession = await sessionDetails();
  if (scientificCapability && !activeSession.scientificCapabilities?.[scientificCapability]) {
    session = null;
    throw new ApiError(SCIENTIFIC_SAVE_RESTART, 405);
  }
  const headers = new Headers(init.headers);
  headers.set('X-HistoPilot-Token', activeSession.token);
  if (init.body) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    credentials: 'same-origin',
    cache: 'no-store',
  });
  if (response.status === 401 && retrySession) {
    session = null;
    return request<T>(path, init, false, scientificCapability);
  }
  if (!response.ok) throw await responseError(response);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
const SCIENTIFIC_SAVE_RESTART = 'The running HistoPilot server does not support this save. Restart HistoPilot, then try saving again here. Your entered tag and note have been kept; you do not need to reload this page.';

/** Gate metadata-aware writes so an older running server cannot freeze without the label. */
export async function requestScientificSave<T>(
  path: string,
  init: RequestInit,
  capability: ScientificCapability,
): Promise<T> {
  try {
    return await request<T>(path, init, true, capability);
  } catch (reason) {
    if (reason instanceof ApiError && (reason.status === 405 || (reason.status === 404 && reason.message === 'Not Found'))) {
      session = null;
      throw new ApiError(SCIENTIFIC_SAVE_RESTART, reason.status);
    }
    throw reason;
  }
}
export const api = {
  workspace: () => request<Workspace>('/workspace'),
  projects: () => request<{ projects: ProjectSummary[]; defaultStoragePath: string }>('/projects'),
  createProject: (input: ProjectInput) =>
    request<ProjectSummary>('/projects', { method: 'POST', body: JSON.stringify(input) }),
  openProject: (path: string) =>
    request<ProjectSummary>('/projects/open', { method: 'POST', body: JSON.stringify({ path }) }),
  projectWorkspace: (id: string) =>
    request<Workspace>(`/projects/${encodeURIComponent(id)}/workspace`),
  updateProject: (id: string, input: { config: InitialConfig }) =>
    request<ProjectSummary>(`/projects/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(input),
    }),
  system: () => request<SystemStatus>('/system'),
  jobs: () => request<Jobs>('/jobs'),
  roots: (purpose: 'data' | 'storage' = 'data') =>
    request<{ roots: FilesystemRoot[] }>(
      `/filesystem/roots?purpose=${purpose === 'data' ? 'source' : purpose}`,
    ),
  directory: (path: string, purpose: 'data' | 'storage' = 'data') =>
    request<DirectoryListing>(
      `/filesystem/list?path=${encodeURIComponent(path)}&purpose=${purpose === 'data' ? 'source' : purpose}`,
    ),
  createDirectory: (parentPath: string, name: string, purpose: 'data' | 'storage' = 'data') =>
    request<{ path: string; name: string; parent: string }>('/filesystem/directories', {
      method: 'POST',
      body: JSON.stringify({ parentPath, name, purpose: purpose === 'data' ? 'source' : purpose }),
    }),
  addSource: (path: string, projectId?: string, role: Source['role'] = 'slides') =>
    request<Source>(projectId ? `/projects/${encodeURIComponent(projectId)}/sources` : '/sources', {
      method: 'POST',
      body: JSON.stringify(projectId ? { path, role } : { path }),
    }),
  saveCohort: (input: CohortInput) =>
    request<Cohort>('/cohorts', { method: 'POST', body: JSON.stringify(input) }),
  createExperiments: (input: ExperimentInput) =>
    request<{ drafts: Experiment[] }>('/experiments', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  deleteExperiment: (id: string) =>
    request<void>(`/experiments/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  experimentManifest: (id: string) =>
    request<Record<string, unknown>>(`/experiments/${encodeURIComponent(id)}/manifest`),
};
