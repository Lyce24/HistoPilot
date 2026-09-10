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
let session: Promise<string> | null = null;
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
async function responseError(response: Response): Promise<ApiError> {
  let message = `The control service returned HTTP ${response.status}.`;
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') message = body.detail;
    else if (Array.isArray(body.detail))
      message = body.detail
        .map((item: { msg?: string }) => item.msg ?? 'Invalid request')
        .join('; ');
  } catch {
    /* A proxy may return a non-JSON error page. */
  }
  return new ApiError(message, response.status);
}
function sessionToken(): Promise<string> {
  if (!session) {
    session = fetch(`${BASE}/session`, { credentials: 'same-origin', cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) throw await responseError(response);
        const body: { token: string } = await response.json();
        if (!body.token) throw new ApiError('The service did not return a session token.', 401);
        return body.token;
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
): Promise<T> {
  const token = await sessionToken();
  const headers = new Headers(init.headers);
  headers.set('X-HistoPilot-Token', token);
  if (init.body) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    credentials: 'same-origin',
    cache: 'no-store',
  });
  if (response.status === 401 && retrySession) {
    session = null;
    return request<T>(path, init, false);
  }
  if (!response.ok) throw await responseError(response);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
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
