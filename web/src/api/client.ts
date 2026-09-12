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
  SystemCompute,
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
/** Transport failures do not prove server rejection. Keep them distinct from
 * ApiError so reviewed mutation UIs retain their original retry operation ID. */
class ServiceResponseError extends Error {
  constructor(message: string, public readonly status: number, public readonly code: string) {
    super(message);
    this.name = 'ServiceResponseError';
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
        .map((item: { msg?: string; loc?: (string | number)[] } | null) => {
          const field = item?.loc?.filter((part) => part !== 'body').join('.');
          return `${field ? `${field}: ` : ''}${item?.msg ?? 'Invalid request'}`;
        })
        .join('; ');
  } catch {
    /* A proxy may return a non-JSON error page. */
  }
  return new ApiError(message, response.status, code);
}
async function serviceFetch(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(`${BASE}${path}`, {
      ...init, credentials: 'same-origin', cache: 'no-store',
    });
  } catch (error) {
    if (init.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) throw error;
    const mutation = !['GET', 'HEAD'].includes((init.method ?? 'GET').toUpperCase());
    throw new ServiceResponseError(
      'Could not reach HistoPilot. Check that the server is running and your connection is available.'
        + (mutation ? ' The request may have reached the server; check the saved record or job status before retrying.' : ''),
      0, 'SERVICE_UNREACHABLE',
    );
  }
}
async function responseJSON<T>(response: Response, init: RequestInit = {}): Promise<T> {
  try {
    return await response.json() as T;
  } catch (error) {
    if (init.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) throw error;
    const mutation = !['GET', 'HEAD'].includes((init.method ?? 'GET').toUpperCase());
    throw new ServiceResponseError(
      'HistoPilot returned an unreadable response. Check the server log and connection.'
        + (mutation ? ' The request may have reached the server; check the saved record or job status before retrying.' : ''),
      response.status, 'INVALID_SERVICE_RESPONSE',
    );
  }
}
function sessionDetails(): Promise<Session> {
  if (!session) {
    const pending = serviceFetch('/session', {})
      .then(async (response) => {
        if (!response.ok) throw await responseError(response);
        const body = await responseJSON<Session | null>(response);
        if (typeof body?.token !== 'string' || !body.token.trim()) {
          throw new ApiError('The service did not return a valid session token.', 401);
        }
        return body;
      })
      .catch((error) => {
        if (session === pending) session = null;
        throw error;
      });
    session = pending;
  }
  return session;
}
async function authenticatedResponse(
  path: string,
  init: RequestInit = {},
  retrySession = true,
  scientificCapability?: ScientificCapability,
): Promise<Response> {
  init.signal?.throwIfAborted();
  const pending = sessionDetails();
  const activeSession = await pending;
  init.signal?.throwIfAborted();
  if (scientificCapability && !activeSession.scientificCapabilities?.[scientificCapability]) {
    if (session === pending) session = null;
    throw new ApiError(SCIENTIFIC_SAVE_RESTART, 405);
  }
  const headers = new Headers(init.headers);
  headers.set('X-HistoPilot-Token', activeSession.token);
  if (init.body) headers.set('Content-Type', 'application/json');
  const response = await serviceFetch(path, { ...init, headers });
  if (response.status === 401 && retrySession) {
    // A late 401 from another request must not discard an already renewed session.
    if (session === pending) session = null;
    return authenticatedResponse(path, init, false, scientificCapability);
  }
  if (!response.ok) throw await responseError(response);
  return response;
}
/** Always use the service session; no scientific fallback data lives in the browser. */
export async function request<T>(
  path: string,
  init: RequestInit = {},
  retrySession = true,
  scientificCapability?: ScientificCapability,
): Promise<T> {
  const response = await authenticatedResponse(path, init, retrySession, scientificCapability);
  if (response.status === 204) return undefined as T;
  return responseJSON<T>(response, init);
}
const SCIENTIFIC_SAVE_RESTART = 'The running HistoPilot server does not support this save. Restart HistoPilot, then try saving again here. Your entered tag and note have been kept; you do not need to reload this page.';

/** Load viewer images with the same session authentication as JSON and downloads. */
export async function fetchArtifactBlob(path: string, signal?: AbortSignal, retrySession = true): Promise<Blob> {
  const response = await authenticatedResponse(path, { signal }, retrySession);
  return response.blob();
}

/** Download a verified artifact using the same session authentication as JSON APIs. */
export async function downloadArtifact(path: string, filename: string, retrySession = true): Promise<void> {
  const response = await authenticatedResponse(path, {}, retrySession);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

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
  systemCompute: (signal?: AbortSignal) => request<SystemCompute>('/system/compute', { signal }),
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
