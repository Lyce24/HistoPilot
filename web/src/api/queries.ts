import { useQuery } from '@tanstack/react-query';
import { api, ApiError } from './client';
export const workspaceKey = ['workspace'] as const;
export function useWorkspace(projectId: string) {
  return useQuery({
    queryKey: [...workspaceKey, projectId],
    queryFn: () => api.projectWorkspace(projectId),
  });
}
export function useSystem() {
  return useQuery({ queryKey: ['system'], queryFn: api.system });
}
export function useSystemCompute(live = true) {
  return useQuery({
    queryKey: ['system', 'compute'],
    queryFn: ({ signal }) => api.systemCompute(signal),
    staleTime: 4000,
    refetchInterval: (query) => live && !(query.state.error instanceof ApiError && query.state.error.status === 404) ? 5000 : false,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: live,
    retry: (attempt, error) => !(error instanceof ApiError && error.status === 404) && attempt < 1,
  });
}
