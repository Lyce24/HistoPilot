import { useQuery } from '@tanstack/react-query';
import { api } from './client';
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
export function useJobs() {
  return useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 15000 });
}
