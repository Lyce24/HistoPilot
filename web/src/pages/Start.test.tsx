import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Start, { savedProjectsByState } from './Start';
import type { ProjectSummary } from '../api/types';

const projects = [
  { id: 'active', name: 'Current project', mode: 'local', storagePath: '/active', createdAt: '2026-09-12', available: true },
  { id: 'archive', name: 'Earlier project', mode: 'local', lifecycleState: 'archived', storagePath: '/archive' },
  { id: 'trash', name: 'Removed project', mode: 'local', lifecycleState: 'trashed', storagePath: '/trash' },
  { id: 'demo', name: 'Synthetic example', mode: 'synthetic-demo' },
] as ProjectSummary[];

describe('saved project lifecycle views', () => {
  it('treats older projects without lifecycle metadata as active and excludes demo records', () => {
    expect(savedProjectsByState(projects, 'active').map((project) => project.id)).toEqual(['active']);
    expect(savedProjectsByState(projects, 'archived').map((project) => project.id)).toEqual(['archive']);
    expect(savedProjectsByState(projects, 'trashed').map((project) => project.id)).toEqual(['trash']);
  });

  it('shows state filters and a cleanup entry without mixing archived names into the active list', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['projects'], { projects, defaultStoragePath: '/projects' });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><Start onOpen={() => {}} /></QueryClientProvider>);
    expect(html).toContain('Current project');
    expect(html).not.toContain('Earlier project');
    expect(html).not.toContain('Removed project');
    expect(html).toContain('aria-label="Project state"');
    expect(html).toContain('Archived');
    expect(html).toContain('Trash');
    expect(html).toContain('Manage cleanup for Current project');
  });
});
