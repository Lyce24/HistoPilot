import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Start, { RECENT_PAGE_SIZE, recentProjectPage, savedProjectsByState } from './Start';
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
    expect(html).not.toContain('Recent project pages');
  });
});

describe('recent project pages', () => {
  const many = Array.from({ length: 12 }, (_, index) => ({
    id: `p${index + 1}`,
    name: `Project ${index + 1}`,
    mode: 'local',
    storagePath: `/p${index + 1}`,
    available: true,
  })) as ProjectSummary[];

  it('splits a list into pages of five and clamps a page past the end', () => {
    expect(RECENT_PAGE_SIZE).toBe(5);
    const first = recentProjectPage(many, 0);
    expect(first.items.map((project) => project.id)).toEqual(['p1', 'p2', 'p3', 'p4', 'p5']);
    expect(first.pageCount).toBe(3);
    expect(recentProjectPage(many, 2).items.map((project) => project.id)).toEqual(['p11', 'p12']);
    expect(recentProjectPage(many, 9).page).toBe(2);
    expect(recentProjectPage(many, -1).page).toBe(0);
    expect(recentProjectPage([], 0)).toEqual({ items: [], page: 0, pageCount: 1 });
  });

  it('shows the first five recent projects with page controls', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['projects'], { projects: many, defaultStoragePath: '/projects' });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><Start onOpen={() => {}} /></QueryClientProvider>);
    expect(html).toContain('Project 5');
    expect(html).not.toContain('Project 6');
    expect(html).toContain('aria-label="Recent project pages"');
    expect(html).toContain('Page 1 of 3');
    expect(html).toContain('aria-current="page"');
    expect(html).toContain('View all');
  });
});
