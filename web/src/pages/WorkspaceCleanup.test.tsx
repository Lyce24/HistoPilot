import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import WorkspaceCleanup, { CleanupReviewPanel, filterCleanupItems, bulkCleanupSelection } from './WorkspaceCleanup';
import type { CleanupInventory, CleanupItem, CleanupPreview } from '../api/lifecycle';
import type { Workspace } from '../api/types';

const items: CleanupItem[] = [
  { key: 'project:p', type: 'project', id: 'p', kind: 'project', name: 'Colon project', state: 'active', dependsOn: [], usedBy: [] },
  { key: 'dataset:d', type: 'dataset', id: 'd', kind: 'dataset', name: 'Development slides', state: 'active', dependsOn: [], usedBy: ['configuration:b'] },
  { key: 'configuration:b', type: 'configuration', id: 'b', kind: 'mil-batch', name: 'Seed comparison', state: 'active', dependsOn: ['dataset:d'], usedBy: [], job: { status: 'running', cancellable: true } },
  { key: 'draft:old', type: 'draft', id: 'old', kind: 'protocol', name: 'Earlier target draft', state: 'trashed', dependsOn: [], usedBy: [] },
];
const preview: CleanupPreview = { action: 'trash', keys: ['dataset:d'], revision: 3, previewHash: 'hash', canApply: true, blockers: [], requiredKeys: [], recordCount: 1, note: '' };
const noOp = () => {};
function panel(changes: Partial<Parameters<typeof CleanupReviewPanel>[0]> = {}) {
  return renderToStaticMarkup(<CleanupReviewPanel preview={preview} items={items} current acknowledged={false} busy={false} uncertain={false} onAcknowledge={noOp} onApply={noOp} onInclude={noOp} onRefresh={noOp} {...changes} />);
}

describe('workspace cleanup review', () => {
  it('shows exact record names and file retention before confirmation is enabled', () => {
    const html = panel();
    expect(html).toContain('Development slides');
    expect(html).not.toContain('Seed comparison');
    expect(html).toContain('recoverable Trash');
    expect(html).toContain('model checkpoints and output files stay on disk');
    expect(html).toContain('does not free disk space');
    expect(html).toMatch(/disabled="">Move to Trash \(1\)/);
    expect(panel({ acknowledged: true })).not.toMatch(/disabled="">Move to Trash \(1\)/);
  });

  it('lists blockers and requires an explicit choice to include dependent records', () => {
    const html = panel({ preview: { ...preview, canApply: false, requiredKeys: ['configuration:b'], blockers: [{ code: 'DEPENDENCY', message: 'The batch still uses these data', keys: ['configuration:b'] }] } });
    expect(html).toContain('The batch still uses these data');
    expect(html).toContain('Seed comparison');
    expect(html).toContain('These records have not been added to your selection');
    expect(html).toContain('Include required records and review again');
    expect(html).not.toContain('I reviewed these records');
  });

  it('cannot confirm a stale or changed review, even after acknowledgement', () => {
    const html = panel({ current: false, acknowledged: true });
    expect(html).toContain('project or selection changed');
    expect(html).not.toContain('Move to Trash (1)');
  });

  it('offers an exact-operation retry when the confirmation result is uncertain', () => {
    const html = panel({ uncertain: true, current: false, acknowledged: true });
    expect(html).toContain('change may already have been applied');
    expect(html).toContain('exact same reviewed request and operation ID');
    expect(html).toContain('Retry this confirmation');
    expect(html).toContain('Refresh records and review again');
    expect(html).not.toContain('I reviewed these records');
  });

  it('distinguishes whole-project visibility from child record changes', () => {
    expect(panel({ preview: { ...preview, keys: ['project:p'] } })).toContain('Its child records keep their current states');
  });

  it('filters by state, kind and searchable name without changing the inventory', () => {
    expect(filterCleanupItems(items, 'active', 'mil-batch', 'seed').map((item) => item.key)).toEqual(['configuration:b']);
    expect(filterCleanupItems(items, 'trashed', '', '')).toHaveLength(1);
    expect(filterCleanupItems(items, 'archived', '', '')).toEqual([]);
    expect(items).toHaveLength(4);
  });

  it('loads Active, Archived and Trash views and exposes cancellation for active jobs', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(['cleanup', 'p'], { projectId: 'p', revision: 3, projectState: 'active', items, audit: [], note: '' } satisfies CleanupInventory);
    const workspace = { mode: 'local', project: { id: 'p', lifecycleState: 'active' } } as Workspace;
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><WorkspaceCleanup workspace={workspace} /></QueryClientProvider>);
    expect(html).toContain('Workspace cleanup');
    expect(html).toContain('Archived');
    expect(html).toContain('Trash');
    expect(html).toContain('Cancel job');
    expect(html).toContain('Seed comparison');
    expect(html).not.toContain('Earlier target draft');
    expect(html).toContain('0 selected');
    expect(html).toMatch(/disabled="">Review selected changes/);
  });

  it('keeps whole-project changes separate from bulk child record selections', () => {
    expect(bulkCleanupSelection(['project:p'], items.slice(0, 3))).toEqual(['dataset:d', 'configuration:b']);
    expect(bulkCleanupSelection(['dataset:d'], [items[0]])).toEqual(['project:p']);
  });
});
