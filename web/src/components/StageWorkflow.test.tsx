import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { StageLibrary, StageLibraryToolbar, StagePage, StageRecordManageButton, StageSteps } from './StageWorkflow';

function renderLibrary(children: ReactNode) {
  const client = new QueryClient();
  try { return renderToStaticMarkup(<QueryClientProvider client={client}>{children}</QueryClientProvider>); }
  finally { client.clear(); }
}

describe('shared stage workflow', () => {
  it('identifies the active page and disables unavailable steps without implying completion', () => {
    const html = renderToStaticMarkup(<StageSteps current="mapping" onChange={() => {}} steps={[
      { id: 'source', title: 'Source', complete: true },
      { id: 'mapping', title: 'Map columns' },
      { id: 'review', title: 'Review', disabled: true },
    ]} />);
    expect(html.match(/aria-current="step"/g)).toHaveLength(1);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>[\s\S]*?<strong>Review<\/strong>/);
    expect(html).toContain('is-complete');
    expect(html).toContain('<nav');
    expect(html).not.toContain('role="tab"');
  });

  it('locks every step during a protected operation', () => {
    const html = renderToStaticMarkup(<StageSteps current="review" disabled onChange={() => {}} steps={[
      { id: 'inputs', title: 'Inputs' }, { id: 'review', title: 'Review' },
    ]} />);
    expect(html.match(/disabled=""/g)).toHaveLength(2);
  });

  it('provides a labelled stage-zero library and a focusable page surface', () => {
    const html = renderLibrary(<StagePage pageKey="library"><StageLibrary project="p" title="Datasets"><p>No datasets yet.</p></StageLibrary></StagePage>);
    expect(html).not.toContain('Stage 0 · Saved records');
    expect(html).not.toContain('<h2');
    expect(html).toContain('aria-label="Datasets"');
    expect(html).toContain('tabindex="-1"');
    expect(html).toContain('No datasets yet.');
  });

  it('labels search and filter results and exposes reset only when supplied', () => {
    const toolbar = (reset = false) => renderToStaticMarkup(<StageLibraryToolbar search="cohort" onSearch={() => {}} searchLabel="Search datasets" count={1} total={8} onReset={reset ? () => {} : undefined}><label>Status<select><option>All</option></select></label></StageLibraryToolbar>);
    expect(toolbar()).toContain('type="search"');
    expect(toolbar()).toContain('aria-label="Search datasets"');
    expect(toolbar()).toContain('1 of 8 records');
    expect(toolbar()).toContain('aria-live="polite"');
    expect(toolbar()).not.toContain('Clear filters');
    expect(toolbar(true)).toContain('Clear filters');
  });

  it('opens local management for the exact record type without linking to cleanup', () => {
    for (const type of ['dataset', 'draft', 'configuration'] as const) {
      const html = renderLibrary(<StageLibrary project="p" title="Datasets"><StageRecordManageButton type={type} id="same-id&next=other" name="Baseline" /></StageLibrary>);
      expect(html).toContain(`data-record-key="${type}:same-id&amp;next=other"`);
      expect(html).toContain('aria-label="Manage Baseline"');
      expect(html).not.toContain('href=');
      expect(html).not.toContain('disabled=');
    }
  });
});
