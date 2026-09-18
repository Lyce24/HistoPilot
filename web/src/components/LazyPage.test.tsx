import { Suspense, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { renderLoadedPage } from '../testFixtures/renderLoadedPage';
import { describe, expect, it, vi } from 'vitest';
import { lazyPage, PageLoadError, PageLoading } from './LazyPage';

describe('on-demand page loading', () => {
  it('waits until the page is selected and reuses successful imports', async () => {
    const load = vi.fn(async () => ({ default: ({ title }: { title: string }) => <h1>{title}</h1> }));
    const Page = lazyPage(load);
    expect(load).not.toHaveBeenCalled();
    expect(await renderLoadedPage(<Suspense fallback={<PageLoading name="Dataset" />}><Page title="Dataset ready" /></Suspense>)).toContain('Dataset ready');
    expect(await renderLoadedPage(<Suspense fallback={<PageLoading name="Dataset" />}><Page title="Other dataset" /></Suspense>)).toContain('Other dataset');
    expect(load).toHaveBeenCalledTimes(1);
  });

  it('announces the current page while its import is pending', () => {
    const Page = lazyPage(() => new Promise<{ default: () => ReactNode }>(() => {}));
    const html = renderToStaticMarkup(<Suspense fallback={<PageLoading name="Slide features" />}><Page /></Suspense>);
    expect(html).toContain('Loading Slide features');
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-busy="true"');
  });

  it('retries a rejected import instead of rethrowing a cached React.lazy rejection', async () => {
    const cause = new Error('The page download failed');
    const load = vi.fn<() => Promise<{ default: () => ReactNode }>>()
      .mockRejectedValueOnce(cause)
      .mockResolvedValue({ default: () => <h1>Recovered page</h1> });
    const Page = lazyPage(load);
    const errors: unknown[] = [];
    const tree = <Suspense fallback={<PageLoading name="Dataset" />}><Page /></Suspense>;
    await renderLoadedPage(tree, (error) => { errors.push(error); });
    expect(errors).toHaveLength(1);
    expect(errors[0]).toBeInstanceOf(PageLoadError);
    expect((errors[0] as Error).cause).toBe(cause);
    (errors[0] as PageLoadError).retryImport();
    expect(await renderLoadedPage(tree)).toContain('Recovered page');
    expect(load).toHaveBeenCalledTimes(2);
  });
});
