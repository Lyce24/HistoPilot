import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import WorkspaceErrorBoundary, { WorkspaceRecovery } from './WorkspaceErrorBoundary';
import { PageLoadError } from './LazyPage';

describe('workspace render recovery', () => {
  it.each([null, undefined, 'display failed', 0])('normalizes an unexpected thrown value: %s', (value) => {
    expect(WorkspaceErrorBoundary.getDerivedStateFromError(value).error).toBeInstanceOf(Error);
  });
  it('renders healthy content unchanged', () => {
    expect(renderToStaticMarkup(<WorkspaceErrorBoundary><p>Project roadmap</p></WorkspaceErrorBoundary>))
      .toBe('<p>Project roadmap</p>');
  });
  it('offers recovery, explains unsaved entries and escapes error details', () => {
    const html = renderToStaticMarkup(<WorkspaceRecovery error={new Error('<script>bad data</script>')} onRetry={() => {}} onExit={() => {}} />);
    expect(html).toContain('role="alert"');
    expect(html).toContain('Try this view again');
    expect(html).toContain('Back to start');
    expect(html).toContain('Unsaved entries');
    expect(html).toContain('Running jobs continue independently');
    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;');
  });
  it('distinguishes a failed page download and offers recovery after an application update', () => {
    const html = renderToStaticMarkup(<WorkspaceRecovery error={new PageLoadError(new Error('Failed to fetch dynamically imported module'))} onRetry={() => {}} onExit={() => {}} />);
    expect(html).toContain('This page could not be loaded');
    expect(html).toContain('Check your connection and try again');
    expect(html).toContain('Reload application');
    expect(html).toContain('Try this view again');
    expect(html).toContain('Back to start');
    expect(html).not.toContain('Unsaved entries');
  });
});
