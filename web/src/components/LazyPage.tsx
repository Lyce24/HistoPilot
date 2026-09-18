import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

/** A page download failure is distinct from a crash in an already loaded editor. */
export class PageLoadError extends Error {
  constructor(cause: unknown, public readonly retryImport: () => void = () => {}) {
    super('The page could not be downloaded. Check your connection to HistoPilot and try again.', { cause });
    this.name = 'PageLoadError';
  }
}

/** Keep successful imports cached, but allow the error boundary to retry failures.
 * React.lazy itself caches rejections, so retrying the same lazy instance would
 * otherwise leave the user stuck on the recovery screen. */
export function lazyPage<Props extends object>(load: () => Promise<{ default: ComponentType<Props> }>) {
  let loaded: LazyExoticComponent<ComponentType<Props>> | null = null;
  return function LazyPage(props: Props) {
    const Component = loaded ??= lazy(() => load().catch((cause: unknown) => {
      // Keep the rejected instance until the user retries. Clearing it here
      // makes React's suspense retry start another request before the error
      // boundary can display recovery, causing an endless loading screen.
      throw new PageLoadError(cause, () => { loaded = null; });
    }));
    return <Component {...props} />;
  };
}

export function PageLoading({ name }: { name: string }) {
  return <section className="roadmap-loading" role="status" aria-live="polite" aria-busy="true">
    <h1>Loading {name}</h1>
    <p>Opening this page. Saved records and running jobs remain available.</p>
  </section>;
}
