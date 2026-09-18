import { useLayoutEffect, useRef } from 'react';

const guards = new Map<symbol, () => string | null>();
let acceptedUrl = '';
let permittedUrl = '';

function scope(href: string) {
  const url = new URL(href);
  const [module, query] = url.hash.split('?');
  return JSON.stringify([url.origin, url.pathname, url.searchParams.get('project'), module, new URLSearchParams(query).get('experiment')]);
}

/** replaceState does not emit a route event; remember intentional in-record changes. */
export function rememberWorkspaceLocation() { if (guards.size) acceptedUrl = window.location.href; }

export function confirmWorkspaceNavigation(): boolean {
  const messages = [...new Set([...guards.values()].map((read) => read()).filter(Boolean))];
  return !messages.length || window.confirm(`${messages.join('\n\n')}\n\nLeave anyway?`);
}

function click(event: MouseEvent) {
  if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const anchor = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
  if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download')) return;
  const destination = new URL(anchor.href);
  if (destination.origin === window.location.origin && destination.pathname === window.location.pathname
    && destination.search === window.location.search && document.getElementById(destination.hash.slice(1))) return;
  if (scope(anchor.href) === scope(window.location.href)) return;
  if (!confirmWorkspaceNavigation()) { event.preventDefault(); event.stopImmediatePropagation(); }
  else permittedUrl = anchor.href;
}

function route(event: Event) {
  const next = window.location.href;
  if (next === acceptedUrl) return;
  const permitted = next === permittedUrl || scope(next) === scope(acceptedUrl);
  permittedUrl = '';
  if (!permitted && !confirmWorkspaceNavigation()) {
    window.history.replaceState(window.history.state, '', acceptedUrl);
    event.stopImmediatePropagation();
    return;
  }
  acceptedUrl = next;
}

function unload(event: BeforeUnloadEvent) {
  if (![...guards.values()].some((read) => read())) return;
  event.preventDefault();
  event.returnValue = '';
}

// Install before router listeners. A router can synchronously unmount its editor
// during a window event; registering only after that editor mounts is too late.
if (typeof window !== 'undefined') {
  acceptedUrl = window.location.href;
  document.addEventListener('click', click, true);
  window.addEventListener('hashchange', route, true);
  window.addEventListener('popstate', route, true);
  window.addEventListener('beforeunload', unload);
}
if (import.meta.hot) import.meta.hot.dispose(() => {
  document.removeEventListener('click', click, true);
  window.removeEventListener('hashchange', route, true);
  window.removeEventListener('popstate', route, true);
  window.removeEventListener('beforeunload', unload);
  guards.clear();
});

/** Protect requests and edits that could not be backed up; recoverable drafts navigate freely. */
export function useWorkspaceNavigationGuard(message: string | null) {
  const latest = useRef(message);
  latest.current = message;
  useLayoutEffect(() => {
    const id = Symbol('workspace-navigation');
    if (guards.size === 0) {
      acceptedUrl = window.location.href;
    }
    guards.set(id, () => latest.current);
    return () => {
      guards.delete(id);
      if (guards.size === 0) {
        permittedUrl = '';
      }
    };
  }, []);
}
