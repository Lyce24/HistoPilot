import { useEffect, useState } from 'react';

export function readHashParameters() {
  return new URLSearchParams(typeof window === 'undefined' ? '' : window.location.hash.split('?')[1] ?? '');
}
export function useHashParameters() {
  const [parameters, setParameters] = useState(readHashParameters);
  useEffect(() => {
    const update = () => setParameters(readHashParameters());
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, []);
  return parameters;
}
/**
 * The hash parameters and a count of hash navigations, so a page can tell a link to the view
 * it already shows (a new visit) from a re-render.
 */
export function useHashVisit() {
  const [state, setState] = useState(() => ({ parameters: readHashParameters(), visit: 0 }));
  useEffect(() => {
    const update = () => setState((current) => ({ parameters: readHashParameters(), visit: current.visit + 1 }));
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, []);
  return state;
}
export const cleanupLink = (id: string) => `#cleanup?key=${encodeURIComponent(`configuration:${id}`)}`;
