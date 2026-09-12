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
export const cleanupLink = (id: string) => `#cleanup?key=${encodeURIComponent(`configuration:${id}`)}`;
