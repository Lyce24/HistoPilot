import { canonicalHash as applyCanonicalHash } from './applyRoutes';

/** Experimental Setup and its earlier names; an experiment's design now lives in Experiments. */
const SETUP_PAGES = new Set(['experimental-setup', 'setup', 'setups', 'experiment-setup']);

/**
 * The canonical hash for a link saved by a module that was merged into another; every other
 * hash is returned unchanged. Experimental Setup became the design steps of an experiment in
 * Experiments and keeps its parameters (the experiment, its tab and the prepared inputs a link
 * carries). The modules Apply models replaced are rewritten by `applyRoutes`.
 */
export function canonicalHash(hash: string) {
  const [path, search = ''] = hash.replace(/^#/, '').split('?');
  if (SETUP_PAGES.has(path)) return `#experiments${search ? `?${search}` : ''}`;
  return applyCanonicalHash(hash);
}
