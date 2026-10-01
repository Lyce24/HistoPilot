/**
 * Apply models is the one place predictors meet cohorts: runs, batches and cohorts, labeled
 * or not. It replaced four modules (Evaluate models, Run inference, Test cohorts and
 * Clinical utility); links saved by them, including Task Center and bookmark links, are
 * rewritten to the canonical `#apply?...` form rather than kept as parallel routes.
 */

export type ApplyView = 'runs' | 'batches' | 'cohorts' | 'methods';
export type RunTab = 'performance' | 'agreement' | 'predictions' | 'cases' | 'compare';
export const RUN_TABS: readonly RunTab[] = ['performance', 'agreement', 'predictions', 'cases', 'compare'];

export interface ApplyLink {
  /** Open one run on a tab, scored against a reference standard instead of its default labels. */
  run?: string; tab?: RunTab; reference?: string;
  /** Open one batch of runs. */
  batch?: string;
  /** A library view, or `new` to start applying predictors (with the context below). */
  view?: ApplyView | 'new';
  /** With `new`, what to apply and where; on the runs view, whose runs to list. */
  experiment?: string; predictor?: string; cohort?: string;
  /** Start a new cohort; `unlabeled` starts one that reads no labels. */
  newCohort?: 'labeled' | 'unlabeled';
  /** Open a saved clinical-utility analysis inside its run. */
  clinical?: string;
}

const ORDER: readonly (keyof ApplyLink)[] = ['view', 'run', 'tab', 'reference', 'batch', 'clinical', 'experiment', 'predictor', 'cohort', 'newCohort'];

export function applyHref(link: ApplyLink = {}) {
  const query = new URLSearchParams();
  for (const key of ORDER) if (link[key]) query.set(key, String(link[key]));
  return `#apply${query.size ? `?${query}` : ''}`;
}

export function readApplyLink(parameters: URLSearchParams): ApplyLink {
  const text = (key: string) => parameters.get(key) || undefined;
  const view = text('view');
  const tab = text('tab');
  const newCohort = text('newCohort');
  return {
    // `evaluation=` is the earlier name of a run link.
    run: text('run') ?? text('evaluation'),
    tab: RUN_TABS.includes(tab as RunTab) ? tab as RunTab : undefined,
    reference: text('reference'),
    batch: text('batch'),
    view: view === 'new' || ['runs', 'batches', 'cohorts', 'methods'].includes(view ?? '') ? view as ApplyLink['view'] : undefined,
    experiment: text('experiment'), predictor: text('predictor'), cohort: text('cohort'),
    newCohort: newCohort === 'labeled' || newCohort === 'unlabeled' ? newCohort : undefined,
    clinical: text('clinical'),
  };
}

const RUN_PAGES = new Set(['evaluation', 'evaluate-models', 'reports', 'inference', 'predict', 'run-inference']);
const COHORT_PAGES = new Set(['test-data', 'test-cohorts']);
const CLINICAL_PAGES = new Set(['clinical-utility', 'clinical']);

function replacement(path: string, parameters: URLSearchParams): string | null {
  const text = (key: string) => parameters.get(key) || undefined;
  const context = { experiment: text('experiment'), predictor: text('predictor'), cohort: text('cohort') };
  if (RUN_PAGES.has(path)) {
    const run = text('evaluation'), batch = text('batch');
    if (run) return applyHref({ run });
    if (batch) return applyHref({ batch });
    return context.experiment || context.predictor || context.cohort ? applyHref({ view: 'new', ...context }) : applyHref();
  }
  if (COHORT_PAGES.has(path)) return applyHref({ view: 'cohorts', newCohort: text('purpose') === 'inference' ? 'unlabeled' : undefined });
  if (CLINICAL_PAGES.has(path)) {
    const clinical = text('clinical'), run = text('evaluation');
    if (clinical) return applyHref({ clinical });
    if (run) return applyHref({ run, tab: 'performance' });
    // A predictor's clinical analyses now live in its runs.
    return context.predictor ? applyHref({ view: 'runs', predictor: context.predictor }) : applyHref();
  }
  return null;
}

/**
 * The canonical hash for one saved by a module Apply models replaced; every other hash is
 * returned unchanged. The run, batch and analysis a link named, the setup context it carried
 * (experiment, predictor, cohort) and a synthetic walkthrough position survive the rewrite.
 */
export function canonicalHash(hash: string) {
  const [path, search = ''] = hash.replace(/^#/, '').split('?');
  const parameters = new URLSearchParams(search);
  const href = replacement(path, parameters);
  if (href === null) return hash;
  const walkthrough = new URLSearchParams(['record', 'step'].flatMap((key) => parameters.get(key) ? [[key, parameters.get(key)!]] : []));
  return walkthrough.size ? `${href}${href.includes('?') ? '&' : '?'}${walkthrough}` : href;
}
