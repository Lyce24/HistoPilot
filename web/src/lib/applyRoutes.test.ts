import { describe, expect, it } from 'vitest';
import { applyHref, canonicalHash, readApplyLink } from './applyRoutes';

describe('Apply models links', () => {
  it('builds canonical links in a stable parameter order', () => {
    expect(applyHref()).toBe('#apply');
    expect(applyHref({ run: 'r/1', tab: 'performance' })).toBe('#apply?run=r%2F1&tab=performance');
    expect(applyHref({ cohort: 'c', predictor: 'p', experiment: 'e', view: 'new' })).toBe('#apply?view=new&experiment=e&predictor=p&cohort=c');
  });

  it('reads links, accepting the earlier run parameter and refusing unknown views and tabs', () => {
    expect(readApplyLink(new URLSearchParams('evaluation=r&tab=cases'))).toMatchObject({ run: 'r', tab: 'cases' });
    expect(readApplyLink(new URLSearchParams('view=elsewhere&tab=metrics'))).toMatchObject({ view: undefined, tab: undefined });
    expect(readApplyLink(new URLSearchParams('view=cohorts&newCohort=unlabeled'))).toMatchObject({ view: 'cohorts', newCohort: 'unlabeled' });
  });

  it('rewrites every link of the modules Apply models replaced', () => {
    expect(canonicalHash('#evaluation?evaluation=r')).toBe('#apply?run=r');
    expect(canonicalHash('#inference?evaluation=r')).toBe('#apply?run=r');
    expect(canonicalHash('#inference?batch=b')).toBe('#apply?batch=b');
    expect(canonicalHash('#evaluate-models?experiment=e&predictor=p')).toBe('#apply?view=new&experiment=e&predictor=p');
    expect(canonicalHash('#evaluation?cohort=c')).toBe('#apply?view=new&cohort=c');
    expect(canonicalHash('#run-inference')).toBe('#apply');
    expect(canonicalHash('#test-data')).toBe('#apply?view=cohorts');
    expect(canonicalHash('#test-data?purpose=inference')).toBe('#apply?view=cohorts&newCohort=unlabeled');
    expect(canonicalHash('#clinical-utility?clinical=a')).toBe('#apply?clinical=a');
    expect(canonicalHash('#clinical-utility?evaluation=r')).toBe('#apply?run=r&tab=performance');
    expect(canonicalHash('#clinical')).toBe('#apply');
    expect(canonicalHash('#clinical-utility?predictor=p')).toBe('#apply?view=runs&predictor=p');
  });

  it('keeps a synthetic walkthrough position through the rewrite', () => {
    expect(canonicalHash('#evaluation?record=blca-evaluation&step=metrics')).toBe('#apply?record=blca-evaluation&step=metrics');
    expect(canonicalHash('#test-data?record=r')).toBe('#apply?view=cohorts&record=r');
  });

  it('leaves every other hash alone', () => {
    for (const hash of ['', '#overview', '#apply?run=r', '#experiments?experiment=e&tab=predictors', '#task-center?owner=o']) expect(canonicalHash(hash)).toBe(hash);
  });
});
