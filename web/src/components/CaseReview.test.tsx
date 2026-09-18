import { describe, expect, it } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import type { ModelEvaluation } from '../api/predictors';
import { defaultCaseQuery, type CasePage, type CaseSlide, type ReviewedCase } from '../api/caseReview';
import { reviewStatusLabels, type SlideReview } from '../api/slideReviews';
import CaseReviewWorkspace, { CaseDetail, CaseSlideDetail, compatibleComparisons } from './CaseReviewWorkspace';
import { isReviewDraft, reviewValues } from './SlideReviewEditor';

const review: SlideReview = { schemaVersion: 1, datasetId: 'dataset', slideId: 's1', revision: 4, status: 'review', notes: 'Inspect tumor', reviewer: 'YL', reasons: ['Limited tumor'], regions: [], evaluationId: null, updatedAt: null, history: [] };
const record = (id: string, cohort = 'cohort'): ModelEvaluation => ({ id, createdAt: '', contentHash: '', lifecycleState: 'active', manifest: { kind: 'model-evaluation', status: 'planned', name: id, predictorId: id, cohortId: cohort, experimentId: 'experiment' }, execution: { status: 'completed' } });

describe('case review and editor recovery', () => {
  it('does not call an unverified slide grouping identity a patient', () => {
    const row = { id: 's1', patientId: 's1', label: 'low', predictedLabel: 'low', probabilities: [.9, .1], slides: [], attributes: {} } as unknown as ReviewedCase;
    const page = { unit: 'slide', classOrder: ['low', 'high'], name: 'Model', positiveClass: null } as CasePage;
    const html = renderToStaticMarkup(<CaseDetail project="project" record={row} page={page} />);
    expect(html).toContain('Case/group ID s1');
    expect(html).not.toContain('Patient s1');
    expect(renderToStaticMarkup(<CaseDetail project="project" record={row} page={{ ...page, unit: 'patient' }} />)).toContain('Patient s1');
  });
  it('offers completed evaluations of the same cohort without self, trash, or unfinished jobs', () => {
    const first = record('first'), other = record('other');
    expect(compatibleComparisons(first, [first, other, record('unrelated', 'different'), { ...record('trash'), lifecycleState: 'trashed' }, { ...record('running'), execution: { status: 'running' } }])).toEqual([other]);
  });

  it('keeps actual/predicted class selection and safe defaults independent of label order', () => {
    expect(defaultCaseQuery()).toMatchObject({ actualClass: null, predictedClass: null, unit: 'selected', minConfidence: 0, outcome: 'all' });
    expect(reviewValues(review, 'evaluation')).toMatchObject({ notes: 'Inspect tumor', status: 'review', evaluationId: 'evaluation' });
    expect(reviewStatusLabels.exclude).toBe('Recommend exclusion');
  });

  it('rejects corrupt recovery copies and preserves revision and annotations in valid drafts', () => {
    const valid = { expectedRevision: review.revision, values: reviewValues(review) };
    expect(isReviewDraft(valid)).toBe(true);
    expect(isReviewDraft({ ...valid, expectedRevision: -1 })).toBe(false);
    expect(isReviewDraft({ ...valid, values: { ...valid.values, notes: 'x'.repeat(8001) } })).toBe(false);
    expect(isReviewDraft({ ...valid, values: { ...valid.values, regions: [{ id: 'roi', label: '', x: 0, y: 0, width: Infinity, height: 1 }] } })).toBe(false);
  });

  it('renders explicit verified-evidence loading and labeled controls without made-up results', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><CaseReviewWorkspace project="project" evaluation={record('evaluation')} comparisons={[]} /></QueryClientProvider>);
    for (const text of ['Case and error review', 'Review unit', 'False negatives', 'Compare with', 'Verifying saved predictions']) expect(html).toContain(text);
    expect(html).not.toContain('No matching cases');
    client.clear();
  });

  it('does not promise patch attention for unsupported models but keeps review available', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const page = { predictorId: 'slide-probe', evaluationId: 'evaluation', supportsAttention: false, comparison: null } as CasePage;
    const slide = { slideId: 's1', datasetId: 'dataset', hasImage: false } as CaseSlide;
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><CaseSlideDetail project="project" slide={slide} page={page} /></QueryClientProvider>);
    expect(html).toContain('Patch attention is unavailable');
    expect(html).toContain('Slide review');
    expect(html).not.toContain('Inspect model attention');
    expect(html).not.toContain('Open interpretation workspace');
    client.clear();
  });

  it('links an attention-capable comparison to its own evaluation instead of the slide probe', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const page = { predictorId: 'slide-probe', evaluationId: 'evaluation', supportsAttention: false, comparison: { predictorId: 'attention-model', id: 'comparison-evaluation', supportsAttention: true, name: 'Comparison' } } as CasePage;
    const html = renderToStaticMarkup(<QueryClientProvider client={client}><CaseSlideDetail project="project" slide={{ slideId: 's1', datasetId: 'dataset', hasImage: false } as CaseSlide} page={page} /></QueryClientProvider>);
    expect(html).toContain('predictor=attention-model');
    expect(html).toContain('evaluation=comparison-evaluation');
    expect(html).toContain('Inspect model attention');
    client.clear();
  });
});
