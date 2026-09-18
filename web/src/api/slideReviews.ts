import { request } from './client';

export type ReviewStatus = 'unreviewed' | 'accept' | 'exclude' | 'review';
export interface ReviewRegion { id: string; label: string; x: number; y: number; width: number; height: number }
export interface SlideReviewValues {
  status: ReviewStatus; notes: string; reviewer: string; reasons: string[]; regions: ReviewRegion[]; evaluationId: string | null;
}
export interface ReviewRevision extends SlideReviewValues { revision: number; updatedAt: string }
export interface SlideReview extends SlideReviewValues {
  schemaVersion: 1; datasetId: string; slideId: string; revision: number; updatedAt: string | null; history: ReviewRevision[];
}
export interface ReviewPage { items: Omit<SlideReview, 'history'>[]; total: number; offset: number; hasMore: boolean }
const base = (project: string, dataset: string) => `/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/slide-reviews`;
export const slideReviews = {
  get: (project: string, dataset: string, slide: string, signal?: AbortSignal) => request<SlideReview>(`${base(project, dataset)}/${encodeURIComponent(slide)}`, { signal }),
  list: (project: string, dataset: string, offset = 0, signal?: AbortSignal) => request<ReviewPage>(`${base(project, dataset)}?${new URLSearchParams({ offset: String(offset), limit: '200' })}`, { signal }),
  save: (project: string, dataset: string, slide: string, values: SlideReviewValues & { expectedRevision: number }) => request<SlideReview>(`${base(project, dataset)}/${encodeURIComponent(slide)}`, { method: 'PUT', body: JSON.stringify(values) }),
};
export const reviewStatusLabels: Record<ReviewStatus, string> = { unreviewed: 'Not reviewed', accept: 'Accept', exclude: 'Recommend exclusion', review: 'Needs review' };
