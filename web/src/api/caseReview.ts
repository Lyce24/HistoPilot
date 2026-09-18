import { downloadRequestedArtifact, request } from './client';
import type { SlideReview } from './slideReviews';

type CaseAttribute = string | number | boolean | null;

export interface CaseQuery {
  unit: 'selected' | 'slide' | 'patient'; outcome: 'all' | 'error' | 'false_positive' | 'false_negative' | 'correct' | 'unlabeled' | 'disagreement';
  comparisonId: string | null; actualClass: number | null; predictedClass: number | null;
  search: string; attribute: string | null; attributeValue: string | null; minConfidence: number; offset: number; limit: number;
}
export interface CaseSlide {
  datasetId: string; slideId: string; slidePath: string | null; hasImage: boolean; attributes: Record<string, CaseAttribute>; review: Omit<SlideReview, 'history'>;
}
export interface ReviewedCase {
  id: string; patientId: string | null; slideIds: string[]; label: string | null; labelIndex: number | null;
  probabilities: number[]; predictedIndex: number; predictedLabel: string; confidence: number; outcome: string;
  comparison: { probabilities: number[]; predictedIndex: number; predictedLabel: string; disagrees: boolean } | null;
  attributes: Record<string, CaseAttribute[]>; slides: CaseSlide[];
}
export interface CasePage {
  supportsAttention?: boolean;
  evaluationId: string; name: string; predictorId: string; cohortId: string; featureBundleId: string | null; classOrder: string[]; positiveClass: string | null;
  decisionThreshold: number; unit: 'slide' | 'patient'; source: { predictionsSha256: string; comparisonSha256: string | null };
  comparison: { id: string; name: string; predictorId: string; decisionThreshold: number; supportsAttention?: boolean } | null;
  attributes: { key: string; label: string; values: string[]; valuesLimited: boolean }[];
  summary: Record<string, number>; items: ReviewedCase[]; total: number; offset: number; hasMore: boolean;
}
export const defaultCaseQuery = (): CaseQuery => ({ unit: 'selected', outcome: 'all', comparisonId: null, actualClass: null, predictedClass: null, search: '', attribute: null, attributeValue: null, minConfidence: 0, offset: 0, limit: 30 });
const base = (project: string, evaluation: string) => `/projects/${encodeURIComponent(project)}/evaluation-runs/${encodeURIComponent(evaluation)}/cases`;
const post = (query: CaseQuery) => ({ method: 'POST', body: JSON.stringify(query) });
export const caseReviews = {
  query: (project: string, evaluation: string, query: CaseQuery, signal?: AbortSignal) => request<CasePage>(`${base(project, evaluation)}/query`, { ...post(query), signal }),
  export: (project: string, evaluation: string, query: CaseQuery) => downloadRequestedArtifact(`${base(project, evaluation)}/export`, 'case-reviews.csv', post(query)),
};
