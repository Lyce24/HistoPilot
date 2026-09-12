import { ApiError } from '../api/client';

/** These failures invalidate scientific evidence; naming conflicts remain retryable. */
export function scientificReviewInvalidated(error: unknown): error is ApiError {
  return error instanceof ApiError && new Set([
    'PREVIEW_STALE', 'STALE_PREVIEW', 'IMPORT_BLOCKED', 'PROTOCOL_PREFLIGHT_BLOCKED',
    'FEATURE_BUNDLE_INVALID', 'EVALUATION_PREFLIGHT_BLOCKED', 'REVISION_CONFLICT', 'DRAFT_FROZEN',
  ]).has(error.code ?? '');
}
