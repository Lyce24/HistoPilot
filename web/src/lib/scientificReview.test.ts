import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import { scientificReviewInvalidated } from './scientificReview';

describe('scientific freeze recovery', () => {
  it.each(['PREVIEW_STALE', 'STALE_PREVIEW', 'IMPORT_BLOCKED', 'PROTOCOL_PREFLIGHT_BLOCKED', 'FEATURE_BUNDLE_INVALID', 'REVISION_CONFLICT', 'DRAFT_FROZEN'])(
    'requires a new review after %s', (code) => {
      expect(scientificReviewInvalidated(new ApiError('Inputs changed', 409, code))).toBe(true);
    },
  );
  it('preserves a reviewed request for retryable naming and connection failures', () => {
    for (const error of [new ApiError('Tag is taken', 409, 'VERSION_TAG_CONFLICT'), new ApiError('Already saved', 409, 'VERSION_LABEL_CONFLICT'), new ApiError('Unavailable', 503), new Error('Offline')]) {
      expect(scientificReviewInvalidated(error)).toBe(false);
    }
  });
});
