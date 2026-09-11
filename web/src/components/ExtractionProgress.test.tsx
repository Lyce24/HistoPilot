import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { ExtractionJob } from '../api/trident';
import ExtractionProgress, { formatExtractionDuration } from './ExtractionProgress';

function job(overrides: Partial<ExtractionJob> = {}): ExtractionJob {
  return {
    id: 'test-run',
    state: 'running',
    createdAt: '2026-09-10T00:00:00Z',
    updatedAt: '2026-09-10T00:10:00Z',
    outputPath: '/test/output',
    logPath: '/test/log',
    sessionName: 'test-session',
    spec: { datasetId: 'dataset', outputPath: '/test/output', options: { task: 'all', patch_encoder: 'uni_v1' } },
    progress: {
      stage: 'segmentation',
      stages: [
        { id: 'preparing', label: 'Prepare', status: 'complete' },
        { id: 'segmentation', label: 'Segment tissue', status: 'active' },
        { id: 'validation', label: 'Validate outputs', status: 'pending' },
      ],
      label: 'Segmenting tissue',
      detail: 'Finding tissue regions in your slides.',
      completed: 5,
      total: 138,
      unit: 'slides',
      percent: 3.6,
      currentSlide: 'T103',
      elapsedSeconds: 181,
      stageElapsedSeconds: 165,
      etaSeconds: 123,
      ratePerSecond: 0.2,
      scope: 'stage',
      warnings: [],
    },
    ...overrides,
  };
}

describe('extraction progress presentation', () => {
  it('distinguishes run duration, current-stage elapsed time and stage ETA', () => {
    const html = renderToStaticMarkup(<ExtractionProgress job={job()} />);
    expect(html).toContain('Run time');
    expect(html).toContain('3m 1s');
    expect(html).toContain('This stage: 2m 45s');
    expect(html).toContain('Stage time remaining');
    expect(html).toContain('~2m 3s');
    expect(html).toContain('Estimate for this step only');
    expect(html).toContain('5 of 138 slides processed');
  });

  it('does not declare extraction complete when a worker reports a 100 percent stage', () => {
    const run = job();
    run.progress = { ...run.progress!, completed: 138, percent: 100, etaSeconds: 0 };
    const html = renderToStaticMarkup(<ExtractionProgress job={run} />);
    expect(html).toContain('100%');
    expect(html).not.toContain('Extraction complete');
    expect(html).not.toContain('slides validated');
  });

  it('labels an inner batch estimate without implying it covers the whole stage', () => {
    const run = job();
    run.progress = { ...run.progress!, scope: 'batch', unit: 'patches' };
    const html = renderToStaticMarkup(<ExtractionProgress job={run} />);
    expect(html).toContain('Batch time remaining');
    expect(html).toContain('This batch only; other slides or stages may follow.');
    expect(html).not.toContain('Stage time remaining');
  });

  it.each(['cancelling', 'cancelled', 'failed', 'interrupted'] as const)('hides stale ETA and activity animation after %s', (state) => {
    const html = renderToStaticMarkup(<ExtractionProgress job={job({ state })} />);
    expect(html).not.toContain('~2m 3s');
    expect(html).not.toContain('is-indeterminate');
    expect(html).not.toContain('Estimating');
    expect(html).toContain('Last reported stage progress');
  });

  it('uses inspected counts during validation and a validated summary on success', () => {
    const run = job();
    run.progress = { ...run.progress!, stage: 'validation' };
    expect(renderToStaticMarkup(<ExtractionProgress job={run} />)).toContain('5 of 138 slides inspected');
    run.state = 'succeeded';
    run.result = { completedSlides: 138 };
    const html = renderToStaticMarkup(<ExtractionProgress job={run} />);
    expect(html).toContain('138 slides validated');
    expect(html).toContain('Extraction complete');
  });

  it('reports usable output coverage when validation finds a missing slide', () => {
    const run = job({ state: 'failed', result: { completedSlides: 137, missingSlides: 1 } });
    run.progress = { ...run.progress!, stage: 'validation', completed: 137, percent: 99.3 };
    const html = renderToStaticMarkup(<ExtractionProgress job={run} />);
    expect(html).toContain('Validated outputs');
    expect(html).toContain('137 of 138 slides validated');
    expect(html).not.toContain('137 of 138 slides inspected');
    expect(html).not.toContain('Extraction complete');
  });

  it('uses an indeterminate display when progress is unavailable', () => {
    const html = renderToStaticMarkup(<ExtractionProgress job={job({ progress: undefined })} />);
    expect(html).toContain('is-indeterminate');
    expect(html).toContain('Not reported');
    expect(html).not.toContain('aria-valuenow');
    expect(html).not.toContain('0%');
  });

  it('formats valid durations and treats missing or invalid values as unknown', () => {
    expect(formatExtractionDuration(0)).toBe('0s');
    expect(formatExtractionDuration(3661)).toBe('1h 1m');
    expect(formatExtractionDuration(90_061)).toBe('1d 1h');
    expect(formatExtractionDuration(null)).toBeNull();
    expect(formatExtractionDuration(-1)).toBeNull();
    expect(formatExtractionDuration(Infinity)).toBeNull();
  });
});
