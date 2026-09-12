import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ApiError } from '../api/client';
import type { TrainingExecution, TrainingResourceHistory, TrainingResourceSample } from '../api/development';
import { ResourceDashboard, ResourceCards, resourceHistoryPollInterval, resourceSamples, resourceSegments } from './RunResourceUsage';

const sample: TrainingResourceSample = {
  at: '2026-09-12T12:00:30Z',
  host: { cpuCount: 8, cpuUtilizationPercent: 25, totalRamGb: 32, availableRamGb: 24, bootId: 'boot', kernel: 'linux' },
  gpus: [{ index: 0, uuid: 'gpu', name: 'A5000', driverVersion: '596', totalMemoryGb: 24, usedMemoryGb: 5, freeMemoryGb: 19, utilizationPercent: 60 }],
  runs: [{ runId: 'run-one', pid: 100, rssGb: 2 }],
};
const execution: TrainingExecution = {
  batchId: 'batch-one', status: 'running', sessionName: 'batch-one', logPath: '/worker.log', outputPath: '/run', runs: [], findings: [],
  createdAt: '2026-09-12T12:00:00Z', updatedAt: sample.at, runCounts: { total: 1, running: 1, queued: 0, completed: 0, failed: 0, cancelled: 0 },
  telemetry: { intervalSeconds: 15, path: '/telemetry.jsonl', latest: sample, peak: { hostUsedRamGb: 8, runRssGb: { 'run-one': 2.5 }, gpuUsedMemoryGb: { 0: 7 } } },
};
const history: TrainingResourceHistory = { batchId: execution.batchId, totalRows: 3, truncated: false, rows: [
  { ...sample, at: '2026-09-12T12:00:00Z' }, { ...sample, at: '2026-09-12T12:00:15Z' }, sample,
] };
afterEach(() => vi.useRealTimers());

describe('recorded run resource usage', () => {
  it('shows separate CPU, RAM, GPU and VRAM cards with measured units and distinct history charts', () => {
    const html = renderToStaticMarkup(<ResourceDashboard execution={{ ...execution, status: 'completed' }} history={history} />);
    expect(html).toContain('>25%</strong>');
    expect(html).toContain('>8.00 GiB</strong>');
    expect(html).toContain('60% utilization');
    expect(html).toContain('>5.00 GiB</strong>');
    expect(html).toContain('CPU history');
    expect(html).toContain('RAM history');
    expect(html).toContain('GPU 0 history');
    expect(html).toContain('VRAM · GPU 0 history');
    expect(html).toContain('CPU and RAM are host-wide');
    expect(html).toContain('GPU usage includes other programs');
    expect(html).toContain('recorded measurements, not live device usage');
    expect(html).not.toContain('NaN');
  });

  it('keeps missing and invalid values unavailable, while measured zeros remain visible', () => {
    const older = structuredClone(execution);
    delete older.telemetry!.latest.host.cpuUtilizationPercent;
    older.telemetry!.latest.host.availableRamGb = Number.NaN;
    older.telemetry!.latest.gpus[0].utilizationPercent = null;
    older.telemetry!.latest.gpus[0].usedMemoryGb = null;
    const html = renderToStaticMarkup(<ResourceCards execution={older} />);
    expect(html).toContain('Not recorded by this worker');
    expect(html).toContain('Utilization unavailable');
    expect(html).not.toContain('NaN');
    expect(html).not.toContain('>0%</strong>');
    expect(html).not.toContain('0% utilization');
    older.telemetry!.latest.host.cpuUtilizationPercent = 0;
    older.telemetry!.latest.gpus[0].utilizationPercent = 0;
    older.telemetry!.latest.gpus[0].usedMemoryGb = 0;
    const zero = renderToStaticMarkup(<ResourceCards execution={older} />);
    expect(zero).toContain('>0%</strong>');
    expect(zero).toContain('0% utilization');
    expect(zero).toContain('>0.00 GiB</strong>');
  });

  it('marks stale active samples but preserves final recordings and handles history failures explicitly', () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date('2026-09-12T12:05:00Z'));
    const stale = renderToStaticMarkup(<ResourceDashboard execution={execution} history={history} error={new Error('offline')} />);
    expect(stale).toContain('resource sample is stale');
    expect(stale).toContain('Resource history could not refresh');
    expect(stale).toContain('60% utilization');
    const finished = renderToStaticMarkup(<ResourceDashboard execution={{ ...execution, status: 'completed' }} history={history} />);
    expect(finished).toContain('Last resource sample');
    expect(finished).not.toContain('resource sample is stale');
    const oldServer = renderToStaticMarkup(<ResourceDashboard execution={execution} error={new ApiError('Not Found', 404)} onRefresh={() => undefined} />);
    expect(oldServer).toContain('Restart HistoPilot manually');
    expect(oldServer).toContain('Retry resource history');
  });

  it('stops polling on completion and unsupported endpoints but permits active transient recovery', () => {
    expect(resourceHistoryPollInterval(true, null)).toBe(15000);
    expect(resourceHistoryPollInterval(true, new Error('offline'))).toBe(15000);
    expect(resourceHistoryPollInterval(true, new ApiError('Not Found', 404))).toBe(false);
    expect(resourceHistoryPollInterval(false, null)).toBe(false);
    expect(resourceHistoryPollInterval(true, null, 1)).toBe(3000);
  });

  it('sorts actual timestamps, merges the snapshot without duplicates, and never invents timestamped measurements', () => {
    const samples = resourceSamples(execution, { ...history, rows: [sample, { ...sample, at: 'invalid' }, history.rows[0]] });
    expect(samples.map((row) => row.at)).toEqual([history.rows[0].at, sample.at]);
    expect(resourceSamples({ ...execution, telemetry: undefined })).toEqual([]);
    expect(renderToStaticMarkup(<ResourceCards execution={{ ...execution, telemetry: undefined }} />)).toContain('not been recorded yet');
  });

  it('breaks curves across missing metrics and sampling gaps, preserving real zero and unequal elapsed time', () => {
    const points = [{ at: 0, value: 2 }, { at: 15000, value: 0 }, { at: 30000, value: null }, { at: 45000, value: 4 }, { at: 60000, value: 5 }, { at: 180000, value: 6 }];
    expect(resourceSegments(points, 37500)).toEqual([[points[0], points[1]], [points[3], points[4]], [points[5]]]);
    expect(resourceSegments([{ at: 0, value: Number.NaN }, { at: 1, value: -1 }], 100)).toEqual([]);
  });

  it('keeps different physical GPUs separate, even when their indices match across resumed hosts', () => {
    const previous = { ...history.rows[0], gpus: [{ ...sample.gpus[0], uuid: 'other-gpu', name: 'Older GPU' }] };
    const html = renderToStaticMarkup(<ResourceDashboard execution={execution} history={{ ...history, rows: [previous, sample] }} />);
    expect(html).toContain('Older GPU');
    expect(html).toContain('A5000');
    expect(html.match(/class="run-resource-gpu"/g)).toHaveLength(2);
    expect(html).toContain('Utilization unavailable');
  });

  it('explains bounded history counts without claiming the count covers the whole file', () => {
    const html = renderToStaticMarkup(<ResourceDashboard execution={execution} history={{ ...history, totalRows: 100, totalRowsIsLowerBound: true, truncated: true, warning: 'Only the recent file window was read.' }} />);
    expect(html).toContain('latest 3 of at least 100 saved samples');
    expect(html).toContain('Only the recent file window was read.');
  });
});
