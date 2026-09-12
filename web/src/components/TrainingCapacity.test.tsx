import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { defaultResources } from '../api/development';
import TrainingCapacity, { trainingCapacity } from './TrainingCapacity';

describe('effective concurrency', () => {
  it('explains why six batch slots still use one run on one GPU', () => {
    const resources = { ...defaultResources(), maxConcurrentRuns: 6 };
    expect(trainingCapacity(resources).configured).toBe(1);
    const html = renderToStaticMarkup(<TrainingCapacity resources={resources} />);
    expect(html).toContain('Configured limit: 1 concurrent run');
    expect(html).toContain('1 selected GPU × 1 run per GPU');
  });

  it('accounts for overlapping training and validation loader pools', () => {
    const resources = { ...defaultResources(), maxConcurrentRuns: 6, runsPerGpu: 6 };
    const runtime = { available: true, cudaAvailable: true, gpuCount: 1, python: 'python', versions: {}, findings: [], host: { cpuCount: 18, availableRamGb: 200, totalRamGb: 256, bootId: 'boot', kernel: 'linux' } };
    expect(trainingCapacity(resources, runtime)).toMatchObject({ configured: 6, cpuSlotsPerRun: 6, available: 3 });
    expect(trainingCapacity(resources, { ...runtime, host: { ...runtime.host, availableRamGb: 16 } }).available).toBe(2);
    expect(trainingCapacity(resources, { ...runtime, gpuCount: 0, cudaAvailable: false }).available).toBe(0);
  });

  it('does not apply a per-GPU limit to CPU execution', () => {
    expect(trainingCapacity({ ...defaultResources(), maxConcurrentRuns: 6, gpuIds: [] }).configured).toBe(6);
  });
});
