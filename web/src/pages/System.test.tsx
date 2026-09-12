import { afterEach, describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiError } from '../api/client';
import type { SystemCompute, SystemStatus } from '../api/types';
import System, { ComputeDashboard, computeErrorMessage, formatComputeBytes, formatUptime } from './System';

const clients: QueryClient[] = [];
afterEach(() => clients.splice(0).forEach((client) => client.clear()));
function render(workers: SystemStatus['workers']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  clients.push(client);
  client.setQueryData(['system'], { mode: 'local', workspace: '/projects', storage: { engine: 'SQLite', journalMode: 'wal', schemaVersion: 1 }, control: { process: 'Control service', cudaModelsLoaded: false }, sourcesReadOnly: true, workers } satisfies SystemStatus);
  return renderToStaticMarkup(<QueryClientProvider client={client}><System /></QueryClientProvider>);
}

describe('system capability reporting', () => {
  it('distinguishes missing extraction runtime from implemented native execution', () => {
    const html = render({ executionEnabled: false, nativeExecutionImplemented: true, tmuxAvailable: true, status: 'TRIDENT runtime unavailable.' });
    expect(html).toContain('Native execution implemented');
    expect(html).toContain('TRIDENT feature extraction');
    expect(html).toContain('Extraction runtime setup required');
    expect(html).toContain('tmux available');
    expect(html).toContain('href="#experiments"');
    expect(html).toContain('href="#features"');
    expect(html).not.toContain('future isolated workers');
    expect(html).not.toContain('Planned');
    expect(html).not.toContain('DuckDB');
  });

  it('reports extraction readiness and unavailable persistence separately', () => {
    const html = render({ executionEnabled: true, nativeExecutionImplemented: true, tmuxAvailable: false, status: 'Review module runtimes.' });
    expect(html).toContain('Runtime ready');
    expect(html).toContain('tmux unavailable');
    expect(html).toContain('Feature tensors &amp; coordinates');
    expect(html).toContain('Dataset and analytical tables');
  });
});

const gib = 1024 ** 3;
const sample: SystemCompute = {
  sampledAt: '2026-09-12T17:00:00Z', sampleIntervalSeconds: 5,
  host: { hostname: 'lab-workstation', platform: 'Linux', release: '6.8', uptimeSeconds: 90061 },
  cpu: { model: 'Xeon', physicalCores: 18, logicalCores: 36, availableCores: 24, utilizationPercent: 37.2, loadAverage: [2.1, 3.2, 4.3], status: 'available', message: null },
  memory: { totalBytes: 128 * gib, usedBytes: 32 * gib, availableBytes: 96 * gib, utilizationPercent: 25, swapTotalBytes: 8 * gib, swapUsedBytes: gib, status: 'available', message: null },
  gpu: { status: 'available', message: null, devices: [
    { index: 0, uuid: 'GPU-0', name: 'NVIDIA RTX A5000', driverVersion: '596.71', utilizationPercent: 89, memoryTotalBytes: 24 * gib, memoryUsedBytes: 18 * gib, memoryFreeBytes: 6 * gib, memoryUtilizationPercent: 75, temperatureCelsius: 65, powerWatts: 175.5, powerLimitWatts: 230 },
    { index: 1, uuid: 'GPU-1', name: 'NVIDIA RTX A5000', driverVersion: '596.71', utilizationPercent: 0, memoryTotalBytes: 24 * gib, memoryUsedBytes: 0, memoryFreeBytes: 24 * gib, memoryUtilizationPercent: 0, temperatureCelsius: null, powerWatts: null, powerLimitWatts: null },
  ] },
  disks: [
    { role: 'workspace', path: '/projects', totalBytes: 1024 * gib, usedBytes: 256 * gib, freeBytes: 768 * gib, utilizationPercent: 25, status: 'available', message: null },
    { role: 'data', path: '/mnt/missing', totalBytes: null, usedBytes: null, freeBytes: null, utilizationPercent: null, status: 'unavailable', message: 'Folder is not mounted.' },
  ],
};

describe('compute statistics', () => {
  it('shows independent GPU readings, CPU capacity, host memory and each disk without confusing host and browser', () => {
    const html = renderToStaticMarkup(<ComputeDashboard data={sample} />);
    for (const text of ['lab-workstation', '37.2%', '18 / 36', '2.1 / 3.2 / 4.3', '32 GiB / 128 GiB', '96 GiB', '1 GiB / 8 GiB', 'GPU 0', 'GPU 1', '89.0%', '18 GiB / 24 GiB', '65 °C', '175.5 W / 230 W', '596.71', '768 GiB', 'Folder is not mounted.', '1d 1h']) expect(html).toContain(text);
    expect(html).toContain('aria-label="GPU 1 utilization"');
    expect(html).toContain('value="0"');
    expect(html).toContain('VRAM free');
    expect(html).toContain('dateTime="2026-09-12T17:00:00Z"');
  });

  it('shows unavailable GPU metrics without implying the machine has no GPUs or a 0% reading', () => {
    const missing: SystemCompute = { ...sample, cpu: { ...sample.cpu, utilizationPercent: null }, gpu: { status: 'unavailable', devices: [], message: 'NVIDIA driver query is unavailable.' } };
    const html = renderToStaticMarkup(<ComputeDashboard data={missing} />);
    expect(html).toContain('Collecting sample');
    expect(html).toContain('Telemetry unavailable');
    expect(html).toContain('NVIDIA driver query is unavailable.');
    expect(html).not.toContain('No NVIDIA GPUs detected.');
    expect(html).not.toContain('>0 GPUs<');
    expect(html).not.toContain('aria-label="CPU utilization"');
  });

  it('distinguishes confirmed absence from failed measurement and preserves real zero readings', () => {
    const empty: SystemCompute = { ...sample, cpu: { ...sample.cpu, utilizationPercent: 0 }, memory: { ...sample.memory, swapTotalBytes: 0, swapUsedBytes: 0 }, gpu: { status: 'available', devices: [], message: null } };
    const html = renderToStaticMarkup(<ComputeDashboard data={empty} />);
    expect(html).toContain('No NVIDIA GPUs detected.');
    expect(html).toContain('No swap configured');
    expect(html).toContain('0.0%');
    expect(html).toContain('aria-label="CPU utilization"');
    expect(html).not.toContain('Collecting sample');
  });

  it('labels stopped and last-known samples without claiming live updates', () => {
    const paused = renderToStaticMarkup(<ComputeDashboard data={sample} live={false} />);
    expect(paused).toContain('Updates paused');
    expect(paused).not.toContain('Live ·');
    const stale = renderToStaticMarkup(<ComputeDashboard data={sample} stale />);
    expect(stale).toContain('Last known sample');
    expect(stale).toContain('37.2%');
    expect(stale).not.toContain('Live ·');
  });

  it('instructs a manual service restart for an old backend without hiding other errors', () => {
    expect(computeErrorMessage(new ApiError('Not Found', 404))).toContain('Restart HistoPilot manually');
    expect(computeErrorMessage(new Error('Connection lost'))).toBe('Compute statistics could not be refreshed. Connection lost');
    expect(computeErrorMessage(new ApiError('Permission denied', 403))).toContain('Permission denied');
    expect(computeErrorMessage(new ApiError('Permission denied', 403))).not.toContain('Restart');
  });

  it('formats binary capacity and uptime without coercing unavailable or invalid numbers into zero', () => {
    expect(formatComputeBytes(0)).toBe('0 B');
    expect(formatComputeBytes(1.5 * gib)).toBe('1.5 GiB');
    expect(formatComputeBytes(1024 * gib)).toBe('1 TiB');
    for (const value of [null, undefined, NaN, Infinity, -1]) {
      expect(formatComputeBytes(value)).toBe('Unavailable');
      expect(formatUptime(value)).toBe('Unavailable');
    }
    expect(formatUptime(0)).toBe('0m');
    expect(formatUptime(3660)).toBe('1h 1m');
  });
});
