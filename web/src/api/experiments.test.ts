import { afterEach, describe, expect, it, vi } from 'vitest';
import { experimentPollInterval, experimentStatusLabel } from './experiments';
import type { ModelExperiment } from './experiments';

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('model experiment identity contracts', () => {
  it('creates the record before inputs exist and retries an uncertain response with the same identity request', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockRejectedValueOnce(new TypeError('Response lost')).mockResolvedValueOnce(response({ id: 'experiment-1' }));
    vi.stubGlobal('fetch', fetcher);
    const { experiments } = await import('./experiments');
    const input = { name: 'Baseline', notes: 'Compare seeds', tags: ['baseline'], operationId: 'same-intent' };
    await expect(experiments.create('project/one', input)).rejects.toMatchObject({ status: 0, code: 'SERVICE_UNREACHABLE' });
    await experiments.create('project/one', input);
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project%2Fone/model-experiments');
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual(input);
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('/launch'))).toBe(false);
  });

  it('reads a specific record and updates complete values using optimistic revision', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response({ token: 'session' })).mockResolvedValueOnce(response({ id: 'experiment/a' })).mockResolvedValueOnce(response({ detail: 'Experiment changed', code: 'STALE_EXPERIMENT' }, 409));
    vi.stubGlobal('fetch', fetcher);
    const { experiments } = await import('./experiments');
    await experiments.get('project', 'experiment/a');
    const update = { name: 'Baseline', notes: 'Keep note', tags: ['seed'], inputs: { protocolId: 'p', featureBundleId: 'f', loadingPolicy: 'native' as const, packArtifactId: null }, expectedRevision: 3 };
    await expect(experiments.update('project', 'experiment/a', update)).rejects.toMatchObject({ status: 409, code: 'STALE_EXPERIMENT' });
    expect(fetcher.mock.calls[1][0]).toBe('/api/v1/projects/project/model-experiments/experiment%2Fa');
    expect(fetcher.mock.calls[2][1].method).toBe('PATCH');
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual(update);
  });

  it('keeps queue scheduling and cancellation truthful and polls active work', () => {
    expect(experimentStatusLabel('queued')).toBe('Scheduled (queued)');
    expect(experimentStatusLabel('completed')).toBe('Finished');
    for (const status of ['queued', 'running', 'cancelling']) expect(experimentPollInterval({ items: [{ status } as ModelExperiment] })).toBe(3000);
    expect(experimentPollInterval({ items: [{ status: 'completed' } as ModelExperiment] })).toBe(15000);
  });
});
