import { beforeEach, describe, expect, it, vi } from 'vitest';
const downloads = vi.hoisted(() => vi.fn());
vi.mock('./client', () => ({ downloadArtifact: downloads, request: vi.fn() }));
import { development } from './development';

beforeEach(() => downloads.mockReset());

describe('authenticated OOF prediction exports', () => {
  it.each(['patient', 'slide'] as const)('downloads %s rows for the exact configuration and both seeds', async (unit) => {
    downloads.mockResolvedValueOnce(undefined);
    await development.downloadOOF('project/one', 'batch/one', 'config/one', 43, 17, unit);
    expect(downloads).toHaveBeenCalledExactlyOnceWith(
      `/projects/project%2Fone/mil-experiments/batches/batch%2Fone/oof/config%2Fone/43/17/${unit}.csv`,
      `oof-config/one-43-17-${unit}.csv`,
    );
  });

  it('propagates backend evidence failures to the download controls', async () => {
    downloads.mockRejectedValueOnce(new Error('OOF evidence changed'));
    await expect(development.downloadOOF('project', 'batch', 'config', 42, 42, 'patient')).rejects.toThrow('OOF evidence changed');
  });
});
