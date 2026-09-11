import { describe, expect, it } from 'vitest';
import { normalizeTridentOptions } from './trident';
import type { TridentOption } from './trident';

const option = (name: string, type: TridentOption['type'], value: unknown): TridentOption => ({
  name,
  flag: `--${name}`,
  label: name,
  group: 'execution',
  type,
  default: value,
  advanced: true,
  description: '',
});

describe('TRIDENT option submission', () => {
  it('preserves zero and false overrides while supplying untouched server defaults', () => {
    const definitions = [
      option('gpu', 'integer', 1),
      option('remove_holes', 'boolean', true),
      option('mag', 'number', 20),
    ];
    expect(normalizeTridentOptions(definitions, { gpu: '0', remove_holes: false })).toEqual({
      gpu: 0,
      remove_holes: false,
      mag: 20,
    });
  });

  it('parses list controls after editing without sending commas or string GPU IDs', () => {
    const definitions = [option('gpus', 'integer[]', null), option('wsi_ext', 'string[]', null)];
    expect(normalizeTridentOptions(definitions, { gpus: '-1, 0, 2, ', wsi_ext: '.svs, .tiff\n.ndpi' })).toEqual({
      gpus: [-1, 0, 2],
      wsi_ext: ['.svs', '.tiff', '.ndpi'],
    });
  });

  it('accepts saved array values and fractional magnification when resuming jobs', () => {
    const definitions = [option('gpus', 'integer[]', null), option('mag', 'number', 20)];
    expect(normalizeTridentOptions(definitions, { gpus: [0, 1], mag: ' 2.5 ' })).toEqual({
      gpus: [0, 1],
      mag: 2.5,
    });
  });

  it('clears optional controls and omits stale options absent from the current catalog', () => {
    const definitions = [option('slide_encoder', 'string', null), option('max_workers', 'integer', null)];
    expect(normalizeTridentOptions(definitions, { slide_encoder: '   ', max_workers: '', unsupported: true })).toEqual({
      slide_encoder: null,
      max_workers: null,
    });
  });
});
