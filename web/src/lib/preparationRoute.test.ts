import { describe, expect, it } from 'vitest';
import { preparationContext, preparationLink } from './preparationRoute';

describe('prepared input navigation', () => {
  it('keeps exact prepared IDs when opening an experiment or selecting a tab', () => {
    const context = { datasetId: 'older dataset', protocolId: 'protocol/one', bundleId: 'bundle&two', saved: 'bundle' as const };
    const link = preparationLink('experiments', context, { experiment: 'question/one', tab: 'inputs' });
    const parameters = new URLSearchParams(link.split('?')[1]);
    expect(preparationContext(parameters)).toEqual(context);
    expect(parameters.get('experiment')).toBe('question/one');
    expect(parameters.get('tab')).toBe('inputs');
  });

  it('keeps plain navigation plain and ignores unrecognized save notices', () => {
    expect(preparationLink('features', {})).toBe('#features');
    expect(preparationContext(new URLSearchParams('dataset=&saved=anything'))).toEqual({ datasetId: undefined, protocolId: undefined, bundleId: undefined, saved: undefined });
  });
});
