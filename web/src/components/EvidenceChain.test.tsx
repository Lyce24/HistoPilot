import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import EvidenceChain, { evidenceLink } from './EvidenceChain';

describe('experiment-owned predictor evidence chain', () => {
  it('links Experiments directly to evaluation while retaining predictor identity', () => {
    const html = renderToStaticMarkup(<EvidenceChain current="evaluation" experimentId="experiment-one" predictorId="ready-one" />);
    expect(html).toContain('Experiments'); expect(html).toContain('Model evaluation');
    expect(html).not.toContain('>Predictors<'); expect(html).not.toContain('#post-development');
    expect(evidenceLink('post-development', { experimentId: 'one', predictorId: 'two' })).toBe('#experiments?experiment=one&predictor=two&tab=predictors');
  });
  it('does not present model interpretation as a step after evaluation, clinical utility or inference', () => {
    for (const current of ['evaluation', 'clinical-utility', 'inference'] as const) {
      expect(renderToStaticMarkup(<EvidenceChain current={current} predictorId="ready-one" evaluationId="run" />)).not.toContain('Model interpretation');
    }
  });
});
