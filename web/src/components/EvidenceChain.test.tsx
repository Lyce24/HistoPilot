import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import EvidenceChain, { evidenceLink } from './EvidenceChain';

describe('experiment-owned predictor evidence chain', () => {
  it('links Experiments directly to Apply models while retaining predictor identity', () => {
    const html = renderToStaticMarkup(<EvidenceChain current="post-development" experimentId="experiment-one" predictorId="ready-one" />);
    expect(html).toContain('Experiments'); expect(html).toContain('href="#apply?view=new&amp;experiment=experiment-one&amp;predictor=ready-one"');
    expect(html).not.toContain('>Predictors<'); expect(html).not.toContain('#post-development');
    expect(evidenceLink('post-development', { experimentId: 'one', predictorId: 'two' })).toBe('#experiments?experiment=one&predictor=two&tab=predictors');
  });
  it('opens a run, and its clinical analysis, in Apply models', () => {
    expect(evidenceLink('apply', { predictorId: 'p', evaluationId: 'run 1' })).toBe('#apply?run=run+1');
    expect(evidenceLink('apply', { evaluationId: 'run', clinicalAnalysisId: 'c' })).toBe('#apply?run=run&clinical=c');
    expect(evidenceLink('apply')).toBe('#apply');
  });
  it('does not present model interpretation as a step after applying models', () => {
    expect(renderToStaticMarkup(<EvidenceChain current="apply" predictorId="ready-one" evaluationId="run" />)).not.toContain('Model interpretation');
  });
});
