import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { RefitBuild } from '../api/predictors';
import type { Workspace } from '../api/types';
import { fixturePredictor } from '../testFixtures/predictors';
import LegacyPredictorRoute from './LegacyPredictorRoute';
import { legacyRecordNote } from './LegacyRecordNote';

afterEach(() => vi.unstubAllGlobals());

function render(hash: string, builds: RefitBuild[] = []) {
  vi.stubGlobal('window', { location: { hash } });
  const client = new QueryClient();
  const source = { ...fixturePredictor(1, 11, 'ensemble', 'legacy/one'), id: 'old-predictor', lifecycleState: 'archived' as const };
  client.setQueryData(['predictors', 'p'], { items: [source] });
  client.setQueryData(['refit-builds', 'p'], { items: builds });
  try { return renderToStaticMarkup(<QueryClientProvider client={client}><LegacyPredictorRoute workspace={{ mode: 'local', project: { id: 'p' } } as Workspace} /></QueryClientProvider>); }
  finally { client.clear(); }
}

describe('historical predictor links', () => {
  it('opens the exact refit from a compute job link, read-only when it ran before the Task Center', () => {
    const predictor = fixturePredictor(1, 11, 'refit');
    const build = { ...predictor, id: 'refit/one', manifest: { ...predictor.manifest, kind: 'refit-build' }, execution: { status: 'interrupted', progress: { epoch: 3, maxEpochs: 12 } } } as unknown as RefitBuild;
    const html = render('#post-development?tab=refits&refit=refit%2Fone', [build]);
    expect(html).toContain('Epoch 3 / 12');
    expect(html).toContain(legacyRecordNote);
    expect(html).not.toContain('Cancel job');
    expect(html).not.toContain('Predictors now belong to their experiment');
  });

  it('preserves the experiment and selected predictor when handing off to Experiments', () => {
    const html = render('#post-development?experiment=legacy%2Fone&predictor=old-predictor');
    expect(html).toContain('#experiments?experiment=legacy%2Fone&amp;tab=predictors&amp;predictor=old-predictor');
    expect(html).not.toContain('Build predictors');
  });
  it('resolves an archived predictor-only link to its original experiment', () => {
    expect(render('#predictor?predictor=old-predictor')).toContain('#experiments?experiment=legacy%2Fone&amp;tab=predictors&amp;predictor=old-predictor');
  });
  it('retains unscoped history and legacy refit recovery without offering new builds', () => {
    for (const hash of ['#post-development', '#post-development?experiment=legacy%2Fone&tab=refits']) {
      const html = render(hash);
      expect(html).toContain('Historical predictors'); expect(html).toContain('Refit jobs');
      expect(html).not.toContain('>Build predictors<');
      expect(html).not.toContain('Review predictor');
    }
  });
});
