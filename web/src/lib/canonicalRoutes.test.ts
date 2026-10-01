import { describe, expect, it } from 'vitest';
import { canonicalHash } from './canonicalRoutes';

describe('links saved by merged modules', () => {
  it('opens an Experimental Setup link as the same experiment design in Experiments', () => {
    expect(canonicalHash('#experimental-setup')).toBe('#experiments');
    expect(canonicalHash('#experimental-setup?experiment=e&tab=batches')).toBe('#experiments?experiment=e&tab=batches');
    expect(canonicalHash('#experimental-setup?experiment=e&tab=review')).toBe('#experiments?experiment=e&tab=review');
    // Prepared inputs carried by a link survive, so a new experiment starts from them.
    expect(canonicalHash('#experimental-setup?dataset=d&targetSplit=t&saved=target-split')).toBe('#experiments?dataset=d&targetSplit=t&saved=target-split');
    for (const alias of ['#setup', '#setups', '#experiment-setup']) expect(canonicalHash(alias)).toBe('#experiments');
  });

  it('still rewrites the modules Apply models replaced, and leaves every other link alone', () => {
    expect(canonicalHash('#inference?evaluation=r')).toBe('#apply?run=r');
    for (const hash of ['', '#overview', '#experiments?experiment=e&tab=runs', '#apply?run=r&reference=x']) expect(canonicalHash(hash)).toBe(hash);
  });
});
