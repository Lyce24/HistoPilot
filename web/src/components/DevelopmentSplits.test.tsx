import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ProtocolPreview } from '../api/scientific';
import { newDevelopmentSplit } from '../lib/protocol';
import { SplitStrategy } from './SplitStrategy';
import { SplitPools } from './SplitPools';
import { PartitionTable } from '../pages/LocalProtocol';

describe('development-only split interface', () => {
  it.each(['kfold', 'monte_carlo', 'nested_kfold', 'held_out', 'leave_one_domain_out'] as const)(
    'shows development controls without a reserved test set for %s', (mode) => {
      const split = { ...newDevelopmentSplit(), mode };
      const client = new QueryClient();
      const html = renderToStaticMarkup(
        <QueryClientProvider client={client}>
          <SplitStrategy
            split={split} onChange={() => {}} seedsText="42" onSeedsChange={() => {}} seedsValid
            fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }}
            rules={null} imported={null}
            pools={<SplitPools pools={split.pools!} development validationFraction={0.15}
              onChange={() => {}} onFractionChange={() => {}} renderConditions={() => null}
              imported={null} live={{ loading: false, error: null }} targetField="label" />}
          />
        </QueryClientProvider>,
      );
      expect(html).toContain('Development assessment');
      expect(html).toContain('Split seeds');
      expect(html).not.toMatch(/final test|reported test|test set|test sources|Set aside test/i);
      client.clear();
    },
  );

  it('labels the internal assessment assignment as development in partition review', () => {
    const counts = { slides: 8, patients: 4, groups: 4, classes: { negative: 2, positive: 2 } };
    const partitions: ProtocolPreview['partitions'] = [{
      seed: 42, fold: 0, planId: 'seed:42/fold:0', phase: 'evaluation', pool: 'development',
      train: counts, val: counts, test: counts,
    }];
    const html = renderToStaticMarkup(<PartitionTable partitions={partitions} />);
    expect(html).toContain('Development assessment');
    expect(html).toContain('Assessment fold 1');
    expect(html).not.toMatch(/Reported test|Test fold|Final test/);
  });
});
