import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { newDevelopmentSplit } from '../lib/protocol';
import { SplitStrategy } from './SplitStrategy';

describe('development-only split interface', () => {
  it.each(['kfold', 'monte_carlo', 'nested_kfold', 'held_out', 'leave_one_domain_out'] as const)(
    'shows development controls without a reserved test set for %s', (mode) => {
      const split = { ...newDevelopmentSplit(), mode };
      const client = new QueryClient();
      const html = renderToStaticMarkup(
        <QueryClientProvider client={client}>
          <SplitStrategy
            split={split} onChange={() => {}} seedsText="42" onSeedsChange={() => {}} seedsError=""
            fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }}
          />
        </QueryClientProvider>,
      );
      expect(html).toContain('Development assessment');
      expect(html).toContain('Split seeds');
      expect(html).not.toMatch(/final test|reported test|test set|test sources|Set aside test/i);
      client.clear();
    },
  );

  it('shows why split seeds cannot be saved, next to the field', () => {
    const client = new QueryClient();
    const html = renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <SplitStrategy split={newDevelopmentSplit()} onChange={() => {}} seedsText="1,2,3,4,5,6,7,8,9,10,11" onSeedsChange={() => {}} seedsError="Enter at most 10 split seeds; you entered 11."
          fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }} />
      </QueryClientProvider>,
    );
    client.clear();
    expect(html).toContain('Up to 10 seeds.');
    expect(html).toMatch(/aria-invalid="true" aria-describedby="([^"]+)"[^>]*>[\s\S]*<p id="\1" class="callout callout-warning" role="alert">Enter at most 10 split seeds; you entered 11.<\/p>/);
  });

  it('offers case-grouped folds only for development slide targets, keeping slide labels', () => {
    const render = (splitUnit: 'slide' | 'patient', groupByPatient?: boolean) => {
      const split = { ...newDevelopmentSplit(), ...(groupByPatient ? { groupByPatient } : {}) };
      const client = new QueryClient();
      const html = renderToStaticMarkup(
        <QueryClientProvider client={client}>
          <SplitStrategy split={split} onChange={() => {}} seedsText="42" onSeedsChange={() => {}} seedsError="" splitUnit={splitUnit}
            fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [] }} />
        </QueryClientProvider>,
      );
      client.clear();
      return html;
    };
    const independent = render('slide');
    expect(independent).toContain('Keep all slides of a case in the same fold');
    expect(independent).not.toMatch(/<input type="checkbox" checked=""\/><span>Keep all slides of a case/);
    expect(independent).toContain('Each slide is assigned independently');
    const grouped = render('slide', true);
    expect(grouped).toMatch(/<input type="checkbox" checked=""\/><span>Keep all slides of a case in the same fold/);
    expect(grouped).toContain('Slide labels, case-grouped folds');
    expect(grouped).toContain('Every training case rotates through one assessment fold');
    expect(grouped).not.toMatch(/patient/i);
    expect(render('patient')).not.toContain('Keep all slides of a case in the same fold');
  });
});
