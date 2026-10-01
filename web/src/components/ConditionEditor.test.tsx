import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Condition, DatasetQueryResult } from '../api/scientific';
import ConditionEditor from './ConditionEditor';
import { composeConditions, conditionComposition, conditionFields, describeCondition } from '../lib/conditions';

function render(conditions: Condition[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(['scientific', 'project', 'field-profile', 'dataset', 'cohort'], {
    valueCounts: ['Site A', 'Site B', 'Site C'].map((value) => ({ value, count: 10 })),
    valuesTruncated: false, records: [],
  } as unknown as DatasetQueryResult);
  try {
    return renderToStaticMarkup(<QueryClientProvider client={client}>
      <ConditionEditor title="Study filters" description="Choose cohort values." emptyMessage="All slides included."
        conditions={conditions} columns={['cohort', 'cohort', 'Slide_ID']} onChange={() => {}}
        fieldContext={{ project: 'project', datasetId: 'dataset', dictionary: [{ key: 'cohort', type: 'categorical', sourceColumn: 'cohort', owner: 'slide' }] }} />
    </QueryClientProvider>);
  } finally { client.clear(); }
}

describe('shared composable filters', () => {
  it('offers cohort values directly and marks two sites as an OR selection', () => {
    const html = render([{ field: 'cohort', op: 'in', value: ['Site A', 'Site B'] }]);
    expect(html).toContain('Include any selected value');
    expect(html.match(/type="checkbox" checked=""/g)).toHaveLength(2);
    expect(html).toContain('Site A');
    expect(html).toContain('Site B');
    expect(html).toContain('Site C');
    expect(html).toContain('A slide can match any one of these values (OR).');
    expect(html.match(/<option[^>]*>cohort<\/option>/g)).toHaveLength(1);
    expect(html).not.toContain('Values, separated by');
  });

  it('preserves nested grouping in editor and review descriptions', () => {
    const conditions: Condition[] = [{ op: 'any', conditions: [
      { field: 'cohort', op: 'eq', value: 'Site A' },
      { op: 'all', conditions: [{ field: 'cohort', op: 'eq', value: 'Site B' }, { field: 'Slide_ID', op: 'eq', value: 's1' }] },
    ] }];
    const html = render(conditions);
    expect(html).toContain('<option value="any" selected="">Any condition (OR)</option>');
    expect(html).toContain('Condition group');
    expect(conditionFields(conditions)).toEqual(['cohort', 'Slide_ID']);
    expect(describeCondition(conditions[0])).toBe('(cohort equals Site A OR (cohort equals Site B AND Slide_ID equals s1))');
    expect(composeConditions(conditionComposition(conditions).op, conditionComposition(conditions).conditions)).toEqual(conditions);
  });

  it('keeps old AND arrays unchanged and clears an emptied OR selection to all slides', () => {
    const original: Condition[] = [{ field: 'cohort', op: 'eq', value: 'Site A' }];
    expect(conditionComposition(original)).toEqual({ op: 'all', conditions: original });
    expect(composeConditions('all', original)).toBe(original);
    expect(composeConditions('any', [])).toEqual([]);
    expect(render([])).toContain('All slides included.');
    expect(render([])).toContain('Add condition');
  });
});
