import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { EmptyState, Metric } from './ui';

describe('empty state', () => {
  it('shows what is missing and the action that fills it', () => {
    const html = renderToStaticMarkup(
      <EmptyState icon="dataset" title="No datasets yet" description="Import a table to begin." action={<button type="button">Create dataset</button>} />,
    );
    expect(html).toContain('No datasets yet');
    expect(html).toContain('Import a table to begin.');
    expect(html).toContain('class="empty-state-action"');
    expect(html).toContain('Create dataset');
    expect(html).toContain('class="empty-state-icon"');
  });

  it('stays a plain explanation when no action applies', () => {
    const html = renderToStaticMarkup(<EmptyState title="No matching records" description="Try another search." />);
    expect(html).not.toContain('empty-state-action');
    expect(html).toContain('No matching records');
  });
});

describe('metric card', () => {
  it('sets a measurement in the numeric display size', () => {
    const html = renderToStaticMarkup(<Metric label="Selected training set" value="138 slides" note="138 groups" />);
    expect(html).toContain('class="metric-value"');
    expect(html).not.toContain('metric-value-text');
  });

  it('sets a state without digits as readable text, not as a number', () => {
    for (const value of ['Sampled from training', 'Unavailable']) {
      const html = renderToStaticMarkup(<Metric label="Early-stop validation" value={value} />);
      expect(html).toContain('metric-value-text');
    }
  });

  it('lets a caller state the kind explicitly', () => {
    expect(renderToStaticMarkup(<Metric label="Runs" value="3 of 5" variant="text" />)).toContain('metric-value-text');
    expect(renderToStaticMarkup(<Metric label="Cohort" value="Grade 2" variant="value" />)).not.toContain('metric-value-text');
  });
});
