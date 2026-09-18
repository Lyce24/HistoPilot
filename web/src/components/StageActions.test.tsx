import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { StageBackButton, StageContinueButton, StageCreateButton } from './StageActions';

describe('stage action contract', () => {
  it('keeps navigation from submitting a form and retains explicit submit actions', () => {
    const navigation = renderToStaticMarkup(<StageContinueButton>Continue to review</StageContinueButton>);
    const creation = renderToStaticMarkup(<StageCreateButton type="submit" disabled>Create experiment</StageCreateButton>);
    expect(navigation).toContain('type="button"');
    expect(creation).toContain('type="submit"');
    expect(creation).toContain('disabled=""');
    expect(creation).toContain('btn-primary');
  });

  it('keeps route changes as links and icons out of the accessible name', () => {
    const link = renderToStaticMarkup(<StageContinueButton href="#experiments">Continue to experiments</StageContinueButton>);
    expect(link).toMatch(/^<a /);
    expect(link).toContain('href="#experiments"');
    expect(link).not.toContain('type="button"');
    expect(link).toContain('aria-hidden="true"');
    expect(link.indexOf('Continue to experiments')).toBeLessThan(link.indexOf('<svg'));
    const back = renderToStaticMarkup(<StageBackButton>Back to datasets</StageBackButton>);
    expect(back).toContain('btn-secondary');
    expect(back.indexOf('<svg')).toBeLessThan(back.indexOf('Back to datasets'));
  });
});
