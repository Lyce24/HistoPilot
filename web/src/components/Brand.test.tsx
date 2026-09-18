import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import favicon from '../../public/favicon.svg?raw';
import standaloneMark from '../../public/histopilot-mark.svg?raw';
import { Brand, HistoPilotMark } from './Brand';

/** Every path with the paint that draws it, so a changed colour cannot slip through. */
function artwork(svg: string) {
  return [...svg.matchAll(/<path\b([^>]*?)\s*\/?>/g)].map(([, attributes]) =>
    ['fill', 'stroke', 'stroke-width', 'strokeWidth', 'd']
      .map((name) => attributes.match(new RegExp(`\\b${name}="([^"]+)"`))?.[1] ?? '')
      .join('|'));
}

describe('shared HistoPilot brand', () => {
  it('keeps the favicon, standalone asset and every inline mark identical', () => {
    const paths = artwork(renderToStaticMarkup(<HistoPilotMark />));
    expect(paths).toHaveLength(3);
    expect(artwork(favicon)).toEqual(paths);
    expect(artwork(standaloneMark)).toEqual(paths);
    // The outline and the needle's tail take their surroundings; only the tip is fixed.
    expect(paths.filter((path) => path.includes('currentColor'))).toHaveLength(2);
    expect(paths.some((path) => path.includes('#ffc72c'))).toBe(true);
    for (const source of [favicon, standaloneMark]) {
      expect(source).toContain('viewBox="0 0 48 48"');
      expect(source).not.toMatch(/<(?:rect|image|text|script)\b/);
    }
  });

  it('provides one accessible name with the wordmark and no nested interactive control', () => {
    const html = renderToStaticMarkup(<Brand subtitle="Research workspace" tone="inverse" size={40} />);
    expect(html).toContain('Histo<strong>Pilot</strong>');
    expect(html).toContain('Research workspace');
    expect(html).toContain('aria-hidden="true"');
    expect(html).not.toContain('aria-label="HistoPilot"');
    expect(html).not.toMatch(/<(?:a|button)\b/);
    expect(html).toContain('hp-brand--inverse');
    expect(html).toContain('width="40" height="40"');
  });

  it('names the standalone and compact symbols while allowing decorative use', () => {
    const compact = renderToStaticMarkup(<Brand iconOnly subtitle="Hidden subtitle" size={24} />);
    expect(compact).toContain('role="img" aria-label="HistoPilot"');
    expect(compact).not.toContain('hp-brand__wordmark');
    expect(compact).not.toContain('Hidden subtitle');
    expect(renderToStaticMarkup(<HistoPilotMark label="HistoPilot home" />)).toContain('aria-label="HistoPilot home"');
    const decorative = renderToStaticMarkup(<HistoPilotMark decorative />);
    expect(decorative).toContain('aria-hidden="true"');
    expect(decorative).not.toContain('role="img"');
    expect(decorative).not.toContain('aria-label');
  });
});
