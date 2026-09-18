/** Check the interface palette's contrast. Run with node web/scripts/verify-palette.mjs. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const theme = await readFile(join(web, 'src/theme.css'), 'utf8');

/** Every `--token: #value` declared by the palette. */
const palette = Object.fromEntries(
  [...theme.matchAll(/(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map(([, name, value]) => [name, value.toLowerCase()]),
);

function luminance(hex) {
  const channels = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const linear = channels.map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4));
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}
function contrast(a, b) {
  const [first, second] = [luminance(a), luminance(b)];
  return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
}
const value = (token) => {
  assert.ok(palette[token], `Palette is missing ${token}`);
  return palette[token];
};
/** WCAG AA is 4.5:1 for text and 3:1 for a graphic that carries meaning. */
function atLeast(need, foreground, background, what) {
  const ratio = contrast(value(foreground), typeof background === 'string' && background.startsWith('#') ? background : value(background));
  assert.ok(ratio >= need, `${what}: ${foreground} on ${background} is ${ratio.toFixed(2)}:1, below ${need}:1`);
  return ratio;
}

const lightSurfaces = ['--theme-surface', '--theme-canvas', '--theme-soft', '--theme-surface-hover', '--theme-settled', '--theme-next-soft', '--theme-accent-soft', '--theme-status-warning-soft', '--theme-alert-soft'];
const darkSurfaces = ['--brand-deep', '--brand-deepest', '--theme-nav-soft', '--theme-nav-hover'];
const checks = [];

// Text the interface reads on its light surfaces.
for (const surface of lightSurfaces) {
  for (const text of ['--theme-ink', '--theme-muted', '--brand-primary', '--brand-deep']) {
    checks.push(atLeast(4.5, text, surface, 'Body, muted and action text'));
  }
}
// Text inside the application shell.
for (const surface of darkSurfaces) {
  checks.push(atLeast(4.5, '--theme-on-dark', surface, 'Shell text'));
  checks.push(atLeast(4.5, '--theme-on-dark-muted', surface, 'Secondary shell text'));
}
// Each phase accent carries the step number on it.
for (const phase of ['--theme-phase-prepare', '--theme-phase-develop', '--theme-phase-evaluate']) {
  checks.push(atLeast(4.5, '--theme-on-dark', phase, 'Step number on its phase accent'));
}
// Filled controls and chips.
checks.push(atLeast(4.5, '--theme-surface', '--brand-primary', 'Primary button label'));
checks.push(atLeast(4.5, '--theme-surface', '--brand-primary-hover', 'Primary button label on hover'));
checks.push(atLeast(4.5, '--brand-deepest', '--brand-accent', 'Label on the shell chip'));

// State text, each on the surface it belongs to.
checks.push(atLeast(4.5, '--theme-alert-ink', '--theme-alert-soft', 'Needs-attention text'));
checks.push(atLeast(4.5, '--theme-next-ink', '--theme-next-soft', 'Next-step text'));
checks.push(atLeast(4.5, '--theme-settled-ink', '--theme-settled', 'Frozen text'));
checks.push(atLeast(4.5, '--theme-status-warning', '--theme-status-warning-soft', 'Warning text'));
checks.push(atLeast(4.5, '--theme-status-warning', '--theme-surface', 'Warning text on a card'));
// Pathology accents: hematoxylin labels interpretation, eosin only marks tissue.
checks.push(atLeast(4.5, '--brand-hematoxylin', '--theme-surface', 'Interpretation label'));
checks.push(atLeast(3, '--brand-eosin', '--theme-surface', 'Tissue mark'));

// The viewer is deliberately dark.
checks.push(atLeast(4.5, '--theme-viewer-text', '--theme-viewer-bg', 'Viewer text'));
checks.push(atLeast(4.5, '--theme-viewer-text', '--theme-viewer-panel', 'Viewer panel text'));

// Marks that are not text still have to be seen.
checks.push(atLeast(3, '--brand-accent', '--brand-deep', 'In-progress mark on the shell'));
checks.push(atLeast(3, '--brand-accent', '--theme-surface', 'In-progress rail'));
checks.push(atLeast(3, '--theme-next-ink', '--theme-canvas', 'In-progress mark on canvas'));
checks.push(atLeast(3, '--theme-alert-mark', '--theme-alert-soft', 'Needs-attention mark'));
checks.push(atLeast(3, '--brand-gray', '--theme-canvas', 'Not-started mark'));
checks.push(atLeast(3, '--brand-primary', '--theme-surface', 'Focus ring'));

/** Two meanings that sit next to each other must not share a hue. */
function hue(hex) {
  const [r, g, b] = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const high = Math.max(r, g, b), low = Math.min(r, g, b), span = high - low;
  if (!span) return -1;
  const sector = high === r ? ((g - b) / span) % 6 : high === g ? (b - r) / span + 2 : (r - g) / span + 4;
  return (sector * 60 + 360) % 360;
}
for (const [a, b, what] of [
  ['--brand-primary', '--theme-alert-mark', 'Action and attention'],
  ['--brand-primary', '--theme-settled-ink', 'Action and frozen'],
  ['--theme-phase-prepare', '--theme-phase-develop', 'Prepare and Develop'],
  ['--theme-phase-develop', '--theme-phase-evaluate', 'Develop and Evaluate'],
  ['--theme-status-warning', '--theme-alert-mark', 'Warning and failure'],
]) {
  const apart = Math.abs(hue(value(a)) - hue(value(b)));
  const degrees = Math.min(apart, 360 - apart);
  assert.ok(degrees >= 25, `${what} are ${degrees.toFixed(0)}deg apart on the hue wheel, too close to tell apart`);
}

/**
 * The visual grammar. Six colours carry meaning and nothing else may:
 *
 *   charcoal  navigation and the application shell
 *   emerald   HistoPilot, action, active, complete
 *   violet    development, modelling, interpretation
 *   blue      evaluation, frozen, immutable
 *   gold      a small institutional accent, inside the logo only
 *   pink      histology visualization only
 *
 * plus warm neutrals for everything else. Two semantic colours sit outside the
 * grammar because failure has to be legible: amber warns and red alerts. They are
 * allowed only on the tokens named for them, so a seventh colour cannot arrive by
 * being called a status.
 */
function family(hex) {
  const [r, g, b] = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
  const high = Math.max(r, g, b), low = Math.min(r, g, b), span = high - low;
  const lightness = (high + low) / 2;
  if (!span) return 'neutral';
  const saturation = span / (1 - Math.abs(2 * lightness - 1));
  if (saturation < 0.12) return 'neutral';
  // At the extremes of lightness a three-unit channel spread reads as saturated
  // without being a colour anyone can see, so judge those on spread instead.
  if ((lightness > 0.95 || lightness < 0.1) && span < 0.04) return 'neutral';
  const sector = high === r ? ((g - b) / span) % 6 : high === g ? (b - r) / span + 2 : (r - g) / span + 4;
  const hue = (sector * 60 + 360) % 360;
  // The shell is a charcoal carrying a deliberate hematoxylin undertone.
  if (saturation < 0.22) return hue >= 230 && hue <= 290 ? 'charcoal' : 'neutral';
  if (hue >= 140 && hue < 195) return 'emerald';
  if (hue >= 195 && hue < 245) return 'blue';
  if (hue >= 245 && hue < 300) return 'violet';
  if (hue >= 300 && hue < 345) return 'pink';
  if (hue >= 345 || hue < 20) return 'alert';
  if (hue >= 20 && hue < 65) return 'warning';
  return `unplaced (hue ${hue.toFixed(0)})`;
}
const grammar = [
  [/^--brand-deep(est)?$|^--theme-nav-/, 'charcoal'],
  [/^--brand-(primary|accent)|^--theme-(accent|next)-|^--theme-phase-prepare$/, 'emerald'],
  [/^--brand-hematoxylin$|^--theme-phase-develop$/, 'violet'],
  [/^--theme-settled|^--theme-phase-evaluate$/, 'blue'],
  [/^--brand-eosin$/, 'pink'],
  [/^--theme-status-warning/, 'warning'],
  [/^--theme-alert-/, 'alert'],
];
for (const [token, hex] of Object.entries(palette)) {
  const found = family(hex);
  if (found === 'neutral') continue;
  const rule = grammar.find(([pattern]) => pattern.test(token));
  assert.ok(rule, `${token} is ${hex}, a saturated ${found} that the visual grammar has no role for`);
  assert.equal(found, rule[1], `${token} is ${hex}, which reads ${found}; the grammar puts it in ${rule[1]}`);
}

/** Brown University's brown and gold are an affiliation mark, not a UI colour. */
assert.doesNotMatch(theme, /#(4e3629|ffc72c)/i, 'Institutional colours belong to the logo, not the palette');

// Data encodings keep their own scales and never live in this palette.
assert.doesNotMatch(theme, /--(train|validation|test|attention)/, 'Scientific scales belong with their charts');
for (const token of Object.keys(palette)) {
  assert.match(token, /^--(brand|theme)-/, `Unexpected palette token ${token}`);
}

console.log(`PASS: ${checks.length} palette contrast pairs, lowest ${Math.min(...checks).toFixed(2)}:1 (${Object.keys(palette).length} tokens).`);
