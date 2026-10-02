import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Objective } from '../api/objectives';
import type { EvaluationExperimentOption } from '../lib/evaluationSelection';
import EvaluationExperimentPicker from './EvaluationExperimentPicker';

const aims: Objective[] = [{ id: 'aim-1', name: 'Aim 1 · Grade', rank: 'primary', description: '', revision: 1, createdAt: '', updatedAt: '' }, { id: 'aim-2', name: 'Aim 2 · Transport', rank: 'secondary', description: '', revision: 1, createdAt: '', updatedAt: '' }];
const option = (id: string, values: Partial<EvaluationExperimentOption> = {}): EvaluationExperimentOption => ({ id, name: id, description: 'Finished · 1 ensemble ready', ensemble: 1, refit: 0, seedEnsemble: 0, ready: 1, ...values });
const options = [
  option('main', { objectiveId: 'aim-1', role: 'primary' }),
  option('loco-tcga', { objectiveId: 'aim-2', role: 'generalization', series: 'LOCO', seriesMember: 'tcga' }),
  option('loco-cptac', { objectiveId: 'aim-2', role: 'generalization', series: 'LOCO', seriesMember: 'cptac', ready: 0 }),
  option('pilot', {}),
];
const render = (selected: string[] = [], scope = '', objectives = aims) => renderToStaticMarkup(<EvaluationExperimentPicker options={options} selected={selected} onChange={() => {}} disabled={false} objectives={objectives} scope={scope} />);

describe('experiments to apply, by objective', () => {
  it('lists objectives in order with series under one heading that selects their ready members', () => {
    const html = render();
    const order = ['Aim 1 · Grade', 'Apply experiment main', 'Aim 2 · Transport', 'Series LOCO', 'Apply experiment loco-cptac', 'Apply experiment loco-tcga', 'No objective', 'Apply experiment pilot'].map((text) => html.indexOf(text));
    expect(order.every((index) => index >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(html).toContain('2 experiments · cptac, tcga');
    expect(html.match(/>Select series</g)).toHaveLength(1);
    expect(html.match(/>Select all with ready predictors</g)).toHaveLength(3);
    // Once its ready members are selected, a heading offers nothing more to select.
    expect(render(['loco-tcga']).match(/>Select series</g)).toBeNull();
  });

  it('keeps the focused objective, and selected experiments outside it, in view', () => {
    const html = render(['pilot'], 'aim-2');
    expect(html).toContain('Aim 2 · Transport'); expect(html).not.toContain('Aim 1 · Grade');
    expect(html).not.toContain('Apply experiment main');
    expect(html).toContain('Apply experiment pilot');
    // Without objectives the list is flat, as before.
    const flat = render([], '', []);
    expect(flat).not.toContain('evaluation-objective-heading'); expect(flat).toContain('Apply experiment main');
  });
});
