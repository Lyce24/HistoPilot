import { describe, expect, it } from 'vitest';
import type { Objective } from '../api/objectives';
import { groupExperiments, libraryEntries, memberSummary, sectionByObjective, SEVERAL, sharedObjective, UNSORTED, type Filed } from './experimentLibrary';

const objective = (id: string, name: string, rank: Objective['rank']): Objective => ({ id, name, rank, description: '', revision: 1, createdAt: '', updatedAt: '' });
const aims = [objective('aim-1', 'Aim 1 · Validity', 'primary'), objective('aim-2', 'Aim 2 · Transport', 'secondary'), objective('aim-3', 'Aim 3 · Resolution', 'exploratory')];
const filed = (id: string, values: Partial<Filed> = {}): Filed => ({ id, name: id, ...values });

describe('experiment library by objective', () => {
  it('keeps the service order of objectives and puts unfiled experiments last', () => {
    const items = [filed('loose'), filed('loco', { objectiveId: 'aim-2' }), filed('main', { objectiveId: 'aim-1', role: 'primary' }), filed('gone', { objectiveId: 'removed' })];
    const groups = groupExperiments(items, aims, { showEmpty: false });
    expect(groups.map((group) => group.key)).toEqual(['aim-1', 'aim-2', UNSORTED]);
    expect(groups.at(-1)?.entries.map((entry) => entry.key)).toEqual(['loose', 'gone']);
    // An objective with nothing filed yet is shown only while nothing is filtered.
    expect(groupExperiments(items, aims, { showEmpty: true }).map((group) => group.key)).toEqual(['aim-1', 'aim-2', 'aim-3', UNSORTED]);
    expect(groupExperiments([filed('main', { objectiveId: 'aim-1' })], aims, { showEmpty: false }).map((group) => group.key)).toEqual(['aim-1']);
  });

  it('orders roles, keeps the sort within a role and folds tuning away', () => {
    const items = [
      filed('pilot', { objectiveId: 'aim-1', role: 'tuning' }),
      filed('clinical', { objectiveId: 'aim-1', role: 'comparison' }),
      filed('unassigned', { objectiveId: 'aim-1' }),
      filed('control', { objectiveId: 'aim-1', role: 'control' }),
      filed('encoder', { objectiveId: 'aim-1', role: 'comparison' }),
      filed('main', { objectiveId: 'aim-1', role: 'primary' }),
    ];
    const [group] = groupExperiments(items, aims, { showEmpty: false });
    expect(group.entries.map((entry) => entry.key)).toEqual(['main', 'clinical', 'encoder', 'control', 'unassigned']);
    expect(group.tuning.map((entry) => entry.key)).toEqual(['pilot']);
    expect(group.count).toBe(6);
  });

  it('shows a series of two or more as one row where its first member would sit, members in their order', () => {
    const items = [
      filed('A2 LOCO tcga', { role: 'generalization', series: 'LOCO', seriesMember: 'tcga' }),
      filed('A2 LOCO rih', { role: 'generalization', series: 'LOCO', seriesMember: 'rih' }),
      filed('alone', { role: 'generalization', series: 'Single', seriesMember: 'only' }),
      filed('A2 LOCO cptac', { role: 'generalization', series: 'LOCO', seriesMember: 'cptac' }),
      filed('control wt10', { role: 'control', series: 'LOCO', seriesMember: 'wt10' }),
      filed('control wt9', { role: 'control', series: 'LOCO', seriesMember: 'wt9' }),
    ];
    const entries = libraryEntries(items, 'aim-2');
    expect(entries.map((entry) => entry.kind === 'series' ? `${entry.series}:${entry.role}` : entry.key)).toEqual(['LOCO:generalization', 'alone', 'LOCO:control']);
    const [loco, , controls] = entries;
    expect(loco.kind === 'series' && loco.items.map((item) => item.seriesMember)).toEqual(['cptac', 'rih', 'tcga']);
    // Members sort as numbers where they hold numbers.
    expect(controls.kind === 'series' && controls.items.map((item) => item.seriesMember)).toEqual(['wt9', 'wt10']);
  });

  it('names a series briefly', () => {
    const items = ['cptac', 'rih', 'rih_sm', 'sr386', 'tcga'].map((member) => filed(member, { seriesMember: member }));
    expect(memberSummary(items)).toBe('cptac, rih, rih_sm, sr386 and 1 more');
    expect(memberSummary(items.slice(0, 2))).toBe('cptac, rih');
    expect(memberSummary([filed('named only')])).toBe('named only');
  });

  it('sections downstream records by objective, then work spanning several, then work under none', () => {
    const runs = [{ id: 'r1', objective: 'aim-2' }, { id: 'r2', objective: null }, { id: 'r3', objective: 'aim-1' }, { id: 'r4', objective: SEVERAL }, { id: 'r5', objective: 'removed' }, { id: 'r6', objective: 'aim-1' }];
    const sections = sectionByObjective(runs, (run) => run.objective, aims);
    expect(sections.map((section) => [section.key, section.items.map((item) => item.id)])).toEqual([['aim-1', ['r3', 'r6']], ['aim-2', ['r1']], [SEVERAL, ['r4']], [UNSORTED, ['r2', 'r5']]]);
    expect(sections[0].objective?.name).toBe('Aim 1 · Validity');
    expect(sections.slice(2).map((section) => section.objective)).toEqual([null, null]);
    expect(sectionByObjective([], () => null, aims)).toEqual([]);
  });

  it('names one objective for work from several experiments only when they agree', () => {
    expect(sharedObjective([])).toBeNull();
    expect(sharedObjective(['aim-1', 'aim-1'])).toBe('aim-1');
    expect(sharedObjective([null, undefined])).toBeNull();
    expect(sharedObjective(['aim-1', undefined])).toBe(SEVERAL);
    expect(sharedObjective(['aim-1', 'aim-2'])).toBe(SEVERAL);
  });
});
