import { describe, expect, it } from 'vitest';
import type { Objective } from '../api/objectives';
import { groupExperiments, libraryEntries, memberSummary, UNSORTED, type Filed } from './experimentLibrary';

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
});
