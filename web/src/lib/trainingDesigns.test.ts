import { describe, expect, it } from 'vitest';
import { splitPlanLabel } from '../api/development';
import { assessesEveryUnit, designLabel, foldLabel } from '../api/experimentResults';
import type { ProtocolSpec } from '../api/scientific';
import { plansPerSeed } from './experimentPredictors';
import { changeSplitStrategy, trainingDesignIssue, trainingDesignText, TRAINABLE_MODES } from './split';
import { newDevelopmentSplit } from './protocol';

type Split = ProtocolSpec['split'];
const split = (changes: Partial<Split>): Split => ({ ...newDevelopmentSplit(), ...changes });
const design = { strategy: 'folds' as const, splitUnit: 'patient' as const, groupByPatient: false, folds: 3, splitSeeds: [42], slideCount: 36, resamplingUnit: 'patient' as const };

describe('training designs beyond generated k-fold', () => {
  it('trains the designs that assess each unit at most once per seed, once they are complete', () => {
    expect(TRAINABLE_MODES).toEqual(['kfold', 'predefined_folds', 'leave_one_domain_out', 'held_out']);
    expect(trainingDesignIssue(split({ mode: 'kfold' }))).toBe('');
    expect(trainingDesignIssue(split({ mode: 'monte_carlo' }))).toMatch(/strategy that trains/);
    expect(trainingDesignIssue(split({ mode: 'predefined_folds' }))).toMatch(/column that assigns/);
    expect(trainingDesignIssue(split({ mode: 'predefined_folds', foldField: 'fold' }))).toBe('');
    expect(trainingDesignIssue(split({ mode: 'leave_one_domain_out' }))).toMatch(/site or cohort column/);
    expect(trainingDesignIssue(split({ mode: 'leave_one_domain_out', domainField: 'site', domainPolicy: 'selected', heldOutDomains: [] }))).toMatch(/at least one site/);
    expect(trainingDesignIssue(split({ mode: 'held_out', seeds: [1, 2] }))).toMatch(/one split seed/);
    expect(trainingDesignIssue(split({ mode: 'held_out', seeds: [1] }))).toBe('');
    // Changing strategy clears the column of another one.
    expect(changeSplitStrategy(split({ mode: 'predefined_folds', foldField: 'fold' }), 'kfold').foldField).toBeUndefined();
  });

  it('summarizes each design in a few words', () => {
    expect(trainingDesignText(split({ mode: 'kfold', folds: 5 }))).toBe('5 folds · 1 split seed · 15% early-stop validation');
    expect(trainingDesignText(split({ mode: 'predefined_folds', foldField: 'fold', seeds: [1, 2] }))).toBe('Predefined folds from fold · 2 split seeds · 15% early-stop validation');
    expect(trainingDesignText(split({ mode: 'leave_one_domain_out', domainField: 'site', domainPolicy: 'selected', heldOutDomains: ['B', 'C'] }))).toMatch(/^Leave one site out by site \(B, C\)/);
    expect(trainingDesignText(split({ mode: 'held_out', testFraction: 0.25 }))).toMatch(/^Held-out assessment of 25%/);
  });

  it('counts plans per seed from the design, or from the derived design when its values come from data', () => {
    expect(plansPerSeed(split({ mode: 'kfold', folds: 4 }))).toBe(4);
    expect(plansPerSeed(split({ mode: 'held_out' }))).toBe(1);
    expect(plansPerSeed(split({ mode: 'leave_one_domain_out', domainPolicy: 'selected', heldOutDomains: ['A', 'B'] }))).toBe(2);
    expect(plansPerSeed(split({ mode: 'predefined_folds', seeds: [1, 2] }))).toBeUndefined();
    expect(plansPerSeed(split({ mode: 'predefined_folds', seeds: [1, 2] }), 6)).toBe(3);
    expect(plansPerSeed(split({ mode: 'monte_carlo' }), 10)).toBeUndefined();
  });

  it('names plans and designs by what they assess', () => {
    expect(splitPlanLabel({ fold: 2 })).toBe('Fold 3');
    expect(splitPlanLabel({ fold: 0, domain: 'Hospital B' })).toBe('Held-out Hospital B');
    expect(splitPlanLabel({ fold: null })).toBe('Held-out assessment');
    expect(foldLabel({ fold: 0 }, 'held_out')).toBe('Held-out assessment');
    expect(foldLabel({ fold: 1, domain: 'B' }, 'leave_one_domain_out')).toBe('Held-out B');
    expect(designLabel(design)).toBe('3-fold cross-validation');
    expect(designLabel(design, { mode: 'predefined_folds', foldField: 'fold' })).toBe('3 predefined folds (fold)');
    expect(designLabel({ ...design, strategy: 'leave_one_domain_out' })).toBe('Leave one site out · 3 sites');
    expect(designLabel({ ...design, strategy: 'held_out', folds: 1 })).toBe('Held-out assessment');
    expect(assessesEveryUnit(design)).toBe(true);
    expect(assessesEveryUnit({ ...design, strategy: 'held_out' })).toBe(false);
    expect(assessesEveryUnit({ ...design, strategy: 'leave_one_domain_out' }, { domainPolicy: 'selected' })).toBe(false);
  });
});
