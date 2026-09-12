import { useId, type ReactNode } from 'react';
import {
  changeHeldOutSource,
  changeSplitStrategy,
  validationFractionDefault,
} from '../lib/split';
import { useQuery } from '@tanstack/react-query';
import { scientific, type ProtocolSpec } from '../api/scientific';
import {
  DistributionBars,
  FieldProfile,
  type ProtocolFieldContext,
} from './ProtocolExploration';
import { ErrorNotice } from './ui';
import { scienceKey } from './ScientificUI';
import './SplitStrategy.css';

type Split = ProtocolSpec['split'];
export const strategyNames: Partial<Record<Split['mode'], string>> = {
  kfold: 'K-fold cross-validation',
  monte_carlo: 'Monte Carlo cross-validation',
  leave_one_domain_out: 'Leave-one-site/cohort-out CV (LOSO/LOCO)',
  nested_kfold: 'Nested K-fold cross-validation',
  held_out: 'Development holdout',
};
const strategyDescriptions: Partial<Record<Split['mode'], string>> = {
  kfold: 'Rotate held-out assessment folds.',
  monte_carlo: 'Repeat independent random splits.',
  leave_one_domain_out: 'Assess one unseen site or cohort at a time.',
  nested_kfold: 'Separate model tuning from assessment.',
  held_out: 'Use one development assessment split.',
};
export { newDevelopmentSplit as newSplit } from '../lib/protocol';
const percent = (value: number) => `${Number((value * 100).toFixed(1))}%`;

export function SplitStrategy({
  split,
  onChange,
  seedsText,
  onSeedsChange,
  seedsValid,
  fieldContext,
  rules,
  imported,
  pools,
}: {
  split: Split;
  onChange: (value: Partial<Split>) => void;
  seedsText: string;
  onSeedsChange: (text: string) => void;
  seedsValid: boolean;
  fieldContext: ProtocolFieldContext;
  rules: ReactNode;
  imported: ReactNode;
  pools?: ReactNode;
}) {
  const strategyGroup = useId();
  const explicitPools = (split.version ?? 1) >= 3;
  const development = split.version === 4;
  const validation = split.validationFraction ?? validationFractionDefault(split.version);
  const test =
    split.mode === 'kfold'
      ? 1 / split.folds
      : split.mode === 'nested_kfold'
        ? 1 / (split.outerFolds ?? 5)
        : (split.testFraction ?? 0.2);
  const randomHeldOut =
    (!explicitPools || development) &&
    split.mode === 'held_out' &&
    (split.heldOutSource ?? 'fractions') === 'fractions';
  const showPercentages =
    split.mode === 'kfold' ||
    split.mode === 'monte_carlo' ||
    split.mode === 'nested_kfold' ||
    randomHeldOut;
  const valFromRules =
    split.mode === 'held_out' && split.heldOutSource === 'rules' && split.rules.val.length > 0;
  const predefined =
    !explicitPools && split.mode === 'held_out' && split.heldOutSource === 'imported';
  return (
    <div className="stack split-strategy-workbench">
      {explicitPools ? pools : null}
      <div className="split-strategy-heading">
        <div>
          <span className="split-section-caption">
            {development ? 'B · DEVELOPMENT STRATEGY' : explicitPools ? 'B · EVALUATION STRATEGY' : 'EVALUATION STRATEGY'}
          </span>
          <h3>{development ? 'Choose how to compare models' : 'Choose how to evaluate'}</h3>
          <p className="muted">
            {development
              ? 'Create reproducible development assessments within the selected training data.'
              : explicitPools
              ? 'Cross-validation runs within your training pool. The final test set stays reserved.'
              : 'Choose how training, early-stop validation and reported test sets are assigned.'}
          </p>
        </div>
      </div>
      <fieldset className="split-strategy-choices">
        <legend className="split-visually-hidden">Split strategy</legend>
        {Object.entries(strategyNames).map(([value, name]) => (
          <label
            className={`split-strategy-choice${split.mode === value ? ' is-selected' : ''}`}
            key={value}
          >
            <input
              type="radio"
              name={strategyGroup}
              value={value}
              checked={split.mode === value}
              onChange={(event) =>
                onChange(changeSplitStrategy(split, event.target.value as Split['mode']))
              }
            />
            <span
              className={`split-strategy-motif motif-${value}${explicitPools && value !== 'held_out' ? ' motif-cv-assessment' : ''}`}
              aria-hidden="true"
            >
              {[0, 1, 2, 3, 4].map((part) => (
                <i key={part} />
              ))}
            </span>
            <strong>{name}</strong>
            <small>{strategyDescriptions[value as Split['mode']]}</small>
          </label>
        ))}
      </fieldset>
      <div className="split-role-guide">
        <div className="split-role-training">
          <strong>Training</strong>
          <span>Fits the model.</span>
        </div>
        <div className="split-role-validation">
          <strong>Early-stop validation</strong>
          <span>Decides when training stops.</span>
        </div>
        {split.mode === 'nested_kfold' ? (
          <div className="split-role-tuning">
            <strong>Inner tuning</strong>
            <span>Compares model settings inside the outer training data.</span>
          </div>
        ) : null}
        {(development || (explicitPools && split.mode !== 'held_out')) ? (
          <div className="split-role-assessment">
            <strong>Development assessment</strong>
            <span>Held out within the training pool for each CV plan.</span>
          </div>
        ) : null}
        {!development ? <div className="split-role-test">
          <strong>{explicitPools ? 'Final test' : 'Reported test'}</strong>
          <span>Evaluates the selected model on unseen data.</span>
        </div> : null}
      </div>
      <p className="split-group-note">
        <span>Patient grouping</span> Known patients stay together. Each confirmed Slide ID
        fallback forms one group. Training seeds and stopping metrics belong to Experiments.
      </p>
      <div className="science-grid-two">
        <label className="label">
          Split seeds, separated by commas
          <input
            className="field"
            value={seedsText}
            onChange={(event) => onSeedsChange(event.target.value)}
          />
        </label>
        {split.mode === 'kfold' ? (
          <NumberSetting
            label="Number of folds"
            value={split.folds}
            min={2}
            max={10}
            onChange={(folds) => onChange({ folds })}
          />
        ) : null}
        {split.mode === 'monte_carlo' ? (
          <NumberSetting
            label="Number of repeats per seed"
            value={split.repeats ?? 5}
            min={1}
            max={100}
            onChange={(repeats) => onChange({ repeats })}
          />
        ) : null}
        {split.mode === 'nested_kfold' ? (
          <>
            <NumberSetting
              label="Outer folds"
              value={split.outerFolds ?? 5}
              min={2}
              max={10}
              onChange={(outerFolds) => onChange({ outerFolds })}
            />
            <NumberSetting
              label="Inner folds per outer fold"
              value={split.innerFolds ?? 3}
              min={2}
              max={10}
              onChange={(innerFolds) => onChange({ innerFolds })}
            />
          </>
        ) : null}
        {!explicitPools && split.mode === 'held_out' ? (
          <label className="label">
            Choose the held-out sets
            <select
              className="field"
              value={split.heldOutSource ?? 'fractions'}
              onChange={(event) =>
                onChange(
                  changeHeldOutSource(split, event.target.value as Split['heldOutSource']),
                )
              }
            >
              <option value="fractions">Generate from percentages</option>
              <option value="rules">Select with rules</option>
              <option value="imported">Use a predefined split column</option>
            </select>
          </label>
        ) : null}
        {split.mode === 'monte_carlo' || randomHeldOut ? (
          <NumberSetting
            label={
              development
                ? 'Development assessment (% of the selected training set)'
                : explicitPools
                ? 'CV assessment (% of the selected training set)'
                : 'Test (% of the eligible cohort)'
            }
            value={(split.testFraction ?? 0.2) * 100}
            min={1}
            max={90}
            onChange={(value) => onChange({ testFraction: value / 100 })}
          />
        ) : null}
        {!explicitPools && !valFromRules && !predefined ? (
          <NumberSetting
            label="Early-stop validation (% of the remaining training pool)"
            value={validation * 100}
            min={1}
            max={90}
            onChange={(value) => onChange({ validationFraction: value / 100 })}
          />
        ) : null}
      </div>
      {!seedsValid ? (
        <p className="callout callout-warning" role="alert">
          Enter integer seeds from 0 to 4294967295, separated by commas.
        </p>
      ) : null}
      <label className="science-check">
        <input
          type="checkbox"
          checked={split.stratify ?? true}
          onChange={(event) => onChange({ stratify: event.target.checked })}
        />
        <span>
          Stratify by target class
          <small>Keep class proportions similar where groups and set sizes allow.</small>
        </span>
      </label>
      {showPercentages && !(explicitPools && split.pools?.validationSource === 'fixed') ? (
        <AllocationPreview
          train={(1 - test) * (1 - validation)}
          val={(1 - test) * validation}
          test={test}
          nested={split.mode === 'nested_kfold'}
          trainingOnly={explicitPools}
          development={development}
        />
      ) : null}
      {split.mode === 'kfold' ? (
        <p className="callout">
          {development
            ? 'Every development group rotates through one assessment fold per split seed. Early-stop validation is drawn from the fitting groups or uses your fixed validation source.'
            : explicitPools
            ? 'Rotate assessment folds within your training set. Your selected test set stays reserved for final evaluation.'
            : 'Each fold takes a turn as the reported test set. Early-stop validation comes from the other folds. Each group is assigned to the reported test set once per seed.'}
        </p>
      ) : null}
      {split.mode === 'monte_carlo' ? (
        <p className="callout">
          {development
            ? 'Each repeat samples a new development assessment subset. Some groups may be assessed more than once or never; preview shows coverage.'
            : explicitPools
            ? 'Each repeat samples fitting and assessment groups from your training set. Your selected test set stays reserved for final evaluation.'
            : 'Each repeat draws a new train, validation and test split. A group may appear in test more than once, or never. Preview shows test coverage.'}
        </p>
      ) : null}
      {split.mode === 'nested_kfold' ? (
        <div className="callout">
          <p>
            {explicitPools
              ? 'Within your selected training set, hold out an outer assessment fold.'
              : 'Hold out an outer test fold.'}{' '}
            Inside the remaining data, rotate inner tuning folds.
          </p>
          <p>
            {development
              ? 'Compare configurations using inner folds and assess the selection procedure using outer folds. Outer assessment groups never enter inner training or tuning.'
              : explicitPools
              ? 'Use inner folds to compare settings and outer folds to assess them. Your selected test set stays reserved for final evaluation.'
              : 'After choosing settings, refit on the outer training pool with its own early-stop validation, then evaluate on the untouched outer test fold.'}
          </p>
        </div>
      ) : null}
      {split.mode === 'leave_one_domain_out' ? (
        <>
          {explicitPools ? (
            <p className="callout">
              {development
                ? 'Rotate sites or cohorts within development data. Early-stop validation uses fitting sites only.'
                : 'Rotate sites or cohorts within your selected training set. Early-stop validation uses source sites only. Your selected test set stays reserved for final evaluation.'}
            </p>
          ) : null}
          <DomainSettings split={split} onChange={onChange} fieldContext={fieldContext} />
        </>
      ) : null}
      {!explicitPools && split.mode === 'held_out' && split.heldOutSource === 'rules'
        ? rules
        : null}
      {predefined ? (
        <>
          <p className="muted">
            Map source values to training, validation and test. If no rows are assigned to
            validation, early-stop validation is sampled from the training pool.
          </p>
          <NumberSetting
            label="Early-stop validation if absent (% of the training pool)"
            value={validation * 100}
            min={1}
            max={90}
            onChange={(value) => onChange({ validationFraction: value / 100 })}
          />
          {imported}
        </>
      ) : null}
      <p className="muted">
        Preview & preflight calculates exact group assignments, class counts and feasibility.
        Percentages are approximate because a group stays intact.
      </p>
    </div>
  );
}
function NumberSetting({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="label">
      {label}
      <input
        className="field"
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
function AllocationPreview({
  train,
  val,
  test,
  nested,
  trainingOnly = false,
  development = false,
}: {
  train: number;
  val: number;
  test: number;
  nested: boolean;
  trainingOnly?: boolean;
  development?: boolean;
}) {
  if (![train, val, test].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))
    return null;
  return (
    <figure className="split-allocation">
      <figcaption>
        {development
          ? 'Approximate share of development data per assessment plan'
          : trainingOnly
          ? 'Approximate share of the selected training set per CV plan'
          : nested
            ? 'Outer refit and evaluation · approximate share of the cohort'
            : 'Approximate share of the cohort per evaluation'}
      </figcaption>
      <div className="split-allocation-bar" aria-hidden="true">
        <span className="split-train" style={{ width: percent(train) }} />
        <span className="split-val" style={{ width: percent(val) }} />
        <span
          className={trainingOnly ? 'split-assessment' : 'split-test'}
          style={{ width: percent(test) }}
        />
      </div>
      <div className="split-allocation-legend">
        <span>
          <i className="split-train" />
          Training <strong>{percent(train)}</strong>
        </span>
        <span>
          <i className="split-val" />
          Early-stop validation <strong>{percent(val)}</strong>
        </span>
        <span>
          <i className={trainingOnly ? 'split-assessment' : 'split-test'} />
          {development ? 'Development assessment' : trainingOnly ? 'CV assessment' : 'Reported test'} <strong>{percent(test)}</strong>
        </span>
      </div>
      <small className="muted">
        {development
          ? 'Assessment groups stay separate from fitting and early stopping. Percentages are applied within the selected development cohort.'
          : trainingOnly
          ? 'The final test set is separate. Within each CV plan, take early-stop validation from the remaining training fold.'
          : 'Remove the test set first, then take the selected validation percentage from the remaining training pool.'}
      </small>
    </figure>
  );
}
function DomainSettings({
  split,
  onChange,
  fieldContext,
}: {
  split: Split;
  onChange: (value: Partial<Split>) => void;
  fieldContext: ProtocolFieldContext;
}) {
  const { project, datasetId, dictionary } = fieldContext;
  const field = split.domainField ?? '';
  const values = useQuery({
    queryKey: [...scienceKey(project), 'domain-values', datasetId, field],
    queryFn: () =>
      scientific.queryDataset(project, datasetId, {
        field,
        search: '',
        filters: [],
        offset: 0,
        limit: 1,
      }),
    enabled: Boolean(datasetId && field && dictionary.some((item) => item.key === field)),
  });
  return (
    <div className="stack">
      <div className="science-grid-two">
        <label className="label">
          Site or cohort column
          <select
            className="field"
            value={field}
            onChange={(event) =>
              onChange({ domainField: event.target.value || undefined, heldOutDomains: [] })
            }
          >
            <option value="">Choose a column</option>
            {dictionary.map((item) => (
              <option key={item.key}>{item.key}</option>
            ))}
          </select>
        </label>
        <label className="label">
          Held-out domain policy
          <select
            className="field"
            value={split.domainPolicy ?? 'all'}
            onChange={(event) =>
              onChange({
                domainPolicy: event.target.value as Split['domainPolicy'],
                heldOutDomains: [],
              })
            }
          >
            <option value="all">Rotate through every site / cohort</option>
            <option value="selected">Evaluate selected sites / cohorts</option>
          </select>
        </label>
      </div>
      {field ? <FieldProfile {...fieldContext} field={field} /> : null}
      <ErrorNotice error={values.error} />
      {values.data ? (
        <DistributionBars
          caption={`${field} · slides across the full dataset, before eligibility rules`}
          values={values.data.valueCounts}
        />
      ) : null}
      {split.domainPolicy === 'selected' && field ? (
        <div className="stack">
          <p className="muted">
            Each selected value gets a separate evaluation. Its entire site or cohort is held
            out.
          </p>
          <div className="science-checkbox-grid">
            {values.data?.valueCounts
              .filter((item) => item.value !== null)
              .map((item) => (
                <label className="science-check" key={item.value}>
                  <input
                    type="checkbox"
                    checked={(split.heldOutDomains ?? []).includes(item.value!)}
                    onChange={(event) =>
                      onChange({
                        heldOutDomains: event.target.checked
                          ? [...(split.heldOutDomains ?? []), item.value!]
                          : (split.heldOutDomains ?? []).filter(
                              (value) => value !== item.value,
                            ),
                      })
                    }
                  />
                  <span>
                    {item.value}
                    <small>{item.count} dataset slides</small>
                  </span>
                </label>
              ))}
          </div>
          {values.data?.valuesTruncated ? (
            <label className="label">
              Additional domain values, separated by |
              <input
                className="field"
                value={(split.heldOutDomains ?? []).join(' | ')}
                onChange={(event) =>
                  onChange({
                    heldOutDomains: event.target.value
                      .split('|')
                      .map((value) => value.trim())
                      .filter(Boolean),
                  })
                }
              />
            </label>
          ) : null}
        </div>
      ) : null}
      <p className="callout">
        {split.version === 4
          ? 'The held-out site or cohort supplies development assessment.'
          : split.version === 3
          ? 'The held-out site or cohort supplies CV assessment within the selected training set. The final test set stays reserved.'
          : 'The held-out site or cohort supplies the reported test set.'}{' '}
        Training and early-stop validation use source sites only. Groups must have one
        consistent site or cohort value.
      </p>
    </div>
  );
}
