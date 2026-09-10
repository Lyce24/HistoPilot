import type { ReactNode } from 'react';
import {
  changeHeldOutSource,
  changeSplitStrategy,
  DEFAULT_VALIDATION_FRACTION,
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
  held_out: 'Held-out validation',
};
export function newSplit(seeds: number[] = [42], folds = 5): Split {
  return {
    version: 3,
    pools: {
      source: 'rules',
      trainSelection: 'remaining',
      validationSource: 'training_fraction',
      rules: { train: [], val: [], test: [] },
    },
    mode: 'kfold',
    folds,
    seeds,
    stratify: true,
    validationFraction: DEFAULT_VALIDATION_FRACTION,
    testFraction: 0.2,
    repeats: 5,
    outerFolds: 5,
    innerFolds: 3,
    domainPolicy: 'all',
    heldOutDomains: [],
    heldOutSource: 'fractions',
    ratios: { train: 0.8, val: 0.2, test: 0 },
    rules: { train: [], val: [], test: [] },
  };
}
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
  const explicitPools = split.version === 3;
  const validation = split.validationFraction ?? validationFractionDefault(split.version);
  const test =
    split.mode === 'kfold'
      ? 1 / split.folds
      : split.mode === 'nested_kfold'
        ? 1 / (split.outerFolds ?? 5)
        : (split.testFraction ?? 0.2);
  const randomHeldOut =
    !explicitPools &&
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
    <div className="stack">
      <label className="label">
        Split strategy
        <select
          className="field"
          value={split.mode}
          onChange={(event) =>
            onChange(changeSplitStrategy(split, event.target.value as Split['mode']))
          }
        >
          {Object.entries(strategyNames).map(([value, name]) => (
            <option value={value} key={value}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <div className="split-role-guide">
        <div>
          <strong>Training</strong>
          <span>Fits the model.</span>
        </div>
        <div>
          <strong>Early-stop validation</strong>
          <span>Decides when training stops.</span>
        </div>
        {split.mode === 'nested_kfold' ? (
          <div>
            <strong>Inner tuning</strong>
            <span>Compares model settings inside the outer training data.</span>
          </div>
        ) : null}
        <div>
          <strong>{explicitPools ? 'Final test' : 'Reported test'}</strong>
          <span>Evaluates the selected model on unseen data.</span>
        </div>
      </div>
      <p className="muted">
        Patients stay together in every set. Confirmed Slide ID fallbacks each form a separate
        group. Choose the stopping metric and model settings later in MIL experiments.
      </p>
      {explicitPools ? pools : null}
      <div className="science-grid-two">
        <label className="label">
          Seeds, separated by commas
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
              explicitPools
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
        />
      ) : null}
      {split.mode === 'kfold' ? (
        <p className="callout">
          {explicitPools
            ? 'Rotate assessment folds within your training set. Your selected test set stays reserved for final evaluation.'
            : 'Each fold takes a turn as the reported test set. Early-stop validation comes from the other folds. Each group is assigned to the reported test set once per seed.'}
        </p>
      ) : null}
      {split.mode === 'monte_carlo' ? (
        <p className="callout">
          {explicitPools
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
            {explicitPools
              ? 'Use inner folds to compare settings and outer folds to assess them. Your selected test set stays reserved for final evaluation.'
              : 'After choosing settings, refit on the outer training pool with its own early-stop validation, then evaluate on the untouched outer test fold.'}
          </p>
        </div>
      ) : null}
      {split.mode === 'leave_one_domain_out' ? (
        <>
          {explicitPools ? (
            <p className="callout">
              Rotate sites or cohorts within your selected training set. Early-stop validation
              uses source sites only. Your selected test set stays reserved for final
              evaluation.
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
}: {
  train: number;
  val: number;
  test: number;
  nested: boolean;
  trainingOnly?: boolean;
}) {
  if (![train, val, test].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))
    return null;
  return (
    <figure className="split-allocation">
      <figcaption>
        {trainingOnly
          ? 'Approximate share of the selected training set per CV plan'
          : nested
            ? 'Outer refit and evaluation · approximate share of the cohort'
            : 'Approximate share of the cohort per evaluation'}
      </figcaption>
      <div className="split-allocation-bar" aria-hidden="true">
        <span className="split-train" style={{ width: percent(train) }} />
        <span className="split-val" style={{ width: percent(val) }} />
        <span className="split-test" style={{ width: percent(test) }} />
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
          <i className="split-test" />
          {trainingOnly ? 'CV assessment' : 'Reported test'} <strong>{percent(test)}</strong>
        </span>
      </div>
      <small className="muted">
        {trainingOnly
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
        {split.version === 3
          ? 'The held-out site or cohort supplies CV assessment within the selected training set. The final test set stays reserved.'
          : 'The held-out site or cohort supplies the reported test set.'}{' '}
        Training and early-stop validation use source sites only. Groups must have one
        consistent site or cohort value.
      </p>
    </div>
  );
}
