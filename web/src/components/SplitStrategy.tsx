import { useId } from 'react';
import { changeSplitStrategy, DEFAULT_VALIDATION_FRACTION, MAX_SPLIT_SEEDS } from '../lib/split';
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
  predefined_folds: 'Predefined folds',
  leave_one_domain_out: 'Leave one site/cohort out (LOSO/LOCO)',
  held_out: 'Held-out assessment',
  monte_carlo: 'Monte Carlo cross-validation',
  nested_kfold: 'Nested K-fold cross-validation',
};
const strategyDescriptions: Partial<Record<Split['mode'], string>> = {
  kfold: 'Rotate generated assessment folds.',
  predefined_folds: 'Use the fold each unit has in a dataset column, such as a published split.',
  leave_one_domain_out: 'Assess one unseen site or cohort at a time.',
  held_out: 'Assess one held-out share of the training set.',
  monte_carlo: 'Repeat independent random splits.',
  nested_kfold: 'Separate model tuning from assessment.',
};
/** Why a strategy only plans: training needs each unit assessed at most once per split seed. */
const planOnlyReasons: Partial<Record<Split['mode'], string>> = {
  monte_carlo: 'Planning only: a unit can be assessed more than once per seed.',
  nested_kfold: 'Planning only: needs a search inside each outer fold.',
};
const percent = (value: number) => `${Number((value * 100).toFixed(1))}%`;

/** Edits a development-only (version 4) split. HistoricalProtocols shows older designs. */
export function SplitStrategy({
  split,
  onChange,
  seedsText,
  onSeedsChange,
  seedsError = '',
  fieldContext,
  supportedModes,
  splitUnit = 'patient',
}: {
  split: Split;
  onChange: (value: Partial<Split>) => void;
  seedsText: string;
  onSeedsChange: (text: string) => void;
  /** Why the seeds cannot be saved; empty when they can (see `splitSeedsError`). */
  seedsError?: string;
  fieldContext: ProtocolFieldContext;
  supportedModes?: readonly Split['mode'][];
  splitUnit?: 'slide' | 'patient' | 'unknown';
}) {
  const strategyGroup = useId();
  const seedsErrorId = useId();
  const validation = split.validationFraction ?? DEFAULT_VALIDATION_FRACTION;
  const test =
    split.mode === 'kfold'
      ? 1 / split.folds
      : split.mode === 'nested_kfold'
        ? 1 / (split.outerFolds ?? 5)
        : (split.testFraction ?? 0.2);
  const showPercentages =
    split.mode === 'kfold' ||
    split.mode === 'monte_carlo' ||
    split.mode === 'nested_kfold' ||
    split.mode === 'held_out';
  return (
    <div className="stack split-strategy-workbench">
      <div className="split-strategy-heading">
        <div>
          <span className="split-section-caption">B · DEVELOPMENT STRATEGY</span>
          <h3>Choose how to compare models</h3>
          <p className="muted">
            Create reproducible development assessments within the selected training data.
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
              disabled={Boolean(supportedModes && !supportedModes.includes(value as Split['mode']))}
              onChange={(event) =>
                onChange(changeSplitStrategy(split, event.target.value as Split['mode']))
              }
            />
            <span
              className={`split-strategy-motif motif-${value}${value !== 'held_out' ? ' motif-cv-assessment' : ''}`}
              aria-hidden="true"
            >
              {[0, 1, 2, 3, 4].map((part) => (
                <i key={part} />
              ))}
            </span>
            <strong>{name}</strong>
            <small>{supportedModes && !supportedModes.includes(value as Split['mode']) ? planOnlyReasons[value as Split['mode']] ?? 'Planning only.' : strategyDescriptions[value as Split['mode']]}</small>
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
        <div className="split-role-assessment">
          <strong>Development assessment</strong>
          <span>Held out within the training pool for each CV plan.</span>
        </div>
      </div>
      <p className="split-group-note">
        {splitUnit === 'slide' ? split.groupByPatient ? <><span>Slide labels, case-grouped folds</span> Each slide keeps its own label and is scored on its own; all slides of a case share one fold and one side of early-stop validation.</> : <><span>Slide splitting</span> Each slide is assigned independently to folds and early-stop validation.</> : splitUnit === 'patient' ? <><span>Patient grouping</span> Known patients stay together. Each confirmed Slide ID fallback forms one group.</> : <><span>Split unit</span> Choose saved targets and splits to establish the unit used for folds and early-stop validation.</>} Choose training seeds and stopping metrics in the hyperparameters step.
      </p>
      <div className="science-grid-two">
        <label className="label">
          Split seeds, separated by commas
          <input
            className="field"
            value={seedsText}
            aria-invalid={seedsError ? true : undefined}
            aria-describedby={seedsError ? seedsErrorId : undefined}
            onChange={(event) => onSeedsChange(event.target.value)}
          />
          <small>{split.mode === 'held_out' ? 'One seed: a held-out assessment trains with one split. Repeat it over training seeds in the hyperparameters.' : split.mode === 'predefined_folds' || split.mode === 'leave_one_domain_out' ? `Up to ${MAX_SPLIT_SEEDS} seeds. The assessment folds are fixed; each seed draws a different early-stop validation.` : `Up to ${MAX_SPLIT_SEEDS} seeds. Each seed gives a different fold assignment.`}</small>
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
        {split.mode === 'monte_carlo' || split.mode === 'held_out' ? (
          <NumberSetting
            label="Development assessment (% of the selected training set)"
            value={(split.testFraction ?? 0.2) * 100}
            min={1}
            max={90}
            onChange={(value) => onChange({ testFraction: value / 100 })}
          />
        ) : null}
      </div>
      {seedsError ? (
        <p id={seedsErrorId} className="callout callout-warning" role="alert">
          {seedsError}
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
          <small>{splitUnit === 'slide' && !split.groupByPatient ? 'Keep slide class proportions similar where set sizes allow.' : splitUnit === 'patient' || split.groupByPatient ? 'Keep class proportions similar where groups and set sizes allow.' : 'Keep class proportions similar where set sizes allow.'}</small>
        </span>
      </label>
      {splitUnit === 'slide' ? <label className="science-check">
        <input
          type="checkbox"
          checked={Boolean(split.groupByPatient)}
          onChange={(event) => onChange({ groupByPatient: event.target.checked || undefined })}
        />
        <span>
          Keep all slides of a case in the same fold
          <small>Uses each slide's case identifier from the dataset (such as a case or accession number). No case is both trained on and assessed, which matches applying the model to new cases. Labels, targets and scoring stay per slide; a case whose parts have different grades keeps each slide's grade. Every training slide needs a case identifier.</small>
        </span>
      </label> : null}
      {showPercentages && split.pools?.validationSource !== 'fixed' ? (
        <AllocationPreview
          train={(1 - test) * (1 - validation)}
          val={(1 - test) * validation}
          test={test}
          splitUnit={splitUnit}
        />
      ) : null}
      {split.mode === 'kfold' ? (
        <p className="callout">
          {splitUnit === 'slide'
            ? split.groupByPatient
              ? 'Every training case rotates through one assessment fold per split seed, taking all its slides with it. Early-stop validation is drawn from the remaining fitting cases.'
              : 'Every training slide rotates through one assessment fold per split seed. Early-stop validation is drawn from the remaining fitting slides.'
            : splitUnit === 'unknown' ? 'Each split seed creates assessment folds and early-stop validation from the selected training records.'
            : 'Every development group rotates through one assessment fold per split seed. Early-stop validation is drawn from the fitting groups or uses your fixed validation source.'}
        </p>
      ) : null}
      {split.mode === 'monte_carlo' ? (
        <p className="callout">
          Each repeat samples a new development assessment subset. Some groups may be assessed
          more than once or never; preview shows coverage.
        </p>
      ) : null}
      {split.mode === 'nested_kfold' ? (
        <div className="callout">
          <p>
            Within your selected training set, hold out an outer assessment fold. Inside the
            remaining data, rotate inner tuning folds.
          </p>
          <p>
            Compare configurations using inner folds and assess the selection procedure using
            outer folds. Outer assessment groups never enter inner training or tuning.
          </p>
        </div>
      ) : null}
      {split.mode === 'leave_one_domain_out' ? (
        <>
          <p className="callout">
            Rotate sites or cohorts within development data. Early-stop validation uses fitting
            sites only.{split.domainPolicy === 'selected' ? ' Only the selected sites are assessed; the others always train, so out-of-fold results cover the selected sites only.' : ''}
          </p>
          <DomainSettings split={split} onChange={onChange} fieldContext={fieldContext} />
        </>
      ) : null}
      {split.mode === 'predefined_folds' ? <FoldSettings split={split} onChange={onChange} fieldContext={fieldContext} splitUnit={splitUnit} /> : null}
      {split.mode === 'held_out' ? (
        <p className="callout">
          One stratified share of the training set is assessed; early-stop validation is drawn from the rest. Out-of-fold results cover that share only, so k-fold gives a more stable estimate when every {splitUnit === 'slide' ? 'slide' : 'group'} should be assessed.
        </p>
      ) : null}
      <p className="muted">
        {splitUnit === 'slide' ? 'Preview calculates exact slide assignments, class counts and feasibility. Percentages are rounded to whole slides.' : splitUnit === 'patient' ? 'Preview & validate calculates exact group assignments, class counts and feasibility. Percentages are approximate because a group stays intact.' : 'Preview calculates exact assignments, class counts and feasibility for the saved split unit.'}
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
  splitUnit = 'patient',
}: {
  train: number;
  val: number;
  test: number;
  splitUnit?: 'slide' | 'patient' | 'unknown';
}) {
  if (![train, val, test].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))
    return null;
  return (
    <figure className="split-allocation">
      <figcaption>Approximate share of development data per assessment plan</figcaption>
      <div className="split-allocation-bar" aria-hidden="true">
        <span className="split-train" style={{ width: percent(train) }} />
        <span className="split-val" style={{ width: percent(val) }} />
        <span className="split-assessment" style={{ width: percent(test) }} />
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
          <i className="split-assessment" />
          Development assessment <strong>{percent(test)}</strong>
        </span>
      </div>
      <small className="muted">
        {`Assessment ${splitUnit === 'slide' ? 'slides' : splitUnit === 'patient' ? 'groups' : 'records'} stay separate from fitting and early stopping. Percentages are applied within the selected development cohort.`}
      </small>
    </figure>
  );
}
/** Predefined folds: the column that assigns each unit's assessment fold, and its values. */
function FoldSettings({
  split,
  onChange,
  fieldContext,
  splitUnit,
}: {
  split: Split;
  onChange: (value: Partial<Split>) => void;
  fieldContext: ProtocolFieldContext;
  splitUnit: 'slide' | 'patient' | 'unknown';
}) {
  const { project, datasetId, dictionary } = fieldContext;
  const field = split.foldField ?? '';
  const values = useQuery({
    queryKey: [...scienceKey(project), 'fold-values', datasetId, field],
    queryFn: () => scientific.queryDataset(project, datasetId, { field, search: '', filters: [], offset: 0, limit: 1 }),
    enabled: Boolean(datasetId && field && dictionary.some((item) => item.key === field)),
  });
  return (
    <div className="stack">
      <label className="label">
        Fold column
        <select className="field" value={field} onChange={(event) => onChange({ foldField: event.target.value || undefined })}>
          <option value="">Choose a column</option>
          {dictionary.map((item) => <option key={item.key}>{item.key}</option>)}
        </select>
        <small>For example the fold numbers of a published study, imported with the dataset.</small>
      </label>
      {field ? <FieldProfile {...fieldContext} field={field} /> : null}
      <ErrorNotice error={values.error} />
      {values.data ? <DistributionBars caption={`${field} · slides across the full dataset, before eligibility rules`} values={values.data.valueCounts} /> : null}
      <p className="callout">
        Each value is one assessment fold: its {splitUnit === 'slide' && !split.groupByPatient ? 'slides are' : 'groups are'} assessed by the model trained on every other fold, and early-stop validation is drawn from those. Every training {splitUnit === 'slide' && !split.groupByPatient ? 'slide' : 'group'} needs a value, and the slides of a group must share one. The folds are the same for every split seed.
      </p>
    </div>
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
        The held-out site or cohort supplies development assessment. Training and early-stop
        validation use source sites only. Groups must have one consistent site or cohort value.
      </p>
    </div>
  );
}
