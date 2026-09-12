import { useState } from 'react';
import type { ProtocolSpec } from '../api/scientific';
import { inferTargetSettings, preservePositiveClass } from '../lib/protocol';
import { DistributionBars, FieldProfile, type ProtocolFieldContext } from './ProtocolExploration';
import { ErrorNotice, Icon } from './ui';

export interface PredictionTargetEditorProps {
  target: ProtocolSpec['target'];
  fieldContext: ProtocolFieldContext;
  unlinkedSlideCount?: number;
  labelValues: {
    data?: {
      valueCounts: { value: string | null; count: number }[];
      valuesTruncated: boolean;
    };
    isPending: boolean;
    error: Error | null;
  };
  rawValues: string[];
  dataLabel: string;
  onChooseTarget: (field: string) => void | Promise<void>;
  onChange: (update: Partial<ProtocolSpec['target']>) => void;
}

/** Keep development and independent test-cohort targets on the same editing workflow. */
export default function PredictionTargetEditor({
  target, fieldContext, unlinkedSlideCount, labelValues, rawValues, dataLabel,
  onChooseTarget, onChange,
}: PredictionTargetEditorProps) {
  const columns = fieldContext.dictionary.map((item) => item.key);
  const [mappingError, setMappingError] = useState<{
    labels: Record<string, string>;
    message: string;
  } | null>(null);
  function updateTarget(update: Partial<ProtocolSpec['target']>) {
    setMappingError(null);
    onChange(update);
  }
  return (
    <div className="stack">
      {unlinkedSlideCount ? (
        <div className="callout callout-warning">
          <strong>
            {unlinkedSlideCount} slides have unresolved
            Patient_ID.
          </strong>{' '}
          Revise this dataset to supply a patient mapping or explicitly confirm Slide ID
          fallback. Fallback creates one group per unresolved slide; it cannot establish
          which slides belong to the same patient.
        </div>
      ) : null}
      <div className="science-grid-two">
        <label className="label">
          Target attribute
          <small>The column containing the answer you want to predict.</small>
          <select
            className="field"
            value={target.field}
            onChange={(event) => void onChooseTarget(event.target.value)}
          >
            <option value="">Choose a target</option>
            {columns.map((column) => (
              <option key={column}>{column}</option>
            ))}
          </select>
        </label>
        <label className="label">
          Task
          <select
            className="field"
            value={target.task}
            onChange={(event) =>
              updateTarget({
                task: event.target.value as ProtocolSpec['target']['task'],
                positiveClass:
                  event.target.value === 'binary_classification'
                    ? preservePositiveClass(
                        target.positiveClass,
                        target.classes,
                      )
                    : undefined,
              })
            }
          >
            <option value="">Choose a task</option>
            <option value="binary_classification">Binary classification</option>
            <option value="multiclass_classification">Multiclass classification</option>
          </select>
        </label>
        <label className="label">
          Label unit
          <select
            className="field"
            value={target.unit}
            onChange={(event) =>
              updateTarget({ unit: event.target.value as 'patient' | 'slide' })
            }
          >
            <option value="patient">
              Patient — consistent label across their slides
            </option>
            <option value="slide">Slide / case — keep known patients together</option>
          </select>
        </label>
        <label className="label">
          Class names, separated by |
          <input
            className="field"
            value={target.classes.join(' | ')}
            onChange={(event) =>
              updateTarget({
                classes:
                  event.target.value === ''
                    ? []
                    : event.target.value.split('|').map((value) => value.trim()),
                positiveClass: preservePositiveClass(
                  target.positiveClass,
                  event.target.value.split('|').map((value) => value.trim()),
                ),
              })
            }
            placeholder="Filled from the selected target"
          />
        </label>
        <label className="label">
          Positive class
          <select
            className="field"
            value={
              target.task === 'binary_classification'
                ? (target.positiveClass ?? '')
                : ''
            }
            disabled={target.task !== 'binary_classification'}
            onChange={(event) =>
              updateTarget({ positiveClass: event.target.value || undefined })
            }
          >
            <option value="">
              {target.task === 'multiclass_classification'
                ? 'Not used for multiclass'
                : 'Choose positive class'}
            </option>
            {target.task === 'binary_classification'
              ? target.classes.map((value, index) => (
                  <option key={index}>{value}</option>
                ))
              : null}
          </select>
          <small>Choose explicitly for binary tasks; source value order does not determine the positive outcome.</small>
        </label>
      </div>
      {target.field ? (
        <div className="stack">
          <FieldProfile {...fieldContext} field={target.field} />
          {labelValues.isPending ? (
            <p className="protocol-live-status" role="status">
              Reading target values…
            </p>
          ) : labelValues.data ? (
            <DistributionBars
              values={labelValues.data.valueCounts}
              caption={`${target.field} · ${dataLabel}`}
              distinctCount={
                labelValues.data.valuesTruncated
                  ? undefined
                  : labelValues.data.valueCounts.length
              }
            />
          ) : null}
        </div>
      ) : null}
      <div className="science-subheading">
        <h3>Match source values to classes</h3>
        <button
          type="button"
          className="btn btn-secondary btn-small"
          disabled={!rawValues.length || labelValues.data?.valuesTruncated}
          onClick={() =>
            updateTarget({
              ...inferTargetSettings(rawValues),
              positiveClass:
                rawValues.length === 2
                  ? preservePositiveClass(target.positiveClass, rawValues)
                  : undefined,
            })
          }
        >
          Use observed source values
        </button>
      </div>
      <p className="muted">
        Classes are filled from the source values. Choose the positive class for a binary
        task, and edit any mapping below. Original dataset values stay unchanged.
      </p>
      <ErrorNotice error={labelValues.error} />
      {labelValues.data?.valuesTruncated ? (
        <p className="callout callout-warning">
          This field has more source values than the preview can display. Choose a categorical
          target or enter the task, classes and label mapping yourself.
        </p>
      ) : null}
      {target.field &&
      labelValues.data &&
      !labelValues.data.valuesTruncated &&
      rawValues.length < 2 ? (
        <p className="callout callout-warning">
          {rawValues.length
            ? 'This target has only one non-missing value.'
            : 'This target has no non-missing values.'}{' '}
          Choose a target with at least two classes for classification.
        </p>
      ) : null}
      {mappingError?.labels === target.labels ? (
        <p className="callout callout-warning" role="alert">{mappingError.message}</p>
      ) : null}
      <div className="science-label-map">
        {Object.entries(target.labels).map(([raw, mapped], index) => (
          <div className="science-label-row" key={index}>
            <label className="label">
              Source value
              <input
                className="field"
                value={raw}
                onChange={(event) => {
                  const nextRaw = event.target.value;
                  if (nextRaw !== raw && Object.hasOwn(target.labels, nextRaw)) {
                    setMappingError({
                      labels: target.labels,
                      message: `Source value “${nextRaw}” already has a mapping. Choose a unique source value.`,
                    });
                    return;
                  }
                  updateTarget({
                    labels: Object.fromEntries(
                      Object.entries(target.labels).map(([key, value]) => [
                        key === raw ? nextRaw : key,
                        value,
                      ]),
                    ),
                  });
                }}
              />
            </label>
            <Icon name="arrow" />
            <label className="label">
              Class
              <select
                className="field"
                value={mapped}
                onChange={(event) =>
                  updateTarget({
                    labels: {
                      ...target.labels,
                      [raw]: event.target.value,
                    },
                  })
                }
              >
                <option value="">Choose class</option>
                {target.classes.map((value, at) => (
                  <option key={at}>{value}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="icon-button"
              aria-label={`Remove mapping ${raw}`}
              onClick={() =>
                updateTarget({
                  labels: Object.fromEntries(
                    Object.entries(target.labels).filter(([key]) => key !== raw),
                  ),
                })
              }
            >
              <Icon name="close" />
            </button>
          </div>
        ))}
      </div>
      <button
        type="button"
        className="btn btn-secondary btn-small science-fit"
        onClick={() => {
          let raw = rawValues.find((value) => !Object.hasOwn(target.labels, value));
          if (raw === undefined) {
            let suffix = Object.keys(target.labels).length + 1;
            while (Object.hasOwn(target.labels, `value_${suffix}`)) suffix += 1;
            raw = `value_${suffix}`;
          }
          updateTarget({
            labels: {
              ...target.labels,
              [raw]: target.classes[0] ?? '',
            },
          });
        }}
      >
        <Icon name="plus" size={15} /> Add label mapping
      </button>
      <div className="science-grid-two">
        <label className="label">
          Missing target values
          <select
            className="field"
            value={target.missing}
            onChange={(event) =>
              updateTarget({
                missing: event.target.value as 'block' | 'exclude',
              })
            }
          >
            <option value="block">Block until resolved</option>
            <option value="exclude">Exclude and record the reason</option>
          </select>
        </label>
        <label className="label">
          Unmapped target values
          <select
            className="field"
            value={target.unmapped}
            onChange={(event) =>
              updateTarget({
                unmapped: event.target.value as 'block' | 'exclude',
              })
            }
          >
            <option value="block">Block until resolved</option>
            <option value="exclude">Exclude and record the reason</option>
          </select>
        </label>
      </div>
    </div>
  );
}
