import type { ReactNode } from 'react';
import { changeValidationSource } from '../lib/split';
import type { ProtocolExploration, ProtocolSpec } from '../api/scientific';
import { ErrorNotice, Icon } from './ui';
import { Findings } from './ScientificUI';
import { PartitionLive } from './ProtocolExploration';

type Pools = NonNullable<ProtocolSpec['split']['pools']>;
type Role = 'train' | 'val' | 'test';

export function SplitPools({
  pools,
  validationFraction,
  onChange,
  onFractionChange,
  renderConditions,
  imported,
  live,
  targetField,
}: {
  pools: Pools;
  validationFraction: number;
  onChange: (update: Partial<Pools>, validationFraction?: number) => void;
  onFractionChange: (fraction: number) => void;
  renderConditions: (role: Role) => ReactNode;
  imported: ReactNode;
  live: { data?: ProtocolExploration; loading: boolean; error: Error | null };
  targetField: string;
}) {
  const fixedValidation = pools.validationSource === 'fixed';
  const roles: Role[] = fixedValidation ? ['test', 'train', 'val'] : ['test', 'train'];
  return (
    <section className="stack split-pools" aria-label="Training and test sources">
      <div className="split-strategy-heading">
        <span className="split-section-caption">A · DATA SELECTION</span>
        <h3>Set aside test data first</h3>
        <p className="muted">
          The remaining eligible groups become your training pool. You can also use your own
          training conditions.
        </p>
      </div>
      <div className="science-grid-two">
        <label className="label">
          How are your sets defined?
          <select
            className="field"
            value={pools.source}
            onChange={(event) =>
              onChange({
                source: event.target.value as Pools['source'],
                trainSelection: 'remaining',
                rules: { train: [], val: [], test: [] },
                imported:
                  event.target.value === 'imported'
                    ? { partitionLabels: {}, foldLabels: {}, testFoldLabels: [] }
                    : undefined,
              })
            }
          >
            <option value="rules">Select with conditions</option>
            <option value="imported">Use a predefined split column</option>
          </select>
        </label>
        <label className="label">
          Early-stop validation source
          <select
            className="field"
            value={pools.validationSource}
            onChange={(event) => {
              const next = changeValidationSource(
                pools,
                event.target.value as Pools['validationSource'],
                validationFraction,
              );
              onChange(next.pools, next.validationFraction);
            }}
          >
            <option value="training_fraction">Sample a percentage of training</option>
            <option value="fixed">Use my fixed validation set</option>
          </select>
        </label>
      </div>
      {pools.source === 'imported' ? (
        <>
          {imported}
          <p className="muted">
            Map the source values to training and test
            {fixedValidation ? ', plus validation' : ''}.
            {fixedValidation
              ? ' Existing validation assignments are preserved.'
              : ' If your file has a validation set, choose fixed validation above or explicitly map those rows into training.'}
          </p>
        </>
      ) : null}
      {roles.map((role) => (
        <div className={`stack split-pool-card split-pool-${role}`} key={role}>
          <div className="split-pool-heading">
            <span className="split-pool-symbol">
              <Icon
                name={role === 'test' ? 'lock' : role === 'train' ? 'features' : 'clock'}
                size={19}
              />
            </span>
            <div>
              <h4>
                {role === 'test'
                  ? '1. Final test set'
                  : role === 'train'
                    ? '2. Training pool'
                    : '3. Fixed validation set'}
              </h4>
              <p>
                {role === 'test'
                  ? 'Reserved for the final evaluation. Never used to fit or select the model.'
                  : role === 'train'
                    ? 'Available for model fitting and cross-validation.'
                    : 'A separate set for early stopping.'}
              </p>
            </div>
          </div>
          {role === 'train' && pools.source === 'rules' ? (
            <>
              <label className="label">
                Training set selection
                <select
                  className="field"
                  value={pools.trainSelection}
                  onChange={(event) =>
                    onChange({
                      trainSelection: event.target.value as Pools['trainSelection'],
                      rules: { ...pools.rules, train: [] },
                    })
                  }
                >
                  <option value="remaining">Use all remaining eligible slides</option>
                  <option value="rules">Define my own training conditions</option>
                </select>
              </label>
              {pools.trainSelection === 'remaining' ? (
                <p className="callout">
                  Training includes every eligible group outside your test and fixed validation
                  sets.
                </p>
              ) : null}
            </>
          ) : null}
          {pools.source === 'rules' &&
          !(role === 'train' && pools.trainSelection === 'remaining')
            ? renderConditions(role)
            : null}
          {live.loading ? (
            <p className="protocol-live-status" role="status">
              Updating selection…
            </p>
          ) : live.data?.partitions && live.data.cohort ? (
            <PartitionLive
              partition={live.data.partitions[role]}
              label={
                role === 'train'
                  ? 'selected training set'
                  : role === 'test'
                    ? 'reserved final test set'
                    : 'fixed validation set'
              }
              fields={[
                ...pools.rules[role].map((item) => item.field),
                ...(pools.imported?.partitionField ? [pools.imported.partitionField] : []),
                targetField,
              ]}
              total={live.data.cohort.totalSlides}
            />
          ) : (
            <p className="muted">Define both training and test to see set counts.</p>
          )}
        </div>
      ))}
      <div className="split-validation-control">
        <div className="split-pool-heading">
          <span className="split-pool-symbol">
            <Icon name="clock" size={19} />
          </span>
          <div>
            <h4>Early stopping</h4>
            <p>A validation set helps decide when training should stop.</p>
          </div>
        </div>
        {fixedValidation ? (
          <p className="muted">
            Your fixed validation set stays separate from fitting, CV assessment and final test.
            In site/cohort CV, validation groups from the held-out site are omitted from that
            plan.
          </p>
        ) : (
          <label className="label">
            Early-stop validation (% of each training set or fold)
            <span className="split-percent-field">
              <input
                className="field"
                type="number"
                min={1}
                max={90}
                value={validationFraction * 100}
                onChange={(event) => onFractionChange(Number(event.target.value) / 100)}
              />
              <span aria-hidden="true">%</span>
            </span>
            <small>Default: 15%. The final test set is never used for early stopping.</small>
          </label>
        )}
      </div>
      <ErrorNotice error={live.error} />
      {live.data?.findings.length ? <Findings findings={live.data.findings} /> : null}
      <p className="muted">
        These source selections apply to every strategy. Preview shows the exact assignments
        within each plan.
      </p>
    </section>
  );
}
