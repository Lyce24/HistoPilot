import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { Workspace } from '../api/types';
import { api } from '../api/client';
import { workspaceKey } from '../api/queries';
import { Badge, EmptyState, ErrorNotice, Icon, PageHeader, Panel } from '../components/ui';
import { useUIStore } from '../store/ui';
import { downloadJSON } from '../lib/download';

export default function Experiments({ workspace: w }: { workspace: Workspace }) {
  const encoderId = useUIStore((state) => state.encoderId) ?? w.encoders[0]?.id;
  const milId = useUIStore((state) => state.milId) ?? w.milModels[0]?.id;
  const setMil = useUIStore((state) => state.setMilId);
  const cohortId = useUIStore((state) => state.selectedCohortId);
  const setCohort = useUIStore((state) => state.setSelectedCohortId);
  const setResult = useUIStore((state) => state.setSelectedResultId);
  const cohort = w.cohortSnapshots.find((item) => item.id === cohortId) ?? w.cohortSnapshots.at(-1);
  const [matrix, setMatrix] = useState(() =>
    w.encoders
      .slice(0, 2)
      .flatMap((encoder, index) =>
        w.milModels.slice(0, index === 0 ? 2 : 1).map((model) => `${encoder.id}:${model.id}`),
      ),
  );
  const [seeds, setSeeds] = useState('42, 84');
  const [folds, setFolds] = useState('5');
  const [aggregation, setAggregation] = useState<'mean' | 'max'>('mean');
  const [formError, setFormError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const client = useQueryClient();
  const create = useMutation({
    mutationFn: api.createExperiments,
    onSuccess: async (result) => {
      await client.invalidateQueries({ queryKey: workspaceKey });
      setMessage(
        `${result.drafts.length} experiment drafts saved in the workspace. No jobs were started.`,
      );
    },
  });
  const remove = useMutation({
    mutationFn: api.deleteExperiment,
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: workspaceKey });
    },
  });
  const manifest = useMutation({
    mutationFn: api.experimentManifest,
    onSuccess: (value) => downloadJSON('histopilot-experiment-manifest.json', value),
  });
  const parsedSeeds = seeds.split(',').map((seed) => seed.trim());
  const seedsValid =
    parsedSeeds.length > 0 &&
    parsedSeeds.every(
      (seed) =>
        /^\d+$/.test(seed) && Number.isSafeInteger(Number(seed)) && Number(seed) <= 4294967295,
    );
  const foldsValid = Number.isInteger(Number(folds)) && Number(folds) >= 2 && Number(folds) <= 10;
  const estimate =
    seedsValid && foldsValid
      ? matrix.length * new Set(parsedSeeds.map(Number)).size * Number(folds)
      : null;
  const encoderName = (id: string) => w.encoders.find((encoder) => encoder.id === id)?.name ?? id;
  const modelName = (id: string) => w.milModels.find((model) => model.id === id)?.name ?? id;
  function saveDrafts() {
    setFormError(null);
    setMessage('');
    if (!cohort) {
      setFormError(
        new Error('Save a cohort first so every experiment has an explicit patient population.'),
      );
      return;
    }
    if (!seedsValid) {
      setFormError(new Error('Enter comma-separated integer seeds, such as 42, 84.'));
      return;
    }
    if (!folds.trim() || !Number.isInteger(Number(folds))) {
      setFormError(new Error('Enter an integer fold count.'));
      return;
    }
    create.mutate({
      cohortId: cohort.id,
      pairs: matrix,
      seeds: parsedSeeds.map(Number),
      folds: Number(folds),
      aggregation,
    });
  }
  return (
    <>
      <PageHeader
        eyebrow="04 / EXPERIMENT DESIGN"
        title="MIL experiments"
        description="Compose a controlled PFM × MIL comparison. The service validates and saves each experiment against a persisted cohort and feature set."
        actions={
          <>
            <button
              className="btn btn-secondary"
              disabled={!w.drafts.length}
              onClick={() =>
                downloadJSON('histopilot-experiment-drafts.json', {
                  schema_version: '0.1.0',
                  mode: 'synthetic-demo',
                  executable: false,
                  dataset: w.dataset,
                  cohorts: w.cohortSnapshots,
                  split: w.split,
                  featureSets: w.featureSets,
                  experiments: w.drafts,
                })
              }
            >
              <Icon name="download" />
              Export drafts
            </button>
            <button className="btn btn-primary" disabled={create.isPending} onClick={saveDrafts}>
              <Icon name="plus" />
              {create.isPending ? 'Saving drafts…' : 'Create experiment drafts'}
            </button>
          </>
        }
      />
      <ErrorNotice error={formError ?? create.error ?? remove.error ?? manifest.error} />
      {message ? (
        <div className="callout" role="status">
          {message}
        </div>
      ) : null}
      <div className="section-heading">
        <div>
          <h2>MIL registry</h2>
          <p className="muted">Select a model, then include it with your encoder in the matrix</p>
        </div>
        <Badge>Training adapters not connected</Badge>
      </div>
      <div className="grid-3">
        {w.milModels.map((model, index) => (
          <button
            key={model.id}
            className={`card select-card ${milId === model.id ? 'selected' : ''}`}
            onClick={() => setMil(model.id)}
            aria-pressed={milId === model.id}
          >
            <span className="model-card-top">
              <span className={`model-mark model-mark-${index}`}>
                <Icon name={index === 0 ? 'cohort' : 'provenance'} size={23} />
              </span>
              <span className="selection-indicator">
                {milId === model.id ? <Icon name="check" size={14} /> : null}
              </span>
            </span>
            <h3>{model.name}</h3>
            <p className="muted">{model.description}</p>
            <span className="model-card-footer">
              <Badge>{model.adapter}</Badge>
              <span>{milId === model.id ? 'Selected model' : 'Select model'}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="grid-2">
        <Panel title="PFM × MIL matrix" subtitle="Each selected pair becomes one experiment draft">
          <div className="table-wrap">
            <table className="experiment-matrix">
              <thead>
                <tr>
                  <th>Encoder</th>
                  {w.milModels.map((model) => (
                    <th key={model.id}>{model.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {w.encoders.map((encoder) => (
                  <tr key={encoder.id}>
                    <th scope="row">{encoder.name}</th>
                    {w.milModels.map((model) => {
                      const key = `${encoder.id}:${model.id}`;
                      const checked = matrix.includes(key);
                      return (
                        <td key={model.id}>
                          <button
                            className={`matrix-cell ${checked ? 'selected' : ''}`}
                            aria-label={`Include ${encoder.name} with ${model.name}`}
                            aria-pressed={checked}
                            onClick={() =>
                              setMatrix((current) =>
                                current.includes(key)
                                  ? current.filter((pair) => pair !== key)
                                  : [...current, key],
                              )
                            }
                          >
                            {checked ? <Icon name="check" size={16} /> : '+'}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="matrix-summary">
            <span>
              <strong>{matrix.length}</strong> model pairs selected
            </span>
            <button
              className="btn btn-secondary btn-small"
              disabled={!encoderId || !milId}
              onClick={() =>
                setMatrix((current) => [...new Set([...current, `${encoderId}:${milId}`])])
              }
            >
              <Icon name="plus" size={14} />
              Add {encoderName(encoderId)} × {modelName(milId)}
            </button>
          </div>
        </Panel>
        <Panel
          title="Training contract"
          subtitle="Unsaved form values · draft fields and references validated by the service"
        >
          <div className="grid-2 training-fields">
            <label className="label">
              Cross-validation folds
              <input
                className="field"
                type="number"
                min="2"
                max="10"
                step="1"
                value={folds}
                onChange={(event) => setFolds(event.target.value)}
              />
              <small className="muted">Planned patient-grouped CV</small>
            </label>
            <label className="label">
              Random seeds
              <input
                className="field"
                type="text"
                value={seeds}
                onChange={(event) => setSeeds(event.target.value)}
              />
              <small className="muted">Comma-separated integers</small>
            </label>
            <label className="label">
              Patient aggregation
              <select
                className="field"
                value={aggregation}
                onChange={(event) => setAggregation(event.target.value as 'mean' | 'max')}
              >
                <option value="mean">Mean across slide predictions</option>
                <option value="max">Max across slide predictions</option>
              </select>
            </label>
            <div className="label">
              Split grouping<div className="field field-readonly">Patient ID</div>
              <small className="muted">Keep patient slides together</small>
            </div>
          </div>
          <div className="contract-total">
            <span>Planned fold / seed combinations</span>
            <strong>{estimate ?? '—'}</strong>
          </div>
          <p className="muted">
            This estimate describes the draft. CV folds, checkpoints, and training jobs are not
            generated yet.
          </p>
        </Panel>
      </div>
      <div className="experiment-context">
        <div>
          <Icon name="lock" />
          <label className="label">
            Saved cohort
            <select
              className="field"
              value={cohort?.id ?? ''}
              onChange={(event) => setCohort(event.target.value)}
            >
              {!cohort ? <option value="">Save a cohort first</option> : null}
              {w.cohortSnapshots.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.patientIds.length} patients · {item.id}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div>
          <span>Dataset</span>
          <strong className="mono">{w.dataset.id}</strong>
        </div>
        <div>
          <span>Source split</span>
          <strong className="mono">{w.split.id}</strong>
        </div>
        <a className="btn btn-secondary btn-small" href="#cohort">
          Build cohort →
        </a>
      </div>
      <Panel
        title="Experiment drafts"
        subtitle="Persisted by the service · execution is not implemented"
        actions={<Badge tone="purple">{w.drafts.length} saved</Badge>}
      >
        {w.drafts.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Encoder</th>
                  <th>MIL</th>
                  <th>Training plan</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {w.drafts.map((draft) => (
                  <tr key={draft.id}>
                    <td>
                      <strong className="mono">{draft.id}</strong>
                      <br />
                      <span className="muted mono">{draft.cohortId}</span>
                    </td>
                    <td>{encoderName(draft.encoderId)}</td>
                    <td>{modelName(draft.milId)}</td>
                    <td>
                      {draft.folds} folds × {draft.seeds.length} seeds
                    </td>
                    <td>
                      <Badge tone="purple">{draft.status}</Badge>
                    </td>
                    <td>
                      <div className="inline-actions">
                        <button
                          className="btn btn-secondary btn-small"
                          disabled={manifest.isPending}
                          onClick={() => manifest.mutate(draft.id)}
                          aria-label={`Export manifest ${draft.id}`}
                        >
                          <Icon name="download" size={15} />
                        </button>
                        <button
                          className="btn btn-secondary btn-small"
                          disabled={remove.isPending}
                          onClick={() => remove.mutate(draft.id)}
                          aria-label={`Remove draft ${draft.id}`}
                        >
                          <Icon name="close" size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="Ready when your design is"
            description="Select model pairs and save a cohort, then create drafts. Saved configurations remain available after refresh."
          />
        )}
      </Panel>
      <Panel
        title="Example comparison"
        subtitle="Synthetic metrics to demonstrate the evaluation workflow"
        actions={
          <a className="btn btn-secondary btn-small" href="#example-results">
            Compare results →
          </a>
        }
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Example run</th>
                <th>Encoder</th>
                <th>MIL model</th>
                <th>AUROC</th>
                <th>AUPRC</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {w.results.map((result) => (
                <tr key={result.id}>
                  <td>
                    <a
                      className="table-link mono"
                      href="#example-results"
                      onClick={() => setResult(result.id)}
                    >
                      {result.id} <Icon name="arrowUp" size={13} />
                    </a>
                  </td>
                  <td>{encoderName(result.encoderId)}</td>
                  <td>{modelName(result.milId)}</td>
                  <td>
                    <strong>{result.auroc.toFixed(3)}</strong>
                  </td>
                  <td>{result.auprc.toFixed(3)}</td>
                  <td>
                    <Badge>Synthetic result</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
