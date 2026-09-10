import { useState } from 'react';
import type { Workspace } from '../api/types';
import { scientific } from '../api/scientific';
import type { FeaturePreview, FeatureSpec } from '../api/scientific';
import { Badge, EmptyState, ErrorNotice, Icon, Metric, PageHeader, Panel } from '../components/ui';
import {
  DatasetSelect,
  Findings,
  SavedNotice,
  useConfigurations,
  useDatasets,
  useRefreshScientific,
} from '../components/ScientificUI';
import ServerFolderPicker from '../components/ServerFolderPicker';
export default function LocalFeatures({ workspace: w }: { workspace: Workspace }) {
  const project = w.project.id;
  const datasets = useDatasets(project);
  const configurations = useConfigurations(project, 'feature');
  const refresh = useRefreshScientific(project);
  const [spec, setSpec] = useState<FeatureSpec>({
    datasetId: w.dataset.id,
    path: w.sources.find((source) => source.role === 'features')?.path ?? '',
    encoderId: w.project.config.encoderId,
    fileSuffix: '.h5',
    idSuffix: '',
    recursive: false,
  });
  const [preview, setPreview] = useState<FeaturePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [message, setMessage] = useState('');
  const [selected, setSelected] = useState('');
  const configuration = configurations.data?.configurations.find((item) => item.id === selected);
  function edit(update: Partial<FeatureSpec>) {
    setSpec((current) => ({ ...current, ...update }));
    setPreview(null);
    setMessage('');
    setError(null);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setMessage('');
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error('Feature inspection failed.'));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="EXISTING EMBEDDINGS"
        title="PFM & features"
        description="Attach an existing HDF5 feature folder to a frozen dataset and review its slide coverage and array headers."
      />
      <ErrorNotice error={error ?? datasets.error ?? configurations.error} />
      <SavedNotice>{message}</SavedNotice>
      <div className="callout">
        <strong>Header and coverage validation.</strong> This check inspects existing files. Full
        feature-value validation and extraction are not connected, and no model execution is
        enabled.
      </div>
      <fieldset className="science-fieldset" disabled={busy}>
        <Panel
          title="Connect a feature folder"
          subtitle="Choose one encoder and representation per feature configuration."
        >
          <div className="stack">
            <div className="science-grid-two">
              <DatasetSelect
                versions={datasets.data?.datasets ?? []}
                value={spec.datasetId}
                onChange={(datasetId) => edit({ datasetId })}
              />
              <label className="label">
                Encoder identity (optional)
                <select
                  className="field"
                  value={spec.encoderId ?? ''}
                  onChange={(event) => edit({ encoderId: event.target.value || undefined })}
                >
                  <option value="">Unspecified — review provenance later</option>
                  {w.encoders.map((encoder) => (
                    <option key={encoder.id} value={encoder.id}>
                      {encoder.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="label">
                Existing HDF5 feature folder
                <input
                  className="field mono"
                  value={spec.path}
                  placeholder="/path/to/features_encoder"
                  onChange={(event) => edit({ path: event.target.value })}
                />
              </label>
              <div className="science-field-actions">
                <ServerFolderPicker
                  label="Browse feature folders"
                  onSelect={(path) => edit({ path })}
                />
              </div>
              <label className="label">
                Suffix to remove from filename stem (optional)
                <input
                  className="field mono"
                  value={spec.idSuffix}
                  placeholder="e.g. _features"
                  onChange={(event) => edit({ idSuffix: event.target.value })}
                />
              </label>
              <label className="science-check">
                <input
                  type="checkbox"
                  checked={spec.recursive}
                  onChange={(event) => edit({ recursive: event.target.checked })}
                />{' '}
                Include subfolders
              </label>
            </div>
            <p className="muted">
              Files ending in .h5 are matched to Slide_ID. Select the precise feature directory; do
              not combine encoders or patch and slide representations in one configuration.
            </p>
            <button
              type="button"
              className="btn btn-primary science-fit"
              disabled={!spec.datasetId || !spec.path.trim()}
              onClick={() =>
                void run(async () => setPreview(await scientific.featurePreview(project, spec)))
              }
            >
              {busy ? 'Inspecting features…' : 'Preview feature coverage'} <Icon name="arrow" />
            </button>
          </div>
        </Panel>
      </fieldset>
      {preview ? (
        <Panel
          title="Feature preflight"
          subtitle="Coverage reflects the selected dataset; files remain in place."
          actions={
            <Badge tone={preview.canFreeze ? 'green' : 'orange'}>
              {preview.canFreeze ? 'Can attach configuration' : 'Resolve blocking findings'}
            </Badge>
          }
        >
          <div className="science-metrics">
            <Metric
              label="Matched slides"
              value={preview.summary.matchedSlides}
              note={`${preview.summary.slideCount} dataset slides`}
            />
            <Metric
              label="Missing features"
              value={preview.summary.missingSlides}
              note={`${preview.summary.orphanFiles} files have no matching slide`}
            />
            <Metric
              label="Feature dimension"
              value={preview.summary.dimensions ?? 'Unresolved'}
              note="Read from array headers"
            />
            <Metric
              label="Patch count"
              value={preview.summary.patchCount.toLocaleString()}
              note="Header totals only"
            />
          </div>
          <Findings findings={preview.findings} />
          <details>
            <summary>Inspect matched file headers</summary>
            <FeatureFiles files={preview.files} />
          </details>
          <div className="science-savebar">
            <p>
              Attach preserves this configuration. A protocol must separately validate feature
              coverage for all included slides.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || !preview.canFreeze}
              onClick={() =>
                void run(async () => {
                  const result = await scientific.featureFreeze(project, spec, preview.previewHash);
                  await refresh();
                  setSelected(result.id);
                  setPreview(null);
                  setMessage('Feature configuration attached to this dataset version.');
                })
              }
            >
              <Icon name="lock" /> {busy ? 'Attaching…' : 'Attach feature configuration'}
            </button>
          </div>
        </Panel>
      ) : null}
      <Panel
        title="Saved feature configurations"
        subtitle="Each configuration belongs to one frozen dataset."
      >
        {!configurations.data?.configurations.length ? (
          <EmptyState
            title="No existing features attached"
            description="Choose a dataset and feature folder, then inspect its coverage above."
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Dataset</th>
                  <th>Encoder</th>
                  <th>Created</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {configurations.data.configurations.map((item) => (
                  <tr key={item.id}>
                    <td className="mono">{item.id.slice(-12)}</td>
                    <td className="mono">{item.manifest.datasetId.slice(-10)}</td>
                    <td>{(item.manifest.spec as FeatureSpec).encoderId ?? 'Unspecified'}</td>
                    <td>{new Date(item.createdAt).toLocaleString()}</td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-secondary btn-small"
                        onClick={() => setSelected(selected === item.id ? '' : item.id)}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {configuration ? (
          <div className="stack science-crosswalk">
            <Badge tone="purple">Frozen feature configuration</Badge>
            <Findings findings={configuration.manifest.findings ?? []} />
            <FeatureFiles files={configuration.manifest.files ?? []} />
            <details>
              <summary>Configuration provenance</summary>
              <pre className="code-block">{JSON.stringify(configuration, null, 2)}</pre>
            </details>
            <button
              type="button"
              className="btn btn-secondary science-fit"
              onClick={() => {
                edit(configuration.manifest.spec as FeatureSpec);
                setSelected('');
              }}
            >
              Use these settings for a new inspection
            </button>
          </div>
        ) : null}
      </Panel>
    </>
  );
}
function FeatureFiles({ files }: { files: FeaturePreview['files'] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Slide_ID</th>
            <th>Patches</th>
            <th>Dimensions</th>
            <th>Type</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file.slideId}>
              <td className="mono">{file.slideId}</td>
              <td>{file.patchCount.toLocaleString()}</td>
              <td>{file.dimensions}</td>
              <td>{file.dtype}</td>
              <td className="mono local-path">{file.path}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
