import { useState } from 'react';
import type { Workspace } from '../api/types';
import { Badge, EmptyState, Icon, PageHeader, Panel } from '../components/ui';
import { useUIStore } from '../store/ui';
import { downloadJSON } from '../lib/download';

export default function Features({ workspace: w }: { workspace: Workspace }) {
  const encoderId = useUIStore((state) => state.encoderId);
  const selectEncoder = useUIStore((state) => state.setEncoderId);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [contract, setContract] = useState(false);
  const selected =
    w.encoders.find((encoder) => encoder.id === (encoderId ?? w.project.config?.encoderId)) ??
    w.encoders[0];
  const detail = w.featureSets.find((feature) => feature.id === detailId);
  return (
    <>
      <PageHeader
        eyebrow="03 / REPRESENTATIONS"
        title="PFM & feature store"
        description="Choose a pathology foundation model from the service registry. Engines stay behind an adapter; lineage stays in HistoPilot."
        actions={
          <>
            <button className="btn btn-secondary" onClick={() => setContract(!contract)}>
              <Icon name="branch" />
              {contract ? 'Hide' : 'View'} encoder contract
            </button>
            <a className="btn btn-primary" href="#experiments">
              Configure MIL <Icon name="arrow" />
            </a>
          </>
        }
      />
      <div className="section-heading">
        <div>
          <h2>Encoder registry</h2>
          <p className="muted">Registry records supplied by the Python control service</p>
        </div>
        <Badge>Execution adapters not connected</Badge>
      </div>
      <div className="grid-3">
        {w.encoders.map((encoder, index) => (
          <button
            key={encoder.id}
            className={`card select-card ${selected?.id === encoder.id ? 'selected' : ''}`}
            onClick={() => selectEncoder(encoder.id)}
            aria-pressed={selected?.id === encoder.id}
          >
            <span className="model-card-top">
              <span className={`model-mark model-mark-${index}`}>
                <Icon name="features" size={23} />
              </span>
              <span className="selection-indicator">
                {selected?.id === encoder.id ? <Icon name="check" size={14} /> : null}
              </span>
            </span>
            <h3>{encoder.name}</h3>
            <p className="muted">{encoder.description}</p>
            <span className="spec-grid">
              <span>
                <small>Representation</small>
                <strong>{encoder.dimensions} dimensions</strong>
              </span>
              <span>
                <small>Planned adapter</small>
                <strong>{encoder.adapter}</strong>
              </span>
            </span>
            <span className="model-card-footer">
              <Badge>Not connected</Badge>
              <span>{selected?.id === encoder.id ? 'Selected encoder' : 'Select encoder'}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="grid-2">
        <Panel
          title="Extraction contract"
          subtitle={`Registry preview · ${selected?.name ?? 'No encoder'}`}
        >
          <ul className="detail-list">
            <li>
              <span>Dataset</span>
              <strong className="mono">{w.dataset.id}</strong>
            </li>
            <li>
              <span>Encoder</span>
              <strong>{selected?.name}</strong>
            </li>
            <li>
              <span>Planned adapter</span>
              <strong>{selected?.adapter}</strong>
            </li>
            <li>
              <span>Storage direction</span>
              <strong>HDF5 + metadata + coordinates</strong>
            </li>
          </ul>
          <p className="muted">
            The service will validate model access, checkpoint identity, patch size, and resolution
            before dispatching an isolated extraction worker.
          </p>
        </Panel>
        <Panel
          title="Reuse features across experiments"
          subtitle="Encode once, compare multiple MIL models"
        >
          <div className="feature-flow">
            <span className="feature-node">
              <Icon name="explorer" size={21} />
              <strong>Slides</strong>
              <small>Versioned source</small>
            </span>
            <span>→</span>
            <span className="feature-node feature-node-accent">
              <Icon name="features" size={21} />
              <strong>{selected?.name}</strong>
              <small>Patch encoder</small>
            </span>
            <span>→</span>
            <span className="feature-node">
              <Icon name="dataset" size={21} />
              <strong>Feature set</strong>
              <small>Reusable artifact</small>
            </span>
          </div>
          <div className="callout">
            Features belong on the local artifact filesystem. The browser receives registry and
            lineage metadata, not embedding arrays.
          </div>
        </Panel>
      </div>
      {contract ? (
        <Panel
          title="Encoder contract preview"
          subtitle="Service registry metadata · no extraction plan is persisted"
        >
          <pre className="code-block">{JSON.stringify(selected, null, 2)}</pre>
          <button
            className="btn btn-secondary"
            onClick={() =>
              downloadJSON('histopilot-encoder-registry-record.json', {
                mode: w.mode,
                executable: false,
                encoder: selected,
              })
            }
          >
            <Icon name="download" />
            Export metadata
          </button>
        </Panel>
      ) : null}
      <Panel
        title="Feature artifacts"
        subtitle={
          w.mode === 'synthetic-demo'
            ? 'Synthetic metadata illustrating the future feature store'
            : 'Features associated with this experiment'
        }
        actions={
          <Badge>
            {w.featureSets.length} {w.mode === 'synthetic-demo' ? 'demo records' : 'feature sets'}
          </Badge>
        }
      >
        {w.featureSets.length === 0 ? (
          <EmptyState
            title="No feature sets yet"
            description="You can save an existing feature folder in Dataset. Feature import and extraction are not connected yet."
          />
        ) : null}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Feature set</th>
                <th>Encoder</th>
                <th>Slides</th>
                <th>Dimensions</th>
                <th>Status</th>
                <th>
                  <span className="sr-only">Details</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {w.featureSets.map((feature) => (
                <tr key={feature.id} className={detailId === feature.id ? 'row-selected' : ''}>
                  <td>
                    <strong className="mono">{feature.id}</strong>
                    <br />
                    <span className="muted">{w.dataset.id}</span>
                  </td>
                  <td>{w.encoders.find((encoder) => encoder.id === feature.encoderId)?.name}</td>
                  <td>
                    {feature.slides} / {w.dataset.slideCount}
                  </td>
                  <td>{feature.dimensions}</td>
                  <td>
                    <Badge tone="purple">{feature.status}</Badge>
                  </td>
                  <td>
                    <button
                      className="btn btn-secondary btn-small"
                      aria-expanded={detailId === feature.id}
                      onClick={() => setDetailId(detailId === feature.id ? null : feature.id)}
                    >
                      Inspect <Icon name="arrow" size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {detail ? (
          <div className="artifact-detail">
            <h3>{detail.id}</h3>
            <p className="muted">
              Metadata only. No embeddings or model checkpoint are bundled with this example.
            </p>
            <pre className="code-block">{JSON.stringify(detail, null, 2)}</pre>
            <button
              className="btn btn-secondary btn-small"
              onClick={() =>
                downloadJSON(`${detail.id}.json`, {
                  mode: w.mode,
                  executable: false,
                  ...detail,
                })
              }
            >
              Export demo metadata
            </button>
          </div>
        ) : null}
      </Panel>
    </>
  );
}
