import { Fragment, useId, useState } from 'react';
import type { Page, Result, Workspace } from '../api/types';
import { Badge, EmptyState, Icon, Metric, PageHeader, Panel } from '../components/ui';
import { downloadJSON } from '../lib/download';
import { useUIStore } from '../store/ui';

type PageProps = { workspace: Workspace };
type MetricKey = 'auroc' | 'auprc' | 'accuracy';

const metricNames: Record<MetricKey, string> = {
  auroc: 'AUROC',
  auprc: 'AUPRC',
  accuracy: 'Accuracy',
};
const metricKeys: MetricKey[] = ['auroc', 'auprc', 'accuracy'];
const workflow: { page: Page; name: string; detail: string; tone: string }[] = [
  { page: 'dataset', name: 'Dataset', detail: 'Slides + tables', tone: 'green' },
  { page: 'cohort', name: 'Cohort', detail: 'Labels + splits', tone: 'green' },
  { page: 'features', name: 'Features', detail: 'PFM embeddings', tone: 'purple' },
  { page: 'experiments', name: 'Experiments', detail: 'MIL + validation', tone: 'neutral' },
  { page: 'evaluation', name: 'Evaluation', detail: 'Compare results', tone: 'neutral' },
  { page: 'explorer', name: 'Explorer', detail: 'Inspect regions', tone: 'neutral' },
  { page: 'provenance', name: 'Provenance', detail: 'Trace every result', tone: 'neutral' },
];

function resultModels(workspace: Workspace, result: Result) {
  return {
    encoder:
      workspace.encoders.find((model) => model.id === result.encoderId)?.name ?? result.encoderId,
    mil: workspace.milModels.find((model) => model.id === result.milId)?.name ?? result.milId,
  };
}

export function OverviewPage({ workspace }: PageProps) {
  const selectResult = useUIStore((state) => state.setSelectedResultId);
  const { dataset, patients, slides, results } = workspace;
  const labeledPatients = patients.filter((patient) => patient.kras.trim().length > 0).length;
  const coverage = patients.length
    ? `${Math.round((labeledPatients / patients.length) * 100)}%`
    : '—';
  const reviewCount = slides.filter((slide) => slide.status === 'Review').length;
  const partitionCount = (partition: string) =>
    patients.filter((patient) => patient.partition === partition).length;

  return (
    <>
      <PageHeader
        eyebrow="YOUR RESEARCH WORKSPACE"
        title="A clearer path from slides to insight."
        description="Build your cohort. Compare your models. Keep the full story behind every result."
        actions={
          <>
            <a className="btn btn-secondary" href="#dataset">
              <Icon name="dataset" /> Explore dataset
            </a>
            <a className="btn btn-primary" href="#experiments">
              <Icon name="plus" /> New experiment
            </a>
          </>
        }
      />
      <section className="workspace-banner">
        <div className="workspace-mark">
          <Icon name="experiments" size={24} />
        </div>
        <div>
          <div className="inline-actions">
            <h2>{workspace.project.name}</h2>
            <Badge tone="purple">Synthetic project</Badge>
          </div>
          <p>Colorectal cancer · binary classification · patient-level analysis</p>
        </div>
        <a href="#provenance" className="version-link">
          <Icon name="branch" size={15} /> {dataset.id} <Icon name="chevron" size={14} />
        </a>
      </section>
      <div className="grid-4 metrics">
        <Metric
          label="Patients"
          value={dataset.patientCount}
          note={`${dataset.specimenCount} specimens · ${dataset.slideCount} synthetic slide records`}
        />
        <Metric
          label="Label coverage"
          value={coverage}
          note="KRAS labels in the synthetic source table"
        />
        <Metric
          label="Feature sets"
          value={workspace.featureSets.length}
          note="Illustrative artifacts from the service registry"
        />
        <Metric
          label="Experiment drafts"
          value={workspace.drafts.length}
          note={`${results.length} example results · drafts saved locally`}
        />
      </div>
      <section className="workflow-section">
        <div className="section-heading">
          <div>
            <h2>Your experiment workflow</h2>
            <p>One connected workspace, from source data to provenance.</p>
          </div>
          <span className="quiet-label">CLICK A STAGE TO EXPLORE</span>
        </div>
        <div className="pipeline">
          {workflow.map((stage, index) => (
            <a key={stage.page} className={`stage stage-${stage.tone}`} href={`#${stage.page}`}>
              <div className="stage-top">
                <span>{String(index + 1).padStart(2, '0')}</span>
                <Icon name={stage.page} size={19} />
              </div>
              <strong>{stage.name}</strong>
              <small>{stage.detail}</small>
              <span className="stage-arrow">
                <Icon name="arrow" size={15} />
              </span>
            </a>
          ))}
        </div>
      </section>
      <div className="overview-columns">
        <Panel
          title="Example results"
          subtitle="Illustrative comparisons on a shared dataset and split."
          actions={
            <a className="text-link" href="#example-results">
              View evaluation →
            </a>
          }
        >
          {results.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Experiment</th>
                    <th scope="col">Encoder / MIL</th>
                    <th scope="col">AUROC</th>
                    <th scope="col">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => {
                    const models = resultModels(workspace, result);
                    return (
                      <tr key={result.id}>
                        <td>
                          <a
                            className="table-link"
                            href="#example-results"
                            onClick={() => selectResult(result.id)}
                          >
                            EXP-{String(index + 1).padStart(3, '0')}
                          </a>
                          <small>KRAS · {result.splitId}</small>
                        </td>
                        <td>
                          {models.encoder}
                          <small>{models.mil}</small>
                        </td>
                        <td>
                          <strong className="mono">{result.auroc.toFixed(3)}</strong>
                        </td>
                        <td>
                          <Badge tone="purple">Example</Badge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No example results"
              description="The service has not returned any results for this workspace."
            />
          )}
        </Panel>
        <Panel title="Dataset at a glance" subtitle="Inspect the source before you build.">
          <div className="audit-row">
            <span className="audit-symbol green">
              <Icon name="check" size={15} />
            </span>
            <div>
              <strong>Patient → specimen → slide</strong>
              <p>
                {patients.length} patients resolve to {slides.length} synthetic slide records.
              </p>
            </div>
            <a href="#dataset" aria-label="Inspect patient hierarchy">
              <Icon name="chevron" size={16} />
            </a>
          </div>
          <div className="audit-row">
            <span className="audit-symbol purple">
              <Icon name="branch" size={15} />
            </span>
            <div>
              <strong>Patient grouping is explicit</strong>
              <p>
                {partitionCount('train')} train · {partitionCount('validation')} validation ·{' '}
                {partitionCount('test')} test patients in the fixed example.
              </p>
            </div>
            <a href="#cohort" aria-label="Inspect cohort and split">
              <Icon name="chevron" size={16} />
            </a>
          </div>
          <div className="audit-row">
            <span className="audit-symbol amber">
              <Icon name="info" size={15} />
            </span>
            <div>
              <strong>{reviewCount} slide records marked for review</strong>
              <p>Example QC flags to explore in the dataset.</p>
            </div>
            <a href="#dataset" aria-label="Review dataset quality flags">
              <Icon name="chevron" size={16} />
            </a>
          </div>
        </Panel>
      </div>
      <section className="lineage-note">
        <div className="lineage-note-icon">
          <Icon name="provenance" size={22} />
        </div>
        <div>
          <h3>No orphan results.</h3>
          <p>Every future result must connect to its run, models, features, and source data.</p>
        </div>
        <a href="#provenance" className="text-link">
          Follow the example lineage <Icon name="arrow" size={16} />
        </a>
      </section>
    </>
  );
}

export function EvaluationPage({ workspace }: PageProps) {
  const selectedId = useUIStore((state) => state.selectedResultId);
  const selectResult = useUIStore((state) => state.setSelectedResultId);
  const [metric, setMetric] = useState<MetricKey>('auroc');
  const selected =
    workspace.results.find((result) => result.id === selectedId) ?? workspace.results[0];
  if (!selected)
    return (
      <EmptyState
        title="No results to compare"
        description="Evaluation will appear here when the service provides result records."
      />
    );
  const models = resultModels(workspace, selected);

  return (
    <>
      <PageHeader
        eyebrow="COMPARE & UNDERSTAND"
        title="Evaluation"
        description="Compare example results with a shared dataset, target, and patient split."
        actions={<Badge tone="purple">Illustrative metrics</Badge>}
      />
      <div className="callout">
        <Icon name="info" />
        These values are synthetic examples supplied by the service. They were not computed from the
        demo patients and do not represent model performance.
      </div>
      <div className="grid-3 metrics">
        {metricKeys.map((key) => (
          <Metric
            key={key}
            label={`${metricNames[key]} · ${selected.id}`}
            value={selected[key].toFixed(3)}
            note={`${models.encoder} × ${models.mil}`}
          />
        ))}
      </div>
      <Panel
        title="Compare model combinations"
        subtitle="Select a metric and a result to inspect its context."
        actions={
          <label className="compact-field">
            <span className="sr-only">Comparison metric</span>
            <select
              className="field"
              value={metric}
              onChange={(event) => {
                const value = event.currentTarget.value;
                if (value === 'auroc' || value === 'auprc' || value === 'accuracy')
                  setMetric(value);
              }}
            >
              {metricKeys.map((key) => (
                <option key={key} value={key}>
                  {metricNames[key]}
                </option>
              ))}
            </select>
          </label>
        }
      >
        <div className="comparison-chart">
          {workspace.results.map((result) => {
            const names = resultModels(workspace, result);
            return (
              <button
                key={result.id}
                type="button"
                className={`comparison-row ${selected.id === result.id ? 'selected' : ''}`}
                onClick={() => selectResult(result.id)}
                aria-pressed={selected.id === result.id}
              >
                <span>
                  {names.encoder}
                  <small>{names.mil}</small>
                </span>
                <span className="bar-track" aria-hidden="true">
                  <span
                    className="bar-fill"
                    style={{ width: `${Math.max(0, Math.min(1, result[metric])) * 100}%` }}
                  />
                </span>
                <strong className="mono">
                  {result[metric].toFixed(3)}
                  <span className="sr-only"> {metricNames[metric]}</span>
                </strong>
              </button>
            );
          })}
          <div className="chart-axis" aria-hidden="true">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </div>
      </Panel>
      <div className="grid-2">
        <Panel title="Comparison contract" subtitle="References for the selected example.">
          <dl className="detail-list">
            <div>
              <dt>Dataset</dt>
              <dd className="mono">{workspace.dataset.id}</dd>
            </div>
            <div>
              <dt>Split</dt>
              <dd className="mono">{selected.splitId}</dd>
            </div>
            <div>
              <dt>Analysis unit</dt>
              <dd>Patient</dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>KRAS status</dd>
            </div>
            <div>
              <dt>Seed</dt>
              <dd>{selected.seed}</dd>
            </div>
            <div>
              <dt>Feature set</dt>
              <dd className="mono">{selected.featureSetId}</dd>
            </div>
          </dl>
        </Panel>
        <Panel title="Inspect this result" subtitle={selected.id}>
          <p className="muted">
            Connect metrics with their run and source artifacts. Real predictions, confidence
            intervals, calibration, and Plotly visualizations are planned.
          </p>
          <div className="inline-actions">
            <a className="btn btn-primary" href="#provenance">
              <Icon name="provenance" /> View lineage
            </a>
            <a className="btn btn-secondary" href="#explorer">
              <Icon name="explorer" /> Open explorer
            </a>
          </div>
        </Panel>
      </div>
    </>
  );
}

// Decorative geometry only. A future OpenSeadragon tile source will consume server WSI tiles.
const tissueCells = Array.from({ length: 95 }, (_, index) => ({
  x: 120 + ((index * 83) % 550),
  y: 110 + ((index * 61) % 270),
  rx: 12 + (index % 11),
  ry: 7 + (index % 8),
  rotation: index * 19,
  fill: ['#dbafc7', '#ce91b4', '#b781a8', '#ecd0dc'][index % 4],
}));
const previewRegions = [1, 2, 3, 4, 5, 6];

function TissueIllustration() {
  const identity = useId();
  const clipId = `${identity}-tissue`;
  const heatId = `${identity}-heat`;
  return (
    <svg
      className="tissue-art"
      viewBox="0 0 800 500"
      role="img"
      aria-label="Schematic tissue illustration, not a real whole-slide image"
    >
      <defs>
        <clipPath id={clipId}>
          <path d="M97 221C61 118 220 62 323 96S457 32 611 110s125 205 53 278-195 19-275 22-244 50-278-55 0-65-14-134Z" />
        </clipPath>
        <radialGradient id={heatId}>
          <stop stopColor="#e87736" stopOpacity=".75" />
          <stop offset=".45" stopColor="#f6bc4e" stopOpacity=".5" />
          <stop offset="1" stopColor="#f6bc4e" stopOpacity="0" />
        </radialGradient>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <rect width="800" height="500" fill="#eed8e3" />
        {tissueCells.map((cell, index) => (
          <Fragment key={index}>
            <ellipse
              cx={cell.x}
              cy={cell.y}
              rx={cell.rx}
              ry={cell.ry}
              transform={`rotate(${cell.rotation} ${cell.x} ${cell.y})`}
              fill={cell.fill}
              opacity=".7"
            />
            <ellipse cx={cell.x} cy={cell.y} rx="3" ry="5" fill="#79548c" opacity=".7" />
          </Fragment>
        ))}
      </g>
      <g className="attention-layer">
        <circle cx="390" cy="247" r="105" fill={`url(#${heatId})`} />
        <circle cx="227" cy="310" r="70" fill={`url(#${heatId})`} />
        <circle cx="584" cy="186" r="75" fill={`url(#${heatId})`} />
      </g>
    </svg>
  );
}

export function ExplorerPage({ workspace }: PageProps) {
  const selectedSlideId = useUIStore((state) => state.selectedSlideId);
  const selectSlide = useUIStore((state) => state.setSelectedSlideId);
  const selectedResultId = useUIStore((state) => state.selectedResultId);
  const [attention, setAttention] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [region, setRegion] = useState(0);
  const slide = workspace.slides.find((item) => item.id === selectedSlideId) ?? workspace.slides[0];
  const result =
    workspace.results.find((item) => item.id === selectedResultId) ?? workspace.results[0];
  if (!slide)
    return (
      <EmptyState
        title="No slides to explore"
        description="The workspace has no slide records. A real WSI reader is not connected yet."
      />
    );
  const patient = workspace.patients.find((item) => item.id === slide.patientId);

  return (
    <>
      <PageHeader
        eyebrow="EXPLORE & INTERPRET"
        title="Slide explorer"
        description="Keep the slide, its patient context, and the generating experiment in one view."
        actions={<Badge tone="purple">Schematic preview</Badge>}
      />
      <div className="explorer-layout">
        <section className="card viewer-card">
          <div className="panel-header">
            <div>
              <h2>Synthetic tissue preview</h2>
              <p>No WSI or model output is loaded.</p>
            </div>
            <label className="compact-field">
              <span className="sr-only">Select slide</span>
              <select
                className="field"
                value={slide.id}
                onChange={(event) => {
                  selectSlide(event.currentTarget.value);
                  setRegion(0);
                  setZoom(1);
                }}
              >
                {workspace.slides.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className={`viewer ${attention ? '' : 'hide-attention'}`}>
            <div className="viewer-chip">
              <Icon name="explorer" size={14} /> {slide.id} · illustration
            </div>
            <div className="tissue-transform" style={{ transform: `scale(${zoom})` }}>
              <TissueIllustration />
            </div>
            <span
              className="region-focus"
              aria-hidden="true"
              style={{
                left: `${30 + (region % 3) * 15}%`,
                top: `${36 + Math.floor(region / 3) * 20}%`,
              }}
            />
            <div className="viewer-bottom">
              <span>SCHEMATIC · NO PHYSICAL SCALE</span>
              <div className="zoom-controls">
                <button
                  type="button"
                  aria-label="Zoom out"
                  disabled={zoom <= 1}
                  onClick={() => setZoom((value) => Math.max(1, value - 0.25))}
                >
                  <Icon name="minus" size={16} />
                </button>
                <span aria-live="polite">{zoom.toFixed(2)}×</span>
                <button
                  type="button"
                  aria-label="Zoom in"
                  disabled={zoom >= 2.5}
                  onClick={() => setZoom((value) => Math.min(2.5, value + 0.25))}
                >
                  <Icon name="plus" size={16} />
                </button>
                <button
                  type="button"
                  aria-label="Reset zoom and region"
                  onClick={() => {
                    setZoom(1);
                    setRegion(0);
                  }}
                >
                  <Icon name="reset" size={15} />
                </button>
              </div>
            </div>
          </div>
          <div className="viewer-toolbar">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={attention}
                onChange={(event) => setAttention(event.currentTarget.checked)}
              />{' '}
              Show illustrative attention overlay
            </label>
            <span className="attention-scale" aria-hidden="true">
              Low <i /> High
            </span>
          </div>
        </section>
        <div className="stack">
          <Panel title="Example regions" subtitle="Select a tile to move the region marker.">
            <div className="tile-grid">
              {previewRegions.map((number, index) => (
                <button
                  key={number}
                  type="button"
                  className={`tissue-tile ${region === index ? 'selected' : ''}`}
                  onClick={() => setRegion(index)}
                  aria-label={`Inspect example region ${number}`}
                  aria-pressed={region === index}
                >
                  <span>R{number}</span>
                  <small>Preview</small>
                </button>
              ))}
            </div>
            <p className="small muted" aria-live="polite">
              Region {region + 1} selected · decorative illustration only.
            </p>
          </Panel>
          <Panel title="Source context" subtitle="Patient → specimen → slide">
            <dl className="detail-list">
              <div>
                <dt>Patient</dt>
                <dd>{slide.patientId}</dd>
              </div>
              <div>
                <dt>Specimen</dt>
                <dd>{slide.specimenId}</dd>
              </div>
              <div>
                <dt>Site</dt>
                <dd>{patient?.site ?? slide.site}</dd>
              </div>
              <div>
                <dt>Label</dt>
                <dd>{patient ? `KRAS ${patient.kras}` : 'Unavailable'}</dd>
              </div>
              <div>
                <dt>Result reference</dt>
                <dd>{result?.id ?? 'No result available'}</dd>
              </div>
            </dl>
            {result ? (
              <a href="#provenance" className="text-link">
                Trace result lineage <Icon name="arrow" size={15} />
              </a>
            ) : null}
          </Panel>
        </div>
      </div>
      <div className="callout">
        <Icon name="info" />
        This is a geometric preview. OpenSeadragon tiles with OpenSlide / cuCIM readers are planned;
        real patches and attention will use level-0 WSI coordinates from the service.
      </div>
    </>
  );
}

export function ProvenancePage({ workspace }: PageProps) {
  const selectedId = useUIStore((state) => state.selectedResultId);
  const result = workspace.results.find((item) => item.id === selectedId) ?? workspace.results[0];
  const manifest = result
    ? workspace.exampleManifests.find((item) => item.result.id === result.id)
    : undefined;
  const [downloadNotice, setDownloadNotice] = useState('');
  if (!result || !manifest)
    return (
      <EmptyState
        title="No lineage manifest"
        description="The service has not supplied an example manifest for the selected result."
      />
    );
  const lineage: { page: Page; label: string; id: string; description: string }[] = [
    {
      page: 'evaluation',
      label: 'Result',
      id: manifest.result.id,
      description: 'Metrics · predictions · attention',
    },
    {
      page: 'experiments',
      label: 'Run & experiment',
      id: manifest.run.id,
      description: `${manifest.experiment.id} · seed ${result.seed}`,
    },
    {
      page: 'features',
      label: 'Feature set',
      id: manifest.features.id,
      description: 'PFM checkpoint + patch coordinates',
    },
    {
      page: 'cohort',
      label: 'Cohort & split',
      id: manifest.split.id,
      description: 'Patient assignments + target definition',
    },
    {
      page: 'dataset',
      label: 'Dataset version',
      id: manifest.dataset.id,
      description: 'Slide → specimen → patient → label source',
    },
  ];
  return (
    <>
      <PageHeader
        eyebrow="REPRODUCIBILITY BY DESIGN"
        title="Provenance"
        description="Follow the references behind a result, all the way back to its source."
        actions={
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              downloadJSON(`${manifest.experiment.id}.json`, manifest);
              setDownloadNotice('Example manifest downloaded. Artifact paths are placeholders.');
            }}
          >
            <Icon name="download" /> Export example manifest
          </button>
        }
      />
      {downloadNotice ? (
        <p role="status" className="small muted">
          {downloadNotice}
        </p>
      ) : null}
      <div className="provenance-layout">
        <Panel
          title="No orphan results."
          subtitle="An explicit chain of references for every output."
        >
          <div className="lineage-chain">
            {lineage.map((item, index) => (
              <a key={item.page} href={`#${item.page}`} className="lineage-node">
                <span className="lineage-step">
                  <Icon name={item.page} size={19} />
                </span>
                <div>
                  <span className="eyebrow">
                    0{index + 1} · {item.label}
                  </span>
                  <strong>{item.id}</strong>
                  <p>{item.description}</p>
                </div>
                <Icon name="chevron" size={16} />
              </a>
            ))}
          </div>
          <div className="callout small">
            These service-provided references describe synthetic artifacts. Real hashes,
            checkpoints, and source availability still require backend verification.
          </div>
        </Panel>
        <section className="card manifest-card">
          <div className="panel-header">
            <div>
              <h2>Example manifest</h2>
              <p>
                {result.id} · illustrative schema {String(manifest.schema_version ?? 'unknown')}
              </p>
            </div>
            <Badge tone="amber">Non-executable</Badge>
          </div>
          <pre className="manifest" tabIndex={0} aria-label="Example lineage manifest">
            {JSON.stringify(manifest, null, 2)}
          </pre>
        </section>
      </div>
      <div className="grid-3 principles">
        <article className="card">
          <span className="principle-icon">
            <Icon name="lock" size={21} />
          </span>
          <h3>Version the data</h3>
          <p>
            Future changes create new dataset versions. Cohorts and splits retain their source
            references.
          </p>
        </article>
        <article className="card">
          <span className="principle-icon">
            <Icon name="features" size={21} />
          </span>
          <h3>Keep engines replaceable</h3>
          <p>PFM, MIL, and WSI ports keep external tools outside the domain model.</p>
        </article>
        <article className="card">
          <span className="principle-icon">
            <Icon name="branch" size={21} />
          </span>
          <h3>Carry context forward</h3>
          <p>Checkpoints, coordinates, seeds, and the environment belong with each future run.</p>
        </article>
      </div>
    </>
  );
}
