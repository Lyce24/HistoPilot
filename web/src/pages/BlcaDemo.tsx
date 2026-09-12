import { useEffect, useState } from 'react';
import type { Workspace } from '../api/types';
import type { DemoPipeline, DemoRecord, DemoStep } from '../api/demo';
import { ROADMAP_MODULES, type RoadmapModuleId } from '../lib/roadmap';
import { Badge, Icon } from '../components/ui';
import { StageLibraryToolbar, StagePage, StageSteps, useStageLibrary } from '../components/StageWorkflow';
import { RunTable } from '../components/ExperimentTracking';
import { ResourceDashboard } from '../components/RunResourceUsage';
import CurveChart from '../components/CurveChart';
import './BlcaDemo.css';

export function filterDemoRecords(records: DemoRecord[], search: string, tag: string) {
  const query = search.trim().toLocaleLowerCase();
  return records.filter((record) => (!tag || record.tags.includes(tag)) && (!query || `${record.name} ${record.id} ${record.description} ${record.tags.join(' ')}`.toLocaleLowerCase().includes(query)));
}

export function demoLocation(hash: string, module: RoadmapModuleId) {
  const [page, query = ''] = hash.replace(/^#/, '').split('?');
  const params = new URLSearchParams(query);
  return page === module ? { recordId: params.get('record'), stepId: params.get('step') } : { recordId: null, stepId: null };
}

function DemoNotice() {
  return <div className="blca-demo-notice" role="note"><Badge>Synthetic · Read-only</Badge><span>Generated examples explain the bladder workflow. Scores, curves, resource measurements, and identifiers are illustrative; they are not real study results.</span></div>;
}

export function BlcaDemoOverview({ workspace }: { workspace: Workspace }) {
  const pipeline = workspace.demoPipeline!;
  const runCount = pipeline.records.flatMap((record) => record.steps).reduce((total, step) => total + (step.runs?.batch.manifest.runs.length ?? 0), 0);
  return <div className="blca-demo">
    <header className="blca-demo-hero"><div><span className="eyebrow">An illustrated research workflow</span><h1>BLCA demo</h1><p>Follow a bladder cancer project from slide records to model evaluation and clinical utility, one step at a time.</p><a className="btn btn-primary" href="#dataset">Explore the pipeline <Icon name="arrow" size={16} /></a></div><div className="blca-demo-hero-mark" aria-hidden="true">BLCA<small>FROM SLIDES<br />TO EVIDENCE</small></div></header>
    <DemoNotice />
    <dl className="blca-demo-summary"><div><dt>Synthetic slide records</dt><dd>{workspace.dataset.slideCount}</dd></div><div><dt>Stages + interpretation</dt><dd>7 + 1</dd></div><div><dt>Illustrative training runs</dt><dd>{runCount}</dd></div></dl>
    <section aria-labelledby="blca-pipeline-title"><div className="blca-demo-section-heading"><h2 id="blca-pipeline-title">Explore each stage</h2><p>Open a record, then use the steps to follow its inputs and outputs.</p></div><div className="blca-demo-pipeline">{ROADMAP_MODULES.map((module, index) => {
      const records = pipeline.records.filter((record) => record.module === module.id);
      return <a href={`#${module.id}`} className="blca-demo-stage" key={module.id}><span className="blca-demo-stage-number">{String(index + 1).padStart(2, '0')}</span><div><h3>{module.shortTitle}</h3><p>{module.id === 'interpretation' ? 'Learn how attention maps connect predictions to slide regions. No slide pixels are included.' : module.description}</p><small>{records.length ? `${records.length} illustrative ${records.length === 1 ? 'record' : 'records'}` : 'Workflow explanation'}</small></div><Icon name="arrow" size={17} /></a>;
    })}</div></section>
    <section className="card blca-demo-basis"><h2>What this demo is based on</h2><ul>{pipeline.sourceBasis.map((basis, index) => <li key={index}>{basis}</li>)}</ul><p className="muted">Only the workflow structure and stated aggregate characteristics inform this example. Slide pixels, patient records, real feature tensors, checkpoints, predictions, and local file paths are not included.</p><p className="muted">Reproducible synthetic generation · Seed {pipeline.seed}</p></section>
  </div>;
}

export default function BlcaDemo({ pipeline, module }: { pipeline: DemoPipeline; module: RoadmapModuleId }) {
  const [location, setLocation] = useState(() => demoLocation(typeof window === 'undefined' ? '' : window.location.hash, module));
  const [search, setSearch] = useState('');
  const [tag, setTag] = useState('');
  useEffect(() => {
    const update = () => setLocation(demoLocation(window.location.hash, module));
    window.addEventListener('hashchange', update);
    window.addEventListener('popstate', update);
    return () => { window.removeEventListener('hashchange', update); window.removeEventListener('popstate', update); };
  }, [module]);
  function navigate(recordId: string | null, stepId?: string) {
    const params = new URLSearchParams();
    if (recordId) params.set('record', recordId);
    if (stepId) params.set('step', stepId);
    window.location.hash = `${module}${params.size ? `?${params}` : ''}`;
    setLocation({ recordId, stepId: stepId ?? null });
  }
  useStageLibrary(() => navigate(null));
  const definition = ROADMAP_MODULES.find((item) => item.id === module)!;
  const records = pipeline.records.filter((record) => record.module === module);
  const selected = records.find((record) => record.id === location.recordId);
  const filtered = filterDemoRecords(records, search, tag);
  const tags = [...new Set(records.flatMap((record) => record.tags))].sort();
  return <div className="blca-demo">
    <header className="blca-demo-page-heading"><span className="eyebrow">BLCA demo</span><h1>{definition.shortTitle}</h1><p>{selected ? selected.description : 'Explore the saved examples to see how this stage fits into the bladder workflow.'}</p></header>
    <DemoNotice />
    <StagePage pageKey={selected?.id ?? 'library'}>{selected ? <DemoRecordView key={selected.id} record={selected} stepId={location.stepId ?? selected.steps[0]?.id} onStepChange={(stepId) => navigate(selected.id, stepId)} onBack={() => navigate(null)} /> : <section className="stage-library card" aria-label={`${definition.shortTitle} demo records`}><div className="stage-library-body">
      <StageLibraryToolbar search={search} onSearch={setSearch} searchLabel={`Search ${definition.shortTitle.toLowerCase()} demo records`} placeholder="Name, description, or tag" count={filtered.length} total={records.length} onReset={search || tag ? () => { setSearch(''); setTag(''); } : undefined}>
        <label className="label">Filter by tag<select className="field" value={tag} onChange={(event) => setTag(event.target.value)}><option value="">All tags</option>{tags.map((item) => <option key={item}>{item}</option>)}</select></label>
      </StageLibraryToolbar>
      {filtered.length ? <div className="table-wrap"><table className="blca-demo-records"><thead><tr><th scope="col">Record</th><th scope="col">Tags</th><th scope="col">Steps</th><th scope="col">Status</th></tr></thead><tbody>{filtered.map((record) => <tr key={record.id}><th scope="row"><button className="text-link blca-demo-record-link" data-demo-record={record.id} type="button" onClick={() => navigate(record.id, record.steps[0]?.id)}>{record.name}<Icon name="arrow" size={14} /></button><p>{record.description}</p></th><td><div className="blca-demo-tags">{record.tags.map((item) => <Badge key={item}>{item}</Badge>)}</div></td><td>{record.steps.length}</td><td><Badge>Illustrative</Badge></td></tr>)}</tbody></table></div> : <p className="blca-demo-empty" role="status">{records.length ? 'No examples match these filters.' : 'This part of the workflow is explained in the roadmap. No synthetic artifact is included.'}</p>}
    </div></section>}</StagePage>
  </div>;
}

export function DemoRecordView({ record, onBack, initialStep = 0, stepId, onStepChange }: { record: DemoRecord; onBack: () => void; initialStep?: number; stepId?: string; onStepChange?: (id: string) => void }) {
  const [localIndex, setLocalIndex] = useState(Math.min(Math.max(0, initialStep), Math.max(0, record.steps.length - 1)));
  const index = stepId === undefined ? localIndex : Math.max(0, record.steps.findIndex((step) => step.id === stepId));
  function setIndex(value: number) { setLocalIndex(value); if (record.steps[value]) onStepChange?.(record.steps[value].id); }
  const step = record.steps[index];
  return <div className="blca-demo-record-view"><div className="blca-demo-record-heading"><button className="text-button" type="button" onClick={onBack}>← Back to records</button><h2>{record.name}</h2></div>
    <StageSteps steps={record.steps.map((item) => ({ id: item.id, title: item.title }))} current={step?.id ?? ''} onChange={(id) => setIndex(record.steps.findIndex((item) => item.id === id))} label={`${record.name} walkthrough steps`} />
    {step ? <><StagePage pageKey={`${record.id}:${step.id}`}><DemoStepView step={step} /></StagePage><nav className="blca-demo-step-actions" aria-label="Walkthrough page navigation"><button type="button" className="btn btn-secondary" onClick={() => index > 0 ? setIndex(index - 1) : onBack()}>{index > 0 ? 'Back' : 'Back to records'}</button><span>Step {index + 1} of {record.steps.length}</span>{index + 1 < record.steps.length ? <button type="button" className="btn btn-primary" onClick={() => setIndex(index + 1)}>Next: {record.steps[index + 1].title} <Icon name="arrow" size={16} /></button> : <button type="button" className="btn btn-primary" onClick={onBack}>Back to records</button>}</nav></> : <p>No walkthrough steps are available.</p>}
  </div>;
}

export function DemoStepView({ step }: { step: DemoStep }) {
  const xValues = step.chart?.series.flatMap((series) => series.points.map((point) => point.x)).filter(Number.isFinite) ?? [];
  const xMin = xValues.length ? Math.min(...xValues) : 0;
  const xMax = xValues.length ? Math.max(...xValues) : 1;
  return <section className="card blca-demo-step" data-demo-step={step.id} aria-label={step.title}><h2>{step.title}</h2><p className="blca-demo-step-description">{step.description}</p>
    {step.notice ? <p className="callout blca-demo-step-notice">{step.notice}</p> : null}
    {step.facts?.length ? <dl className="blca-demo-facts">{step.facts.map((fact, index) => <div key={index}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}</dl> : null}
    {step.chart ? <div className="blca-demo-chart"><CurveChart {...step.chart} description="Illustrative synthetic values generated for this walkthrough." xRange={[xMin, Math.max(xMin + 0.001, xMax)]} /></div> : null}
    {step.runs ? <><div className="blca-demo-run-status"><Badge>{step.runs.execution.status}</Badge><strong>{step.runs.execution.runCounts.completed} / {step.runs.execution.runCounts.total} completed</strong><span>Synthetic batch · {step.runs.batch.manifest.spec.batchName}</span></div><p className="blca-demo-synthetic-label">Illustrative training evidence · Every metric, checkpoint reference, and resource sample below is synthetic.</p><RunTable illustrative batch={step.runs.batch} execution={step.runs.execution} histories={step.runs.histories} /><div className="blca-demo-resources"><ResourceDashboard execution={step.runs.execution} history={step.runs.resources} /><details className="blca-demo-worker"><summary>Worker, device &amp; runtime</summary><p>These samples illustrate the monitoring interface. This demo does not start a worker or measure your current machine.</p><dl className="blca-demo-facts"><div><dt>Worker</dt><dd>Simulated training worker</dd></div><div><dt>Device</dt><dd>{step.runs.execution.telemetry?.latest?.gpus.map((gpu) => gpu.name).join(', ') || 'Synthetic compute device'}</dd></div><div><dt>Runtime</dt><dd>Static, deterministic example</dd></div></dl></details></div></> : null}
    {step.table ? <section className="blca-demo-table-section" aria-label={step.table.title ?? `${step.title} records`}>{step.table.title ? <h3>{step.table.title}</h3> : null}<div className="table-wrap"><table><thead><tr>{step.table.columns.map((column, index) => <th scope="col" key={index}>{column}</th>)}</tr></thead><tbody>{step.table.rows.map((row, index) => <tr key={index}>{row.map((value, column) => column === 0 ? <th scope="row" key={column}>{value}</th> : <td key={column}>{value}</td>)}</tr>)}</tbody></table></div></section> : null}
  </section>;
}
