import { useEffect, useRef, useState } from 'react';
import type { Workspace } from '../api/types';
import type { useRoadmap } from '../components/useRoadmap';
import { Icon } from '../components/ui';
import { ROADMAP_CONNECTIONS } from '../lib/roadmap';

export type Roadmap = ReturnType<typeof useRoadmap>;
type Module = Roadmap['modules'][number];
export const moduleIcons: Record<string, string> = {
  dataset: 'dataset', cohort: 'cohort', features: 'features', experiments: 'experiments',
  'source-cv': 'evaluation', selection: 'experiments', predictor: 'experiments',
  'test-data': 'folder', evaluation: 'evaluation', reports: 'provenance',
  'clinical-utility': 'evaluation', interpretation: 'explorer',
};
export const completedModuleLabel = (id: string) => id === 'experiments' ? 'Experiment outputs available' : id === 'interpretation' ? 'Attention maps available' : id === 'clinical-utility' ? 'Analysis saved' : id === 'evaluation' ? 'Evaluation results available' : 'Complete & frozen';
export const completedModuleAction = (id: string) => id === 'experiments' ? 'Review experiment outputs' : id === 'interpretation' ? 'Review attention maps' : id === 'clinical-utility' ? 'Review saved analysis' : id === 'evaluation' ? 'Review evaluation results' : 'Review frozen versions';
export function ModuleStatus({ status, completedLabel = 'Complete & frozen' }: { status: Module['status']; completedLabel?: string }) {
  return <span className={`module-status status-${status}`}>
    <span aria-hidden="true" />
    {status === 'complete' ? completedLabel : status === 'draft' ? 'Saved work' : 'Not started'}
  </span>;
}

type NodeBounds = Pick<DOMRect, 'top' | 'left' | 'right' | 'bottom' | 'width' | 'height'>;
export function roadmapConnectionPath(start: NodeBounds, end: NodeBounds, bounds: Pick<DOMRect, 'left' | 'top'>): string {
  if (Math.abs(start.top - end.top) < 1 && end.left > start.right) {
    const x1 = start.right - bounds.left;
    const y1 = start.top + start.height / 2 - bounds.top;
    const x2 = end.left - bounds.left - 6;
    const y2 = end.top + end.height / 2 - bounds.top;
    const mid = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  }
  const x1 = start.left + start.width / 2 - bounds.left;
  const y1 = start.bottom - bounds.top;
  const x2 = end.left + end.width / 2 - bounds.left;
  const y2 = end.top - bounds.top - 6;
  const mid = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
}

export function RoadmapGraph({ modules }: { modules: Module[] }) {
  const board = useRef<HTMLDivElement>(null);
  const [lines, setLines] = useState<{ id: string; path: string; complete: boolean }[]>([]);
  useEffect(() => {
    const element = board.current;
    if (!element) return;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const bounds = element.getBoundingClientRect();
        const nodes = new Map(Array.from(element.querySelectorAll<HTMLElement>('[data-module]'))
          .map((node) => [node.dataset.module, node.getBoundingClientRect()]));
        const paths = ROADMAP_CONNECTIONS.flatMap(({ from, to }) => {
          const start = nodes.get(from);
          const end = nodes.get(to);
          if (!start || !end) return [];
          return [{ id: `${from}-${to}`, complete: modules.find((item) => item.id === from)?.status === 'complete',
            path: roadmapConnectionPath(start, end, bounds) }];
        });
        setLines(paths);
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    element.querySelectorAll('[data-module]').forEach((node) => observer.observe(node));
    measure();
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [modules]);
  return <div className="roadmap-board" ref={board}>
    <svg className="roadmap-connections" aria-hidden="true">
      <defs>
        <marker id="roadmap-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#b4c2c3" /></marker>
        <marker id="roadmap-arrow-done" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#419980" /></marker>
      </defs>
      {lines.map((line) => <path key={line.id} d={line.path} className={line.complete ? 'is-complete' : ''} markerEnd={`url(#roadmap-arrow${line.complete ? '-done' : ''})`} />)}
    </svg>
    <div className="roadmap-phase phase-prepare"><span>01</span><h3>Prepare</h3><p>Establish your data,<br />targets and representations.</p></div>
    <div className="roadmap-phase phase-develop"><span>02</span><h3>Develop</h3><p>Manage experiments,<br />then freeze their predictors.</p></div>
    <div className="roadmap-phase phase-evaluate"><span>03</span><h3>Evaluate</h3><p>Prepare test data,<br />and track predictor evaluations.</p></div>
    <div className="roadmap-phase phase-insights"><span>04</span><h3>Clinical insights</h3><p>Assess clinical utility,<br />then inspect model attention.</p></div>
    {modules.map((module, index) => {
      const blockers = module.blockers.map((id) => modules.find((item) => item.id === id)?.shortTitle).join(' + ');
      const contents = <>
        <div className="roadmap-node-top"><span className="roadmap-node-icon"><Icon name={moduleIcons[module.id]} size={19} /></span><ModuleStatus status={module.status} completedLabel={completedModuleLabel(module.id)} /></div>
        <h3>{module.title}</h3>
        <p>{module.description}</p>
        {module.id === 'test-data' ? <p className="roadmap-prerequisite-note">Requires a frozen development protocol. Preparation can begin while models train.</p> : null}
        {module.id === 'interpretation' ? <p className="roadmap-prerequisite-note">Use any compatible slides with a saved predictor, or continue from a clinical analysis.</p> : null}
        <div className="roadmap-node-bottom">
          <span>{!module.unlocked ? <><Icon name="lock" size={12} /> {blockers ? `Requires ${blockers}` : 'Review compatible inputs'}</> : module.status === 'complete' ? completedModuleAction(module.id) : module.id === 'experiments' ? 'Open experiments' : module.status === 'draft' ? 'Continue module' : 'Open module'}</span>
          {module.unlocked ? <Icon name="arrow" size={15} /> : null}
        </div>
      </>;
      const className = `roadmap-node node-${module.id} node-${module.status}${module.unlocked ? '' : ' is-locked'}`;
      return module.unlocked ? <a key={module.id} href={`#${module.id}`} className={className} data-module={module.id} aria-label={`${module.title}: ${module.status === 'complete' ? completedModuleLabel(module.id).toLowerCase() : module.status === 'draft' ? 'saved work' : 'not started'}. Open module.`}>{contents}</a>
        : <article key={module.id} className={className} data-module={module.id} aria-label={`${module.title}, locked`}>
            {contents}{module.compatibilityIssue ? <p className="roadmap-compatibility">{module.compatibilityIssue}</p> : null}
            <span className="sr-only">Module {index + 1}. Complete its prerequisites to unlock.</span>
          </article>;
    })}
  </div>;
}

export default function ProjectRoadmap({ workspace, roadmap }: { workspace: Workspace; roadmap: Roadmap }) {
  const complete = roadmap.modules.filter((module) => module.status === 'complete').length;
  const drafts = roadmap.modules.filter((module) => module.status === 'draft').length;
  const available = roadmap.modules.filter((module) => module.unlocked && module.status !== 'complete');
  const next = available.find((module) => module.status === 'draft') ?? available[0];
  const allComplete = complete === roadmap.modules.length;
  const blocked = roadmap.modules.find((module) => module.status !== 'complete');
  const prerequisite = blocked?.prerequisites.map((id) => roadmap.byId[id]).find((module) => module?.unlocked);
  return <div className="project-roadmap">
    <header className="roadmap-heading">
      <div><div className="eyebrow">PROJECT WORKFLOW</div><h1>Your research roadmap</h1><p>Prepare data, develop predictors, evaluate performance, and investigate clinical utility and model attention.</p></div>
      <div className="roadmap-progress"><strong>{roadmap.hasData ? complete : '—'}<span> / {roadmap.modules.length}</span></strong><span>modules complete</span><div role="progressbar" aria-label="Completed modules" aria-valuemin={0} aria-valuemax={roadmap.modules.length} aria-valuenow={roadmap.hasData ? complete : undefined}><span style={{ width: `${roadmap.hasData ? complete / roadmap.modules.length * 100 : 0}%` }} /></div></div>
    </header>
    {workspace.mode === 'synthetic-demo' ? <div className="callout roadmap-demo">Synthetic demonstration. Sample records and illustrative results are separate from real model execution.</div> : null}
    {roadmap.error && roadmap.hasData ? <div className="callout callout-warning roadmap-refresh-warning" role="status"><span>Showing the last loaded progress. Some project records could not refresh.</span><button className="text-button" onClick={() => void roadmap.refetch()}>Retry</button></div> : null}
    {roadmap.isLoading && !roadmap.hasData ? <div className="roadmap-loading" role="status"><Icon name="clock" size={28} /><h2>Reading your project progress</h2><p>Checking saved drafts and frozen versions…</p></div>
      : roadmap.error && !roadmap.hasData ? <div className="roadmap-loading" role="alert"><h2>Project progress could not be loaded</h2><p>{roadmap.error.message}</p><button className="btn btn-primary" onClick={() => void roadmap.refetch()}>Try again</button><a className="btn btn-secondary" href="#dataset">Open data module</a></div>
      : <>
        <div className="roadmap-next"><div className="roadmap-next-symbol"><Icon name={next ? moduleIcons[next.id] : allComplete ? 'check' : 'lock'} size={23} /></div><div><span>{drafts ? 'PICK UP WHERE YOU LEFT OFF' : 'YOUR NEXT STEP'}</span><h2>{next?.title ?? (allComplete ? 'Your roadmap is complete' : 'Review workflow prerequisites')}</h2><p>{next ? `${available.length} module${available.length === 1 ? '' : 's'} available. ${complete ? 'Continue this branch or choose another available module below.' : 'Start with your source data. You can also review later test-data requirements.'}` : allComplete ? 'Open any module to review its saved evidence.' : `${roadmap.modules.length - complete} module${roadmap.modules.length - complete === 1 ? '' : 's'} still need completed evidence. ${blocked?.compatibilityIssue ?? 'Review the required inputs shown below to continue.'}`}</p></div>{next ? <a className="btn btn-primary" href={`#${next.id}`}>{next.status === 'draft' ? 'Continue module' : 'Start module'}<Icon name="arrow" size={16} /></a> : !allComplete && prerequisite ? <a className="btn btn-primary" href={`#${prerequisite.id}`}>Review {prerequisite.shortTitle}<Icon name="arrow" size={16} /></a> : null}</div>
        <section className="roadmap-map" aria-label="Project module dependencies">
          <div className="roadmap-map-heading"><div><Icon name="branch" size={18} /><h2>From data to evidence</h2></div><div className="roadmap-legend" aria-label="Module status legend"><ModuleStatus status="complete" completedLabel="Complete" /><ModuleStatus status="draft" /><ModuleStatus status="not-started" /><span><Icon name="lock" size={12} /> Prerequisites required</span></div></div>
          <RoadmapGraph modules={roadmap.modules} />
          <div className="roadmap-map-note"><Icon name="info" size={15} /><p>Arrows show the main workflow; cards list additional prerequisites. Green modules have completed evidence; yellow modules have saved work. Evaluate models requires both a ready predictor and prepared test data.</p></div>
        </section>
      </>}
  </div>;
}
