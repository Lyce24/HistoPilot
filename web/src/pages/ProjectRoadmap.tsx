import type { Workspace } from '../api/types';
import type { useRoadmap } from '../components/useRoadmap';
import { suggestedRoadmapModule } from '../lib/roadmap';
import { Icon } from '../components/ui';

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

const phases = [
  { id: 'prepare', title: 'Prepare', step: '01' },
  { id: 'develop', title: 'Develop', step: '02' },
  { id: 'evaluate', title: 'Evaluate', step: '03' },
] as const;

/** Modules a project must complete; the optional analyses are not counted. */
const requiredModules = (modules: Module[]) => modules.filter((module) => !module.optional);

/** The modules this one is still waiting for, named, or null when it can proceed. */
function missingInputs(module: Module, modules: Module[]): string | null {
  const inputs = module.blockers.map((id) => modules.find((item) => item.id === id)?.shortTitle).filter(Boolean);
  return inputs.length ? inputs.join(' and ') : null;
}

/**
 * One row per module: what it is, what this project already has in it, and what
 * it is still waiting for. Saved evidence replaces a generic "Saved outputs"
 * label because it answers the reader's actual question in the same line.
 * Module explanations and artifact totals stay off these rows.
 */
function LauncherItem({ module, modules, demo, next }: { module: Module; modules: Module[]; demo: boolean; next?: boolean }) {
  const complete = module.unlocked && !demo && module.status === 'complete';
  // A finished module reports what it produced. Unfinished work names what it is
  // still waiting for, keeping any evidence it has already saved.
  const missing = complete || demo ? null : missingInputs(module, modules);
  const evidence = module.status === 'not-started' ? null : module.evidence;
  const status = demo ? 'Example'
    : missing ? evidence ? `${evidence} · needs ${missing}` : `Needs ${missing}`
      : evidence ?? 'Not started';
  const state = missing ? 'is-blocked' : `status-${module.status}`;
  // The row stays one line; the full compatibility explanation belongs on the
  // element itself, where a pointer or assistive technology can read it.
  const detail = module.compatibilityIssue ?? status;
  const contents = <>
    <span className={`roadmap-item-icon${complete ? ' is-complete' : ''}`} aria-hidden="true">
      <Icon name={moduleIcons[module.id]} size={17} />
    </span>
    <span className="roadmap-item-copy">
      <strong>{module.shortTitle}</strong>
      <span className={`roadmap-item-status ${state}`}>{status}</span>
    </span>
    <span className="roadmap-item-action">{module.unlocked ? 'Open' : 'Locked'}</span>
  </>;
  const className = `roadmap-item${module.unlocked ? '' : ' is-locked'}${next ? ' is-next' : ''}`;
  return <li>{module.unlocked
    ? <a className={className} data-module={module.id} href={`#${module.id}`} title={module.compatibilityIssue} aria-label={`Open ${module.shortTitle}. ${detail}.`}>{contents}</a>
    : <div className={className} data-module={module.id} aria-disabled="true" aria-label={`${module.shortTitle}. ${detail}.`} title={detail}>{contents}</div>}
  </li>;
}

/**
 * How far the project has come and the one step that can be taken next. Progress
 * counts required modules only, so an optional analysis never reads as missing
 * work. A project with every required module complete is told so plainly.
 */
export function RoadmapProgress({ modules, demo = false }: { modules: Module[]; demo?: boolean }) {
  const required = requiredModules(modules);
  const done = required.filter((module) => module.status === 'complete');
  const next = demo ? undefined : suggestedRoadmapModule(modules);
  return <section className="roadmap-state" aria-labelledby="roadmap-state-title">
    <h2 id="roadmap-state-title" className="sr-only">Project progress</h2>
    <div className="roadmap-state-bar">
      <ol className="roadmap-state-steps" aria-hidden="true">
        {required.map((module) => <li key={module.id} className={module.status === 'complete' ? 'is-complete' : module.status === 'draft' ? 'is-draft' : ''} />)}
      </ol>
      <p className="roadmap-state-count">{demo
        ? 'Every stage below holds an illustrative record.'
        : <><strong>{done.length} of {required.length}</strong> required steps complete</>}</p>
    </div>
    {demo ? null : next
      ? <a className="roadmap-state-next" href={`#${next.id}`} data-next={next.id} aria-label={`Continue with ${next.shortTitle}. ${next.description}`}>
        <span className="roadmap-state-next-icon" aria-hidden="true"><Icon name={moduleIcons[next.id]} size={20} /></span>
        <span className="roadmap-state-next-copy">
          <small>{next.status === 'draft' ? 'Continue where you left off' : 'Next step'}</small>
          <strong>{next.shortTitle}</strong>
          <span>{next.description}</span>
        </span>
        <span className="roadmap-state-next-action">Open<Icon name="arrow" size={15} /></span>
      </a>
      : <p className="roadmap-state-done"><Icon name="check" size={16} />Every required step is complete. Review your saved outputs, or continue with an optional analysis.</p>}
  </section>;
}

export function RoadmapLauncher({ modules, demo = false }: { modules: Module[]; demo?: boolean }) {
  const next = demo ? undefined : suggestedRoadmapModule(modules);
  return <>
    <div className="roadmap-launcher">
      {phases.map((phase) => <section key={phase.id} className="roadmap-section" data-phase={phase.id} aria-labelledby={`roadmap-${phase.id}`}>
        {/* The step number is decoration drawn by CSS, so the heading's own text
            stays the phase name for assistive technology and tests. */}
        <h2 id={`roadmap-${phase.id}`} data-step={phase.step}>{phase.title}</h2>
        <ul>{modules.filter((module) => module.phase === phase.id).map((module) => <LauncherItem key={module.id} module={module} modules={modules} demo={demo} next={module.id === next?.id} />)}</ul>
      </section>)}
    </div>
    <section className="roadmap-analysis" aria-labelledby="roadmap-analysis-title">
      <h2 id="roadmap-analysis-title">Optional analyses</h2>
      <ul>{modules.filter((module) => module.phase === 'insights').map((module) => <LauncherItem key={module.id} module={module} modules={modules} demo={demo} />)}</ul>
    </section>
  </>;
}

export default function ProjectRoadmap({ workspace, roadmap }: { workspace: Workspace; roadmap: Roadmap }) {
  return <div className="project-roadmap">
    {/* The next-step card below states what to open, so the heading does not repeat it. */}
    <header className="roadmap-heading"><h1>Project roadmap</h1></header>
    {workspace.mode === 'synthetic-demo' ? <div className="callout roadmap-demo">Synthetic demo. These are illustrative records.</div> : null}
    {roadmap.error && roadmap.hasData ? <div className="callout callout-warning roadmap-refresh-warning" role="status"><span>Some saved records could not refresh.</span><button className="text-button" onClick={() => void roadmap.refetch()}>Retry</button></div> : null}
    {roadmap.isLoading && !roadmap.hasData ? <div className="roadmap-loading" role="status"><Icon name="clock" size={24} /><p>Loading workflow…</p></div>
      : roadmap.error && !roadmap.hasData ? <div className="roadmap-loading" role="alert"><h2>Could not load the workflow</h2><p>{roadmap.error.message}</p><button className="btn btn-primary" onClick={() => void roadmap.refetch()}>Retry</button><a className="btn btn-secondary" href="#dataset">Open datasets</a></div>
      : <>
        <RoadmapProgress modules={roadmap.modules} demo={workspace.mode === 'synthetic-demo'} />
        <RoadmapLauncher modules={roadmap.modules} demo={workspace.mode === 'synthetic-demo'} />
      </>}
  </div>;
}
