import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Workspace } from '../api/types';
import { buildRoadmap } from '../lib/roadmap';
import ProjectRoadmap, { ModuleStatus, RoadmapLauncher, RoadmapProgress, completedModuleLabel, type Roadmap } from './ProjectRoadmap';

const workspace = {
  mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 }, drafts: [], featureSets: [], cohortSnapshots: [],
} as unknown as Workspace;

function roadmap(overrides: Partial<Roadmap> = {}): Roadmap {
  const modules = buildRoadmap(workspace);
  return { modules, byId: Object.fromEntries(modules.map((module) => [module.id, module])), hasData: true, isLoading: false, error: null, ...overrides } as Roadmap;
}

describe('compact project workflow launcher', () => {
  it('groups six primary modules into three phases with two separate optional analyses', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html.match(/data-module="/g)).toHaveLength(8);
    const section = (phase: string) => html.match(new RegExp(`data-phase="${phase}"[^>]*>(.*?)</section>`))?.[1] ?? '';
    expect(section('prepare')).toContain('data-module="dataset"');
    expect(section('prepare')).toContain('data-module="cohort"');
    expect(section('prepare')).toContain('data-module="features"');
    expect(section('develop')).toContain('data-module="experiments"');
    expect(section('evaluate')).toContain('data-module="test-data"');
    expect(section('evaluate')).toContain('data-module="evaluation"');
    expect(html).toContain('Optional analyses');
    expect(html).toContain('data-module="clinical-utility"');
    expect(html).toContain('data-module="interpretation"');
    // A module icon anchors each row; arrows, legends and counts stay off the page.
    expect(html.match(/class="roadmap-item-icon"/g)).toHaveLength(8);
    expect(html).not.toContain('roadmap-legend');
  });

  it('shows missing inputs while keeping preparation and analysis registries accessible', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html).toContain('data-module="cohort" href="#cohort"');
    expect(html).toContain('Needs Datasets');
    expect(html).toContain('Needs Datasets and Slide features');
    // Slide features depend on slide files, so they open with no dataset in the project.
    for (const id of ['dataset', 'features', 'experiments', 'test-data', 'evaluation', 'clinical-utility', 'interpretation']) expect(html).toContain(`href="#${id}"`);
  });

  it('uses the same Open action for saved results and drafts without claiming execution readiness', () => {
    // A project this far along has met every prerequisite, so no row is blocked.
    const modules = buildRoadmap(workspace).map((module) => module.id === 'experiments'
      ? { ...module, unlocked: true, blockers: [], status: 'draft' as const, evidence: '1 saved development plan' }
      : { ...module, unlocked: true, blockers: [], status: 'complete' as const, evidence: '1 frozen dataset' });
    const html = renderToStaticMarkup(<RoadmapLauncher modules={modules} />);
    expect(html.match(/class="roadmap-item-action">Open</g)).toHaveLength(8);
    // Each row states what the project actually holds, in one line.
    expect(html).toContain('1 frozen dataset');
    expect(html).toContain('1 saved development plan');
    expect(html.match(/roadmap-item-status status-complete/g)).toHaveLength(7);
    expect(html.match(/roadmap-item-status status-draft/g)).toHaveLength(1);
    expect(html).not.toContain('Ready');
    expect(html).not.toContain('Review experiment outputs');
  });

  it('keeps saved work visible on a row that is still waiting for an input', () => {
    const modules = buildRoadmap(workspace).map((module) => module.id === 'experiments'
      ? { ...module, status: 'draft' as const, evidence: '1 saved development plan' } : module);
    const html = renderToStaticMarkup(<RoadmapLauncher modules={modules} />);
    expect(html).toContain('1 saved development plan · needs Targets &amp; splits and Slide features');
    // A row with nothing saved names only what it is waiting for.
    expect(html).toContain('>Needs Datasets<');
    expect(html).toContain('roadmap-item-status is-blocked');
  });

  it('reports completion instead of a next step once every required module is done', () => {
    const modules = buildRoadmap(workspace).map((module) => module.optional
      ? module
      : { ...module, unlocked: true, blockers: [], status: 'complete' as const });
    const html = renderToStaticMarkup(<RoadmapProgress modules={modules} />);
    expect(html).toContain('<strong>6 of 6</strong> required steps complete');
    expect(html).toContain('Every required step is complete');
    expect(html).not.toContain('roadmap-state-next');
    // An untouched optional analysis never reads as missing required work.
    expect(html).not.toContain('Clinical utility');
  });

  it('counts progress and suggests a step without claiming execution readiness', () => {
    const html = renderToStaticMarkup(<RoadmapProgress modules={buildRoadmap(workspace)} />);
    expect(html).toContain('<strong>0 of 6</strong> required steps complete');
    expect(html).toContain('data-next="dataset"');
    expect(html).toContain('>Next step<');
    expect(html).not.toContain('Ready');
    // The demo has illustrative records, not a next step to take.
    const demo = renderToStaticMarkup(<RoadmapProgress modules={buildRoadmap(workspace)} demo />);
    expect(demo).toContain('Every stage below holds an illustrative record.');
    expect(demo).not.toContain('roadmap-state-next');
  });

  it('marks the one module that can be worked on next, in place', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html.match(/roadmap-item is-next/g)).toHaveLength(1);
    expect(html).toContain('class="roadmap-item is-next" data-module="dataset"');
  });

  it('keeps the page concise with no chart, totals, recommendation or expanded module explanation', () => {
    const html = renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={roadmap()} />);
    expect(html).toContain('<h1>Project roadmap</h1>');
    // Progress and one next step; no chart, no legend, no aggregate artifact totals.
    expect(html).toContain('<strong>0 of 6</strong> required steps complete');
    expect(html).toContain('data-next="dataset"');
    expect(html).not.toContain('role="progressbar"');
    expect(html).not.toContain('roadmap-legend');
    expect(html).not.toContain('saved outputs');
    // Only the one suggested module explains itself; the rows stay to one line each.
    const explained = buildRoadmap(workspace).filter((module) => html.includes(module.description));
    expect(explained.map((module) => module.id)).toEqual(['dataset']);
  });

  it('keeps the launcher available after a refresh failure but provides recovery when no data loaded', () => {
    const error = new Error('Records could not be loaded');
    const cached = renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={roadmap({ error })} />);
    expect(cached).toContain('Some saved records could not refresh');
    expect(cached).toContain('href="#experiments"');
    const failed = renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={roadmap({ error, hasData: false })} />);
    expect(failed).toContain('role="alert"');
    expect(failed).toContain('Retry');
    expect(failed).toContain('href="#dataset"');
    expect(failed).not.toContain('roadmap-launcher');
    const loading = renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={roadmap({ hasData: false, isLoading: true })} />);
    expect(loading).toContain('role="status"');
    expect(loading).toContain('Loading workflow');
    expect(loading).not.toContain('roadmap-launcher');
  });

  it('labels synthetic records as examples rather than saved real outputs', () => {
    const modules = buildRoadmap(workspace).map((module) => ({ ...module, status: 'complete' as const, unlocked: true }));
    const html = renderToStaticMarkup(<ProjectRoadmap workspace={{ ...workspace, mode: 'synthetic-demo' }} roadmap={roadmap({ modules })} />);
    expect(html).toContain('Synthetic demo. These are illustrative records.');
    expect(html.match(/>Example</g)).toHaveLength(8);
    expect(html).not.toContain('frozen dataset');
  });

  it('preserves the scientific status labels shared with module pages', () => {
    const html = renderToStaticMarkup(<ModuleStatus status="complete" completedLabel={completedModuleLabel('experiments')} />);
    expect(html).toContain('Experiment outputs available');
    expect(html).not.toContain('frozen');
    expect(completedModuleLabel('evaluation')).toBe('Evaluation results available');
  });
});
