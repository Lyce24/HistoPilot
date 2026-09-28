import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Workspace } from '../api/types';
import { buildRoadmap } from '../lib/roadmap';
import ProjectRoadmap, { RoadmapLauncher, RoadmapProgress, completedModuleLabel, type Roadmap } from './ProjectRoadmap';

const workspace = { mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 } } as unknown as Workspace;

function roadmap(overrides: Partial<Roadmap> = {}): Roadmap {
  const modules = buildRoadmap(workspace);
  return { modules, byId: Object.fromEntries(modules.map((module) => [module.id, module])), hasData: true, isLoading: false, error: null, ...overrides } as Roadmap;
}

describe('compact project workflow launcher', () => {
  it('groups the pipeline into seven ordered stages with parallel preparation', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html.match(/data-module="/g)).toHaveLength(9);
    const section = (phase: string) => html.match(new RegExp(`data-phase="${phase}"[^>]*>(.*?)</section>`))?.[1] ?? '';
    expect(section('datasets')).toContain('data-module="dataset"');
    expect([...html.matchAll(/data-phase="([^"]+)"/g)].map((match) => match[1])).toEqual(['datasets', 'prepare', 'setup', 'develop', 'evaluate', 'clinical', 'interpret']);
    expect(section('prepare')).toContain('data-module="cohort"');
    expect(section('prepare')).toContain('data-module="features"');
    expect(section('develop')).toContain('data-module="experiments"');
    expect(section('setup')).toContain('data-module="experimental-setup"');
    expect(html).not.toContain('data-module="test-data"');
    expect(section('evaluate')).toContain('data-module="evaluation"');
    expect(section('evaluate')).toContain('data-module="inference"');
    expect(html).toContain('Slide features and Targets &amp; splits are independent.');
    expect(html).toContain('data-module="clinical-utility"');
    expect(html).toContain('data-module="interpretation"');
    // A module icon anchors each row; arrows, legends and counts stay off the page.
    expect(html.match(/class="roadmap-item-icon"/g)).toHaveLength(9);
    expect(html).not.toContain('roadmap-legend');
  });

  it('shows missing inputs while keeping preparation and analysis registries accessible', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html).toContain('data-module="cohort" href="#cohort"');
    expect(html).toContain('Needs Datasets');
    expect(html).not.toContain('Needs Datasets and Slide features');
    expect(html).toContain('Needs Datasets, Slide features and Experiments');
    expect(html).toContain('Needs Targets &amp; splits and Slide features');
    // Slide features depend on slide files, so they open with no dataset in the project.
    for (const id of ['dataset', 'features', 'experimental-setup', 'experiments', 'evaluation', 'clinical-utility', 'interpretation']) expect(html).toContain(`href="#${id}"`);
  });

  it('uses the same Open action for saved results and drafts without claiming execution readiness', () => {
    // A project this far along has met every prerequisite, so no row is blocked.
    const modules = buildRoadmap(workspace).map((module) => module.id === 'experiments'
      ? { ...module, unlocked: true, blockers: [], status: 'draft' as const, evidence: '1 saved development plan' }
      : { ...module, unlocked: true, blockers: [], status: 'complete' as const, evidence: '1 frozen dataset' });
    const html = renderToStaticMarkup(<RoadmapLauncher modules={modules} />);
    expect(html.match(/class="roadmap-item-action">Open</g)).toHaveLength(9);
    // Each row states what the project actually holds, in one line.
    expect(html).toContain('1 frozen dataset');
    expect(html).toContain('1 saved development plan');
    expect(html.match(/roadmap-item-status status-complete/g)).toHaveLength(8);
    expect(html.match(/roadmap-item-status status-draft/g)).toHaveLength(1);
    expect(html).not.toContain('Ready');
    expect(html).not.toContain('Review experiment outputs');
  });

  it('keeps saved work visible on a row that is still waiting for an input', () => {
    const modules = buildRoadmap(workspace).map((module) => module.id === 'experiments'
      ? { ...module, status: 'draft' as const, evidence: '1 saved development plan' } : module);
    const html = renderToStaticMarkup(<RoadmapLauncher modules={modules} />);
    expect(html).toContain('1 saved development plan · needs Experimental Setup');
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
  });

  it('marks the one module that can be worked on next, in place', () => {
    const html = renderToStaticMarkup(<RoadmapLauncher modules={buildRoadmap(workspace)} />);
    expect(html.match(/roadmap-item is-next/g)).toHaveLength(1);
    expect(html).toContain('class="roadmap-item is-next" data-module="dataset"');
  });

  it('keeps the page concise with no chart, totals, recommendation or expanded module explanation', () => {
    const html = renderToStaticMarkup(<ProjectRoadmap roadmap={roadmap()} />);
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
    const cached = renderToStaticMarkup(<ProjectRoadmap roadmap={roadmap({ error })} />);
    expect(cached).toContain('Some saved records could not refresh');
    expect(cached).toContain('href="#experiments"');
    const failed = renderToStaticMarkup(<ProjectRoadmap roadmap={roadmap({ error, hasData: false })} />);
    expect(failed).toContain('role="alert"');
    expect(failed).toContain('Retry');
    expect(failed).toContain('href="#dataset"');
    expect(failed).not.toContain('roadmap-launcher');
    const loading = renderToStaticMarkup(<ProjectRoadmap roadmap={roadmap({ hasData: false, isLoading: true })} />);
    expect(loading).toContain('role="status"');
    expect(loading).toContain('Loading workflow');
    expect(loading).not.toContain('roadmap-launcher');
  });

  it('preserves the scientific status labels shared with module pages', () => {
    expect(completedModuleLabel('experiments')).toBe('Experiment outputs available');
    expect(completedModuleLabel('evaluation')).toBe('Evaluation results available');
    expect(completedModuleLabel('inference')).toBe('Predictions available');
  });
});
