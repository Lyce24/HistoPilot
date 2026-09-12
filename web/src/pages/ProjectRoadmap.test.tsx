import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Workspace } from '../api/types';
import { buildRoadmap } from '../lib/roadmap';
import ProjectRoadmap, { ModuleStatus, RoadmapGraph, completedModuleLabel, roadmapConnectionPath, type Roadmap } from './ProjectRoadmap';

const workspace = {
  mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 }, drafts: [], featureSets: [], cohortSnapshots: [],
} as unknown as Workspace;

describe('eight-module roadmap presentation', () => {
  it('draws side-to-side arrows between paired cards and a vertical arrow into evaluation', () => {
    const model = { left: 300, top: 400, right: 500, bottom: 560, width: 200, height: 160 };
    const freeze = { left: 550, top: 400, right: 750, bottom: 560, width: 200, height: 160 };
    const evaluation = { ...freeze, top: 600, bottom: 760 };
    const test = { ...model, top: 600, bottom: 760 };
    const board = { left: 20, top: 100 };
    expect(roadmapConnectionPath(model, freeze, board)).toBe('M 480 380 C 502 380, 502 380, 524 380');
    expect(roadmapConnectionPath(test, evaluation, board)).toBe('M 480 580 C 502 580, 502 580, 524 580');
    expect(roadmapConnectionPath(freeze, evaluation, board)).toBe('M 630 460 C 630 477, 630 477, 630 494');
  });

  it('renders eight cards and explains independent test-cohort preparation', () => {
    const html = renderToStaticMarkup(<RoadmapGraph modules={buildRoadmap(workspace)} />);
    expect(html.match(/data-module="/g)).toHaveLength(8);
    expect(html).not.toContain('data-module="post-development"');
    expect(html).not.toContain('Build predictors');
    expect(html).not.toContain('Requires a frozen development protocol');
    expect(html).toContain('Create test cohorts before model development or feature extraction');
    expect(html).toContain('Compatibility is checked in Evaluate models');
    expect(html).toContain('phase-develop');
    expect(html).toContain('phase-evaluate');
    expect(html).toContain('phase-insights');
    expect(html).toContain('Clinical insights');
    expect(html).toContain('data-module="clinical-utility"');
    expect(html).toContain('data-module="interpretation"');
    expect(html).not.toContain('data-module="selection"');
    expect(html).not.toContain('data-module="predictor"');
  });

  it('labels completed training evidence without implying predictor freezing', () => {
    const html = renderToStaticMarkup(<ModuleStatus status="complete" completedLabel={completedModuleLabel('experiments')} />);
    expect(html).toContain('Experiment outputs available');
    expect(html).not.toContain('frozen');
    expect(completedModuleLabel('post-development')).toBe('Complete & frozen');
  });

  it('uses result-specific actions for completed compute and analysis modules', () => {
    const modules = buildRoadmap(workspace).map((module) => ({ ...module, status: 'complete' as const, unlocked: true }));
    const html = renderToStaticMarkup(<RoadmapGraph modules={modules} />);
    expect(html).toContain('Review experiment outputs');
    expect(html).toContain('Review attention maps');
    expect(html).toContain('Review saved analysis');
    expect(html).toContain('Review evaluation results');
    expect(completedModuleLabel('evaluation')).toBe('Evaluation results available');
  });

  it('keeps blocked incomplete workflows distinct from a completed roadmap', () => {
    const modules = buildRoadmap(workspace).map((module) => ({ ...module, status: module.id === 'dataset' ? 'complete' as const : 'not-started' as const, unlocked: module.id === 'dataset' }));
    const roadmap = { modules, byId: Object.fromEntries(modules.map((module) => [module.id, module])), hasData: true, isLoading: false, error: null } as Roadmap;
    const html = renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={roadmap} />);
    expect(html).toContain('Review workflow prerequisites');
    expect(html).toContain('7 modules still need completed evidence');
    expect(html).not.toContain('Your roadmap is complete');
    expect(html).toContain('href="#dataset"');
    const completed = { ...roadmap, modules: modules.map((module) => ({ ...module, status: 'complete' as const, unlocked: true })) };
    expect(renderToStaticMarkup(<ProjectRoadmap workspace={workspace} roadmap={completed} />)).toContain('Your roadmap is complete');
  });
});
