import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Workspace } from '../api/types';
import { buildRoadmap } from '../lib/roadmap';
import { ModuleStatus, RoadmapGraph, completedModuleLabel, roadmapConnectionPath } from './ProjectRoadmap';

const workspace = {
  mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 }, drafts: [], featureSets: [], cohortSnapshots: [],
} as unknown as Workspace;

describe('nine-module roadmap presentation', () => {
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

  it('renders nine cards and four phases while preserving the test protocol gate', () => {
    const html = renderToStaticMarkup(<RoadmapGraph modules={buildRoadmap(workspace)} />);
    expect(html.match(/data-module="/g)).toHaveLength(9);
    expect(html).toContain('data-module="post-development"');
    expect(html).toContain('Build predictors');
    expect(html).toContain('Requires a frozen development protocol');
    expect(html).toContain('Preparation can begin while models train');
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
    expect(html).toContain('Completed runs available');
    expect(html).not.toContain('frozen');
    expect(completedModuleLabel('post-development')).toBe('Complete & frozen');
  });
});
