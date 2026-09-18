import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Workspace } from '../api/types';
import { buildRoadmap } from '../lib/roadmap';
import RoadmapModule from './RoadmapModule';
import type { Roadmap } from './ProjectRoadmap';

const workspace = {
  mode: 'local', project: { id: 'project' }, dataset: { slideCount: 0 }, drafts: [], featureSets: [], cohortSnapshots: [],
} as unknown as Workspace;

function roadmap(): Roadmap {
  const modules = buildRoadmap(workspace);
  return { modules, byId: Object.fromEntries(modules.map((module) => [module.id, module])), hasData: true, isLoading: false, error: null } as Roadmap;
}
const render = (id: string) => {
  const value = roadmap();
  return renderToStaticMarkup(<RoadmapModule module={value.modules.find((item) => item.id === id)!} roadmap={value} workspace={workspace} />);
};

describe('blocked module page', () => {
  it('makes the prerequisite that can be opened the page action, not the way back', () => {
    const html = render('cohort');
    expect(html).toContain('<a class="btn btn-small btn-primary" href="#dataset">');
    expect(html).toContain('<a class="btn btn-secondary" href="#overview">');
    expect(html).toContain('Required modules');
    expect(html).toContain('No frozen dataset or saved import');
  });

  it('uses the module phase rather than a generic label', () => {
    expect(render('cohort')).toContain('01 PREPARE');
    expect(render('features')).toContain('01 PREPARE');
    expect(render('cohort')).not.toContain('PROJECT MODULE');
  });

  it('omits the outputs panel when the module has no specific outputs to describe', () => {
    expect(render('cohort')).not.toContain('Module outputs');
    expect(render('cohort')).not.toContain('A completed, frozen module output');
    // Modules that do describe their outputs keep the panel.
    const value = roadmap();
    const evaluation = { ...value.modules.find((item) => item.id === 'evaluation')!, unlocked: false, blockers: ['experiments' as const] };
    const html = renderToStaticMarkup(<RoadmapModule module={evaluation} roadmap={value} workspace={workspace} />);
    expect(html).toContain('Module outputs');
    expect(html).toContain('Predictions from the frozen predictor');
  });
});
