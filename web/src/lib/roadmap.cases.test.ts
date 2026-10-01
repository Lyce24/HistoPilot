import { describe, expect, it } from 'vitest';
import type { Workspace } from '../api/types';
import { ROADMAP_MODULES, ROADMAP_STEPS, buildRoadmap, suggestedRoadmapModule, type RoadmapEvidence } from './roadmap';
import cases from './roadmapCases.json';

// The same cases run against histopilot/roadmap.py in tests/test_roadmap.py, so
// `histopilot project roadmap` reports what this page shows. Undefined compares as JSON null.
const plain = (value: unknown) => JSON.parse(JSON.stringify(value ?? null, (_key, item) => item === undefined ? null : item));

describe('roadmap shared with the service', () => {
  it('names the same stages and steps', () => {
    expect(plain(ROADMAP_MODULES)).toEqual(cases.modules);
    expect(plain(ROADMAP_STEPS)).toEqual(cases.steps);
  });

  it('computes the same status, evidence and blockers', () => {
    for (const scenario of cases.scenarios) {
      const stages = buildRoadmap(scenario.workspace as unknown as Workspace, scenario.evidence as unknown as Partial<RoadmapEvidence>);
      const computed = stages.map(({ id, status, unlocked, blockers, evidence, artifactCount, compatibilityIssue, retainedWork }) =>
        ({ id, status, unlocked, blockers, evidence, artifactCount, compatibilityIssue, retainedWork }));
      expect(plain(computed), scenario.name).toEqual(scenario.expected);
      expect(suggestedRoadmapModule(stages)?.id ?? null, scenario.name).toBe(scenario.suggested);
    }
  });
});
