import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { DemoPipeline, DemoRecord } from '../api/demo';
import type { Workspace } from '../api/types';
import type { Roadmap } from './ProjectRoadmap';
import { Content } from '../App';
import { buildRoadmap, ROADMAP_MODULES } from '../lib/roadmap';
import BlcaDemo, { BlcaDemoOverview, DemoRecordView, DemoStepView, demoLocation, filterDemoRecords } from './BlcaDemo';

const records: DemoRecord[] = [
  { id: 'demo-dataset', module: 'dataset', name: 'Synthetic BLCA slides', description: 'Invented identifiers with the development and independent test workflow.', tags: ['Synthetic', 'Slides'], steps: [
    { id: 'inventory', title: 'Slide inventory', description: 'No slide pixels or real patient records are included.', facts: [{ label: 'Slide fallback groups', value: '138 synthetic groups' }], table: { title: 'Illustrative slide records', columns: ['Slide', 'Role'], rows: [['BLCA-DEMO-001', 'Development']] } },
    { id: 'handoff', title: 'Dataset handoff', description: 'Use these synthetic records to explain the next stage.' },
  ] },
  { id: 'demo-experiment-v2', module: 'experiments', name: 'Baseline v2', description: 'Synthetic baseline training', tags: ['Baseline', 'ABMIL'], steps: [{ id: 'inputs', title: 'Inputs', description: 'An illustrative input pair.' }] },
  { id: 'demo-experiment-v3', module: 'experiments', name: 'Baseline v3', description: 'Synthetic revised training', tags: ['Revised', 'ABMIL'], steps: [{ id: 'inputs', title: 'Inputs', description: 'Another illustrative pair.' }] },
];
const pipeline: DemoPipeline = { version: 1, synthetic: true, readOnly: true, seed: 20260912, sourceBasis: ['Aggregate workflow shape only.'], records };
const workspace = { project: { id: 'blca-demo-v1', name: 'BLCA demo' }, mode: 'synthetic-demo', executionEnabled: false, demoPipeline: pipeline, dataset: { slideCount: 138 } } as Workspace;

describe('BLCA synthetic walkthrough', () => {
  it('starts with searchable records and tag filters without edit or compute controls', () => {
    const html = renderToStaticMarkup(<BlcaDemo pipeline={pipeline} module="experiments" />);
    expect(html).toContain('Baseline v2');
    expect(html).toContain('Baseline v3');
    expect(html).toContain('Search experiments demo records');
    expect(html).toContain('Filter by tag');
    expect(html).toContain('Synthetic · Read-only');
    expect(html).toContain('data-demo-record="demo-experiment-v2"');
    expect(html).not.toContain('data-demo-step=');
    for (const action of ['Manage', 'Create experiment', 'Launch', 'Delete', 'Archive']) expect(html).not.toContain(`>${action}`);
  });

  it('combines case-insensitive search with exact tag filters', () => {
    expect(filterDemoRecords(records, '  BASELINE  ', 'ABMIL').map((record) => record.id)).toEqual(['demo-experiment-v2', 'demo-experiment-v3']);
    expect(filterDemoRecords(records, 'v3', 'Baseline')).toEqual([]);
    expect(filterDemoRecords(records, 'development', 'Slides').map((record) => record.id)).toEqual(['demo-dataset']);
    expect(filterDemoRecords(records, 'demo-experiment-v2', '').map((record) => record.id)).toEqual(['demo-experiment-v2']);
  });

  it('renders one step at a time with explicit previous and next navigation', () => {
    const html = renderToStaticMarkup(<DemoRecordView record={records[0]} onBack={() => {}} />);
    expect(html).toContain('data-demo-step="inventory"');
    expect(html).not.toContain('data-demo-step="handoff"');
    expect(html).not.toContain('Use these synthetic records to explain the next stage.');
    expect(html).toContain('Next: Dataset handoff');
    expect(html).toContain('Step 1 of 2');
    const last = renderToStaticMarkup(<DemoRecordView record={records[0]} onBack={() => {}} stepId="handoff" />);
    expect(last).toContain('data-demo-step="handoff"');
    expect(last).not.toContain('data-demo-step="inventory"');
    expect(last).toContain('Step 2 of 2');
    expect(last).not.toContain('Next:');
  });

  it('restores shareable record and step links and scopes them to their module', () => {
    expect(demoLocation('#dataset?record=demo-dataset&step=inventory', 'dataset')).toEqual({ recordId: 'demo-dataset', stepId: 'inventory' });
    expect(demoLocation('#experiments?record=one&step=runs', 'dataset')).toEqual({ recordId: null, stepId: null });
    expect(demoLocation('#dataset', 'dataset')).toEqual({ recordId: null, stepId: null });
    const fallback = renderToStaticMarkup(<DemoRecordView record={records[0]} onBack={() => {}} stepId="missing" />);
    expect(fallback).toContain('data-demo-step="inventory"');
  });

  it('labels the source basis, synthetic values, and absence of real study artifacts', () => {
    const html = renderToStaticMarkup(<BlcaDemoOverview workspace={workspace} />);
    expect(html).toContain('138');
    expect(html).toContain('7 + 1');
    expect(html).toContain('Aggregate workflow shape only.');
    expect(html).toContain('not real study results');
    expect(html).toContain('real feature tensors');
    expect(html).toContain('Seed 20260912');
    for (const module of ROADMAP_MODULES) expect(html).toContain(`href="#${module.id}"`);
  });

  it('routes BLCA modules and old tool links without any live query provider', () => {
    const modules = buildRoadmap(workspace);
    const roadmap = { modules } as Roadmap;
    for (const page of ['system', 'explorer', 'provenance', 'example-results', 'cleanup'] as const) {
      const html = renderToStaticMarkup(<Content page={page} workspace={workspace} roadmap={roadmap} />);
      expect(html).toContain('BLCA demo');
      expect(html).toContain('Explore each stage');
      expect(html).not.toContain('CRC');
    }
    for (const module of ROADMAP_MODULES) {
      const html = renderToStaticMarkup(<Content page={module.id} workspace={workspace} roadmap={roadmap} />);
      expect(html).toContain('Synthetic · Read-only');
      expect(html).toContain('Filter by tag');
    }
  });

  it('provides numeric chart access and distinguishes illustrative evidence', () => {
    const html = renderToStaticMarkup(<DemoStepView step={{ id: 'calibration', title: 'Calibration', description: 'Generated example probabilities.', chart: { title: 'Synthetic calibration', xLabel: 'Predicted probability', yLabel: 'Observed fraction', series: [{ label: 'Example', points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }], yRange: [0, 1] } }} />);
    expect(html).toContain('Illustrative synthetic values');
    expect(html).toContain('View numeric curve data');
    expect(html).not.toContain('NaN');
  });
});
