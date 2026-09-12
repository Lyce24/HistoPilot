import type { Workspace } from '../api/types';
import type { Roadmap } from './ProjectRoadmap';
import { moduleIcons, ModuleStatus, completedModuleLabel } from './ProjectRoadmap';
import { Icon, PageHeader } from '../components/ui';

type Module = Roadmap['modules'][number];
const outputs: Record<string, string[]> = {
  'source-cv': ['Patient-grouped source cross-validation', 'Out-of-fold predictions and source performance reports'],
  selection: ['Model selection using source development evidence', 'Refit models and ensemble specifications'],
  predictor: ['A selected, immutable predictor', 'Pinned preprocessing, feature requirements and model weights'],
  experiments: ['Frozen training plans and per-fold development results', 'Automatic ensemble or refit predictors for each configuration and seed, when enabled'],
  'test-data': ['A filtered test cohort from a combined or separately imported file', 'Verified slide coverage and compatible features', 'Saved inference settings without data splits'],
  evaluation: ['Predictions from the frozen predictor', 'An evaluation record tied to the later test dataset'],
  'clinical-utility': ['Calibration, Brier score and operating curves', 'Decision curves against treat-all and treat-none strategies', 'A saved analysis linked to its evaluation and predictor'],
  interpretation: ['Slide overlays from real ABMIL attention weights', 'Exact slide, feature, coordinate and predictor provenance', 'Inspectable ensemble member and refit attention maps'],
  reports: ['Performance metrics and uncertainty', 'Clinical analyses and reproducible reports'],
};
export default function RoadmapModule({ module, roadmap, workspace }: { module: Module; roadmap: Roadmap; workspace: Workspace }) {
  const prerequisites = module.prerequisites.map((id) => roadmap.modules.find((item) => item.id === id)!);
  return <div className="clinical-workspace roadmap-module-detail">
    <PageHeader eyebrow="PROJECT MODULE" title={module.title} description={module.description} />
    <section className="module-gate">
      <span className="module-gate-icon"><Icon name={module.unlocked ? moduleIcons[module.id] : 'lock'} size={28} /></span>
      <div><h2>{module.unlocked ? 'Review this workflow' : 'Complete the prerequisites to unlock this module'}</h2><p>{module.id === 'evaluation' ? 'Inference requires a ready predictor from Experiments and a verified test cohort. Test-cohort preparation can proceed while models are still developing.' : 'Your saved work stays available. Complete the required inputs below to continue.'}</p></div>
    </section>
    {prerequisites.length ? <section className="module-prerequisites"><h2>Required modules</h2>{prerequisites.map((item) => <div key={item.id}><Icon name={moduleIcons[item.id]} /><span><strong>{item.title}</strong><small>{item.evidence}</small></span><ModuleStatus status={item.status} completedLabel={completedModuleLabel(item.id)} />{item.unlocked ? <a className="btn btn-secondary btn-small" href={`#${item.id}`}>Open<Icon name="arrow" size={14} /></a> : <Icon name="lock" size={15} />}</div>)}</section> : null}
    {module.compatibilityIssue ? <div className="callout callout-warning">{module.compatibilityIssue}</div> : null}
    <section className="module-expected"><h2>{module.id === 'test-data' ? 'What you will need' : 'Module outputs'}</h2><ul>{(outputs[module.id] ?? ['A completed, frozen module output']).map((output) => <li key={output}><Icon name="check" size={16} />{output}</li>)}</ul></section>
    <div className="callout"><Icon name="info" size={17} /><p>{module.id === 'test-data' ? 'Freeze a development protocol first. Then select test rows and verify features; no completed model or test-data split is needed to prepare this cohort.' : workspace.mode === 'synthetic-demo' ? 'This demonstration contains illustrative results. It does not execute training, freeze a predictor, or evaluate later test data.' : 'Supported ABMIL development batches produce fold checkpoints and OOF predictions. Experiments automatically generate the configured ensemble or refit predictors from completed folds. Evaluate ready predictors on a verified test cohort.'}</p></div>
    <a className="btn btn-primary" href="#overview">Return to roadmap<Icon name="arrow" size={16} /></a>
  </div>;
}
