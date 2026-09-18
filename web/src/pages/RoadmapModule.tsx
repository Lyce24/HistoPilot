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
const phaseEyebrow: Record<Module['phase'], string> = {
  prepare: '01 PREPARE', develop: '02 DEVELOP', evaluate: '03 EVALUATE', insights: 'OPTIONAL ANALYSIS',
};
export default function RoadmapModule({ module, roadmap, workspace }: { module: Module; roadmap: Roadmap; workspace: Workspace }) {
  const prerequisites = module.prerequisites.map((id) => roadmap.modules.find((item) => item.id === id)!);
  // The blocked module's own next action is the prerequisite that can be opened,
  // not a return to the roadmap the reader just came from.
  const entry = prerequisites.find((item) => module.blockers.includes(item.id) && item.unlocked)
    ?? prerequisites.find((item) => item.unlocked);
  const expected = outputs[module.id];
  return <div className="clinical-workspace roadmap-module-detail">
    <PageHeader eyebrow={phaseEyebrow[module.phase]} title={module.title} description={module.description} />
    <section className="module-gate">
      <span className="module-gate-icon"><Icon name={module.unlocked ? moduleIcons[module.id] : 'lock'} size={28} /></span>
      <div><h2>{module.unlocked ? 'Review this workflow' : 'Prepare the required inputs'}</h2><p>{module.id === 'evaluation' ? 'Inference requires a ready predictor from Experiments and a verified test cohort. Test-cohort preparation can proceed while models are still developing.' : module.id === 'interpretation' ? 'Attention requires an ABMIL predictor, compatible features and slide images. Evaluation and clinical reports are optional.' : 'Your saved work stays available. Prepare the required inputs below to continue.'}</p></div>
    </section>
    {prerequisites.length ? <section className="module-prerequisites"><h2>Required modules</h2>{prerequisites.map((item) => <div key={item.id}><Icon name={moduleIcons[item.id]} /><span><strong>{item.title}</strong><small>{item.evidence}</small></span><ModuleStatus status={item.status} completedLabel={completedModuleLabel(item.id)} />{item.unlocked ? <a className={`btn btn-small ${item.id === entry?.id ? 'btn-primary' : 'btn-secondary'}`} href={`#${item.id}`}>Open<Icon name="arrow" size={14} /></a> : <Icon name="lock" size={15} />}</div>)}</section> : null}
    {module.compatibilityIssue ? <div className="callout callout-warning">{module.compatibilityIssue}</div> : null}
    {expected ? <section className="module-expected"><h2>{module.id === 'test-data' ? 'What you will need' : 'Module outputs'}</h2><ul>{expected.map((output) => <li key={output}><Icon name="check" size={16} />{output}</li>)}</ul></section> : null}
    <div className="callout"><Icon name="info" size={17} /><p>{module.id === 'test-data' ? 'Import a dataset, select test records and define the prediction target. Model and feature compatibility are checked during evaluation; preparing a test cohort does not require a development protocol.' : workspace.mode === 'synthetic-demo' ? 'This demonstration contains illustrative results. It does not execute training, freeze a predictor, or evaluate later test data.' : 'Supported ABMIL development batches produce fold checkpoints and OOF predictions. Experiments automatically generate the configured ensemble or refit predictors from completed folds. Evaluate ready predictors on a verified test cohort.'}</p></div>
    <a className={`btn ${entry ? 'btn-secondary' : 'btn-primary'}`} href="#overview">Return to roadmap<Icon name="arrow" size={16} /></a>
  </div>;
}
