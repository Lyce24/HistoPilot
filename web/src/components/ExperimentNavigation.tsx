import type { ExperimentStage } from '../api/experiments';
import { StageSteps } from './StageWorkflow';

export type ExperimentPage = 'setup' | 'batches' | 'review' | 'runs' | 'results' | 'predictors';

/** Setup is a short sequence. Submitted records use views without numbered progress. */
export default function ExperimentNavigation({ stage, mode, frozen = false, current, disabled, inputsReady, hasBatches, onChange }: {
  mode: 'setup' | 'execution'; frozen?: boolean; stage: ExperimentStage; current: ExperimentPage; disabled: boolean; inputsReady: boolean;
  hasBatches: boolean; onChange: (page: ExperimentPage) => void;
}) {
  if (mode === 'execution' && stage === 'planning') return <nav className="experiment-views" aria-label="Execution"><span>Frozen setup</span><strong>Ready to run</strong></nav>;
  if (mode === 'setup') return <StageSteps label="Experimental setup" current={current} disabled={disabled} onChange={(next) => onChange(next as ExperimentPage)} steps={[
    { id: 'setup', title: 'Inputs & training design', description: 'Dataset, features, targets and folds', complete: inputsReady },
    { id: 'batches', title: 'Hyperparameters', description: 'Models, training settings and seeds', disabled: !inputsReady && !frozen, complete: hasBatches },
    { id: 'review', title: frozen ? 'Frozen setup' : 'Review & freeze', description: 'Save the design for execution', disabled: !inputsReady && !frozen },
  ]} />;
  const views = [{ id: 'runs', title: 'Runs' }, { id: 'results', title: 'Results' }, { id: 'predictors', title: 'Predictors' }] as const;
  return <nav className="experiment-views" aria-label="Experiment views" onKeyDown={(event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
    const index = buttons.indexOf(event.target as HTMLButtonElement);
    if (index < 0) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus(); buttons[next]?.click();
  }}>
    <span className="experiment-view-caption">Submitted plan</span>
    {views.map((view) => <button type="button" id={`development-tab-${view.id}`} key={view.id}
      aria-current={current === view.id ? 'page' : undefined} disabled={disabled}
      title={view.id === 'results' && stage === 'running' ? 'Partial results: training seeds appear as their folds finish.' : undefined}
      onClick={() => onChange(view.id)}>{view.title}</button>)}
  </nav>;
}
