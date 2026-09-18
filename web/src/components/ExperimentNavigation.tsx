import type { ExperimentStage } from '../api/experiments';
import { StageSteps } from './StageWorkflow';

export type ExperimentPage = 'setup' | 'batches' | 'review' | 'runs' | 'results';

/** Setup is a short sequence. Submitted records use views without numbered progress. */
export default function ExperimentNavigation({ stage, current, disabled, inputsReady, hasBatches, onChange }: {
  stage: ExperimentStage; current: ExperimentPage; disabled: boolean; inputsReady: boolean;
  hasBatches: boolean; onChange: (page: ExperimentPage) => void;
}) {
  if (stage === 'planning') return <div className="experiment-navigation">
    <StageSteps label="Experiment setup" current={current} disabled={disabled} onChange={(next) => onChange(next as ExperimentPage)} steps={[
      { id: 'setup', buttonId: 'development-tab-setup', title: 'Inputs', description: 'Choose targets and features', complete: inputsReady },
      { id: 'batches', buttonId: 'development-tab-batches', title: 'Batches', description: 'Choose training settings', disabled: !inputsReady, complete: hasBatches },
      { id: 'review', title: 'Review & submit', description: 'Confirm and start training', disabled: !inputsReady },
    ]} />
    <div className="experiment-future-views"><span>After submission</span><button id="development-tab-runs" disabled>Runs</button><span aria-hidden="true">·</span><button id="development-tab-results" disabled>Results</button></div>
  </div>;
  const views = [{ id: 'setup', title: 'Inputs' }, { id: 'batches', title: 'Batches' }, { id: 'runs', title: 'Runs' }, { id: 'results', title: 'Results' }] as const;
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
      aria-current={current === view.id ? 'page' : undefined} disabled={disabled || (view.id === 'results' && stage !== 'finished')}
      title={view.id === 'results' && stage !== 'finished' ? 'Results open when training and predictor creation finish.' : undefined}
      onClick={() => onChange(view.id)}>{view.title}</button>)}
  </nav>;
}
