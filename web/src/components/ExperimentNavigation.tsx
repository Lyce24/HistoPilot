import type { ExperimentStage } from '../api/experiments';
import { StageSteps, StageViews } from './StageWorkflow';

export type ExperimentPage = 'setup' | 'batches' | 'review' | 'runs' | 'results' | 'predictors';

/**
 * One experiment from design to results. Before it starts, its design is a short sequence:
 * inputs and training design, hyperparameters, then review and freeze; a frozen design is
 * started from the same last step. A started experiment has views in the same order: its
 * read-only design, then its runs, results and predictors.
 */
export default function ExperimentNavigation({ stage, frozen = false, current, disabled, inputsReady, hasBatches, onChange }: {
  frozen?: boolean; stage: ExperimentStage; current: ExperimentPage; disabled: boolean; inputsReady: boolean;
  hasBatches: boolean; onChange: (page: ExperimentPage) => void;
}) {
  if (stage === 'planning') return <StageSteps label="Experiment design" current={current} disabled={disabled} onChange={(next) => onChange(next as ExperimentPage)} steps={[
    { id: 'setup', title: 'Inputs & training design', description: 'Dataset, features, targets and folds', complete: inputsReady },
    { id: 'batches', title: 'Hyperparameters', description: 'Models, training settings and seeds', disabled: !inputsReady && !frozen, complete: hasBatches },
    { id: 'review', title: frozen ? 'Start' : 'Review & freeze', description: frozen ? 'The design is frozen; start training' : 'Freeze the design, then start it', disabled: !inputsReady && !frozen, complete: frozen },
  ]} />;
  const views = [
    { id: 'setup', title: 'Inputs' },
    { id: 'batches', title: 'Hyperparameters' },
    { id: 'runs', title: 'Runs' },
    { id: 'results', title: 'Results', hint: stage === 'running' ? 'Partial results: training seeds appear as their folds finish.' : undefined },
    { id: 'predictors', title: 'Predictors' },
  ] as const;
  return <StageViews label="Experiment views" caption="Started experiment" idPrefix="development-tab-" views={views} current={current} disabled={disabled} onChange={onChange} />;
}
