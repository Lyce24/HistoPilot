/** What each kind of saved record is called on every page, one name per kind. */
const kindLabels: Record<string, string> = {
  project: 'Whole project', dataset: 'Dataset', protocol: 'Targets & splits', 'analysis-protocol': 'Targets & splits', 'target-split': 'Targets & splits',
  feature: 'Feature inventory', 'feature-bundle': 'Feature bundle', 'mil-batch': 'Training batch',
  'development-batch': 'Training batch draft', 'mil-experiment': 'Saved experiment inputs', 'experiment-setup': 'Frozen design',
  'model-experiment': 'Experiment', 'frozen-predictor': 'Predictor', 'predictor-refit': 'Refit training plan', 'model-evaluation': 'Run · labeled cohort', 'evaluation-batch': 'Batch · labeled cohort',
  'inference-run': 'Run · unlabeled cohort', 'inference-batch': 'Batch · unlabeled cohort',
  'clinical-analysis': 'Clinical utility analysis', 'reference-standard': 'Reference standard', 'model-interpretation': 'Model interpretation',
  'evaluation-cohort': 'Cohort', 'dataset-import': 'Import draft', extraction: 'Feature extraction',
  'feature-validate': 'Feature validation', 'feature-validation': 'Feature validation',
  'feature-pack': 'Feature packing', 'feature-packing': 'Feature packing', packing: 'Feature packing',
};
export const recordKindLabel = (kind: string) => kindLabels[kind] ?? kind.replaceAll('-', ' ').replace(/^./, (letter) => letter.toUpperCase());
/** A kind where whether a run's cohort is labeled is not known: a run or a batch of runs. */
export const neutralRecordKindLabel = (kind: string) =>
  kind === 'model-evaluation' || kind === 'inference-run' ? 'Run'
    : kind === 'evaluation-batch' || kind === 'inference-batch' ? 'Batch of runs'
      : recordKindLabel(kind);
