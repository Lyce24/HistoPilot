type Step = 'experiments' | 'post-development' | 'evaluation' | 'clinical-utility' | 'interpretation';
export interface EvidenceContext { experimentId?: string; predictorId?: string; evaluationId?: string; clinicalAnalysisId?: string }
export function evidenceLink(module: string, context: EvidenceContext = {}) {
  const query = new URLSearchParams();
  if (context.experimentId) query.set('experiment', context.experimentId);
  if (context.predictorId) query.set('predictor', context.predictorId);
  if (context.evaluationId) query.set('evaluation', context.evaluationId);
  if (context.clinicalAnalysisId) query.set('clinical', context.clinicalAnalysisId);
  if (module === 'post-development') { module = 'experiments'; query.set('tab', 'predictors'); }
  return `#${module}${query.size ? `?${query}` : ''}`;
}
export default function EvidenceChain({ current, ...context }: EvidenceContext & { current: Step }) {
  return <nav className="chain-banner" aria-label="Model evidence workflow">{([
    ['experiments', 'Experiments'], ['evaluation', 'Model evaluation'],
    ['clinical-utility', 'Clinical utility'], ['interpretation', 'Model interpretation'],
  ] as const).map(([module, label], index) => <span className="evidence-chain-step" key={module}>{index ? <span aria-hidden="true">→</span> : null}{(current === 'post-development' ? 'experiments' : current) === module ? <strong aria-current="step">{label}</strong> : <a href={evidenceLink(module, context)}>{label}</a>}</span>)}</nav>;
}
