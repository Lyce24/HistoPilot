import { applyHref } from '../lib/applyRoutes';

type Step = 'experiments' | 'post-development' | 'apply';
export interface EvidenceContext { experimentId?: string; predictorId?: string; evaluationId?: string; clinicalAnalysisId?: string }
export function evidenceLink(module: string, context: EvidenceContext = {}) {
  // A run opens in Apply models; a predictor or experiment starts applying it to a cohort.
  if (module === 'apply') {
    if (context.evaluationId) return applyHref({ run: context.evaluationId, clinical: context.clinicalAnalysisId });
    return context.experimentId || context.predictorId ? applyHref({ view: 'new', experiment: context.experimentId, predictor: context.predictorId }) : applyHref();
  }
  const query = new URLSearchParams();
  if (context.experimentId) query.set('experiment', context.experimentId);
  if (context.predictorId) query.set('predictor', context.predictorId);
  if (context.evaluationId) query.set('evaluation', context.evaluationId);
  if (context.clinicalAnalysisId) query.set('clinical', context.clinicalAnalysisId);
  if (module === 'post-development') { module = 'experiments'; query.set('tab', 'predictors'); }
  return `#${module}${query.size ? `?${query}` : ''}`;
}
/** Experiments produce predictors; Apply models runs them on cohorts, labeled or not. Model
 * interpretation needs only trained weights, a dataset and its features, so it is not a step here. */
export default function EvidenceChain({ current, ...context }: EvidenceContext & { current: Step }) {
  const steps = [['experiments', 'Experiments'], ['apply', 'Apply models']] as const;
  return <nav className="chain-banner" aria-label="Model evidence workflow">{steps.map(([module, label], index) => <span className="evidence-chain-step" key={module}>{index ? <span aria-hidden="true">→</span> : null}{(current === 'post-development' ? 'experiments' : current) === module ? <strong aria-current="step">{label}</strong> : <a href={evidenceLink(module, context)}>{label}</a>}</span>)}</nav>;
}
