import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { ModelExperiment } from '../api/experiments';
import { predictors } from '../api/predictors';
import { configurationChoice } from '../lib/applyModels';
import { applyHref } from '../lib/applyRoutes';
import { useSeedEnsembleBuild } from './useSeedEnsembleBuild';
import { ErrorNotice } from './ui';

/**
 * Apply the configuration shown in Results. With several seeds that is its seed ensemble, the
 * deployable form of the seed-ensemble row above (built here from the verified checkpoints
 * when needed); with one seed, its fold ensemble. Apply models opens with that predictor and
 * proposes the testing set its development reserved.
 */
export default function ApplyConfiguration({ project, record, batchId, candidateId, number }: {
  project: string; record: Pick<ModelExperiment, 'id' | 'name'>; batchId: string; candidateId: string; number: number;
}) {
  const seeds = useQuery({ queryKey: ['seed-ensembles', project, record.id], queryFn: () => predictors.seedEnsembles(project, record.id) });
  const registry = useQuery({ queryKey: ['predictors', project], queryFn: () => predictors.list(project) });
  const { busy, error, pending, build } = useSeedEnsembleBuild(project, record);
  if (!seeds.data || !registry.data) return null;
  const plan = configurationChoice(seeds.data.items, registry.data.items, record, batchId, candidateId);
  const link = (predictor: string) => applyHref({ view: 'new', experiment: record.id, predictor });
  const detail = plan.detail;
  let action: ReactNode = null;
  if (plan.action === 'apply') {
    action = <a className="btn btn-primary btn-small" href={link(plan.predictorId)}>Apply this configuration</a>;
  } else if (plan.action === 'build-seed-ensemble') {
    action = <button type="button" className="btn btn-primary btn-small" disabled={Boolean(busy)} onClick={() => void build(plan.choice).then((built) => { if (built) window.location.hash = link(built.id); })}>{busy ? 'Building seed ensemble…' : 'Build seed ensemble and apply'}</button>;
  } else if (plan.action === 'choose') {
    action = <a className="btn btn-secondary btn-small" href={applyHref({ view: 'new', experiment: record.id })}>Choose predictors to apply</a>;
  }
  return <section className="exp-apply" aria-label={`Apply configuration ${number}`}>
    <div><strong>Apply configuration {number} to a cohort</strong><p className="muted">{detail} A labeled cohort is scored; an unlabeled one is predicted only.</p></div>
    {action}
    <ErrorNotice error={error} />
    {pending && !busy ? <p className="callout">The last build response was lost. Building again reuses the same request identity.</p> : null}
  </section>;
}
