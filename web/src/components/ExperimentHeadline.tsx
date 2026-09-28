import type { BatchHeadline } from '../api/experimentResults';
import { fixed, metricPhrase } from '../lib/experimentResults';

/**
 * The experiment list's result cell: the best batch's OOF seed mean ± SD on macro AUROC
 * (AUROC for a binary target). Opening the experiment shows folds, seeds and intervals.
 */
export default function ExperimentHeadline({ batches }: { batches?: BatchHeadline[] }) {
  const ranked = (batches ?? []).filter((batch) => batch.metrics.auroc).sort((a, b) => b.metrics.auroc!.mean - a.metrics.auroc!.mean);
  if (!ranked.length) return <span className="muted">—</span>;
  const best = ranked[0];
  const auroc = best.metrics.auroc!;
  const partial = best.seeds < best.plannedSeeds;
  return <span className="experiment-headline">
    <span className="experiment-headline-value"><strong>{fixed(auroc.mean)}</strong>{auroc.sd !== null && auroc.n > 1 ? <span> ± {fixed(auroc.sd)}</span> : null}</span>
    <small>{best.name} · OOF {metricPhrase('auroc', best.task)}</small>
    <small>{partial ? `${best.seeds} of ${best.plannedSeeds} seeds so far` : `${best.seeds} seed${best.seeds === 1 ? '' : 's'}`}{ranked.length > 1 ? ` · best of ${ranked.length} batches` : ''}</small>
  </span>;
}
