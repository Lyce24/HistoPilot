import type { BatchHeadline } from '../api/experimentResults';
import { fixed, metricPhrase } from '../lib/experimentResults';

/** An experiment's batches with an AUROC, best seed mean first. */
const rankedBatches = (batches?: BatchHeadline[]) =>
  (batches ?? []).filter((batch) => batch.metrics.auroc).sort((a, b) => b.metrics.auroc!.mean - a.metrics.auroc!.mean);

/**
 * The experiment list's result cell: the best batch's OOF seed mean ± SD on macro AUROC
 * (AUROC for a binary target). Opening the experiment shows folds, seeds and intervals.
 */
export default function ExperimentHeadline({ batches, loading = false }: { batches?: BatchHeadline[]; loading?: boolean }) {
  const ranked = rankedBatches(batches);
  // While results load, a dash would read as "no result".
  if (!ranked.length) return loading ? <span className="muted" role="status" aria-label="Loading result">…</span> : <span className="muted">—</span>;
  const best = ranked[0];
  const auroc = best.metrics.auroc!;
  const partial = best.seeds < best.plannedSeeds;
  return <span className="experiment-headline">
    <span className="experiment-headline-value"><strong>{fixed(auroc.mean)}</strong>{auroc.sd !== null && auroc.n > 1 ? <span> ± {fixed(auroc.sd)}</span> : null}</span>
    <small>{best.name} · OOF {metricPhrase('auroc', best.task)}</small>
    <small>{partial ? `${best.seeds} of ${best.plannedSeeds} seeds so far` : `${best.seeds} seed${best.seeds === 1 ? '' : 's'}`}{ranked.length > 1 ? ` · best of ${ranked.length} batches` : ''}</small>
  </span>;
}

/**
 * A series' result cell: the range of its members' headlines, each member's best batch as
 * {@link ExperimentHeadline} picks it, and how many members have one yet.
 */
export function SeriesHeadline({ members, loading = false }: { members: (BatchHeadline[] | undefined)[]; loading?: boolean }) {
  const best = members.map((batches) => rankedBatches(batches)[0]).filter((batch): batch is BatchHeadline => Boolean(batch));
  if (!best.length) return loading ? <span className="muted" role="status" aria-label="Loading results">…</span> : <span className="muted">—</span>;
  const means = best.map((batch) => batch.metrics.auroc!.mean);
  const low = Math.min(...means);
  const high = Math.max(...means);
  return <span className="experiment-headline">
    <span className="experiment-headline-value"><strong>{fixed(low)}</strong>{high > low ? <span> – {fixed(high)}</span> : null}</span>
    <small>Range of best OOF {metricPhrase('auroc', best[0].task)}</small>
    <small>{best.length} of {members.length} with results</small>
  </span>;
}
