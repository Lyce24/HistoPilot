import { experimentRoles, type ExperimentRole, type Objective } from '../api/objectives';

/** What the library reads of an experiment to place it. */
export interface Filed {
  id: string; name: string;
  objectiveId?: string | null; role?: ExperimentRole | null; series?: string | null; seriesMember?: string | null;
}
/** One row of an objective: an experiment, or a series of two or more that differ in one factor. */
export type LibraryEntry<T extends Filed> =
  | { kind: 'experiment'; key: string; item: T }
  | { kind: 'series'; key: string; series: string; role: ExperimentRole | null; items: T[] };
export interface LibraryGroup<T extends Filed> {
  /** The objective's ID, or {@link UNSORTED}. */
  key: string;
  objective: Objective | null;
  /** Everything but tuning, primary first. */
  entries: LibraryEntry<T>[];
  /** Pilots, hyperparameter rounds and earlier versions, folded away by default. */
  tuning: LibraryEntry<T>[];
  count: number;
}
export const UNSORTED = 'unsorted';

const natural = (left: string, right: string) => left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' });
export const entryRole = <T extends Filed>(entry: LibraryEntry<T>) => entry.kind === 'series' ? entry.role : entry.item.role ?? null;
/** Primary, comparison, variant, generalization and control, then experiments without a role, then tuning. */
function roleRank(role: ExperimentRole | null) {
  if (role === null) return experimentRoles.indexOf('tuning') - 0.5;
  return experimentRoles.indexOf(role);
}

/**
 * An objective's rows, in role order. Experiments arrive in the library's sort order, which
 * holds within a role; a series sits where its first member would, and lists its members
 * by their place in it (cptac, rih, tcga…). A series of one is shown as its experiment.
 */
export function libraryEntries<T extends Filed>(items: T[], group: string): LibraryEntry<T>[] {
  const rows: LibraryEntry<T>[] = [];
  const series = new Map<string, Extract<LibraryEntry<T>, { kind: 'series' }>>();
  for (const item of items) {
    if (!item.series) { rows.push({ kind: 'experiment', key: item.id, item }); continue; }
    const key = `${group}\u0000${item.role ?? ''}\u0000${item.series}`;
    const known = series.get(key);
    if (known) { known.items.push(item); continue; }
    const entry = { kind: 'series' as const, key, series: item.series, role: item.role ?? null, items: [item] };
    series.set(key, entry);
    rows.push(entry);
  }
  return rows
    .map((entry): LibraryEntry<T> => entry.kind === 'series' && entry.items.length === 1 ? { kind: 'experiment', key: entry.items[0].id, item: entry.items[0] } : entry)
    .map((entry) => entry.kind === 'series'
      ? { ...entry, items: [...entry.items].sort((a, b) => natural(a.seriesMember ?? '', b.seriesMember ?? '') || natural(a.name, b.name) || a.id.localeCompare(b.id)) }
      : entry)
    .sort((a, b) => roleRank(entryRole(a)) - roleRank(entryRole(b)));
}

/**
 * The library by objective: objectives in the service's order (primary, secondary, then
 * exploratory, by name), then experiments filed under none. `showEmpty` keeps objectives
 * nothing is filed under yet, so a new one is visible before anything joins it; filtered
 * views leave them out.
 */
export function groupExperiments<T extends Filed>(items: T[], objectives: Objective[], { showEmpty }: { showEmpty: boolean }): LibraryGroup<T>[] {
  const known = new Set(objectives.map((objective) => objective.id));
  const buckets = new Map<string, T[]>();
  for (const item of items) {
    const key = item.objectiveId && known.has(item.objectiveId) ? item.objectiveId : UNSORTED;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(item); else buckets.set(key, [item]);
  }
  const groups: { key: string; objective: Objective | null }[] = [...objectives.map((objective) => ({ key: objective.id, objective })), { key: UNSORTED, objective: null }];
  return groups.flatMap(({ key, objective }) => {
    const members = buckets.get(key) ?? [];
    if (!members.length && !(showEmpty && objective)) return [];
    const rows = libraryEntries(members, key);
    return [{ key, objective, count: members.length, entries: rows.filter((entry) => entryRole(entry) !== 'tuning'), tuning: rows.filter((entry) => entryRole(entry) === 'tuning') }];
  });
}

/** Members of a series named briefly: "cptac, rih, tcga and 6 more". */
export function memberSummary(items: Filed[], shown = 4) {
  const names = items.map((item) => item.seriesMember || item.name);
  if (names.length <= shown) return names.join(', ');
  return `${names.slice(0, shown).join(', ')} and ${names.length - shown} more`;
}
