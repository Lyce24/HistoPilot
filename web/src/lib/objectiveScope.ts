import { useSyncExternalStore } from 'react';
import type { Objective } from '../api/objectives';

/**
 * The objective a project is focused on, shared by Experiments, Apply models, Model
 * interpretation and the roadmap. It lives in the page address (`?objective=`) beside
 * `?project=`, so it survives moving between modules (their links change only the hash),
 * reloads, and links shared with others. Empty means every objective; {@link UNFILED_SCOPE}
 * means work filed under no objective.
 */
export const UNFILED_SCOPE = 'none';
const EVENT = 'histopilot:objective-scope';

export function readObjectiveScope(href = typeof window === 'undefined' ? '' : window.location.href): string {
  if (!href) return '';
  return new URL(href).searchParams.get('objective') ?? '';
}

export function setObjectiveScope(value: string) {
  const url = new URL(window.location.href);
  if (value) url.searchParams.set('objective', value); else url.searchParams.delete('objective');
  window.history.replaceState(window.history.state, '', url);
  window.dispatchEvent(new Event(EVENT));
}

function subscribe(callback: () => void) {
  for (const name of [EVENT, 'popstate', 'hashchange']) window.addEventListener(name, callback);
  return () => { for (const name of [EVENT, 'popstate', 'hashchange']) window.removeEventListener(name, callback); };
}

/**
 * The focused objective, checked against the project's objectives: an address naming an
 * objective this project does not have (another project's, or one removed since) reads as
 * every objective, so a stale link never hides everything.
 */
export function useObjectiveScope(objectives: readonly Objective[] | undefined): [string, (value: string) => void] {
  const value = useSyncExternalStore(subscribe, () => readObjectiveScope(), () => '');
  return [effectiveScope(value, objectives), setObjectiveScope];
}

export function effectiveScope(value: string, objectives: readonly Objective[] | undefined) {
  if (!value || value === UNFILED_SCOPE) return value;
  return objectives?.some((item) => item.id === value) ? value : '';
}

/** Whether work from an experiment filed under `objectiveId` is in the focused scope. */
export function inObjectiveScope(scope: string, objectiveId: string | null | undefined) {
  if (!scope) return true;
  return scope === UNFILED_SCOPE ? !objectiveId : objectiveId === scope;
}
