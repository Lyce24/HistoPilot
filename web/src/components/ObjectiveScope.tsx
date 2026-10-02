import { useQuery } from '@tanstack/react-query';
import { objectiveRankLabel, objectives as objectiveRecords, type Objective } from '../api/objectives';
import { UNFILED_SCOPE, useObjectiveScope } from '../lib/objectiveScope';
import './ObjectiveScope.css';

/** Focus the project on one objective. Nothing shows until the project has objectives. */
export function ObjectiveScopeSelect({ objectives, scope, onChange }: {
  objectives: readonly Objective[]; scope: string; onChange: (value: string) => void;
}) {
  if (!objectives.length) return null;
  return <label className="label objective-scope-select"><span>Objective</span><select className="field" aria-label="Focus on an objective" value={scope} onChange={(event) => onChange(event.target.value)}>
    <option value="">All objectives</option>
    {objectives.map((item) => <option key={item.id} value={item.id}>{item.name} · {objectiveRankLabel[item.rank]}</option>)}
    <option value={UNFILED_SCOPE}>No objective</option>
  </select></label>;
}

/** The project-wide focus, in the top bar of the pages it applies to. */
export function TopbarObjectiveScope({ project }: { project: string }) {
  const listed = useQuery({ queryKey: ['objectives', project], queryFn: () => objectiveRecords.list(project), staleTime: 30000 });
  const items = listed.data?.items ?? [];
  const [scope, setScope] = useObjectiveScope(items);
  return <ObjectiveScopeSelect objectives={items} scope={scope} onChange={setScope} />;
}

/** What a focused list leaves out, and the way back to every objective. */
export function ObjectiveScopeNotice({ objectives, scope, hidden, noun, onClear }: {
  objectives: readonly Objective[]; scope: string; hidden: number; noun: string; onClear: () => void;
}) {
  if (!scope) return null;
  const name = scope === UNFILED_SCOPE ? 'work filed under no objective' : objectives.find((item) => item.id === scope)?.name ?? 'one objective';
  return <p className="callout objective-scope-notice" role="status">Focused on {name}{hidden ? ` · ${hidden.toLocaleString()} other ${noun} hidden` : ''}. <button type="button" className="text-button" onClick={onClear}>Show all objectives</button></p>;
}
