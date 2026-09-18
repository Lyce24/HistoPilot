import { useId, useState } from 'react';
import type { Condition, ConditionGroup, ConditionScalar, ScalarCondition } from '../api/scientific';
import { composeConditions, conditionComposition, isConditionGroup, parseConditionValue, scalarConditions } from '../lib/conditions';
import { FieldProfile, useFieldProfile, type ProtocolFieldContext } from './ProtocolExploration';
import { ErrorNotice, Icon } from './ui';
import './ConditionEditor.css';

const OPERATORS: [ScalarCondition['op'], string][] = [
  ['in', 'is any of'], ['not_in', 'is none of'], ['eq', 'equals'], ['ne', 'does not equal'],
  ['lt', 'less than'], ['lte', 'at most'], ['gt', 'greater than'], ['gte', 'at least'],
  ['exists', 'is present / missing'], ['regex', 'matches regex'],
];
const literalKey = (value: ConditionScalar) => JSON.stringify(value);

function MembershipValues({ condition, fieldContext, onChange }: {
  condition: ScalarCondition;
  fieldContext: ProtocolFieldContext;
  onChange: (condition: ScalarCondition) => void;
}) {
  const [search, setSearch] = useState('');
  const [customValue, setCustomValue] = useState('');
  const { query, identity } = useFieldProfile({ ...fieldContext, field: condition.field });
  const selected = Array.isArray(condition.value) ? condition.value : [];
  const values = identity
    ? [...new Set((query.data?.records ?? []).map((row) => condition.field === 'Slide_ID' ? row.slideId : row.patientId))]
    : (query.data?.valueCounts ?? []).map((item) => item.value);
  const options = [...new Map([...selected, ...values].map((value) => [literalKey(value), value])).values()];
  const visible = options.filter((value) => String(value ?? 'Missing').toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  const update = (value: ConditionScalar, checked: boolean) => onChange({
    ...condition,
    value: checked ? [...selected, value] : selected.filter((item) => literalKey(item) !== literalKey(value)),
  });
  const addValue = () => {
    if (customValue.length && selected.length < 100 && !selected.some((value) => value === customValue)) update(customValue, true);
    setCustomValue('');
  };
  return <fieldset className="condition-values">
    <legend>{condition.op === 'in' ? 'Include any selected value' : 'Exclude selected values'}</legend>
    {options.length > 8 ? <label className="label">Find a value<input className="field" type="search" value={search} onChange={(event) => setSearch(event.target.value)} /></label> : null}
    <div className="condition-value-options">
      {visible.map((value) => <label className="condition-value-option" key={literalKey(value)}>
        <input type="checkbox" checked={selected.some((item) => literalKey(item) === literalKey(value))}
          disabled={selected.length >= 100 && !selected.some((item) => literalKey(item) === literalKey(value))}
          onChange={(event) => update(value, event.target.checked)} />
        <span>{value === null ? 'Missing' : value === '' ? '(empty text)' : String(value)}</span>
      </label>)}
    </div>
    {query.isPending && !query.error && fieldContext.datasetId ? <small className="muted">Reading available values…</small> : null}
    {!selected.length ? <small className="muted">Select at least one value.</small> : <small className="muted">{selected.length} selected · {condition.op === 'in' ? 'A slide can match any one of these values (OR).' : 'A slide must match none of these values.'}</small>}
    {query.data?.valuesTruncated || identity ? <small className="muted">Showing example values. You can add any exact value below.</small> : null}
    <div className="condition-custom-value">
      <label className="label">Add an exact value<input className="field" value={customValue} maxLength={4096}
        placeholder="Value from the dataset" onChange={(event) => setCustomValue(event.target.value)}
        onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); addValue(); } }} /></label>
      <button type="button" className="btn btn-secondary btn-small" disabled={!customValue.length || selected.length >= 100} onClick={addValue}>Add value</button>
    </div>
    <ErrorNotice error={query.error} />
  </fieldset>;
}

function ScalarRuleEditor({ condition, fields, fieldContext, onChange, onRemove, label }: {
  condition: ScalarCondition;
  fields: string[];
  fieldContext: ProtocolFieldContext;
  onChange: (condition: ScalarCondition) => void;
  onRemove: () => void;
  label: string;
}) {
  const suggestionsId = useId();
  const { query } = useFieldProfile({ ...fieldContext, field: condition.field });
  const membership = condition.op === 'in' || condition.op === 'not_in';
  return <div className="protocol-condition">
    <div className="condition-row">
      <label className="label">Field<select className="field" value={condition.field} onChange={(event) => onChange({ ...condition, field: event.target.value, value: membership ? [] : condition.op === 'exists' ? true : '' })}>
        {!fields.includes(condition.field) ? <option value={condition.field}>{condition.field} (unavailable)</option> : null}
        {fields.map((field) => <option key={field}>{field}</option>)}
      </select></label>
      <label className="label">Condition<select className="field" value={condition.op} onChange={(event) => {
        const op = event.target.value as ScalarCondition['op'];
        const previous = Array.isArray(condition.value) ? condition.value : [condition.value];
        onChange({ ...condition, op, value: op === 'exists' ? true : op === 'in' || op === 'not_in' ? previous.filter((value) => value !== '') : previous[0] ?? '' });
      }}>{OPERATORS.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select></label>
      {membership ? <span className="condition-selected-count">{Array.isArray(condition.value) ? condition.value.length : 0} values selected</span> : <label className="label">Value
        {condition.op === 'exists' ? <select className="field" value={String(condition.value)} onChange={(event) => onChange({ ...condition, value: event.target.value === 'true' })}>
          <option value="true">Is present</option><option value="false">Is missing</option>
        </select> : <input className="field" list={condition.op === 'eq' || condition.op === 'ne' ? suggestionsId : undefined}
          value={String(condition.value ?? '')} placeholder={condition.op === 'regex' ? '^pattern$' : ''}
          inputMode={['lt', 'lte', 'gt', 'gte'].includes(condition.op) ? 'decimal' : undefined}
          onChange={(event) => onChange({ ...condition, value: parseConditionValue(event.target.value, condition.op) })} />}
      </label>}
      <button type="button" className="icon-button" aria-label={`Remove ${label}`} onClick={onRemove}><Icon name="close" /></button>
    </div>
    {membership ? <MembershipValues key={`${condition.field}:${condition.op}`} condition={condition} fieldContext={fieldContext} onChange={onChange} /> : <>
      <datalist id={suggestionsId}>{(query.data?.valueCounts ?? []).filter((item) => item.value !== null).map((item) => <option key={item.value!} value={item.value!} />)}</datalist>
      <FieldProfile {...fieldContext} field={condition.field} />
    </>}
    {condition.op === 'regex' ? <small className="muted">Matches text, preserving spelling and case. For example, ^T matches values beginning with T.</small> : null}
  </div>;
}

function RuleGroupEditor({ group, fields, fieldContext, onChange, newCondition, depth, available, label }: {
  group: ConditionGroup;
  fields: string[];
  fieldContext: ProtocolFieldContext;
  onChange: (group: ConditionGroup) => void;
  newCondition: () => ScalarCondition;
  depth: number;
  available: boolean;
  label: string;
}) {
  const update = (conditions: Condition[]) => onChange({ ...group, conditions });
  return <div className={depth ? 'condition-group' : 'condition-root'}>
    <div className="condition-composition">
      {group.conditions.length ? <label className="label">Match<select className="field" aria-label={`${label} match mode`} value={group.op} onChange={(event) => onChange({ ...group, op: event.target.value as ConditionGroup['op'] })}>
        <option value="all">All conditions (AND)</option><option value="any">Any condition (OR)</option>
      </select></label> : null}
      <button type="button" className="btn btn-secondary btn-small" disabled={!available} onClick={() => update([...group.conditions, newCondition()])}><Icon name="plus" size={14} />Add condition</button>
      {depth < 3 ? <button type="button" className="btn btn-secondary btn-small" disabled={!available} onClick={() => update([...group.conditions, { op: group.op === 'all' ? 'any' : 'all', conditions: [newCondition()] }])}>Add group</button> : null}
    </div>
    {group.conditions.length ? <small className="muted">{group.op === 'all' ? 'A slide must match every condition in this group.' : 'A slide needs to match at least one condition in this group.'}</small> : null}
    {group.conditions.map((condition, index) => isConditionGroup(condition)
      ? <div className="condition-subgroup" key={index}>
        <div className="condition-subgroup-heading"><strong>Condition group</strong><button type="button" className="icon-button" aria-label={`Remove ${label} group ${index + 1}`} onClick={() => update(group.conditions.filter((_, at) => at !== index))}><Icon name="close" /></button></div>
        <RuleGroupEditor group={condition} fields={fields} fieldContext={fieldContext} newCondition={newCondition} depth={depth + 1} available={available} label={`${label} group ${index + 1}`}
          onChange={(changed) => update(changed.conditions.length ? group.conditions.map((item, at) => at === index ? changed : item) : group.conditions.filter((_, at) => at !== index))} />
      </div>
      : <ScalarRuleEditor key={index} condition={condition} fields={fields} fieldContext={fieldContext} label={`${label} condition ${index + 1}`}
        onChange={(changed) => update(group.conditions.map((item, at) => at === index ? changed : item))}
        onRemove={() => update(group.conditions.filter((_, at) => at !== index))} />)}
  </div>;
}

export function ConditionEditor({ title, description, emptyMessage, conditions, columns, fieldContext, onChange }: {
  title: string;
  description: string;
  emptyMessage: string;
  conditions: Condition[];
  columns: string[];
  fieldContext: ProtocolFieldContext;
  onChange: (conditions: Condition[]) => void;
}) {
  const fields = [...new Set([...columns, 'Slide_ID', 'Patient_ID'])];
  const newCondition = (): ScalarCondition => ({ field: columns.find((field) => field.toLocaleLowerCase() === 'cohort') ?? columns[0] ?? 'Slide_ID', op: 'in', value: [] });
  return <div className="condition-editor">
    <h3>{title}</h3><p className="muted">{description}</p>
    <RuleGroupEditor group={conditionComposition(conditions)} fields={fields} fieldContext={fieldContext} newCondition={newCondition} depth={0}
      available={scalarConditions(conditions).length < 30} label={title}
      onChange={(group) => onChange(composeConditions(group.op, group.conditions))} />
    {!conditions.length ? <p className="muted">{emptyMessage}</p> : null}
  </div>;
}

export default ConditionEditor;
