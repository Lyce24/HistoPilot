import type { AttributeMapping, Condition, ConditionGroup, ConditionValue, ScalarCondition } from '../api/scientific';

export function isConditionGroup(condition: Condition): condition is ConditionGroup {
  return condition.op === 'all' || condition.op === 'any';
}

export function scalarConditions(conditions: Condition[]): ScalarCondition[] {
  return conditions.flatMap((condition) => isConditionGroup(condition)
    ? scalarConditions(condition.conditions) : [condition]);
}

export function conditionFields(conditions: Condition[]): string[] {
  return [...new Set(scalarConditions(conditions).map((condition) => condition.field))];
}

export function conditionComposition(conditions: Condition[]): ConditionGroup {
  return conditions.length === 1 && isConditionGroup(conditions[0])
    ? conditions[0] : { op: 'all', conditions };
}

export function composeConditions(op: ConditionGroup['op'], conditions: Condition[]): Condition[] {
  // Keep empty filters and ordinary AND lists compatible with existing frozen versions.
  return op === 'all' || conditions.length === 0 ? conditions : [{ op, conditions }];
}

export function parseConditionValue(text: string, op: ScalarCondition['op'], type?: AttributeMapping['type']): ConditionValue {
  const numeric = ['lt', 'lte', 'gt', 'gte'].includes(op) || ['integer', 'decimal'].includes(type ?? '');
  const scalar = (value: string) => numeric && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : value;
  if (op === 'in' || op === 'not_in') return text.split('|').map((value) => scalar(value.trim()));
  if (op === 'regex') return text;
  if (type === 'boolean' && (text === 'true' || text === 'false')) return text === 'true';
  return scalar(text);
}

export function describeCondition(condition: Condition): string {
  if (isConditionGroup(condition)) {
    return `(${condition.conditions.map(describeCondition).join(condition.op === 'all' ? ' AND ' : ' OR ')})`;
  }
  if (condition.op === 'exists') return `${condition.field} ${condition.value ? 'is present' : 'is missing'}`;
  const labels = { eq: 'equals', ne: 'does not equal', in: 'is one of', not_in: 'is not one of', regex: 'matches', lt: 'is less than', lte: 'is at most', gt: 'is greater than', gte: 'is at least' };
  return `${condition.field} ${labels[condition.op]} ${Array.isArray(condition.value) ? condition.value.map((value) => value ?? 'Missing').join(' OR ') : String(condition.value ?? 'Missing')}`;
}
