/** Retain an existing positive-class choice only while it is still a class. */
export function preservePositiveClass(
  current: string | undefined,
  classes: string[],
): string | undefined {
  return current !== undefined && classes.includes(current) ? current : undefined;
}

/** Initial suggestions use complete, non-missing source values without changing their spelling. */
export function inferTargetSettings(values: (string | null)[], truncated = false) {
  const classes = truncated
    ? []
    : [
        ...new Set(
          values.filter((value): value is string => value !== null && value.trim() !== ''),
        ),
      ];
  const task: '' | 'binary_classification' | 'multiclass_classification' =
    classes.length === 2
      ? 'binary_classification'
      : classes.length > 2
        ? 'multiclass_classification'
        : '';
  return {
    task,
    classes,
    labels: Object.fromEntries(classes.map((value) => [value, value])),
    positiveClass: task === 'binary_classification' ? classes[0] : undefined,
  };
}
