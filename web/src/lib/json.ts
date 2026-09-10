/** Compare the JSON values sent to the service, independent of object key order. */
export function sameJSON(left: unknown, right: unknown): boolean {
  const canonical = (value: unknown) =>
    JSON.stringify(value, (_key, item: unknown) =>
      item !== null && typeof item === 'object' && !Array.isArray(item)
        ? Object.fromEntries(
            Object.entries(item).sort(([leftKey], [rightKey]) =>
              leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0,
            ),
          )
        : item,
    );
  return canonical(left) === canonical(right);
}
