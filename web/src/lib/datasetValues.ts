/** Quote literal categories so a source string cannot impersonate missing data. */
export function categoryLabel(value: string | null): string {
  return value === null ? '(missing value)' : JSON.stringify(value);
}
