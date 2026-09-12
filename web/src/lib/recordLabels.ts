/** Compact presentation only: full IDs remain in links, exports and saved inputs. */
export function shortRecordId(id: string): string {
  return id.length > 28 ? `${id.slice(0, 12)}…${id.slice(-10)}` : id;
}
