/**
 * Saved times as people read them, in their own locale and time zone. A library's
 * "Updated" column shows the date ("Oct 2, 2026"); a job or worker detail adds the clock
 * time ("Oct 2, 2026, 1:15 PM"). A missing or unreadable time shows as a dash.
 */
const parsed = (value?: string | null) => {
  if (!value) return null;
  const time = new Date(value);
  return Number.isNaN(time.getTime()) ? null : time;
};

export function formatDate(value?: string | null): string {
  return parsed(value)?.toLocaleDateString(undefined, { dateStyle: 'medium' }) ?? '—';
}

export function formatDateTime(value?: string | null): string {
  return parsed(value)?.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) ?? '—';
}
