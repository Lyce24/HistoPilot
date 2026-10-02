/** A job or run status as a word for people: "Running", "Completed", never "not_started". */
const labels: Record<string, string> = {
  queued: 'Queued', pending: 'Pending', starting: 'Starting', running: 'Running', stopping: 'Stopping',
  cancelling: 'Cancelling', completed: 'Completed', succeeded: 'Completed', failed: 'Failed',
  cancelled: 'Cancelled', interrupted: 'Interrupted', not_started: 'Ready to run', planned: 'Planned',
  skipped: 'Skipped', partial: 'Partly failed', held: 'Held', waiting: 'Waiting',
};

export const statusLabel = (status?: string | null) =>
  !status ? '' : labels[status] ?? status.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase());
