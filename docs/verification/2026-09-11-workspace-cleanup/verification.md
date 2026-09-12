# HistoPilot workspace cleanup verification

Final frontend verification: TypeScript `tsc --noEmit` passed; Vitest **35 files / 236 tests passed** (1.93 seconds). Includes retained-work direct routes with archived prerequisites absent, recovery-only routes for trashed projects, review/dependency guards, lifecycle API payloads, stable network retry requests, polling and project filters.

Browser verification used an offline `file://` page bundling the actual WorkspaceCleanup and Start React components and application CSS, with in-memory lifecycle and project APIs. Global fetch was intercepted; **zero fetch calls** occurred. No server was started, and no real project records, jobs or files were changed. Browser runtime error list was empty.

Passed browser interactions:
- Bulk selection excludes the whole-project row. Selecting the project clears child selections, and selecting a child clears the project.
- Deleting a dataset reports its dependent training batch; the batch stays unselected until the explicit Include required records action triggers a new preview.
- Confirmation remains disabled until the reviewed-file-retention acknowledgement is checked.
- Simulated committed change with lost response offers retry; both attempts use the identical action, keys, preview hash and operation ID, while the mock revision increments once.
- Active job cancellation is polled through its terminal status. Cancellation disappears once the job stops; notice does not continue to claim an active wait.
- Final cancellation fix: a job resumed before the previous terminal status was observed receives a different operation ID on its next cancellation after an acknowledged response.
- Changed inventory revision removes the confirmation checkbox/action; backend stale-preview rejection clears the review and acknowledgement and displays the error.
- Changing action clears the previous preview.
- Restore uses the same explicit preview and acknowledgement and returns the selected record to Active.
- Start defaults to active projects. Archived and Trash filters display only the chosen state. Archived cleanup links and the trashed-project main action navigate to cleanup without a project-open request.
- Start and cleanup have no horizontal overflow at a 600 x 900 viewport. Desktop dependency review and mobile screenshots visually inspected.

Artifacts:
- `dependency-review.png`: desktop explicit dependency review.
- `start-trash-mobile.png`: narrow Start project Trash filter.
- `cleanup-mobile.png`: narrow cleanup view after stale-preview rejection.
- `harness.tsx`, `build.mjs`, `index.html`: reproducible isolated harness. The Start screenshot has an absent favicon because root-relative public assets are not served under file://; this is a harness limitation.

Browser checks exercise mocked service contracts; actual endpoint, dependency, storage, cancellation and concurrency behavior are covered by the separately reported Python tests. No live-server end-to-end request was attempted under the user server-start preference.
