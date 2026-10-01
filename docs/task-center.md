# Task Center

The Task Center runs all of HistoPilot's long-running compute: training folds and result collection, predictor coordination, refits, predictor runs on cohorts, interpretation, feature extraction and its validation, feature packing, and study archives. There is one queue per operating-system user on a machine, shared by every project and workspace, because they all share the same GPUs.

The work is split in two:

- **Stage pages own the science.** Experiments, Apply models, Slide features and the other pages hold inputs, previews and results, plus one science action (**Run**, or **Resume**/**Retry** when work needs attention). Each shows a status chip that links to its tasks.
- **The Task Center owns operations.** It decides when and where each task runs, and holds queue order, hold, cancel, retry, logs, measured resources, attempts and history.

A single **runner** process owns the queue. It admits tasks against GPU slots, GPU memory, RAM and CPU threads, starts each in its own process group, records outcomes, and resumes interrupted work after a restart. It knows nothing about MIL; it runs commands and reads their result files.

## The page

Open **Task Center** under **Project tools**. It has these panels:

| Panel | Contents |
| --- | --- |
| Running | Running tasks with progress, elapsed time and measured VRAM and RAM. **Details** opens a drawer with the log, command, working folder, filtered environment, measurements, dependencies and attempts. A failed task explains in plain language whether it is safe to retry, above the traceback. |
| Queue | Owners (an experiment, batch or record) in the order they will run, with counts per state and the first task's waiting reason, such as "waiting for GPU memory". Hold, release, move and cancel act on a whole owner. |
| Capacity | **Parallel GPU tasks**, the suggested value and a **Why N?** breakdown, with meters for GPU slots, GPU memory, RAM and CPU. **Pause queue** stops new starts; running tasks continue. |
| Planned in this project | Frozen setups that have not been started. Start them from Experiments; their tasks then join the end of the queue. |
| History | Finished tasks grouped by owner, newest first, filtered by result, project and kind. Failed, cancelled and interrupted work can be retried. |

## Tasks

| Kind | What it runs | Lane | Priority |
| --- | --- | --- | --- |
| `mil-fold` | One training fold (`histopilot.workers.managed_fold`) | GPU, or CPU without a GPU | Normal |
| `mil-collect` | A batch's result collection: OOF assembly, configuration selection and results (`managed_collect`) | CPU | Interactive |
| `predictor-coordinator` | An experiment's predictor work: publishes ensembles and launches refits (`workers.experiment_predictors`) | CPU service task | Normal |
| `compute-job` | A refit, predictor run or interpretation (`workers.compute_job`) | GPU, or CPU | Interactive for a single run or interpretation; normal for refits and bulk members |
| `bulk-submit` | Creates the member runs of a batch in Apply models | CPU | Interactive |
| `extraction` | TRIDENT feature extraction | GPU (CPU when every device is −1) | Normal |
| `extraction-validation` | Checks the extraction's outputs, after it succeeds | CPU | Normal |
| `packing` | Feature validation, pack verification or packing (`workers/pack_features.py`) | CPU | Normal |
| `archive` | Study archive export, verification or restore (`workers/portability.py`) | CPU | Normal |

Each task belongs to an **owner**, such as an experiment, a batch, a batch of predictor runs, an extraction, a feature pack or an archive job. Owners group tasks in the queue and receive bulk actions.

Some tasks share an **exclusive key** and never run at the same time: the collections of one batch, extractions writing one output folder, packing jobs of one feature source, and archive jobs of one project.

### States

| State | Meaning |
| --- | --- |
| `blocked` | Waiting for the tasks it depends on |
| `queued` | Ready to start when capacity allows |
| `starting` | Admitted; the process is being spawned |
| `running` | Its process is alive |
| `stopping` | Cancel or pause requested; waiting for the process to exit |
| `succeeded`, `failed`, `cancelled`, `interrupted` | Finished. `interrupted` means the process was lost, for example to a reboot. |

A retry starts a new attempt of the same task at its original queue position. Every kind uses this one vocabulary; kind-specific detail lives in the task's progress and exit record. Stage pages summarize their tasks as one chip state: not started, queued, running, held, stopping, needs attention, runner stopped, completed or cancelled.

### Dependencies

A task starts only when its dependencies are met. The default condition is that each dependency **succeeded**. A batch's final collection and the predictor coordinator instead wait until their dependencies are **finished**, whatever the outcome, so they can record partial results. When a dependency fails or is cancelled, a blocked task that needed it to succeed is cancelled too ("A task it depends on did not succeed"). Retrying the failed task revives those dependents; a cancel you made yourself is kept.

## Queue order and admission

The runner considers queued tasks in this order:

1. Interactive tasks first: single evaluations and interpretations, result collections and bulk submission. They jump ahead of training but never stop running work.
2. Then owners in submission order. A new owner joins the end of the queue. **Move** reorders owners (top, up, down, bottom).
3. Within an owner, plan order: configuration, then training seed, split seed and fold.

Within a lane, admission is strictly in order. A task that does not fit holds back later tasks competing for the same resource, so large tasks are not starved. CPU-lane work can still start while a GPU task waits. A held owner keeps its place but starts nothing, and exclusive-key waits and busy retries do not hold back other owners.

### Capacity

| Setting | Default | Meaning |
| --- | --- | --- |
| Parallel GPU tasks | 4 per GPU | Task slots on each GPU (1–16), also settable per GPU |
| CPU task slots | Logical CPUs ÷ 4 | CPU-lane tasks at once |
| CPU threads per run, data loader workers | 2 and 2 | Given to training runs when they launch |
| Auto-resume | On | Requeue work interrupted by a reboot or crash |
| Paused | Off | Stop starting new tasks |

A task starts only when all of these hold:

- **CPU threads:** committed threads plus its need fit within the logical CPUs, less a reserve of 2.
- **CPU slot**, for CPU-lane tasks.
- **RAM:** available RAM, less a reserve (5% of RAM, between 2 and 16 GiB) and the requests of tasks started in the last 90 seconds, covers its need.
- **GPU:** the GPU has a free slot, its committed requests fit within its memory less a reserve (5%, at least 1 GiB), and its live free memory covers the request.

GPUs with the fewest used slots and the most free memory are tried first. A task larger than any GPU's budget runs alone, and only on a GPU that is really empty (no HistoPilot tasks, and at most 0.5 GiB in use beyond the reserve). Otherwise it waits with a reason such as "Waiting for GPU memory: N GiB of GPU i is in use outside HistoPilot".

RAM requests are scheduling allowances, not operating-system limits. Schedulers of other checkouts that have not yet migrated, and TRIDENT runs started by hand, publish leases in a shared registry under the temporary directory (`histopilot-training-<uid>`). The runner counts those leases as used capacity and writes a lease for each of its own tasks, so no two of them overcommit each other. Every 5 minutes the runner also removes the leases whose owner process (and supervisor, if it names one) is confirmed dead, so a crashed worker's reservation does not count forever; a lease whose owner is alive, or cannot be verified, or that cannot be read, is kept.

### Estimates and the suggestion

Each training fold's GPU memory, RAM and CPU request comes from measurements of earlier runs of the same workload: model, precision, batch and bag sizes, feature dimension, model dimensions and loading policy. The request scales with the fold's bag size and adds one safety margin (×1.15 + 0.3 GiB for GPU memory). Without measurements a formula is used. Extraction requests are refined the same way from earlier extractions.

The **Parallel GPU tasks** suggestion shows each limit separately: GPU memory, RAM, CPU and measured throughput (the smallest concurrency within 5% of the best observed epochs per second). The smallest limit binds, and **Why N?** names it with its evidence. With no throughput evidence the suggestion is capped at 4. Changing capacity takes effect at the next admission and never changes a frozen setup.

## Cancel, hold and stop

- **Cancel** removes queued and blocked tasks at once. A running task gets SIGTERM to its process, then SIGKILL of its whole process group after a grace period: 30 seconds by default, 10 for extraction validation, 60 for packing. Workers save a resumable checkpoint when they can. Cancelled training folds resume from their last checkpoint. A cancel beats a concurrent pause.
- **Hold** stops an owner's queued tasks from starting. **Release** undoes it.
- **Stop** holds the owner and pauses its running tasks gracefully. They return to the queue, still held, and resume from their checkpoints when released.
- **Retry** requeues a failed, cancelled or interrupted task, or every such task of an owner, as a new attempt.

Cancel and retry from the Task Center are routed through the owning service, so the stage record stays consistent. Freed capacity is refilled on the next tick, about a second later.

## Busy workers (exit 75)

A worker that cannot get its project lock (`PROJECT_BUSY`) or its output lock (`OUTPUT_BUSY`) within a short wait exits with code **75** (EX_TEMPFAIL) and leaves its record untouched. The runner requeues it as busy: it waits 5 seconds, doubling to 60 seconds, and 10 minutes after ten busy exits in a row. From the third try the waiting reason names the cause. A busy task never fails for being busy.

| Worker | Waits before exiting 75 |
| --- | --- |
| Refit, predictor run or interpretation, before computing | 60 s. After computing it waits up to 15 minutes, so finished work is not thrown away. |
| Predictor coordinator | 120 s of busy passes |
| Bulk submission | 60 s |

Result collection, packing and archives honour the same contract. Workers pinned before the contract existed are recognised by the lock message on the last line of their log.

## Automatic requeues

| Situation | What happens |
| --- | --- |
| Reboot or crash | Tasks that were running become `interrupted`. With auto-resume on, they are requeued and resume from their checkpoints when the runner starts again. Training folds first check that their training interpreter still works. |
| Out of GPU memory | Requeued once with 1.5× the GPU memory request, capped at the largest GPU. A second OOM, or a request already at the cap, fails with "needs more GPU memory than this machine has". A manual retry starts a fresh budget. |
| Memory taken by another program | If, right after an OOM, at least 1 GiB more is in use on that GPU than tasks and the reserve explain, the task is requeued as GPU contention without raising its request, at most 3 times. Admission then waits until the memory is free. |
| Lost GPU | A lost device (not a device-side assert, which fails only that task) requeues the task, at most 3 times. The GPU takes no work for 60 seconds, doubling per incident up to 30 minutes, and then only after a CUDA probe succeeds. The fence survives a runner restart. |
| No progress | A running task with no progress for 30 minutes is flagged as stalled. |

Automatic requeues apply only to kinds that can resume safely. Requeue decisions are written together with the task's end and retried (30 seconds doubling to 5 minutes, giving up after 2 hours); dependents wait meanwhile.

## Exit records and journaled starts

Every task starts through a small wrapper, `histopilot/taskcenter/wrap.py`. The task is the tracked process and leads its own session; the wrapper stays outside it, waits for it and writes an exit record (return code or signal) under the state directory's `exits/`. A runner that restarts, or that finds tasks which ended while it was down, classifies them from these records. Only a task without a record is treated as lost. Records are deleted when the attempt concludes; leftovers older than 7 days are swept when the runner starts.

Starts are journaled too. The runner moves a task to `starting` before it spawns the process, and to `running` once the process is recorded. On recovery it adopts a task whose process is alive, concludes one whose process already exited, and otherwise requeues it without spending an attempt. Running tasks are identified by process ID, start time and boot ID, so a reused process ID is never mistaken for a task.

## The runner

`histopilot serve` starts the runner in the tmux session `hp-runner-<uid>`, unless you pass `--no-runner`. A file lock keeps one runner per user. The runner ticks every second, writes a heartbeat every 2 seconds and samples telemetry every 10 seconds. Tasks run in their own process groups, so they survive the browser, SSH sessions, the service and the runner itself. The HistoPilot service is never hosted in tmux; start and stop it yourself.

`histopilot serve` also restarts a runner that is outdated: one whose loaded code differs from the checkout, or one started from another checkout ("Started from another checkout" on the page). The restart waits for the current tick to finish; running tasks are adopted by the new runner. Any submission wakes a stopped runner. Set `HISTOPILOT_TASK_CENTER_AUTOSTART=0` to disable automatic start and restart; the tests do this.

The runner inherits `HISTOPILOT_*` variables, `PATH`, `LD_LIBRARY_PATH` and `TMPDIR` from the process that starts it, so set runtime variables such as `HISTOPILOT_TRIDENT_ROOT` before `histopilot serve`. See [deployment](deployment.md#runtime-environments).

```bash
histopilot runner status          # add --json for alive, code-current, other-checkout, counts
histopilot runner start
histopilot runner stop
histopilot runner run             # run in the foreground
```

Each takes `--state-dir`. The page and `POST /api/v1/task-center/runner/restart` restart the runner.

## Deep links

Every task and owner has a link of the form:

```text
#task-center?owner=<key>&task=<id>&project=<id>&kind=<kind>&state=<state>
```

`task=` opens the task's detail drawer. Stage chips link here, and tasks link back to their record. In the web code, `taskCenterHref` and `readTaskCenterRoute` in `web/src/api/taskCenter.ts` build and parse these links. The page polls a snapshot every 2 seconds while work is live and every 10 seconds otherwise; chips poll every 3 seconds while running, 5 seconds while waiting and 30 seconds otherwise.

## State directory and maintenance

The state directory is `$HISTOPILOT_STATE_DIR`, else `$XDG_STATE_HOME/histopilot`, else `~/.local/state/histopilot`. It must belong to you and must not be writable by others.

| File | Contents |
| --- | --- |
| `task-center.sqlite` (with `-wal` and `-shm`) | Owners, groups, tasks, dependencies, events, settings, operation receipts, measurements and the runner row, in WAL mode |
| `runner.lock` | The single-runner lock |
| `runner.log` | The runner's log |
| `exits/` | Wrapper start and exit records |
| `runner-stacks.log`, `server-<port>-stacks.log` | Thread stacks written by `kill -USR1` on the runner or the service |

Only the runner writes task state; the API writes intents and new tasks. Every transition is a guarded update in one transaction. Every 5 minutes the runner truncates the SQLite write-ahead log (`wal_checkpoint(TRUNCATE)`), and connections cap its size at 8 MiB, so the log stays bounded. Tasks, events and history are never pruned, and `runner.log` is not rotated; the store grows slowly with use. Page reads list live owners and page through history, so a large store costs little per poll.

## Work created before the Task Center

Batches, compute jobs, predictor coordinators, extractions, packing jobs and archives created before the Task Center ran in their own tmux sessions. HistoPilot no longer runs, watches or stops them, and they never appear in the Task Center:

- A finished record shows its saved status and results.
- An unfinished one reads as interrupted, with a note that it was created before the Task Center.
- Launch, resume, retry and cancel are refused with `CREATED_BEFORE_TASK_CENTER` (409). Clone the batch or experiment, or preview the extraction or feature job again, to run it as Task Center tasks.

Work submitted from an archive pinned before the Task Center is refused the same way (see [architecture](architecture.md#pinned-compute-archives)). Only the runner itself still runs in tmux.

## Known limitations

- A multi-GPU TRIDENT request runs on one GPU.
- Extraction has normal priority, so one submitted behind a long training queue waits its turn; move its owner up.
- Retrying a failed extraction-validation task only re-validates the same outputs. To rerun TRIDENT, use **Resume extraction** on the Slide features page.
- Refit tasks carry no model label, so their titles lack the model name.
- Busy-retry counters live in memory and reset when the runner restarts.
