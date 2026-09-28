# Task Center: compute verdict and redesign

Status: Phases 0–3 implemented on `pipeline-redesign` (uncommitted), including the
2026-09-27 hardening round ([`PIPELINE_HARDENING.md`](PIPELINE_HARDENING.md)); Phase 4 is
not started. Sections 1–7 are the approved proposal from 2026-09-26. Their verdict (§2)
describes the tree *before* the Task Center, and their line references point at that tree;
where a proposal statement no longer matches the code, a short "As built" note says so.
Section 8 describes what was built and is the reference for current behaviour.

## 1. Summary

HistoPilot runs each kind of compute through a separate launcher, set of state files and
set of cancel and resume rules. The kinds are feature extraction, packing, MIL fold runs,
refits, evaluations, interpretation and study archives.

Parallelism is chosen per training batch and frozen into the scientific setup. Schedulers
enforce it by racing each other for a lease registry kept as files in `/tmp`. The
parallelism suggestion throws away every measurement whenever the code changes, then falls
back to a hard-coded one run per GPU.

This design replaces that with one **Task Center** per machine:

1. **One task model and state machine** for all compute.
2. **One long-lived runner process** that owns the queue, capacity, admission, cancellation
   and recovery.
3. **One capacity setting**, for example "5 parallel GPU tasks". It comes with a measured,
   explained suggestion and can be changed at any time without refreezing anything.
4. **Experiments produce tasks.** Compute settings leave the frozen setup.
5. **One screen** lists planned, queued, running and finished work across all projects.

The resulting flow is:

- **Experimental Setup** decides *what* to compute and freezes it.
- **Experiment Management** submits experiments and follows their progress and results.
- **Task Center** decides *when and where* each task runs.

## 2. Verdict

This verdict describes the application on 2026-09-26, before the Task Center. Most of the
problems below are fixed; §8 describes the current behaviour.

### 2.1 Measured on this machine

The machine has an RTX A5000 (24 GB), 36 logical CPUs and 188 GB RAM. The figures below
come from two completed training batches: ABMIL with 5 runs and nnMIL with 15 runs, both at
4 concurrent runs.

| Quantity | Measured | What HistoPilot assumes |
|---|---|---|
| GPU memory per run | 1.2–1.75 GiB torch reserved peak | 1.9 GiB estimate, then 1 run per GPU regardless |
| Host RAM per run | About 4–6 GiB private. The 15.9 GB fp16 pack is memory-mapped and shared by every run. | 8 GiB reserved. The UI shows 18–21.5 GiB RSS per run because RSS counts the shared pack in every process. |
| CPU | About 9 cores busy with 4 runs (25% host CPU) | 6 slots reserved per run, 24 for 4 runs |
| Speed vs concurrency | nnMIL: 16–18 s/epoch at 4 runs, 16 s at ~2.75. ABMIL: 24–25 s at 4 runs, 29 s when mostly alone. | No throughput model |
| GPU utilisation | 52% mean and 88% peak at 4 runs | Not used |

Four concurrent runs cost each run at most ~10% of its speed, so throughput is close to 4×.
The GPU is not saturated at 4, so 5 is the next setting to try. Nothing on this machine
justifies 1.

### 2.2 The parallelism suggestion

The suggestion logic lives in `runtime_advisor.py`, `runtime_evidence.py` and
`runtime_workload.py`. It says 1 for these reasons:

1. **Any code change discards all history.** Each past batch must match the current
   snapshot of 40+ source files (`runtime_evidence.py:111`). A change to `train_batch.py`
   at 17:22 that does not affect memory invalidated all 20 runs.
2. **The workload key is too strict.** It includes the protocol hash, the bundle hash, the
   class count and recipe fields that do not affect memory (`runtime_workload.py:11-61`).
   A 3-class target can therefore never reuse 4-class measurements.
3. **Sizes are compared incorrectly.** The largest evaluation bag across *all* folds of the
   request is compared with a *single* fold's measurement. As a result 12 of 15 identical
   runs fail the 1.25× rule.
4. **Caps are hard-coded.** The limit is 1 run per GPU with no measurement, 3 with a
   measurement and 4 with an assessment-stage measurement (`runtime_advisor.py:295`). This
   applies even when free VRAM allows about 10.
5. **The RAM and CPU models are inaccurate.** RAM has an 8 GiB floor and does not model the
   shared memory-mapped pack. The CPU slot formula (threads + 2 × workers) over-reserves,
   and the physical core count is never filled in.
6. **The suggestion describes one batch at one moment.** It is computed from the free VRAM,
   free RAM and live leases at that instant, then frozen into a setup that may run hours
   later next to other work.
7. **The inputs are fragile.** Any packing or extraction lease makes lease reading return
   "unknown". Run plans of about 2.2 MB each silently exhaust the evidence scan budget
   after 3–4 batches.
8. **The result is opaque.** The UI shows "additional runs permitted" but never says which
   limit applied (CPU 5, RAM 21, VRAM 10, cap 1).

### 2.3 Scheduling and execution

- **There is no global queue.** Every launched batch runs its own tmux scheduler, which
  polls the file lease registry every 0.5 s. Compute and preparation jobs poll every 1 s.
  - Across batches and experiments, whichever process polls first gets the free capacity.
  - A queued compute job is a live, idle process; a bulk evaluation can create up to 256
    of them.
- **Parallelism is part of the science configuration.** Resources are inside
  `DevelopmentBatchSpec` and its preview hash. Changing concurrency means copying the setup.
- **One cautious job throttles the whole GPU.** A GPU's cap is the smallest `runsPerGpu`
  among all jobs on it (`train_batch.py:105-109`).
  - A single evaluation, refit or extraction at 1 therefore makes the GPU exclusive.
  - New batches also default to 1.
- **The job machinery is fragmented.** There are 9 job kinds with:
  - 6 near-duplicate tmux executors
  - several status vocabularies (`state` vs `status`, `succeeded` vs `completed`,
    `waiting`, `attention`, `partial`, `not_started`)
  - 4 cancel styles, with grace periods of 30 s, 10 s, 5+5 s or cooperative only
  - 4 resume styles
- **Generic job scaffolding exists but is unused.** This covers `/api/v1/jobs`,
  `JobService`, `JobState` and `LocalProcessExecutor`. (2026-09-27: `LocalProcessExecutor`
  and its module `workers/supervisor.py` were removed; the global `/api/v1/jobs` stub,
  `JobService` and `JobState` remain. See [`PIPELINE_HARDENING.md`](PIPELINE_HARDENING.md) 4.2.)
- **Reservations are incomplete.**
  - The predictor coordinator, archive jobs, and fan-outs that run inside HTTP requests
    (bulk evaluation, visualize) take no lease.
  - VRAM is never accounted for.
  - Extraction keeps its GPU lease through CPU-only validation.
  - Evaluations are pinned to GPU 0.

### 2.4 Lifecycle and robustness

- **Status is recomputed on every read and never saved.** Each read probes tmux and
  `/proc`. Reads take the project lock and delete dead leases as a side effect.
- **A host or WSL restart interrupts everything.** Nothing reconciles or resumes at
  startup, and every batch then needs a manual Resume.
- **Recovery has gaps.**
  - Cancelling a batch and then resuming it loses its automatic predictors.
  - A failed batch leaves its experiment marked Running.
- **Disk and temporary files are not managed.**
  - There is no free-disk check before training.
  - `/tmp/histopilot-packing-1000` holds 38,452 leftover claim and lock files (42 MB).
    Every packing submit scans them.

### 2.5 User experience

- **No single place shows all planned and running work.**
  - Operations is per project and only offers Cancel.
  - The Compute jobs pill counts training runs only.
  - Two workspaces on this machine (`~/.histopilot`, `~/.histopilot_inference`) share one
    GPU without seeing each other's queue.
- **Results arrive late.** An experiment's Results stay closed until every batch *and*
  every predictor has finished.
- **The queue cannot be controlled.** There is no pause, no reorder, no ETA and no waiting
  reason ("waiting for GPU slot 5/5").

## 3. Goals and non-goals

Goals:

1. **One queue.** One queue and one capacity model per machine, across workspaces and
   projects.
2. **Capacity is operational.** It is set once, can change at any time and is never frozen
   into science records.
3. **Fill in order.** Queued work fills free slots in submission order across experiments,
   with hold and reorder.
4. **Graceful cancel with immediate refill.** Cancel works at task, batch and experiment
   level. Stopped work is checkpointed and freed slots are refilled at once.
5. **Durable state.** State is saved as it changes and survives server restarts and
   reboots, with automatic resume.
6. **Better suggestions.** Parallelism suggestions come from measurements, are explained,
   and cover every job kind.
7. **Science unchanged.** Frozen hashes, pinned worker code and results stay exactly as
   they are.

Non-goals:

- Multi-machine or cluster scheduling.
- Preempting running tasks by priority, other than an explicit pause.
- Any change to training or evaluation code paths that affects results.

## 4. Design

### 4.1 Layers

```
Experimental Setup ──freeze──► frozen setup (recipes, folds, seeds; no compute settings)
        │
Experiment Management ──submit──► task graph ─┐
Features / Evaluate / Interpret / Archive ────┤ enqueue (one transaction)
                                              ▼
                          Task store (machine-level SQLite, WAL)
                                              ▲ intents        │ state, events
                          API server ─────────┘                ▼
                                                     Runner (one per OS user)
                                   capacity model · admission · dispatch · reconcile
                                                               │ spawn (own process group)
                        workers: fold run · finalize · refit · evaluation · interpretation
                                 extraction · validation · packing · archive
```

The runner knows nothing about MIL; it runs commands. Producers such as experiments,
extractions and evaluations turn scientific records into tasks, and they learn outcomes
through dependencies rather than callbacks.

### 4.2 Task model

| Field | Meaning |
|---|---|
| `id`, `kind` | e.g. `fold-run`, `batch-finalize`, `predictor-publish`, `refit`, `evaluation`, `interpretation`, `extraction`, `extraction-validation`, `packing`, `archive` |
| owner chain | workspace → project → experiment → batch → group, or project → extraction, and so on. Used for grouping, bulk actions and progress. |
| `queueKey`, `priorityClass` | position in the global order; `interactive` or `normal` |
| `dependsOn` | tasks that must succeed first, e.g. finalize depends on all folds of a group |
| `request` | device class (GPU/CPU), estimated VRAM, private RAM, CPU cores, shared files (for example the pack), exclusive flag |
| `command` | interpreter, argv, cwd (the pinned archive for training and compute), env, log/progress/result paths |
| `state`, `attempt` | see below; a retry adds an attempt to the same task |
| process identity | pid, start ticks, boot id, process group |
| `progress` | phase, completed/total, epoch/maxEpochs, message, updatedAt |
| `exit` | reason: `ok`, `oom`, `cuda_failure`, `error`, `cancelled`, `interrupted`, `stalled` |
| measurements | peak VRAM, peak private RAM, shared RAM, mean CPU cores, seconds per epoch, co-tenants over time |

States:

```
blocked ──deps ok──► queued ──admit──► starting ──► running ──► succeeded
   │                  │  ▲                              │──► failed
   │                  │  └── retry / resume (attempt+1) │──► cancelled   (via stopping)
   │                  └──► held (owner paused)          └──► interrupted (process lost)
   └──dependency failed/cancelled──► cancelled
```

Every kind uses this one vocabulary. Kind-specific detail belongs in `progress` and
`exit`, not in new states.

### 4.3 The runner

- **Process.**
  - One runner per OS user per machine: `histopilot runner`, in tmux session `hp-runner`,
    kept single by a file lock.
  - `histopilot serve` and the first submission start it automatically. The CLI offers
    `histopilot runner status|stop`.
  - It is not a systemd service, because systemd is unreliable on WSL.
- **Store.**
  - `$XDG_STATE_HOME/histopilot/task-center.sqlite` in WAL mode.
  - It is machine-level because every workspace shares the same GPU.
  - Per-project folders stay the source of scientific results. The task store is the
    source of execution state.
- **Loop, every 1 s:**
  1. Apply intents: enqueue, cancel, hold, move, retry, capacity change.
  2. Reconcile running tasks by process identity. Read their progress and result files.
  3. Promote `blocked` tasks whose dependencies have succeeded.
  4. Admit tasks in order while capacity fits.
  5. Sample telemetry every 10 s for all kinds alike: VRAM through NVML or torch peaks,
     private RAM through `/proc/<pid>/smaps_rollup`, and CPU time.
  6. Write a heartbeat.
- **Tasks outlive the runner.** Each task runs in its own process group, so it survives a
  runner restart. The runner re-adopts tasks by pid + start ticks + boot id.
- **After a reboot:**
  1. Tasks recorded as running under an older boot id become `interrupted`.
  2. They are re-queued at their original position (auto-resume is on by default).
  3. They resume from `last.ckpt` where the worker supports it.
- **Lease interop.** Existing pinned workers apply the file-lease rules.
  - The runner writes a lease file for every task it starts.
  - It counts foreign leases as used capacity. These come from legacy batch schedulers and
    other workspaces during migration.
  - So old and new schedulers never over-commit each other.

### 4.4 Scheduling policy

- **Order:**
  1. Priority class first.
  2. Then queue position: the owner's submission order, then plan order inside the owner
     (configuration → training seed → split seed → fold).
  3. Owners can be moved to the top, up, down or to the bottom.
- **Fill.**
  - When capacity frees, the runner admits the first queued tasks that fit.
  - A task that does not fit holds back later tasks that need the same resource, which
    prevents starvation.
  - Tasks on other lanes, such as CPU-only work, may run ahead of it.
- **Interactive class.** Short tasks a user is waiting on go ahead of training in the
  queue, but never preempt running work. Examples:
  - a single evaluation
  - interpretation for the viewer
  - batch finalize
  - predictor publish
- **Hold (pause).**
  - A held owner's queued tasks are skipped.
  - "Hold and stop running" also stops its running tasks gracefully and re-queues them at
    the front of that owner. They resume from checkpoints when the owner is released.
- **Cancel.**
  - Queued and blocked tasks are cancelled at once.
  - Running tasks receive SIGTERM to the process group. After a grace period (30 s for
    training, 10 s for extraction) they receive SIGKILL. The state then becomes
    `cancelled`, with a `forced` flag if SIGKILL was needed.
  - Dependents are cancelled too.
  - Freed capacity is refilled on the next tick.
- **Retry.**
  - `failed`, `cancelled` and `interrupted` tasks can be retried as a new attempt at the
    same queue position.
  - The retry resumes from a checkpoint where the worker supports it.
  - Each owner has a "Retry failed" action.
- **OOM backoff.** A task that exits with out-of-memory is re-queued once with its VRAM
  estimate ×1.5. A second OOM fails it.
- **Guards.**
  - The GPU lane pauses on GPU or driver failure; the CPU lane continues.
  - All admission pauses when the output volume is short of disk space.
  - A task with no progress for 30 minutes is flagged `stalled`.

  As built: a device failure fences only that GPU (§8.2); there is no general disk guard,
  only an extraction dispatch check against its preview's size estimate (§8.4).

**Worked example.** Capacity is 5 GPU tasks. E1 (25 runs), E2 (25) and E3 (45) are
submitted in that order.

1. Five E1 runs start. E1 keeps all five slots until its queue is empty.
2. When only E1's last run is still going, the next four queued tasks belong to E2, so
   1 E1 run and 4 E2 runs run together.
3. When an E1 group's folds finish, its finalize and predictor tasks run next as
   interactive tasks, and that group's results appear straight away.
4. The user cancels E2. Its queued runs are cancelled at once. Its four running runs get
   SIGTERM, save `last.ckpt`, exit and become `cancelled`. Within about a second of each
   exit, an E3 run takes the slot.
5. WSL restarts while E3 is running. The runner starts with the server, marks E3's running
   tasks `interrupted`, re-queues them first and resumes them from checkpoints.

### 4.5 Capacity model and the suggestion

1. **Measurement store.** One machine-level table holds a record for each attempt:
   - kind and workload class
   - GPU model and UUID
   - peak VRAM (torch reserved)
   - private RAM (PSS of the process tree)
   - shared file-backed memory per file
   - mean CPU cores
   - seconds per epoch. This needs a timestamp on each history row, which new pinned code
     would add.
   - co-tenants on the GPU over time
   - exit reason

   The runner collects these the same way for every kind.
2. **Workload class key.** The key is: kind, model, precision, batch size, eval batch size,
   feature dimension, embed and attention dimensions, layer count, loading policy and
   gradient checkpointing.
   - It leaves out protocol, bundle, class count, loss and sampler, which do not change
     memory.
   - Size differences are handled by scaling each fold on its own training bag cap and
     evaluation patch count.
   - There is no code-snapshot gate. A deliberate `estimatorVersion` replaces it.
3. **Per-task estimates.**
   - Use the nearest measured class: exact match first, then the same model, then the same
     kind. Scale it by bag size.
   - Fall back to a formula when nothing is measured.
   - Apply the safety margin once: ×1.15 + 0.3 GiB.
4. **Limits.** Each is computed and shown separately:
   - VRAM = (total − reserve) / estimate
   - RAM = (available − reserve − shared files counted once) / private estimate
   - CPU = (cores − reserve) / measured cores per task
5. **Throughput.**
   - For each workload class on this GPU model, record aggregate epochs per second against
     concurrency.
   - Suggest the smallest concurrency within 5% of the best observed.
   - If throughput is still rising at the highest level observed, suggest one more and
     label it "try 5".
   - With no measurements, the default is the smaller of the resource limits and 4.
6. **Output.**
   - One number per GPU, "parallel GPU tasks".
   - A breakdown that marks the binding limit and names its evidence (n runs, dates).
   - For this machine: VRAM 10, RAM 25, CPU 14, and throughput still rising at 4, so the
     suggestion is **5**.
7. **Auto-tune (later).** The runner can apply the suggestion between tasks and back off on
   OOM or slowdown.

The same estimator serves the Task Center (global) and, until Phase 1 ships, the existing
per-batch panel.

### 4.6 Worker contract

**Tasks are opaque commands.** The runner never imports worker code.

- For training and compute, it runs the pinned archive's interpreter and module.
- At dispatch it sets `CUDA_VISIBLE_DEVICES`, the thread variables, and
  `HISTOPILOT_TASK_MANAGED=1`. With that flag, new workers skip their own lease handling.

**Worker obligations:**

1. Write progress JSON at least once a minute.
2. On SIGTERM, save resumable state and write `result.json` with `cancelled`.
3. Always write `result.json` with status, exit reason and measurements.

**Adapters:**

| Work | Entrypoint | Change needed |
|---|---|---|
| Fold run | `train_batch.py --fold <run plan>` | Exists today. The runner writes the device into the environment instead of the run plan. |
| Batch finalize (OOF, selection, results) | `train_batch.py --collect` | New mode in the pinned module. Runs per group as soon as its folds finish. |
| Refit, evaluation, interpretation | `compute_job.py` | Skip self-leasing when managed. |
| Predictor publish (replaces the coordinator) | small API-level worker | Converts coordinator items into tasks. |
| Extraction | TRIDENT, then `verify_extraction` | Split into two tasks so validation releases the GPU. |
| Packing | `pack_features.py` | Hard-kill after grace. |
| Archive | `portability.py` | Report progress through the contract. |

### 4.7 Experiment Management

- **New setups no longer carry compute settings.** Existing frozen setups keep their
  hashes; their `resources` are read but ignored. As built: under the Task Center a launch
  never reads the frozen `resources` (`training._prepare` uses this machine's CUDA runtime
  and the Task Center `defaults` for threads and loader workers); only resumes of legacy
  tmux batches still use them.
  - Before removing `dataLoaderWorkers` from the recipe, verify that augmentation random
    streams (instance dropout, feature noise) do not depend on the worker count. If they
    do, keep that one field as a recipe setting.
- **Submit.** Submitting enqueues the whole task graph in one transaction, at the end of
  the queue:
  - fold runs
  - a finalize per group
  - predictor tasks
- **Experiment page.** It shows:
  - progress per batch
  - results per group as soon as that group is finalized
  - ETA from the estimator
  - actions: Hold, Resume, Cancel, Retry failed, Move up or down, View in Task Center
- **Planned work is visible.** Frozen setups that are not yet submitted appear in the
  Task Center under *Planned*, with run counts and estimated GPU-hours.

### 4.8 Task Center UI

- The Task Center is a workspace-level page. It replaces "Jobs & study backups", and the
  floating Compute jobs pill becomes its summary, e.g. "4 running · 41 queued · ~2 h 10 m".
- **Header.**
  - Capacity control: "Parallel GPU tasks: 5 (suggested 5) · Apply".
  - Meters for GPU slots, VRAM, RAM and CPU.
  - Queue state (running or paused) and time until the queue is empty.
  - An expandable "Why 5?" breakdown.
- **Running.** One row per task showing:
  - project · experiment · batch · configuration · seed · fold
  - progress (epoch 23/100; early stopping may end it sooner), elapsed time, VRAM, RAM
  - Cancel and Log actions
- **Queue.**
  - Grouped by owner, with counts per state and a waiting reason on the first task in
    line ("waiting: GPU slots 5/5").
  - Hold, Cancel, Move, and expand to individual tasks.
- **Planned.** Frozen, unsubmitted setups with a Submit action.
- **History.**
  - Succeeded, failed, cancelled and interrupted tasks, with duration, exit reason, Retry
    and Log.
  - Filterable by project, kind and owner.
- **Links.** Every science page links to its tasks, and every task links back to its
  science record.

As built (§8.5): *Planned* lists the current project's frozen setups with batch and
training-group counts and an **Open in Experiments** link; submission stays in Experiments.
There is no GPU-hours estimate. History is grouped by owner and filtered by result, project
and kind. Running rows open a task detail drawer (log, measurements, attempts) instead of a
separate Log action.

### 4.9 Storage and API

- **Tables:** `tasks`, `task_dependencies`, `task_events` (append-only transitions),
  `intents`, `capacity_settings`, `measurements`, `runner_state`.
- **Who writes what.**
  - The API writes intents and new task graphs.
  - Only the runner writes state transitions. Each transition is a conditional update, for
    example `... WHERE state IN ('queued','held')`.
- **Endpoints**, workspace-level:
  - `GET /task-center/tasks` with filters
  - `GET /task-center/tasks/{id}` with log tail
  - `POST .../{id}/cancel|retry`
  - `POST /task-center/owners/{kind}/{id}/hold|release|cancel|retry|move`
  - `GET|PUT /task-center/capacity`, with the suggestion
- **Live updates:** polling every 2 s at first; server-sent events later.

As built, the API has more read endpoints (summary, snapshot, rollup, history, full task
log); §8.5 lists them. Updates are still polled.

## 5. Migration plan

Each phase ships and is tested on its own. Legacy records stay readable throughout.

| Phase | Size | Scope | Outcome |
|---|---|---|---|
| 0 | S | Fix the suggestion inside today's architecture. Estimator steps 1–4 read today's run files. Remove the code gate and hard caps. Scale per fold. Take RAM from measured private memory. Show every limit and the binding one. Fix the lease reader. | The suggestion on this machine becomes 4–5 and is explained. |
| 1 | L | Task store, runner and Task Center UI. MIL fold runs and finalize for newly submitted experiments. Global capacity. Hold, cancel, reorder and retry. Reconciliation and auto-resume. Lease interop. Compute settings leave new setups. | One queue for training; the worked example in 4.4 works. Legacy batches finish on their own schedulers and show as external load. |
| 2 | M | Refits, evaluations, bulk evaluations, interpretations and predictor publishing become tasks. Retire the coordinator and the fan-outs that run inside HTTP requests. | Per-group results appear early. Predictors survive cancel and resume. |
| 3 | M | Extraction (with a separate validation task), packing and archives become tasks. Unified logs and progress. Housekeeping of old claims and leases. | Every job kind is in the Task Center. |
| 4 | S–M | Throughput auto-tune, notifications, multi-GPU placement, per-project limits. | Hands-off tuning. |

## 6. Risks

| Risk | Mitigation |
|---|---|
| The runner becomes a single point of failure | State lives in SQLite. Tasks run in their own process groups and survive a runner crash. The API restarts the runner and it re-adopts tasks. |
| SQLite contention between the API and the runner | WAL, short transactions, runner-only transitions, conditional updates. |
| Old pinned workers do not know the new rules | Opaque commands plus lease interop. Legacy schedulers keep running until their batches finish. |
| Reproducibility | Science hashes are unchanged. Effective threads, workers and device are recorded in every run plan. Verify the worker-count point in 4.7. |
| OOM from over-subscription | Measured estimates, a single margin, a VRAM budget and OOM backoff. |
| Scope creep | Phase 0 and Phase 1 deliver the requested behaviour for training. Later phases migrate one job kind at a time. |

## 7. Decisions needed

1. **Scope:** one Task Center per machine, covering both workspaces. *Recommended.*
2. **Order:** first come, first served by submission, with manual reorder and hold.
   *Recommended.* The alternative is round-robin across experiments.
3. **Compute settings leave frozen setups** for all new setups. *Recommended.*
4. **Auto-resume after reboot:** on by default. *Recommended.*
5. **Cancel:** graceful, with a 30 s grace period for training. Cancelled runs can be
   retried from their checkpoint. *Recommended.*
6. **Sequencing:** Phase 0 first as a small standalone fix, then Phase 1. *Recommended.*

Decisions 1–5 were approved on 2026-09-26 as recommended; Phases 0–2 were built together.
Phase 3 and the hardening round followed on 2026-09-27.

## 8. As built (Phases 0–3)

### 8.1 Components

| Module | Role |
|---|---|
| `taskcenter/store.py` | SQLite store (WAL): owners, task groups, tasks, dependencies, events, settings, operations, measurements, runner row. Every state change is a guarded transition in one `BEGIN IMMEDIATE` transaction. The runner row also records the code hash, the checkout root and the persisted GPU fences. |
| `taskcenter/runner.py` | The single runner per OS user. Each tick: host sample, lease housekeeping, resolve journaled starts, reconcile exits, bookkeeping retries, promote ready tasks, admit in queue order, telemetry, heartbeat. |
| `taskcenter/procs.py`, `taskcenter/wrap.py` | Spawn each task in its own session through a small stdlib-only wrapper that journals the start and the exit (§8.2); identity by pid + start ticks + boot id; the CUDA-init probe used by GPU fences. |
| `taskcenter/capacity.py` | Admission: GPU slots, committed and pending VRAM, RAM (own and foreign reservations), CPU threads, exclusivity. Work that fits no GPU runs alone only on a GPU that is actually empty. |
| `taskcenter/estimator.py` | Per-task VRAM/RAM/CPU estimates from run history and measurements, and the parallelism suggestion with every limit and the binding one. |
| `taskcenter/leases.py` | Interop with the legacy lease registry (`$TMPDIR/histopilot-training-<uid>`), under the same writer lock legacy schedulers use. |
| `taskcenter/adapters/` | Per-kind hooks: `mil-fold`/`mil-collect`, `compute-job` (refit, evaluation, interpretation), `predictor-coordinator`, `bulk-submit`/`generic`, and the Phase 3 kinds `extraction`, `extraction-validation`, `packing`, `archive`. `base.py` holds the busy contract. |
| `taskcenter/client.py`, `application/task_records.py` | Import-light facade that application services (and pinned coordinator copies) use to submit and steer tasks; shared plumbing for the extraction, packing and archive records. |
| `taskcenter/service.py`, `api/task_center.py` | Read models and actions for the web page and the stage chips (`/api/v1/task-center/...`, §8.5). |
| `taskcenter/launcher.py`, `runner_cli.py` | Start/stop the runner in tmux session `hp-runner-<uid>`; `histopilot runner` `run`, `start`, `stop`, `status`. |
| `workers/managed_fold.py`, `managed_collect.py` | Pinned workers for one fold and for result collection; `compute_job.py`, the predictor coordinator, the TRIDENT runner, `pack_features.py` and `portability.py` gained a managed mode. |

State lives in `$HISTOPILOT_STATE_DIR`, else `$XDG_STATE_HOME/histopilot`, else
`~/.local/state/histopilot` (`task-center.sqlite`, `runner.lock`, `runner.log`, and
`exits/` for the wrapper journals).

### 8.2 Behaviour

- **Queue order.** Interactive tasks (single evaluations and interpretations, result
  collections, bulk submission) first, then owners in submission order (`queue_seq`), then
  group and plan order. Hold, release, move (up/down/top/bottom), stop & hold, cancel and
  retry act on an owner; cancel and retry also act on one task. Tasks that share an
  `exclusiveKey` never run at the same time (one batch's collections, one extraction
  output, one packing feature source, one project's archive jobs).
- **Parallelism.** GPU task slots per GPU (default 4, set in the Task Center), bounded by
  VRAM, RAM and CPU. On this machine the suggestion is 5, limited by measured throughput.
  A task that fits no GPU may run alone on a GPU without HistoPilot tasks only when that
  GPU is really empty (used memory at most its reserve + 0.5 GiB). Otherwise it waits with
  "Waiting for GPU memory: N GiB of GPU i is in use outside HistoPilot".
- **Starts are journaled.** `starting` is the crash journal: the runner moves a task
  `queued → starting` before it spawns it and `starting → running` once the process is
  recorded. A runner that finds a task in `starting` adopts it when the wrapper's start
  record names a live process, concludes it when that process already exited, and
  otherwise requeues it without spending an attempt.
- **Exit codes survive runner restarts.** Every task starts through `taskcenter/wrap.py`.
  The task stays the tracked pid and session leader; the wrapper stays outside its session,
  waits for it and writes `exits/<hash>-<attempt>.exit.json` (return code or signal).
  A runner that adopts a task, or starts after tasks ended while none ran, classifies them
  from that record; only a task with no record is `lost`. Journals are deleted when the
  attempt concludes and swept after 7 days.
- **Busy contract.** A managed worker that cannot get its project lock (`PROJECT_BUSY`) or
  output lock (`OUTPUT_BUSY`) within a short wait exits **75** (EX_TEMPFAIL) and leaves its
  record untouched, or (coordinator records) records the error code. Adapters map both to
  `requeue "busy"`; the task waits 5 s, doubling to 60 s, then runs again. It never fails
  for being busy. Waits before giving up: compute pre-check 60 s (the post-compute check
  keeps 15 min so finished work is not thrown away), coordinator 120 s of busy passes, bulk
  submit 60 s. Workers pinned before the contract are recognised by the lock message on the
  last line of their log.
- **Cancel.** Cancel markers (`cancel.json`, `cancel.requested`) are written before any
  signal; SIGTERM, then SIGKILL of the process group after the grace (30 s). Cancelled runs
  are recorded cancelled and resume from their checkpoint. Cancelling an owner releases its
  hold so its collections can record the outcome. A cancel that races a pause wins.
- **Dependencies.** The default condition is `succeeded` (training and coordinator edges use
  `terminal`). A blocked task whose `succeeded` dependency failed or was cancelled is
  cancelled too ("A task it depends on did not succeed"), in cascade. Retrying the failed
  task revives the dependents this rule cancelled; a user's own cancel is kept.
- **Recovery.** Tasks run in their own process groups and survive a runner restart; a new
  runner adopts live processes and concludes lost ones. Requeue decisions (auto-resume,
  OOM backoff) are written in the same transaction as the task's end and retried until
  decided (30 s doubling to 5 min, giving up after 2 h); dependents wait meanwhile.
  Auto-resume checks the fold's own training interpreter. Adapter hooks that fail
  transiently are retried per task (2 s doubling to 60 s), so one stuck hook does not slow
  every tick; the training-runtime probe runs off the runner loop.
- **Out of memory.** An OOM is retried once with 1.5× VRAM, capped at the largest GPU. The
  budget is counted separately from attempts, so pauses and busy requeues do not use it,
  and a manual retry starts a new one. Refit and evaluation OOMs requeue the same way. An
  OOM on a GPU where, right after the exit, at least 1 GiB more is in use than the reserve
  and the other tasks' requests account for (and no legacy lease is on it) is requeued as
  `gpu-contention` instead: the request is not raised and the OOM retry is not spent, at
  most 3 times per run. Admission then waits until that memory is free.
- **Device loss.** A lost device (not a device-side assert or illegal access, which fail only
  the task) ends the task `interrupted` and requeues it as `device-lost`, at most 3 times
  per run. That GPU takes no work for 60 s, doubling per incident up to 30 min, and then
  only after a CUDA-init probe (`procs.cuda_probe`: `libcuda` through ctypes, no Torch)
  succeeds on it. The fence is saved on the runner row, so a restarted runner keeps it.
- **Legacy interop.** Admission holds the legacy registry lock while it re-checks and
  writes its lease, and counts legacy leases (slots, CPU and reserved-but-unused RAM).
  Batches, compute jobs, coordinators, extractions, packing jobs and archives created
  before the change keep their tmux workers.
- **Experiments.** New submissions record `executionMode: task-center`. Each batch is one
  task per fold plus a final collection; the predictor coordinator waits for every batch's
  final collection and then decides per batch. Every ready refit is launched as its own
  compute task and the Task Center admits them (no longer one refit at a time). A
  coordinator that finds the project busy keeps looping; after 120 s it exits 75 and is
  requeued, and items that failed only because the project was busy are reopened, so it
  resumes where it stopped. Launches ignore a frozen setup's legacy `resources`.
  Cancelled Task Center experiments stay resumable, including their predictors.

### 8.3 Operations

- `histopilot serve` starts the runner unless `--no-runner`. It restarts a runner whose
  loaded code changed, or one started from another checkout (the runner row records
  `checkoutRoot`; `histopilot runner status` reports `otherCheckout`, and the Task Center
  says "Started from another checkout"). Submissions from any page wake a stopped runner.
- `HISTOPILOT_EXECUTION_MODE` is no longer read: new work always launches as Task Center
  tasks. Records created before (no `executionMode`) keep their tmux path for status,
  cancel and resume. `HISTOPILOT_TASK_CENTER_AUTOSTART=0` disables automatic runner start
  (tests set this).
- The runner inherits `HISTOPILOT_*`, `PATH` and `LD_LIBRARY_PATH` from the process that
  starts it. Extraction needs a TRIDENT checkout: `HISTOPILOT_TRIDENT_ROOT`, or
  `<checkout>/.local/TRIDENT`; a worktree has neither by default, and the preview and the
  System page then name the roots searched and any sibling checkout found.
- New web code reaches the live server at the next start through `bash serve.sh`. The
  launcher rebuilds and bundles the frontend when `web/` changed (see
  [`deployment.md`](deployment.md) and [`PIPELINE_HARDENING.md`](PIPELINE_HARDENING.md) §6).

### 8.4 Phase 3: extraction, validation, packing and archives

| Task kind | Owner kind | Lane | Depends on | `exclusiveKey` |
|---|---|---|---|---|
| `extraction` | `extraction` | GPU (CPU when every TRIDENT device is −1) | – | `trident-output:<hash of output path>` |
| `extraction-validation` | `extraction` | CPU | `extraction` (succeeded) | – |
| `packing` (validate, attach or pack) | `feature-pack` | CPU | – | `feature-source:<hash of project and feature set>` |
| `archive` (export, verify or restore) | `archive` | CPU | – | `archive-project:<hash of project>` |

- Records keep their own `state` vocabulary, derived from the task store, and gain
  `executor`, `task`, `ownerKey` (extraction also `tasks.extraction`/`tasks.validation`).
- **Extraction** shares the GPU: it requests VRAM (estimated from its options, then
  refined from the measured peak of the same workload × 1.15 + 0.3 GiB) instead of taking
  the whole GPU; legacy preparation leases now publish 4 runs per GPU. Before every attempt
  the TRIDENT runner clears locks left by dead writers and moves their outputs aside to
  `<name>.stale-<epoch>` (nothing is deleted), so TRIDENT redoes those slides. At dispatch
  it waits when free disk falls below the preview's size estimate. The runner writes
  `progress.json` from the log. **Resume extraction** (`POST
  /projects/{id}/extractions/{job}/resume`) requeues the TRIDENT task, which re-arms
  validation; extractions from before the Task Center resume through a new preview on the
  same output.
- **Validation** runs as a separate CPU task, so it no longer holds the GPU.
- **Packing** replaces claim files for new jobs: the `exclusiveKey` serializes jobs of one
  feature source, and outputs are guarded by live packing tasks plus the worker's output
  lock (busy → exit 75 → requeue). A retry moves the earlier receipt aside and removes
  stale `.packing-*` staging folders.
- **Archives**: an export that finds the project busy writes `queued` and exits 75.
  Operations "Retry" and the Task Center requeue the same task.
- **Housekeeping**: packing preview and submit sweep old claims and locks under
  `/tmp/histopilot-packing-<uid>` and orphan staging folders, at most every 6 h and within
  a 5 s budget.
- **Preflight**: extraction preview reports missing slide-reader modules
  (`SLIDE_READER_UNAVAILABLE`), the disk estimate (`INSUFFICIENT_SPACE`,
  `LOW_DISK_SPACE`) and multi-GPU requests (`SINGLE_GPU_TASK`).

### 8.5 Stage pages, UI and API

- **Split of responsibility.** A stage page owns the science: inputs, previews, results and
  one science action (Run, or Resume/Retry when the work needs attention). The Task Center
  owns everything operational: queue order, hold, cancel, retry, logs, resources, attempts
  and history. Stage pages show one `RunStatusChip` over their tasks: Experiments (Runs tab),
  Evaluate models and Run inference batches, refits, Model interpretation, Slide features
  (extraction and packing), Operations (archives), the job tray (machine-wide) and System.
  Records started in tmux before the Task Center keep their old controls.
- **Chip states**: not started · queued · running · held · stopping · needs attention ·
  runner stopped · completed, plus cancelled. Each links to the Task Center.
- **Deep links**: `#task-center?owner=<key>&task=<id>&project=<id>&kind=<kind>&state=<state>`
  (`taskCenterHref` and `readTaskCenterRoute` in `web/src/api/taskCenter.ts`). `task=` opens
  the task detail drawer: plain-language failure ("safe to retry" or not) above the
  traceback, command, working folder, filtered environment, paths, measured resources,
  labels, dependencies, attempts and the full log. Tasks link back to their record.
- **Experiments** reads one execution status from the task store, without the project lock:
  running · queued · held · waiting · needs-attention · cancelled · completed, plus
  `statusReason`. A coordinator that stopped on a busy project reads "Waiting" or "Needs
  attention", never "Failed".
- **Page.** Running, Queue (owners in run order), Capacity (parallel GPU tasks, suggestion
  and its "Why N?" breakdown), *Planned in this project* (frozen setups not yet started,
  linking to Experiments; they are submitted there) and History (grouped by owner, filtered
  by result, project and kind, paged). Everything polls: one snapshot every 2 s while work
  is live, 10 s otherwise; chips poll every 3 s while running, 5 s while waiting and 30 s
  otherwise.
- **Endpoints** under `/api/v1/task-center`:

  | Method / path | Purpose |
  |---|---|
  | `GET /summary` | Runner, capacity, counts and recent failures |
  | `GET /snapshot` | Summary, running tasks and live owners in one read (the page's poll) |
  | `GET /rollup` | A stage's run status. Scope: `owner`, `ownerKind`+`ownerId`, `recordKind`+`recordId` or `recordIds`, `project` (+`kinds`), or none for the machine. Reads only the task store. |
  | `GET /history` | Finished tasks grouped by owner; `project`, `kind`, `state`, `limit`, `offset` |
  | `GET /tasks` | Tasks with filters; `offset` adds paging |
  | `GET /tasks/{id}` | One task with its log tail, attempts and dependents |
  | `GET /tasks/{id}/log` | The whole log, streamed; `download=true` for a file |
  | `POST /tasks/{id}/cancel`, `/retry` | Task actions |
  | `GET /owners` (`scope=live` or `all`), `GET /owners/{key}` | Owners |
  | `POST /owners/{key}/{action}` | Owner actions: `hold`, `release`, `stop`, `cancel`, `retry`, `move` (with `position`) |
  | `GET /capacity`, `PUT /capacity` | Settings and the suggestion |
  | `POST /runner/start`, `/runner/restart` | Runner control |

  Record endpoints added with Phase 3: `POST /projects/{id}/extractions/{job}/resume`, and
  archive `POST .../operations/archives/{job}/retry` now requeues the same task.

### 8.6 Differences from the proposal and open items

- The coordinator was kept as one task per experiment (Phase 2 planned to retire it); it
  waits on final collections rather than on every fold succeeding.
- Phase 4 (throughput auto-tune, notifications, multi-GPU placement, per-project limits)
  is not started. A multi-GPU TRIDENT request runs on one GPU under the Task Center.
- Legacy tmux jobs that are still running appear only as external load (their leases), not
  as tasks.
- Store hygiene is deferred: tasks, events and owners are never pruned, there is no
  `task_events(task_id)` index, and the runner does no WAL checkpoint, backup,
  `runner.log` rotation or general disk guard. Page reads list live owners only and history
  is paged, so the growing store costs less per poll.
- The live `train_batch.run_batch` scheduler is kept for tmux-era batches and their tests;
  remove it once no such batch must stay resumable. `archive_cli` recovery still launches
  tmux directly.
- The estimator (about 1,160 lines) has not been simplified.
- A Task Center Retry of a failed `extraction-validation` task only re-validates the same
  outputs; resuming TRIDENT is the stage's **Resume extraction**.
- Extraction tasks have normal priority, so an extraction submitted after a long training
  queue waits its owner's turn (move the owner up).
- The Operations inventory shows a generic waiting message for queued Task Center
  extraction and packing jobs instead of the task's own waiting reason.
- Refit tasks carry no `model` label, so their titles lack the model name, and the refit
  page still sends one launch request per refit.
