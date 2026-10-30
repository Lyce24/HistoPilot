# Design decisions

HistoPilot turns pathology slides into trained and applied MIL models on one machine, for one person, with an audit trail strong enough to put in a methods section. Four decisions shape almost everything else. This page records each one: the problem it answers, what was chosen, what it costs, and where to find it in the code. The last section explains how HistoPilot was built and how its behaviour is checked.

For the system as a whole, read [architecture](architecture.md) first.

## 1. Every change is previewed, then confirmed against that preview

**Problem.** The actions that matter in a study are hard to take back. Freezing a design fixes the folds and the feature bundle every later number depends on; starting an experiment spends GPU hours; applying a model to a held-out cohort can only be done blind once. A form that changes things on submit gives no chance to check the consequences, and a check made at review time can be stale by the time the change runs.

**Decision.** Every command that changes state comes in two steps. The **preview** computes what would happen (the folds, coverage of every training slide by the features, overlap between development and testing patients, which tasks would be queued) and returns it with a `previewHash`. The **commit** carries that hash, and the service recomputes the preview and refuses with `PREVIEW_STALE` (409) if anything changed in between. The browser, the CLI and the agent tools all use the same pair. In the CLI, `--yes` only answers the confirmation prompt: blocking findings still stop the command, and `--preview-hash` makes the commit run only if the preview is still the one that was reviewed ([CLI contract](cli-contract.md)).

The same mechanism is what makes AI agents safe to connect. A scoped agent token can preview and save drafts; with the `commit` scope its change still does not run but is parked as a request (`202 CONFIRMATION_PENDING`). A person approves it, and the service replays it under that person's session, rechecking the preview, so a request for something that has since changed is refused. An agent cannot approve its own requests ([AI agents](agents.md#approvals)).

**Cost.** Two round trips for every change, and preview code that must stay exactly in step with commit code. Reviews also do real work (coverage checks read every bundle's index), which is why they are cached by file identity where the inputs cannot change unseen.

**Where.** `histopilot/application/*` (each service's `preview_*` and commit pair), `api/access.py` (parked requests and approvals), `client/` (the shared CLI and agent client), `docs/cli-contract.md`.

## 2. Scientific records are frozen, content-addressed versions

**Problem.** A reported AUROC is only meaningful if you can say exactly which slides, labels, features, folds, code and environment produced it. Mutable records drift: a relabelled patient or a re-extracted slide silently changes what an old experiment "used".

**Decision.** Each stage freezes an immutable record that the next stage consumes by ID: dataset, target-and-split, feature bundle, experiment design, cohort, reference standard. IDs are content hashes over a canonical encoding (`storage/io.py`), so the same inputs always give the same ID and different inputs never share one. Names, tags and notes are mutable labels stored beside the frozen content, outside its hash. Changing anything means making a new version; old records stay readable forever and are never migrated in place.

Code is pinned the same way. A submitted experiment runs from an archived copy of the modules that affect results, with each file's hash and a fingerprint of the runtime (interpreter, CUDA, package versions). Updating HistoPilot never changes the code of submitted work; follow-up work refuses with `EXPERIMENT_RUNTIME_CHANGED` if the environment moved ([pinned compute archives](architecture.md#pinned-compute-archives)).

**Cost.** More records and more storage; a corrected label means a new dataset version and new experiments rather than an edit. Hashes stored in projects can never change their encoding, so three canonical encodings coexist and are pinned by tests. Moving a worker module would strand queued work, so stored names are pinned too (`tests/test_persisted_contracts.py`).

**Where.** `histopilot/storage/` (stores, staged publication, `io.py`), `workers/compute_archive.py`, `workers/training_process.py:compute_snapshot`, `docs/architecture.md#scientific-records`.

## 3. One queue per machine runs all compute

**Problem.** Training folds, refits, predictor runs, attention maps, feature extraction and packing all compete for the same GPU, RAM and CPU threads. Earlier versions gave each batch its own scheduler in its own terminal session; two projects could start work at once, oversubscribe the GPU, and leave nothing that knew what was running after a reboot.

**Decision.** The Task Center is one machine-level queue per operating-system user, shared by every project and workspace. Stage pages own the science and enqueue tasks; a single runner owns operations. It admits tasks in queue order against GPU slots, GPU memory, RAM and CPU threads, starts each in its own process group through a small wrapper, and records outcomes from result and exit files, never from an exit code alone. Work survives restarts: interrupted tasks are recovered and resumed from their checkpoints, a worker that finds its project busy exits with 75 and is requeued, and hold, cancel, retry and history act on whole owners such as an experiment ([Task Center](task-center.md)).

**Cost.** The runner is a single process that must stay healthy, and it serves one machine: there is no cluster scheduler behind it. Resource estimates are advisory, so admission is conservative by default, which on a small machine means less parallelism than the hardware might allow.

**Where.** `histopilot/taskcenter/` (store, runner, wrapper, capacity, adapters), `docs/task-center.md`.

## 4. SQLite and plain files instead of a database server

**Problem.** HistoPilot is a local-first, single-user application for people who should not have to run infrastructure. A project must be self-contained: backed up, moved to another disk or archived for a paper as one folder.

**Decision.** Each project is a folder with its own SQLite store (drafts, frozen configurations, labels, receipts) next to its artifacts on disk; the central workspace and the Task Center keep their own SQLite files. Artifacts are published through a journal: staged files with a checksum manifest are synced and atomically renamed into place, then committed, so a crash leaves either a complete record or a marked interruption, never a partial one. All databases use WAL; one writer lock per project serializes scientific writes across threads and processes, and a busy project fails fast with `PROJECT_BUSY`.

**Cost.** One writer per project and one machine per project. Storage needs POSIX file locks, directory fsync and same-filesystem rename, so network filesystems are not certified. SQLite's own edge cases become HistoPilot's: version 3.51.0 can deadlock when one thread closes the last connection to a WAL database while another opens one, so every connection disables the close-time checkpoint and opens and closes under one process-wide lock ([concurrency](architecture.md#concurrency)).

**Where.** `histopilot/storage/` (`sqlite_connections.py`, `read_sessions.py`, locks, staged publication), `docs/architecture.md#persistence`.

## How HistoPilot was built

I built HistoPilot with Claude Code, Anthropic's coding agent, as a pair programmer. I set the requirements and the decisions on this page, decided what a correct result is for each workflow, reviewed the changes and directed the fixes; the agent wrote much of the code under that direction. The public history is condensed (about 90 commits over five weeks, some of them very large), so it shows where the project arrived rather than every step.

What makes the code trustworthy is how it is checked, not who typed it:

- **CI on every push** (`.github/workflows/checks.yml`): Ruff, the full backend suite (over 3,000 tests) on the CI's Python and core count, the MCP agent tests in their own environment, about 1,000 frontend unit tests, the production web build, and a wheel build that verifies the packaged UI and documentation.
- **Reference parity.** CLAM, TransMIL, DSMIL, DTFD-MIL and ACMIL are checked against outputs recorded from the authors' own implementations with strict parameter-by-parameter loading (`tests/test_mil_parity.py`), and the plain-PyTorch OceanPath Mamba-2 MIL against OceanPath's own model (`tests/test_oceanpath_parity.py`).
- **Stored-contract tests.** Adapter names, worker entry points, archive protocol markers and the three canonical hash encodings are pinned, so a refactor cannot strand saved work (`tests/test_persisted_contracts.py`, `tests/test_storage_io.py`).
- **Agent evaluations.** Scripted sessions check that the agent tools stay inside their token's scope (`tests/test_agent_evals.py`), and real models are run through the MCP server on a fixed task suite that includes requests to change things without approval ([agent evaluation](../evals/agents/README.md)).
- **Independent recomputation.** Numbers HistoPilot reports on real studies have been checked against independent recomputations from its exported predictions.
