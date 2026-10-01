# CLI contract

Version 1 · 2026-09-30. [Command line](cli.md) lists the commands; [agents](agents.md) covers agent access.

This page fixes what a script or an AI agent can rely on when it runs `histopilot`: command shape, output, exit codes, route classes, confirmation and spec files. The Python client in `histopilot/client/` shares the error kinds, route classes and spec rules; prompts and exit codes belong to the CLI. The CLI reaches a running service over HTTP through Python's standard library, so it adds no runtime dependency; tests plug in an in-process service instead.

## Commands

`histopilot <noun> <verb> [RECORD] [options]`, for example `histopilot experiment list` or `histopilot tasks show TASK --json`. Nouns are record kinds such as `project`, `dataset`, `targets`, `experiment`, `predictor`, `cohort`, `run` and `tasks`; `histopilot <noun> --help` lists their verbs. Outside that grid, `status` summarizes the service, `schema` prints a request model's JSON Schema, `use` saves a default project for the current `--url`, and `api` sends a raw request and returns the service's response as `data`, outside the stability promise below. `token`, `confirm`, `login` and `agent` manage agent access.

- **Naming records.** Use the ID, or `@tag` for a record with a version tag: datasets and frozen configurations such as Targets & splits, feature sources and cohorts. A tag names one version of its kind per project, compared without regard to case, and IDs never start with `@`. An experiment can also be named by its name, compared without regard to case; an ID wins over a name, and a name that two experiments share is refused.
- **Options on every command:** `--url` (`HISTOPILOT_URL`, default `http://127.0.0.1:8787`), `--project` (`HISTOPILOT_PROJECT`, else the saved default), `--json`, and `--timeout SECONDS`, which bounds the whole command. An option beats its environment variable, which beats the saved default. Lists also take `--limit` and `--offset`.
- **Commands that commit** also take `--yes`, `--dry-run`, `--preview-hash` and `--operation-id`. Commands that start work take `--wait`.
- **Environment.** `HISTOPILOT_TOKEN` presents a scoped agent token instead of fetching the session; `HISTOPILOT_HOST_HEADER` names the service's own address behind an SSH forward to another port; `HISTOPILOT_LOGIN` supplies the sign-in secret of a service started with `--login`.
- **Older commands keep their behaviour:** `serve`, `doctor`, `runner`, `verify-study`, `restore-study` and `verify-feature-pack`. The flat `train-batch`, `training-status`, `pack-features` and `feature-jobs` keep their JSON output and print a note on standard error naming the replacement.

## Output

- **Text by default,** for people: tables for lists, labelled fields for one record. Wording and columns may change in any release, so scripts must not parse it.
- **`--json` is the contract.** Standard output carries exactly one JSON document, on success and on failure, usage errors included. `--json` also turns off prompts and colour. Progress, notes and prompts always go to standard error.

```json
{"schemaVersion": 1, "ok": true, "data": {"id": "…", "name": "…"}, "warnings": []}
{"schemaVersion": 1, "ok": false, "error": {"code": "PREVIEW_STALE", "kind": "conflict", "message": "…", "status": 409, "findings": []}}
```

| Field | Meaning |
| --- | --- |
| `data` | The result: an object for one record, an array for a list. Field names are the service's own, except the fields the CLI adds below. |
| `page` | Lists only: `{offset, limit, hasMore}`. A list returns at most `--limit` items (default 50), starting at `--offset`. |
| `warnings`, `error.findings` | Findings `{code, message, severity}`, plus `field` when one field is at fault: those that did not stop the command, and those that did. |
| `error.code` | The service's code, listed in [error codes](error-codes.md), or one of the CLI's own, listed with the exit codes below. |
| `error.kind`, `error.status` | The exit-code class below, and the HTTP status (`null` when the CLI raised the error). |
| `runState` | Added to each record that runs: `queued`, `running`, `succeeded`, `failed`, `needs-attention` (held, interrupted without automatic resume, or the runner stopped) or `cancelled`, and `null` before anything starts. It unifies the service's status vocabularies, which stay in their own fields. |
| `model` | Added to each predictor and run, and to a run's summary and the run it is compared with: the predictor's `experiment`, `experimentId`, `batch`, `batchId`, `architecture`, `method` and `description`, as the browser describes it ("nnMIL · Configuration 1 · Fold ensemble · Train 42 / split 42"); a run whose predictor is gone has only `predictorId` and `description`. Predictor names leave out the batch, so this is how two models' predictors and runs are told apart. |
| `execution` | Added to each batch in `batch list`: its execution's `status`, `runCounts`, `updatedAt` and `finishedAt`, from the executions the list returns beside the batches; a batch that never launched has none. |
| `data.preview`, `data.result` | Commit commands only: what the commit will do, and the service's response, `null` until the commit runs. The preview always carries a `previewHash`: the service's, or a SHA-256 of the preview itself. A commit that was refused or not confirmed still returns `data.preview`. |

- Within one `schemaVersion`, fields are only added. Renaming or removing a field, or changing its type or meaning, raises the version. Readers ignore fields they do not know.
- `tasks log --follow --json` is the one stream: JSON Lines of `{"event": "log", "text", "nextOffset"}`, ending with the envelope.

## Exit codes

| Exit | `error.kind` | Meaning, with the CLI's own codes | Next move |
| --- | --- | --- | --- |
| 0 | | Done. With `--dry-run`: the commit would be allowed. | |
| 1 | `internal` | Unexpected failure: a CLI bug (`INTERNAL_ERROR`), a service fault, or a download that does not match its recorded SHA-256 (`DOWNLOAD_MISMATCH`). | Report it with the output |
| 2 | `invalid` | Bad options or arguments (`USAGE_ERROR`, `PROJECT_REQUIRED`), a spec that fails its checks or names an unknown kind (`SPEC_INVALID`, `SPEC_FIELD_REQUIRED`, `SPEC_KIND_UNKNOWN`), or a request the service's schema rejects (`REQUEST_INVALID`). | Fix the command or spec |
| 3 | `refused` | Blocking findings (`PREVIEW_BLOCKED`), a state that forbids the action, a route, project or exposure level a scoped token does not reach (`SCOPE_MISSING`, `PROJECT_OUT_OF_SCOPE`, `EXPOSURE_NONE`, `EXPOSURE_METADATA`), or a name that two experiments share (`EXPERIMENT_AMBIGUOUS`). | Change the request or the state first; resent unchanged, it fails again |
| 4 | `conflict` | Something changed since it was read, such as a stale revision or preview, or a preview that no longer has the `--preview-hash` given (`PREVIEW_CHANGED`); or the project or service stayed busy. | Reload or preview again, then retry |
| 5 | `not-found` | No record, task or route by that name (`TAG_NOT_FOUND` for an unknown `@tag`). An experiment named by a name no experiment has goes to the service as an ID, which answers `EXPERIMENT_NOT_FOUND`. A list filter (`--experiment`, `--cohort`) that names nothing is refused the same way (`EXPERIMENT_NOT_FOUND`, `RECORD_NOT_FOUND`) instead of listing nothing. | Check the ID or tag |
| 6 | `unavailable` | No service at `--url` (`SERVICE_UNREACHABLE`), an answer that is not the service's JSON (`RESPONSE_UNREADABLE`), or it refused the session over Host, Origin, token or sign-in (`HOST_REFUSED`, `SESSION_TOKEN_REQUIRED`, `TOKEN_INVALID`, `LOGIN_REQUIRED`, …). | Start the service, fix `--url`, or sign in |
| 7 | `unconfirmed` | A commit with no `--yes` and no terminal to ask on (`CONFIRMATION_REQUIRED`), the person said no (`CONFIRMATION_DECLINED`), or a scoped token's commit waits for a person (`CONFIRMATION_PENDING`). Nothing changed. | Review `data.preview`, then rerun with `--yes`; a person approves a pending request with `histopilot confirm` |
| 8 | `timeout` | `--timeout` or a request ran out of time (`TIMEOUT`); the work may go on. | Wait again, or rerun to replay |
| 9 | `work-failed` | The awaited work failed, was cancelled or needs attention (`WORK_FAILED`, `WORK_CANCELLED`, `WORK_NEEDS_ATTENTION`). A stopped runner ends a wait here instead of hanging it. | `histopilot tasks show`, then resume or retry |
| 130 | `interrupted` | Interrupted with Ctrl-C (`INTERRUPTED`). | Rerun to replay |

Every service error carries a code, and [error codes](error-codes.md) gives each its kind. For a code the CLI's registry does not know, from an older or newer service, the status decides: 401 → 6, after the token has been renewed once; 400, 405 and 422 → 2; 403 and 413 → 3; 404 → 5; 409 → 4 for a code ending in `_BUSY` or naming a stale revision or preview, otherwise 3; 408, 429, 503 and 504 → 4; any other 5xx → 1. Apart from the token renewal, only read routes retry, and only on `PROJECT_BUSY`: on the browser's schedule (0.25, 0.5, 1 and 2 s), then exit 4.

## Route classes

`histopilot/api/route_classes.py` gives every service route one class. A test fails when a route has no class, or when a class names a route that no longer exists. The class follows the route's effect, not its HTTP method; every GET is read.

| Class | The route… | Routes |
| --- | --- | ---: |
| read | changes nothing: every GET, plus POST queries and exports such as dataset queries, scores, agreement, subgroups and case exports. Caches and housekeeping do not count. | 117 |
| preview | prepares a change without making it: the `…preview` checks, which return findings with a `previewHash` or a `can…` gate, and saves of drafts, which include experiments and their batch plans. Nothing is frozen, queued or removed. | 24 |
| commit | changes one project for real: freeze, publish or save a record; launch, submit, resume, cancel, retry, hold or move work; clean up; relabel a version; write a slide review; edit the project's planning settings. | 50 |
| admin | reaches beyond one project or widens access: create or open projects, register or relink source folders, create folders, study archives, the Task Center's capacity and runner, agent tokens, a project's AI-exposure level, and deciding agents' requests. | 16 |

- A route with several actions takes the class of its strongest. Archive export, verify and restore share one route, and restore writes a new study folder, so the route is admin.
- The CLI uses the classes for confirmation and busy retries. A route missing from the table, for example one sent through `histopilot api`, counts as commit.
- Scoped agent tokens reuse them. A read token reaches read routes; a preview token adds preview routes; a commit token's commits wait for a person (below); only the service's session reaches admin routes. A scoped token reaches only its own project, the Task Center narrowed to it, and a few machine-level reads such as `/version` and `/access`.

## Confirmation

1. Read and preview requests never ask.
2. Before a commit or admin request, the command shows what will happen. That is the preview with its findings or, for an action without one such as cancel or relabel, the record's current state and the change.
3. Blocking findings stop the command with exit 3. `--yes` never overrides them.
4. On a terminal the command then asks, with no as the default. With `--json`, or without a terminal, it needs `--yes`; otherwise it exits 7 and returns the preview. Answering no also exits 7.
5. `--dry-run` stops after step 2 and exits 0, or 3 when findings block the commit.
6. `--preview-hash H` commits only if a fresh preview still hashes to `H` (exit 4 otherwise), so a person can confirm exactly what an agent or a colleague showed them. The service refuses a stale preview in any case.
7. `--yes` has no environment variable or setting; each command must say it.
8. Before sending a request that takes an operation ID (every commit that has one, and the creation of projects and drafts), the CLI writes the ID to a journal in the state directory (`~/.local/state/histopilot/cli/` by default). The ID stays there until the command ends, and a command that stops short (exit 7, 8 or 130, or `--dry-run`) keeps it. Rerunning the same command after a lost response, a timeout or Ctrl-C, including one that came while `--wait` waited, replays that ID, and the service applies the operation once: a replayed creation returns the record the first attempt made. A refusal frees the ID at once, so a new attempt gets a new one. `--operation-id` sets the ID explicitly.
9. A commit made with a scoped token that has commit scope is parked: the service answers 202 with `CONFIRMATION_PENDING` and changes nothing, and the CLI exits 7 with the request in `data.request` (`requestId`, `expiresAt`). A person approves it with `histopilot confirm approve ID`, which replays it under their own session, or declines it. A parked request expires after 24 hours.
10. Prompts guard against mistakes, not against agents, since an agent can pass `--yes`. Token scopes, exposure levels and isolation are what limit an agent; see [agents](agents.md).

## Spec files

1. A spec is a YAML or JSON file in UTF-8. YAML is read with PyYAML's safe loader.
2. It opens with `kind` and `specVersion: 1`. The other keys are the fields of the service's request model for that kind, under the same camelCase names. `histopilot schema KIND` prints the model's JSON Schema, and `histopilot NOUN template` writes a complete starter from HistoPilot's templates, the values the browser starts from. Where the browser proposes values from the project's own records, the CLI proposes the same ones, with a note on each: Apply models' method, predictors and cohort (`apply template --experiment`), a target's classes (`targets infer`), comparison arms (`experiment template --compare-model`) and an extraction's features (`features template --from-extraction`).
3. The CLI checks a spec before sending anything: first the fields below, then `@tag` names, then the request model. An unknown key or a wrong type exits 2 and names the field. Quote values such as `no`, `on` or `1.10`; YAML would otherwise read them as a boolean or a number.
4. Every field that changes the science is written out. `template` and `export` write every field, and `create` refuses a spec that omits a field whose service default differs from the browser's choice, such as `splitUnit`, `includeMissingSlides` or an apply `scope`, naming each one and why (`SPEC_FIELD_REQUIRED`). The service refuses the riskiest omissions from any client: a new experiment without `setupVersion` (`SETUP_VERSION_REQUIRED`), predictor IDs without a `scope`, and an Apply models `inference` object without every setting.
5. Other records are referenced by ID or `@tag`. The CLI resolves tags before sending and reports the IDs it used.
6. Paths are absolute paths on the service's machine, inside its data roots.
7. An experiment design (`kind: experiment`) holds the experiment's `name`, `notes`, `tags` and `predictorPolicy`; its `inputs`, which are the dataset, Targets & splits, feature bundle, loading policy and version 4 training design; and its `batches`, each a development batch spec with an `id` and a complete `recipe`, without the fields the experiment fills in.
8. `export` writes a spec that `create` accepts back unchanged. A recreated experiment reaches the same design hash.
