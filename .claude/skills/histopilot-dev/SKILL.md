---
name: histopilot-dev
description: "Develop HistoPilot, the local-first computational-pathology app: a FastAPI service (histopilot/), a React browser UI (web/) and a machine-wide Task Center runner. Load before changing or reviewing HistoPilot API routes, request schemas, scientific storage or frozen records, Task Center tasks, adapters or workers, CLI commands, browser UI code, tests or docs; when a HistoPilot test fails; or when you must judge whether a change is safe for stored projects, queued tasks, a running service or the public repository."
---

# HistoPilot development

HistoPilot takes a pathology study from slide tables to evaluated MIL models on one workstation. A FastAPI service (`histopilot/`) owns validation and immutable scientific records, the React UI (`web/`) calls it under `/api/v1`, and one Task Center runner per OS user runs all compute in isolated workers.

This file is the working guide. [reference.md](reference.md) holds the repo map, contract tables, commands and recipes; open only the section you need.

## Workflow for a change

1. **Read** the whole path first: route in `histopilot/api/` → request model in `histopilot/schemas/` → service in `histopilot/application/` → `histopilot/storage/` or the Task Center. For the UI: `web/src/api/` → the page or component. Read the doc page that describes the behaviour.
2. **Plan** against "Decisions you must get right". If the change would rename a stored name, change an identity hash or rewrite a frozen record, stop and ask the owner.
3. **Test first where practical**: a failing test in `tests/`, or a `*.test.ts(x)` beside the web source, that pins the intended behaviour.
4. **Implement** the smallest change that follows the neighbouring code.
5. **Verify** with the checklist below.
6. **Sync docs** in the same change, then report what you ran and what you could not run.

## Setup

Environments are built by uv from the committed `uv.lock`; never `pip install`.

| Environment | Build it with | Used by |
| --- | --- | --- |
| `.venv` | `uv sync --locked --extra training --extra imaging` | Tests, Ruff, the service during development |
| `.venv-training` | `UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --extra training` | Training, refit, predictor-run and attention workers started by a real service |
| `.venv-agent` | `UV_PROJECT_ENVIRONMENT=.venv-agent uv sync --locked --extra agent` | `histopilot agent serve` and the MCP tests, which skip in `.venv` |
| `web/node_modules` | `npm --prefix web ci` | Vitest, type check, build, browser checks |

Build one only when it is missing or its lock file changed, and ask before changing an environment someone else uses. First commands in a checkout:

```bash
git status --short                                  # others' changes: never stash, reset or revert them
pgrep -af "histopilot(\.cli)? (serve|runner run)"   # a live service or runner? Leave it alone
.venv/bin/python -c "import histopilot, torch, PIL" # without these, Torch and image tests skip silently
.venv/bin/python -m pytest -q -n 8 -m "not slow"    # baseline before you edit
```

## Decisions you must get right

1. **Never run against live state.** Tests build an in-process app on temporary folders; `tests/conftest.py` isolates `TMPDIR` and gives every test its own `HISTOPILOT_STATE_DIR` with runner autostart off. Do not run `bash serve.sh`, `histopilot serve` or `histopilot runner start|stop|run` with the user's state: there is one runner per OS user, and a service started from your checkout restarts it on your code. Do not run `scripts/bundle_web.py` without `--check`: it swaps `histopilot/static/` under a running service. Checks on real data happen only in a sandbox service (reference.md, "Sandbox"), and only with the owner's permission.
2. **Stored names do not move.** The Task Center starts its runner as `python -m histopilot.cli runner run`, so `histopilot/cli.py` stays a module ending in `if __name__ == "__main__": app()`, and the `runner` group keeps `run`, `start`, `status` and `stop`. Task rows store their `adapter` and `kind` names and their argv, such as `mil-fold` and `python -m histopilot.workers.managed_fold`; pinned archives store code. `tests/test_persisted_contracts.py` pins the adapter and module names and `tests/test_task_center_launcher.py` the runner's argv; no test pins the kinds. A rename needs a shim or a migration, never just an edit.
3. **Identity is content.** A configuration's ID is `configuration-` plus the SHA-256 of its manifest's canonical JSON (`publish_configuration` in `histopilot/storage/scientific.py`). Dataset IDs, preview hashes and job IDs are content hashes too, each with its own canonical encoding (`histopilot/storage/io.py`). A new optional field of a spec must serialize to nothing at its default, as `SplitSpec.omit_default_grouping` does in `histopilot/schemas/protocols.py`. Never add provenance (author, user, host, tool, timestamps) to a spec or manifest: identical designs would split into separate records. Provenance belongs in receipts or logs; tags and notes are version labels, kept outside the content.
4. **Frozen records are immutable.** A change makes a new version. Older records stay readable and are never migrated in place.
5. **Keep the request contract.** Request bodies derive from `RequestModel` (`extra="forbid"`), so unknown fields get 422. An `operationId` replays: the same ID with the same request returns the first result, and with a different request gets 409 `OPERATION_CONFLICT`. A freeze repeats the reviewed `previewHash`; changed inputs get 409 `PREVIEW_STALE`. Draft edits and previews send `expectedRevision`; a stale one gets 409 `REVISION_CONFLICT`. Freezes that publish a version require `versionLabel` `{tag, note?}`.
6. **Classify what you add.** Every route has exactly one class, read, preview, commit or admin, in `histopilot/api/route_classes.py`. Every service error code is registered with its kind in `histopilot/api/error_codes.py`, and `docs/error-codes.md` is generated by `scripts/error_codes.py`, never edited by hand. CLI commands follow `docs/cli-contract.md`.
7. **Keep the service light.** Importing `histopilot.api.app` must not load Torch, Lightning or h5py (`tests/test_architecture.py`). Compute runs as Task Center tasks in workers.
8. **The remote is public.** Tracked files, fixtures, screenshots and commit messages hold no study or cohort names, cohort sizes, slide or patient IDs, results of real studies or personal paths, and nothing from `/.local/` or the private docs listed in `.gitignore`. Use `/path/to/...` placeholders and the synthetic BLCA demo.

## Verify at the end

- [ ] Targeted tests: `.venv/bin/python -m pytest -q -rs tests/test_<area>.py`. The `-rs` summary shows tests skipped for a missing extra.
- [ ] `.venv/bin/python -m pytest -q -n 8 -m "not slow"`. Before hand-over, run the full `.venv/bin/python -m pytest -q -n 8` (about 9 minutes) when you touched storage, training, workers or the Task Center.
- [ ] The contract tests pass without edits to their expectations: `test_persisted_contracts`, `test_storage_io`, `test_route_classes`, `test_error_codes`, `test_private_imports`, `test_architecture`, `test_model_catalog_fixture`, `test_task_center_launcher`.
- [ ] `.venv/bin/ruff check .` and `.venv/bin/ruff format --check <the files you changed>`.
- [ ] If `web/` changed: `npm --prefix web test`, `npm --prefix web run typecheck`, `npm --prefix web run build`, and `node web/scripts/verify-<surface>.mjs` for each walkthrough that renders what you changed.
- [ ] Docs synced (reference.md, "Docs sync"); `docs/error-codes.md` regenerated if error codes changed.
- [ ] `git diff` reviewed: only your files, no weakened assertions, no unrelated reformatting, nothing private.
- [ ] Nothing committed, pushed, served, bundled or restarted unless the owner asked.

## Pitfalls

- `uv sync` is exact: without `--extra training --extra imaging` it removes Torch and Pillow from `.venv`, and their tests then skip instead of failing.
- When `VIRTUAL_ENV` points at another checkout, `uv run` warns and ignores it. `.venv/bin/python` is unambiguous.
- A new git worktree has no `.venv-training`; real training there fails with "No module named 'torch'".
- `ruff format .` rewrites dozens of files you did not touch, because the tree is not format-clean. Format only your files. Ruff also formats Python code blocks inside Markdown.
- An in-process app needs `with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:`. The `with` runs the lifespan that creates the workspace database, and any other Host gets 400. Send the token from `GET /api/v1/session` as `X-HistoPilot-Token`.
- A test module imports fixtures from `tests/support/` by name, or pytest does not register them.
- "Worker source changed while creating its archive" means files under `histopilot/` changed during the run, for example another agent editing the same tree. Rerun those tests once the tree is quiet.
- Task IDs hash the kind name (`histopilot/taskcenter/ids.py`); renaming a kind orphans stored rows and deep links.
- An adapter name missing from `ADAPTERS` falls back to the generic adapter with only a warning, which misreports that task's outcomes.
- `node web/scripts/verify-blca-demo.mjs --readme` and `python -m histopilot.application.blca_demo --write` rewrite tracked files; run them only when asked.
- A change to `histopilot/models/catalog.py` needs `web/src/lib/modelCatalog.json` regenerated (reference.md, "Browser checks").
- The Vite dev server proxies `/api` to `127.0.0.1:8787`; if a real service listens there, the dev UI edits real projects.
- `npm ci` deletes `web/node_modules` first, and that folder may be a link shared with other worktrees.
- A helper that another module imports must be public; `tests/test_private_imports.py` lets the count of private imports only fall.
- `verify-case-review.mjs`, `verify-operations.mjs` and `verify-morphology.mjs` load a prebuilt harness from their output folder and fail without one; the other walkthroughs build their own fixture.
