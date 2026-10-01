# HistoPilot developer reference

The detail behind [SKILL.md](SKILL.md). Read only the section you need. Nothing keeps this file in sync with the code: when you change something it describes, update it in the same change. If a file named here does not exist yet, its rule still applies; say so in your hand-over.

1. [Repository map](#repository-map)
2. [Persisted contracts](#persisted-contracts)
3. [Request and error conventions](#request-and-error-conventions)
4. [Route classes, error codes and the CLI contract](#route-classes-error-codes-and-the-cli-contract)
5. [Tests](#tests)
6. [Browser checks](#browser-checks)
7. [Format and lint](#format-and-lint)
8. [Docs sync](#docs-sync)
9. [Recipes](#recipes)
10. [Environments](#environments)
11. [Safety: live systems, the sandbox and privacy](#safety-live-systems-the-sandbox-and-privacy)

## Repository map

### Top level

| Path | Holds |
| --- | --- |
| `histopilot/` | The Python package: service, storage, Task Center, workers and CLI |
| `web/` | The React UI (Vite, TypeScript, TanStack Query, Vitest) and offline browser checks in `web/scripts/` |
| `tests/` | The pytest suite; `tests/support/` holds shared helpers, `tests/fixtures/` static inputs |
| `docs/` | User and developer documentation; `docs/assets/` holds screenshots of synthetic data |
| `scripts/bundle_web.py` | Copies `web/dist/` into `histopilot/static/`; `--build` builds first, `--check` only reports |
| `scripts/error_codes.py` | Generates `docs/error-codes.md` from the error-code registry |
| `serve.sh` | Launcher: rebuilds the UI when `web/` changed, then runs `histopilot serve` in the foreground |
| `examples/config.toml` | Service configuration: `[server]` (`host`, `port`) and `[storage]` (`workspace`, `data_roots`) |
| `pyproject.toml`, `uv.lock` | Dependencies, the `training`, `imaging` and `sdpc` extras, the `dev` group, the pytest `slow` marker, Ruff settings |
| `.github/workflows/checks.yml` | CI: `uv sync --locked --extra imaging`, `ruff check .`, the full pytest suite, the agent tests in a `.venv-agent`, `npm ci && npm test && npm run build` in `web/`, bundling and a wheel check (the frontend, the demo workspace and exactly the MCP doc pages) |
| `histopilot/static/` | The built UI a checkout serves; gitignored |
| `/.local/` | Private, gitignored notes. Never copy anything from it into tracked files. |

### Python packages

| Package | Holds | Rule |
| --- | --- | --- |
| `histopilot/api/` | FastAPI routers by area; `app.py` (`create_app(settings)`), `security.py` (loopback Host, Origin and token boundary), `route_classes.py`, `error_codes.py` | Never imports Torch, Lightning or h5py. `histopilot.api` imports `create_app` lazily so the route and error tables load without FastAPI. |
| `histopilot/schemas/` | Pydantic request models and spec contracts; `workspace.RequestModel` is the strict base; `version_labels.py` | Specs that enter manifests keep their serialized shape |
| `histopilot/application/` | Services, one module per area (see the area map below); `task_records.py` holds the Task Center plumbing they share (`TaskCenterAccess`, `owner`, `enqueue`) | Business logic lives here, not in routers |
| `histopilot/storage/` | `scientific.py` (drafts, datasets, configurations, version labels, publications), `database.py` (central workspace database), `lifecycle.py` (archive, Trash, restore), `project_lock.py` (`StorageError`, writer lock), `filesystem.py` (root confinement), `packed.py` and `pack_import.py` (feature packs), `io.py` (canonical JSON, content hashes, timestamps, bounded JSON files) | Managed paths reject symlinks |
| `histopilot/taskcenter/` | `store.py` (SQLite task store), `runner.py`, `launcher.py` (tmux start, stop, restart), `wrap.py` (per-task wrapper), `capacity.py`, `estimator.py`, `leases.py`, `paths.py`, `ids.py`, `model.py` (states and task-spec validation), `service.py` (API read models and routed actions), `client.py`, `jobs.py` (live-code entry points), `adapters/` | Nothing here is in the pinned compute snapshot; the runner preloads every adapter at start |
| `histopilot/workers/` | Task entry points: `managed_fold`, `managed_collect`, `compute_job`, `experiment_predictors`, `pack_features`, `portability`, `verify_extraction`; `train_batch.py`, `training_process.py` (process identity, `compute_snapshot`), `compute_archive.py` | Only workers may initialize CUDA |
| `histopilot/training/`, `models/`, `datasets/` | Lightning fold training, refit, inference and attention; model implementations (`catalog.py` is Torch-free, `registry.py` builds models); MIL data loading and sampling | Imported by workers only |
| `histopilot/adapters/` | `native/` finds the training interpreter; `trident/` plans and supervises TRIDENT extraction (`runner.py` is started by path under TRIDENT's own Python) | |
| `histopilot/viewer/` | Isolated slide readers, image cache, attention arrays | Readers run in child processes |
| `histopilot/client/`, `histopilot/commands/` | The CLI's client library (standard-library HTTP behind a replaceable transport) and its noun-verb command groups | Follow `docs/cli-contract.md` |
| `histopilot/resources/` | The synthetic BLCA demo fixture (`blca_demo_workspace.json`) | Generated by `application/blca_demo.py` |
| Top-level modules | `cli.py` (the Typer app: `serve`, `doctor`, the flat legacy commands, `python -m` entry), `runner_cli.py`, `archive_cli.py`, `config.py` (`Settings`, `load_settings`), `service_lock.py`, `web_bundle.py`, `doctor.py`, `diagnostics.py`; Torch-free science: `cv_summary.py`, `statistics.py`, `scoring.py`, `candidate_selection.py`, `inference_summary.py`, `calibration.py`, `clinical_features.py` | |

### Areas: routes, schemas, services and web clients

`{P}` stands for `/api/v1/projects/{identity}`. Router, schema and service files sit in `histopilot/api/`, `histopilot/schemas/` and `histopilot/application/`; web clients in `web/src/api/`.

| Area (UI module) | Routes | Router | Schemas | Services | Web client |
| --- | --- | --- | --- | --- | --- |
| Service, projects, drafts, datasets | `/health`, `/session`, `/system`, `/filesystem/*`, `/projects`, `{P}`, `{P}/drafts`, `{P}/datasets` | `app.py`, `scientific.py` | `workspace.py`, `scientific.py`, `imports.py` | `project_workspace.py`, `imports.py`, `exploration.py` | `client.ts`, `scientific.ts`, `queries.ts` |
| Targets & splits | `{P}/target-splits`, `{P}/protocols/explore`, `{P}/configurations` | `scientific.py` | `target_splits.py`, `protocols.py` | `target_splits.py`, `protocols.py`, `modern_splits.py` | `targetSplits.ts` |
| Slide features | `{P}/features`, `{P}/feature-bundles`, `{P}/feature-packs`, `{P}/extractions` | `scientific.py` | `features.py`, `feature_bundles.py`, `feature_packs.py`, `extractions.py`, `slide_lists.py` | `features.py`, `feature_bundles.py`, `feature_packs.py`, `extractions.py` | `bundles.ts`, `packing.ts`, `trident.ts` |
| Experiments | `{P}/model-experiments`, `{P}/mil-experiments` | `model_experiments.py`, `mil.py` | `model_experiments.py`, `development.py`, `training_controls.py`, `nnmil.py` | `model_experiments.py`, `development.py`, `training.py`, `experiment_results.py`, `experiment_predictors.py`, `comparisons.py` | `experiments.ts`, `development.ts`, `experimentResults.ts`, `mil.ts` |
| Predictors | `{P}/predictors` | `predictors.py` | `predictors.py` | `predictors.py`, `predictor_builds.py`, `refits.py` | `predictors.ts`, `predictorBuilds.ts` |
| Apply models: cohorts | `{P}/evaluation-cohorts`, `{P}/drafts/{draft_id}/evaluation-preview`, `…/evaluation-freeze` | `evaluations.py` | `evaluations.py` | `evaluations.py` | `evaluation.ts` |
| Apply models: runs | `{P}/evaluation-runs` and its sub-routes | `predictors.py` (`evaluation_run_router`), `inference.py`, `case_review.py`, `performance.py`, `references.py` | `predictors.py`, `bulk_evaluations.py`, `inference.py`, `case_review.py`, `performance.py`, `references.py` | `evaluation_runs.py`, `bulk_evaluations.py`, `run_evidence.py`, `run_metrics.py`, `run_performance.py`, `inference_analysis.py`, `case_review.py` | `evaluation.ts`, `bulkEvaluations.ts`, `inference.ts`, `caseReview.ts`, `performance.ts`, `recalibration.ts` |
| Reference standards | `{P}/reference-standards` | `references.py` | `references.py` | `references.py` | `references.ts` |
| Clinical utility | `{P}/clinical-analyses` | `clinical.py` | `clinical.py` | `clinical.py`, `clinical_inputs.py` | `clinicalUtility.ts` |
| Interpretation | `{P}/interpretations` | `interpretation.py` | `interpretation.py` | `interpretation.py`, `interpretation_gallery.py` | `interpretation.ts` |
| Slide viewing and reviews | `{P}/morphology`, `{P}/datasets/{dataset_id}/slide-reviews` | `morphology.py`, `slide_reviews.py` | `morphology.py`, `slide_reviews.py` | `morphology.py`, `slide_reviews.py` | `morphology.ts`, `slideReviews.ts` |
| Cleanup, backups, sources | `{P}/cleanup`, `{P}/operations` | `lifecycle.py`, `operations.py` | `lifecycle.py`, `operations.py` | `lifecycle.py`, `operations.py`, `portability_jobs.py` | `lifecycle.ts`, `operations.ts` |
| Task Center | `/task-center/*` (machine-wide) | `task_center.py` (request models inline) | | `histopilot/taskcenter/service.py` | `taskCenter.ts` |

Route names keep older wording: an `evaluation-cohort` is a cohort of Apply models and an `evaluation-run` is one predictor applied to one cohort.

### Web

| Path | Holds |
| --- | --- |
| `web/src/api/` | Typed HTTP clients per area over `client.ts` (`request`, `requestScientificSave`, downloads); `types.ts`; tests beside them as `*.test.ts` |
| `web/src/lib/` | Workflow helpers: routing (`hashRoute.ts`, `canonicalRoutes.ts`, `applyRoutes.ts`), roadmap, drafts and recovery, `taskCenterActions.tsx` (operation-ID receipts), charts, slide tiles; `modelCatalog.json` is generated |
| `web/src/pages/`, `web/src/components/` | Pages (`Local*.tsx`, `TaskCenter.tsx`, `System.tsx`, `Start.tsx`, `BlcaDemo.tsx`) and shared components, with `*.test.tsx` beside them |
| `web/src/testFixtures/` | Fixtures shared by Vitest tests and walkthroughs |
| `web/scripts/` | Offline browser checks (`verify-*.mjs`) and `measure-page-loading.mjs` |
| `web/dist/` | Build output (gitignored), copied into `histopilot/static/` by `scripts/bundle_web.py` |

### Test layout

| Path | Holds |
| --- | --- |
| `tests/conftest.py` | Session-wide private `TMPDIR` and runner autostart off; a private `HISTOPILOT_STATE_DIR` per test; the `task_center` fixture |
| `tests/support/task_center.py` | `Center`, `fake_host`, `begin` and `conclude` |
| `tests/support/workers.py` | Run a queued task's worker in-process (`run_task`, `run_pack`, `run_archive`) or as a subprocess (`run_compute_worker`) |
| `tests/support/projects.py` | Datasets, feature bundles packed through a real packing task, cohorts |
| `tests/support/predictors.py`, `training.py`, `compute.py` | Candidates, frozen predictors, refits; training runtime probes and batches; compute jobs and the `managed_study` fixture |
| `tests/support/cli.py` | `Service`: drives the CLI and its client against an in-process app |
| `tests/test_persisted_contracts.py`, `test_storage_io.py`, `test_route_classes.py`, `test_error_codes.py`, `test_private_imports.py`, `test_architecture.py`, `test_model_catalog_fixture.py`, `test_task_center_launcher.py` | Contract tests: fix the code, never their expectations |

## Persisted contracts

Queued tasks, pinned archives and frozen records outlive the code that wrote them. Anything below that moves strands work on users' machines. If a change needs one of them to move, stop and ask the owner; the answer is a shim or a migration plus a test update, never an edit alone.

### Names

| Contract | Defined in | Pinned by | What depends on it |
| --- | --- | --- | --- |
| The runner starts as `python -u -m histopilot.cli runner run` in the tmux session `hp-runner-<uid>`, appending to `runner.log` in the state directory | `taskcenter/launcher.py` (`ensure_runner`); `runner_cli.py`; the `if __name__ == "__main__": app()` line of `cli.py` | `tests/test_task_center_launcher.py`, which checks the argv string only | `histopilot serve`, `histopilot runner start`, the Task Center's start and restart actions, and new submissions. One runner serves every checkout of an OS user. |
| CLI commands `serve`, `doctor`, `runner` (`run`, `start`, `status`, `stop`), `verify-study`, `restore-study`, `verify-feature-pack`; the flat `train-batch`, `training-status`, `pack-features` and `feature-jobs`, kept as deprecated aliases | `cli.py`, `runner_cli.py`, `archive_cli.py` | `docs/cli-contract.md` | Docs, user scripts, messages such as the archive commands' "Follow it with `histopilot runner status`" |
| `histopilot.cli:create_dev_app` (the uvicorn factory for `serve --dev`) and the console script `histopilot = "histopilot.cli:app"` | `cli.py`, `pyproject.toml` | `tests/test_config_cli.py` | `serve --dev`, installed wheels |
| Adapter names stored in task rows: `generic`, `mil-fold`, `mil-collect`, `compute-job`, `predictor-coordinator`, `bulk-submit`, `extraction`, `extraction-validation`, `packing`, `archive` | `taskcenter/adapters/__init__.py` (`ADAPTERS`, as `"module:Class"` strings) | `STORED_ADAPTERS` in `tests/test_persisted_contracts.py` | The runner resolves each row's adapter by name. An unknown name falls back to `GenericAdapter` with only a warning. |
| Task kinds stored in task rows: `mil-fold`, `mil-collect`, `compute-job`, `predictor-coordinator`, `bulk-submit`, `extraction`, `extraction-validation`, `packing`, `archive` | The producers in `application/` | No dedicated test | Task IDs (`taskcenter/ids.py` hashes the kind), `taskcenter/service.py` (links, cancel and retry routing by kind), history and task filters, deep links (`kind=`), the labels in `web/src/api/taskCenter.ts`, `docs/task-center.md` |
| Worker modules started with `python -m`: `histopilot.workers.managed_fold`, `managed_collect`, `compute_job`, `experiment_predictors`, `pack_features`, `verify_extraction`, and `histopilot.taskcenter.jobs` | `workers/`, `taskcenter/jobs.py` | `ENTRY_MODULES` | The argv of queued tasks and of pinned archives |
| Worker files started by path: `workers/pack_features.py`, `workers/portability.py`, `adapters/trident/runner.py` | `feature_packs.WORKER`, `portability_jobs.WORKER` | `test_worker_files_started_by_path_exist` | Queued argv |
| `TASK_CENTER_PROTOCOL = 1` in `workers/compute_job.py` and `application/experiment_predictors.py` | | `test_archived_modules_declare_the_protocol_where_launches_read_it` | Launches read it from pinned archives without importing them; an archive without it is refused with `CREATED_BEFORE_TASK_CENTER` |
| The files of the training code fingerprint | `workers/training_process.py:compute_snapshot` | `test_the_code_fingerprint_names_only_existing_files`, `tests/test_mil_compute_fingerprint.py` | Submitted experiments check their pinned code against it. A fingerprinted module that starts importing another HistoPilot module adds that module to the list; new submissions then get a new fingerprint. |
| Project folder layout: `histopilot-project.json`, `histopilot-state.sqlite` (scientific store, schema 4), `histopilot-lifecycle.json`, the lock files, `datasets/`, `training/`, `compute-jobs/` and so on | `docs/architecture.md`, "Persistence" | `tests/test_scientific_storage.py`, `tests/test_lifecycle_storage.py` | Projects move between machines and versions; older stores are upgraded by adding tables, never reset |

### Identities and hashes

| Identity | Computed as | Where |
| --- | --- | --- |
| Configuration ID | `configuration-` + SHA-256 of `encode_document(manifest)`, compact UTF-8 canonical JSON | `ScientificStore.publish_configuration`; an identical manifest returns the existing record (`ON CONFLICT(id) DO NOTHING`) |
| Dataset ID | `dataset-` + SHA-256 of the manifest and its artifact checksums | `_hash_content` in `storage/scientific.py`; timestamps, project IDs and locations are envelope fields, outside the hash |
| Most preview, plan and evidence hashes | `content_hash(...)`: compact ASCII, the default | Each service's `preview` |
| Import previews and artifacts | Compact UTF-8, as for configurations | `application/imports.py` |
| Extraction and feature-pack job IDs and preview hashes | `content_hash(..., compact=False)`: default separators | `application/extractions.py`, `feature_packs.py` |
| Task IDs and owner keys | `task-` or `owner-` + the first 32 hex digits of a SHA-256 over the kind and its parts | `taskcenter/ids.py` |

`tests/test_storage_io.py` pins the digest of each canonical encoding on fixed inputs. A call site keeps the encoding it has; a new one uses `canonical_json` or `content_hash` from `storage/io.py`, never its own `json.dumps`.

What enters a manifest: the service's preview result, which includes `spec.model_dump(mode="json")`. The same dump feeds the preview hash and the replay check of an `operationId`. So any change in how a spec serializes changes preview hashes (reviewed drafts go stale), configuration IDs (identical designs become new records) and replays (a retried operation conflicts). The shape is protected by wrap serializers that drop fields at their historical default:

| Model | Serializer | Drops |
| --- | --- | --- |
| `SplitSpec` (`schemas/protocols.py`) | `omit_default_grouping` | `groupByPatient` when false, `foldField` when unset |
| `TargetSplitSettings` (`schemas/target_splits.py`) | `omit_default_remainder` | `testRemaining` when false |
| `TargetSplitSpec` | `preserve_inherited_testing_target` | `testTarget` and `splitUnit` when not given |
| `SlideListSpec` (`schemas/slide_lists.py`) | `preserve_existing_selection` | `slideList` when unset |
| `FeatureSpec` (`schemas/features.py`) | `preserve_existing_selection`, which replaces the parent's | `slideList` when unset, `featureKind` when `patch` |
| `TrainingRecipe`, `DevelopmentBatchSpec` (`schemas/development.py`) | `retain_legacy_shape` | Experimental controls at their defaults, and newer optional fields left unset |
| `EvaluationSpec` (`schemas/evaluations.py`) | `preserve_legacy_purpose` | `purpose` when `independent`, `splitUnit` when not given, `sourceTargetSplitId` when unset |

Tests that guard this shape: `tests/test_slide_patient_grouped_folds.py::test_historical_designs_serialize_unchanged`, `tests/test_mil_experimental_schema.py::test_unused_controls_preserve_legacy_serialization_and_candidate_identity`, `tests/test_features.py::test_legacy_feature_preview_and_freeze_retry_keep_their_identity`, `tests/test_extractions.py::test_legacy_extraction_preview_and_retry_keep_their_hashes`.

### Records and provenance

- A frozen dataset or configuration never changes. A change is a new version; the old one stays readable. Records from earlier versions are never migrated in place.
- A draft is intent only. It has an integer `revision` and a status, `editable` or `frozen`; a frozen draft refuses edits (`DRAFT_FROZEN`).
- Mutable state lives beside the content: version labels in the store's `version_labels` table, archive and Trash state in `histopilot-lifecycle.json`, execution state in job folders and the Task Center store.
- A configuration document is `{id, projectId, contentHash, manifest, createdAt}`. Only `manifest` is hashed. Anything about who or what made a record, such as author, user, host, tool, CLI or agent, or time, goes in a receipt, a log or the envelope, never in a spec or manifest. Otherwise two identical designs frozen by different people, or at different times, become different records.

## Request and error conventions

| Convention | Rule | Where |
| --- | --- | --- |
| Session | `GET /api/v1/session` returns `{token, scientificCapabilities}`. Every other `/api/` route needs the token in `X-HistoPilot-Token`, except `/api/v1/health` and `/api/v1/session`. Tokens belong to one service process. | `api/security.py`, `api/app.py` |
| Boundary | The Host header must be a loopback name with the service's port, such as `127.0.0.1:8787`; other Hosts get 400. Foreign Origins and cross-site fetches get 403, a missing or wrong token 401. | `api/security.py` |
| Strict bodies | Request models derive from `RequestModel` (`schemas/workspace.py`, `extra="forbid"`); the few that derive from `BaseModel` set `extra="forbid"` themselves. An unknown field gets 422. | `histopilot/schemas/` |
| `operationId` | Publications, launches and every Task Center action take one. The same ID with the same request returns the original result; with a different request, 409 `OPERATION_CONFLICT`. | `configuration_publications` in `storage/scientific.py`; operation receipts in the task store |
| `previewHash` | A freeze or commit repeats the hash of the preview the user reviewed. The service recomputes it; if inputs changed, 409 `PREVIEW_STALE` (older paths say `STALE_PREVIEW`; cleanup says `CLEANUP_PREVIEW_STALE`). | Each service's `freeze` or `apply` |
| `expectedRevision` | Draft updates and previews name the revision they read. A stale one gets 409 `REVISION_CONFLICT`. Version labels have their own revision (`SetVersionLabelRequest`). | `ScientificStore._editable` |
| `versionLabel` | Freezes of datasets (imports), feature sources, feature bundles, Targets & splits, cohorts and development batches require `{tag, note?}`: a tag of 1–80 characters, unique per kind and project regardless of case (`VERSION_TAG_CONFLICT`), and a note of up to 2,000 characters. Labels are outside the content: they never change a configuration ID. Identical contents already saved under another label give `VERSION_LABEL_MISMATCH`. | `schemas/version_labels.py`, `storage/scientific.py` |
| Status codes | Reads, previews and queries return 200; publications and creations 201; most launches, resumes and cancels 202. Follow the neighbouring routes. | Routers |
| Errors | Every API error body is `{"detail", "code"}`. `StorageError(message, code, status_code=409, findings=...)` is the usual carrier and adds `findings` when a commit is refused over review findings; `WorkspaceError` and `FilesystemError` carry codes too. Schema validation returns 422 `REQUEST_INVALID` with `detail` as FastAPI's list of rejected fields; unknown API routes return 404 `API_ENDPOINT_UNKNOWN`; unexpected exceptions 500 `INTERNAL_ERROR`. Clients branch on `code`, never on message text. | `storage/project_lock.py`, `api/app.py`, `docs/error-codes.md` |

Error messages are plain sentences that say what to do next, such as "Targets or partitions changed. Preview again."

## Route classes, error codes and the CLI contract

These contracts are recent. They make the API usable by the CLI and, later, by scoped agent tokens.

### Route classes

`histopilot/api/route_classes.py` gives every `/api/v1` route exactly one class. Paths in its `_TABLE` are relative to `/api/v1`, with `{P}` for `/projects/{identity}`. The class follows the route's effect, not its method; a route with several actions takes its strongest.

| Class | Meaning | Examples |
| --- | --- | --- |
| `read` | Changes nothing, including POST queries and exports | Every GET; `evaluation-runs/{id}/scores`, `…/cases/query`, `datasets/{id}/query` |
| `preview` | Prepares a change without making it: `…preview` routes and draft saves | `features/preview`; `POST drafts`, `PATCH drafts/{id}`, `POST model-experiments` |
| `commit` | Changes one project for real | Freeze, publish, launch, submit, resume, cancel, retry, relabel, slide reviews, cleanup, Task Center task and owner actions |
| `admin` | Reaches beyond one project or widens access | Create or open projects, register or relink sources, create folders, study archives, Task Center capacity and runner |

`tests/test_route_classes.py` fails when a route has no class, when a class names a route that no longer exists, when a GET is not `read`, when a path ending in `preview` is not `preview`, when the set of preview routes not named `…preview` changes, when any API module uses `include_in_schema`, or when importing the table pulls in FastAPI. The table stays standard-library only. The CLI treats a route missing from the table as commit.

### Error codes

Every error code the service returns is registered with its kind in `histopilot/api/error_codes.py` (`ERROR_CODES`, with the kinds in `KINDS`: `invalid`, `refused`, `conflict`, `not-found`, `unavailable`, `internal`). `STALE_PREVIEW` is registered as an alias of `PREVIEW_STALE`. `docs/error-codes.md` is generated:

```bash
.venv/bin/python scripts/error_codes.py          # rewrite docs/error-codes.md
.venv/bin/python scripts/error_codes.py --check  # exit 1 when the page or the registry is out of date
.venv/bin/python scripts/error_codes.py --sites  # every code with its statuses and raise sites
```

`tests/test_error_codes.py` scans the raise sites in `histopilot/` and fails when a raised code is not registered, when a registered code is no longer raised anywhere, when a code built at runtime is not listed in `COMPUTED` under its module, when a class that sets `self.code` is missing from `CARRIERS` in `scripts/error_codes.py`, when `docs/error-codes.md` differs from the generator's output, or when the registry imports more than the standard library. The CLI and the client map each kind to an exit code (`docs/cli-contract.md`, "Exit codes").

### CLI contract

`docs/cli-contract.md` fixes command shape (`histopilot <noun> <verb>`), the `--json` envelope, exit codes, confirmation before commit and admin routes, the operation journal and spec files. Read it before adding or changing a command; see [the recipe](#add-or-change-a-cli-command).

## Tests

### Commands

| Goal | Command |
| --- | --- |
| One module or test, with skip reasons | `.venv/bin/python -m pytest -q -rs tests/test_route_classes.py` or `…/test_file.py::test_name` |
| Fast tier (excludes `slow`) | `.venv/bin/python -m pytest -q -n 8 -m "not slow"` |
| Full suite | `.venv/bin/python -m pytest -q -n 8`, about 9 minutes with 8 workers |
| List the slow tests | `.venv/bin/python -m pytest --collect-only -q -m slow` |
| Contract tests | `.venv/bin/python -m pytest -q tests/test_persisted_contracts.py tests/test_storage_io.py tests/test_route_classes.py tests/test_error_codes.py tests/test_private_imports.py tests/test_architecture.py tests/test_model_catalog_fixture.py tests/test_task_center_launcher.py` |
| As CONTRIBUTING and README write it | `uv run pytest -n auto --dist worksteal -m "not slow"`, then without `-m` for the full suite |

- `pyproject.toml` sets `testpaths = ["tests"]`, `pythonpath = ["."]` and `--strict-markers`. The only marker is `slow`: a test that trains real models or drives the real runner, 5 seconds or more. Mark such tests `@pytest.mark.slow`.
- Tests that need an extra call `pytest.importorskip("torch")`, `("lightning")`, `("PIL.Image")` or `("sklearn.metrics")`, so without the extra they skip rather than fail. CI installs only `--extra imaging`, so CI never runs the Torch tests; run them locally with the `training` extra when you touch training, workers or predictors.
- With many workers, `-n auto` on a busy machine, or another agent editing `histopilot/`, tests that archive the package fail with "Worker source changed while creating its archive" (`workers/compute_archive.py`). Rerun them once the tree is quiet; they are not real failures.

### What `tests/conftest.py` does

- A session fixture gives the run a private `TMPDIR` (so lease registries and output locks are private too) and sets `HISTOPILOT_TASK_CENTER_AUTOSTART=0`. The folder is removed when no test failed.
- An autouse fixture sets `HISTOPILOT_STATE_DIR` to the test's own `tmp_path` and resets the default Task Center client.
- `task_center` yields a `Center` over this test's store and cleans up any runner it started.

### An in-process app

```python
from fastapi.testclient import TestClient
from support.projects import dataset

from histopilot.api import create_app
from histopilot.config import Settings

BASE_URL = "http://127.0.0.1:8787"
API = "/api/v1"


def test_an_in_process_app_serves_a_frozen_dataset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,))
    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        assert client.get(f"{API}/projects").status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        (settings.workspace / "projects").mkdir(parents=True)
        project = client.post(
            f"{API}/projects",
            json={"name": "Study", "storagePath": str(settings.workspace / "projects" / "Study")},
        ).json()
        store = client.app.state.projects.scientific_store(project["id"])
        document, _rows = dataset(store)
        response = client.get(f"{API}/projects/{project['id']}/datasets/{document['id']}")
        assert response.status_code == 200, response.text
        missing = client.get(f"{API}/projects/{project['id']}/datasets/dataset-{'0' * 64}")
        assert missing.status_code == 404 and missing.json()["code"] == "DATASET_NOT_FOUND"
```

- `with` runs the app's lifespan, which creates the workspace database; without it, project routes fail.
- `base_url` must match the settings' port (8787 by default), or the boundary answers 400.
- A new project needs a new or empty folder whose parent exists inside the workspace or a data root.
- `client.app.state.projects.scientific_store(identity)` reaches the project's store, so tests can publish records with the `tests/support` helpers instead of driving every preview.
- Services can also be tested directly on a `ScientificStore(folder, project_id)`, as `tests/test_target_splits.py` does.

### Work that goes through the Task Center

A service that launches compute queues tasks exactly as in production; tests observe and drive them with the `task_center` fixture:

```python
def test_submission_queues_one_task_per_fold(task_center, tmp_path):
    ...  # call the service under test
    folds = task_center.tasks(adapter="mil-fold")
    assert len(folds) == 5
    task_center.finish(folds[0]["id"])  # record a successful exit
    task_center.finish(folds[1]["id"], "failed")  # or a failure
```

`Center` offers `tasks(kind=, adapter=, state=, group=)`, `task(id)`, `state(id)`, `start(id)`, `finish(id, state, returncode=, reason=, error=)`, `runner(**options)` with `tick_until(runner, predicate)` for a real runner over this store on a fake host, and `context(host=)` for adapter hooks. `support.workers` runs a task's worker as the runner would.

A test module that uses a fixture from `tests/support/` imports it by name, for example `from support.predictors import registry as registry`, so pytest registers it.

### A failing test

- Find out what the test protects before touching it. Contract tests (above) protect stored names, hashes and boundaries; their expectations change only with an owner-approved migration.
- Fix the code. Never delete an assertion, loosen a comparison, add a skip or an `xfail`, or re-pin a digest to make a test pass. If you believe the test is wrong, say why in your hand-over and leave it failing, or ask.
- Rerun the whole module, then the fast tier.

## Browser checks

| Check | Command | Notes |
| --- | --- | --- |
| Unit and component tests | `npm --prefix web test` | Vitest (`vitest run`); the whole suite takes seconds. One file: `npm --prefix web test -- src/pages/TaskCenter.test.tsx`. |
| Type check | `npm --prefix web run typecheck` | `tsc --noEmit`, strict mode; `cd web && npx tsc --noEmit` is the same |
| Build | `npm --prefix web run build` | `tsc --noEmit && vite build`; writes only `web/dist/` |
| Walkthroughs | `node web/scripts/verify-<name>.mjs [output-folder]` | Offline; see below |
| Palette contrast | `node web/scripts/verify-palette.mjs` | No browser needed |
| Is the served bundle current? | `.venv/bin/python scripts/bundle_web.py --check` | Read-only; exit 1 when `histopilot/static/` is older than `web/` |
| BLCA demo fixture | `.venv/bin/python -m histopilot.application.blca_demo --check` | Also run by `tests/test_blca_demo.py`; `--write` regenerates the tracked fixture |
| Model catalog fixture | `.venv/bin/python -c "import json; from histopilot.models import catalog; print(json.dumps(catalog.describe(), indent=2))" > web/src/lib/modelCatalog.json` | After any change to `histopilot/models/catalog.py`; `tests/test_model_catalog_fixture.py` compares the two |

Walkthroughs build a page from the real React components with mocked API modules, drive it in headless Chromium and start no HistoPilot server. They need a Chromium headless shell under `~/.cache/ms-playwright/chromium_headless_shell-*`, or its path in `HISTOPILOT_CHROMIUM`. Several fixtures make any unmocked request throw, and `verify-task-center.mjs` also fails on `window.confirm`; a new API call on a surface needs a mock in each walkthrough that renders it.

| Script | Surface |
| --- | --- |
| `verify-apply-models.mjs` | Apply models: runs library and run views (`pages/LocalApplyModels.tsx`) |
| `verify-blca-demo.mjs` | The whole BLCA demo; `--readme` replaces the tracked images in `docs/assets/blca/` |
| `verify-brand.mjs` | Branding in the app shell and Start page |
| `verify-case-review.mjs` | Case review. Loads a prebuilt harness from its output folder; fails without one |
| `verify-experiment-workflow.mjs` | Experiments, from design to start (`pages/LocalExperiments.tsx`) |
| `verify-extraction-jobs.mjs` | Extraction jobs in Slide features and the job tray |
| `verify-feature-workflow.mjs` | Slide features (`pages/LocalFeatures.tsx`) |
| `verify-mil-recipes.mjs` | Recipe editing and save payloads (`components/DevelopmentBatches.tsx`) |
| `verify-morphology.mjs` | Slide viewer and morphology explorer. Needs a prebuilt harness |
| `verify-operations.mjs` | Study backups & sources. Needs a prebuilt harness |
| `verify-page-loading.mjs` | Lazy pages and error boundaries |
| `verify-preparation-pages.mjs` | Datasets (`pages/LocalDataset.tsx`) |
| `verify-record-management.mjs` | Record management and stage workflow components |
| `verify-registry-recovery.mjs` | Experiment registry: create operations and metadata recovery |
| `verify-roadmap-guidance.mjs` | Project roadmap |
| `verify-run-resource-usage.mjs` | Run resource usage charts |
| `verify-system-compute.mjs` | System & storage (`pages/System.tsx`) |
| `verify-target-split-workflow.mjs` | Targets & splits (`pages/LocalTargetSplit.tsx`) |
| `verify-task-center.mjs` | Task Center page, detail drawer and job tray |
| `verify-test-cohort-workflow.mjs` | Cohort setup (`pages/LocalEvaluationSetup.tsx`) |

`measure-page-loading.mjs` compares route loading against an older `App.tsx`; it is a measurement, not a check.

Serving the UI: `histopilot serve` serves `histopilot/static/`. `bash serve.sh` rebuilds it when the fingerprint of `web/` changed, and refuses to while a service from the same checkout runs. For UI work with hot reload, start the [sandbox](#sandbox) service with `--dev`, which admits the Vite origin, and run `npm --prefix web run dev`. Vite's own address and its proxy are fixed in `web/vite.config.ts`: it proxies `/api` to 127.0.0.1:8787, so the sandbox must listen on 8787, and only when no other service does.

## Format and lint

| Tool | Rule |
| --- | --- |
| `ruff check .` | Must pass: CI and CONTRIBUTING require it. Rules `E4`, `E7`, `E9`, `F`, `I` (import order) and `UP`, target Python 3.11, line length 100 (`pyproject.toml`). `tests/support` imports sort with third-party packages, `histopilot` imports in their own block. |
| `ruff format` | Not enforced, and the tree is not format-clean. Keep new files and your own hunks formatted (`.venv/bin/ruff format --check <files>`); never run `ruff format .`, which rewrites dozens of files you did not touch. Ruff also formats Python code blocks in Markdown. |
| TypeScript | `tsc` strict mode is the only gate; there is no ESLint or Prettier. Follow the surrounding style. |
| Names | A helper that another module imports is public (no leading underscore); `tests/test_private_imports.py` only lets the count of private cross-module imports fall. |
| Docs | Plain, direct sentences and tables, generic placeholders such as `/path/to/slides`, and the synthetic BLCA demo for examples and screenshots. |

## Docs sync

Update the page that describes a behaviour in the same change. Keep `docs/` current rather than adding review or verification reports.

| Page | Covers | Update when you change |
| --- | --- | --- |
| `README.md` | What HistoPilot is, how to start it, the workflow at a glance | Setup commands, the headline workflow, requirements |
| `docs/user-guide.md` | The workflow stage by stage, as users see it; the command line | Anything a user sees or does in a stage page, or a CLI command |
| `docs/methods.md` | Split units, cross-validation, selection, OOF predictions, metrics, intervals, predictors, applying models, reference standards, clinical utility, recalibration | Anything that changes a scientific definition or result |
| `docs/api.md` | Session and conventions, every route group, the behaviour of each route | Any route, request body, response field or error a client handles |
| `docs/error-codes.md` | Every service error code and its kind | Never by hand: regenerate with `scripts/error_codes.py` |
| `docs/task-center.md` | Page, task kinds, states, dependencies, admission, capacity, cancel, hold and stop, busy exits, requeues, runner, deep links, state directory | Task kinds, states, runner behaviour, capacity rules |
| `docs/architecture.md` | Runtime and ownership, scientific records, execution model, pinned archives, persistence, locks, code map | New packages or stages, storage layout, execution rules |
| `docs/deployment.md` | Installation, `serve.sh`, configuration, runtime environments, environment variables, slide viewing, SSH, Vite, wheels, diagnostics, API protections | Setup, launcher, environment variables, security boundary |
| `docs/cli-contract.md` | CLI command shape, output envelope, exit codes, route classes, confirmation, spec files | Any CLI or route-class rule |
| `docs/blca-demo.md` | The synthetic walkthrough | The demo generator or its pages |
| `CONTRIBUTING.md` | Environments, tests and helpers, stored names and hashes, frontend checks, docs and privacy | Development workflow or test helpers |

Also update this skill (`SKILL.md`, `reference.md`) and its evaluation tasks (`tests/skill_evals/histopilot-dev.md`) when a rule they state changes.

## Recipes

### Add an API route

1. Put the handler in the router of its area ([area map](#areas-routes-schemas-services-and-web-clients)). Project routes hang off `APIRouter(prefix="/api/v1/projects/{identity}")` and reach the project with `projects.scientific_store(identity)`. A new area gets `histopilot/api/<area>.py` with `def <area>_router(projects, filesystem) -> APIRouter`, included in `create_app` with a local import, as the others are.
2. Validate the body with a `RequestModel` subclass in `histopilot/schemas/<area>.py`, and bound query parameters (`Query(max_length=…)`, `ge`, `le`).
3. Keep logic in a service in `histopilot/application/`, and raise `StorageError(message, "CODE", status)` for anything a client should handle. Import heavy modules lazily inside the handler; `tests/test_architecture.py` fails if building the app imports Torch, Lightning or h5py.
4. Return 200 for reads, previews and queries, 201 for publications, and 202 for queued work, as neighbouring routes do. Never pass `include_in_schema=False`.
5. Classify it in `_TABLE` of `histopilot/api/route_classes.py`, for example `("GET", "{P}/datasets/{dataset_id}/records"): READ`. A preview route's path ends in `preview`. A new draft-saving route of class preview must also join the expected set in `tests/test_route_classes.py`; that is a deliberate decision, not a fix.
6. Register any new error code ([below](#add-a-service-error-code)).
7. Test in-process ([recipe](#an-in-process-app)): the success case, 401 without the token, the not-found or conflict codes, and 422 for an unknown body field.
8. Add a row to the right table of `docs/api.md`, and to its route map for a new group. Describe user-visible behaviour in `docs/user-guide.md`.
9. If the UI uses it, add the browser call ([below](#add-a-browser-api-call)).

### Add a Task Center task kind

1. **Producer.** An application service builds each task spec as `taskcenter/model.py:normalize_task` expects: `id` from `taskcenter/ids.py` (`task_id(kind, …)`), `kind`, `adapter`, `title`, `group` `{kind, id}`, `labels`, `request` (`lane` `gpu` or `cpu`, `cpuThreads`, `ramGb`, `vramGb`, `graceSeconds`), `command` (`argv`, absolute `cwd` and `log`, optional absolute `progress` and `result`, `env`), `adapterData`, `dependsOn`, and `exclusiveKey` when two tasks must never run together. Build the owner with `application/task_records.owner(...)` and queue through `task_records.enqueue(access, owner, specs)`, as extraction, packing and archives do. It wakes the runner, except for injected test clients and code already running inside a task. `application/portability_jobs.py` is a compact example.
2. **Worker.** Start it as `python -u -m histopilot.workers.<name>` or by file path. It writes its log, a bounded progress JSON (at most 1 MiB) and a result file the adapter reads; it stops cleanly on SIGTERM within the task's grace period; and when its project or output lock is busy (`PROJECT_BUSY`, `OUTPUT_BUSY`) it changes nothing and exits 75, which the runner requeues with a backoff. Existing workers read `HISTOPILOT_TASK_MANAGED=1` from `command.env` to know the Task Center runs them. Decide whether it runs from live code or from a pinned archive: extraction, validation, packing, archives and bulk submission run the checkout's code; folds, collections, compute jobs and the predictor coordinator run archived code.
3. **Adapter.** Reuse `generic` for an opaque command (set `adapterData.requeueSafe` only if rerunning is harmless), or add `histopilot/taskcenter/adapters/<name>.py` subclassing `Adapter` from `adapters/base.py` (`prepare`, `on_started`, `progress`, `on_exit` returning `outcome(...)`, `can_requeue`, `on_requeue`; raise `AdapterError` for transient or fatal hook failures). Register it in `ADAPTERS` as `"name": "module:Class"`. Keep adapter imports light: the runner preloads every adapter when it starts.
4. **Contracts.** Add the new adapter name to `STORED_ADAPTERS` and a new entry module to `ENTRY_MODULES` (or a file started by path to `test_worker_files_started_by_path_exist`) in `tests/test_persisted_contracts.py`. Adding names is right; removing or renaming them is not. From now on the kind and adapter names are persisted.
5. **Task Center service.** In `taskcenter/service.py`, add stage links (`_task_link`, `_owner_link`) and, if the kind owns a stage record, route cancel and retry through its owning service (`_cancel_tasks`, `_retry_tasks`) so the record stays consistent. Other kinds are cancelled and retried directly in the store.
6. **Browser.** Add labels to `taskKinds` (and `ownerKinds` for a new owner kind) in `web/src/api/taskCenter.ts`; show it on its stage page through the rollup status chip.
7. **Tests.** With the `task_center` fixture, assert the queued specs, then drive them with `finish(...)` or run the worker through `support.workers`. Mark a test that runs a real runner `slow`.
8. **Docs.** Add the kind to the "Tasks" table of `docs/task-center.md` (what it runs, lane, priority), and update `docs/architecture.md` if the execution model changed.

### Add a field to a frozen-record spec

1. Find where the spec enters a manifest: the service's preview includes `spec.model_dump(mode="json")`, the preview hash covers it, `publish_configuration` hashes the manifest, and an `operationId` replay compares it.
2. Add the field with a default that means "behave as before", for example `field: int | None = None`, with bounds.
3. Make the default serialize to nothing: extend the model's `@model_serializer(mode="wrap")` to pop the key at its default, as `SplitSpec.omit_default_grouping` and `TargetSplitSettings.omit_default_remainder` do. If the model has no serializer, add one. A subclass serializer may replace its parent's (`FeatureSpec` replaces `SlideListSpec`'s), so extend the one that actually runs.
4. Code that reads stored manifests uses `.get(key, default)`: old records never carry the key.
5. Test, as `tests/test_slide_patient_grouped_folds.py::test_historical_designs_serialize_unchanged` does: an existing spec serializes exactly as before and without the key; an explicit default equals omission; a non-default value appears and changes the preview hash and the configuration ID. Freezing an unchanged design must give the same configuration ID as before your change.
6. If the model is in the training fingerprint (`compute_snapshot` lists `schemas/development.py`, `schemas/model_experiments.py` and others), new submissions get a new fingerprint; submitted experiments keep their archived code.
7. Document the setting in `docs/api.md` and `docs/user-guide.md`, and in `docs/methods.md` if it changes the science. Update the TypeScript types in `web/src/api/` if the UI sends it.
8. Never add provenance (author, user, host, tool, time) as a field: see [Records and provenance](#records-and-provenance).

### Add a browser API call

1. Add a typed function to the area's module in `web/src/api/` (or a new module) that calls `request<T>(path, init)` from `./client`. The path is relative to `/api/v1`; wrap every ID in `encodeURIComponent`; send bodies with `JSON.stringify`. The client adds the token, renews it once on 401, rereads a GET after `PROJECT_BUSY` (0.25, 0.5, 1 and 2 seconds), and never replays a mutation.
2. A tagged freeze goes through `requestScientificSave(path, init, 'taggedFreeze')` and a relabel through `requestScientificSave(path, init, 'versionLabels')`; both refuse cleanly when the running server lacks that capability (`scientificCapabilities` from `/session`).
3. Generate one operation ID per reviewed intent (`crypto.randomUUID()`), reuse it when the response was lost or the server failed (5xx), and take a new one only after a definite rejection (4xx other than 408). `runTaskCenterOperation` and `definiteRejection` in `web/src/lib/taskCenterActions.tsx` implement this.
4. Declare response types beside the call with the service's own field names. Read data through TanStack Query with keys scoped by project and record, as `web/src/api/queries.ts` and `taskCenterKeys` do.
5. Test it in `<module>.test.ts`: `vi.stubGlobal('fetch', fetcher)` with the session response first, `vi.resetModules()` and `vi.unstubAllGlobals()` in `afterEach`, then `await import('./module')`, and assert the exact path and body (`web/src/api/lifecycle.test.ts` is an example). Component tests render with `renderToStaticMarkup` and a seeded `QueryClient`; fixtures live in `web/src/testFixtures/`.
6. Mock the call in every walkthrough fixture that renders the surface, then run those walkthroughs.
7. The route must exist, be classified, and be documented on the server side.

### Add a service error code

1. Raise `StorageError("A sentence that says what to do.", "AREA_PROBLEM", status)` (or pass `code=` to `WorkspaceError` and `FilesystemError`): 404 for a missing record, 409 for a conflict or a forbidden state (the default), 422 for invalid input, 403 for an unsafe path or refused access, 413 for a size limit, 503 or 504 when a dependency is unavailable.
2. Register the code with its kind in `ERROR_CODES` (`histopilot/api/error_codes.py`); a code built at runtime is also listed in `COMPUTED` under the module that raises it, and a code whose last raise site you removed leaves the registry. Then regenerate `docs/error-codes.md` with `.venv/bin/python scripts/error_codes.py`, confirm with `--check`, and run `tests/test_error_codes.py`.
3. Never rename or reuse a code. The browser, the CLI and tests branch on codes; for example `web/src/lib/scientificReview.ts` treats `PREVIEW_STALE`, `STALE_PREVIEW` and `REVISION_CONFLICT` as "review again".
4. Preview findings (`{code, message, severity}` in `findings`) are not HTTP errors. An `error` finding blocks the freeze through `canFreeze`.

### Add or change a CLI command

1. Read `docs/cli-contract.md` first. New commands are noun-verb groups in `histopilot/commands/`, registered by `register_commands(app)` from `histopilot/cli.py`; they call the service through `histopilot/client/`, never by building the app in-process.
2. Commands that reach commit or admin routes show a preview, ask for confirmation (`--yes` to skip, `--dry-run` to stop after the preview) and journal their operation ID before sending.
3. Keep `serve`, `doctor`, `runner`, `verify-study`, `restore-study` and `verify-feature-pack` as they are; keep the flat `train-batch`, `training-status`, `pack-features` and `feature-jobs` as deprecated aliases. Keep `histopilot/cli.py` a module with its `__main__` line, and `create_dev_app` importable from it.
4. Test through `tests/support/cli.py` (`Service`), which points the client's transport at an in-process app, so every route, boundary check and error body is real.
5. Document user-facing commands in `docs/user-guide.md`, "Command line", and contract changes in `docs/cli-contract.md`.

## Environments

| Environment | Build | Notes |
| --- | --- | --- |
| `.venv` (development) | `uv sync --locked --extra training --extra imaging` | The service alone needs only `uv sync --locked`. `training` adds Torch, Lightning, TorchMetrics and scikit-learn; `imaging` adds Pillow and OpenSlide; `sdpc` pins OpenSDPC. The `dev` group (pytest, httpx, Ruff, pytest-xdist) is installed by default. |
| `.venv-agent` | `UV_PROJECT_ENVIRONMENT=.venv-agent uv sync --locked --extra agent` | The MCP server, `histopilot agent serve`. Tests that need `mcp` call `pytest.importorskip("mcp")` and skip in `.venv`; run `tests/test_agent_tools.py` and `tests/test_agent_evals.py` with `.venv-agent/bin/python` when you touch `histopilot/agent/`. |
| `.venv-training` | `UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --extra training` | The interpreter of training, refit, predictor-run and attention workers. Found through `HISTOPILOT_TRAINING_PYTHON`, then `<checkout>/.venv-training/bin/python`, then the service's Python; a new one is picked up without a restart. Each worktree needs its own. |
| `web/node_modules` | `npm --prefix web ci` | Node.js 22.12 or newer (`web/package.json` engines). `npm ci` deletes the folder first, and it may be a link shared between worktrees. |
| TRIDENT, SDPC reader | Outside this repository | `HISTOPILOT_TRIDENT_PYTHON`, `HISTOPILOT_TRIDENT_ROOT`, `HISTOPILOT_SDPC_PYTHON`; see `docs/deployment.md` |

- `uv sync` is exact: it removes whatever the requested extras do not include. Preview any sync with `--dry-run`. `uv run` syncs inexactly and never removes packages.
- If `VIRTUAL_ENV` points at another checkout's environment, uv warns and uses this project's `.venv`. Calling `.venv/bin/python` directly avoids the question.
- Never `pip install` into these environments, and never borrow another checkout's environment.

| Variable | Effect |
| --- | --- |
| `HISTOPILOT_STATE_DIR` | Task Center state directory; default `$XDG_STATE_HOME/histopilot`, else `~/.local/state/histopilot` |
| `HISTOPILOT_TASK_CENTER_AUTOSTART=0` | Never start or restart the runner automatically |
| `HISTOPILOT_TRAINING_PYTHON` | Worker interpreter for training and compute |
| `HISTOPILOT_TRIDENT_PYTHON`, `HISTOPILOT_TRIDENT_ROOT` | TRIDENT interpreter and checkout |
| `HISTOPILOT_SDPC_PYTHON`, `HISTOPILOT_SDPC_OPTIMIZATIONS=0` | SDPC reader interpreter; turn off the OpenSDPC extraction fixes |
| `HISTOPILOT_CHROMIUM` | Chromium for the browser walkthroughs |
| `TMPDIR` | Holds the per-user lease registry (`histopilot-training-<uid>`) that every runner of the user shares |

The runner inherits `HISTOPILOT_*` (except the autostart switch and per-task variables), `PATH`, `LD_LIBRARY_PATH` and `TMPDIR` from the process that starts it.

## Safety: live systems, the sandbox and privacy

### Live systems

A live system is anything someone relies on: their workspace (by default `~/.histopilot/workspace`) and project folders, their Task Center state directory, a running service on any port, the runner in the tmux session `hp-runner-<uid>`, and the `histopilot/static/` folder of a checkout that serves.

| Never, outside a sandbox | Because |
| --- | --- |
| `bash serve.sh`, `histopilot serve` | It serves the user's default workspace and state, and restarts the user's runner from your checkout's code |
| `histopilot runner start`, `stop`, `run` | There is one runner per OS user, shared by every checkout |
| `scripts/bundle_web.py` without `--check` | It swaps `histopilot/static/` under a running service |
| `restore-study`, `verify-study`, or any command with `--url` pointing at a service someone uses | They queue real tasks or change real projects |
| `npm --prefix web run dev` while a real service listens on 127.0.0.1:8787 | Vite proxies `/api` there |

Editing a checkout that a live service or runner was started from is also a risk. Extraction, validation, packing, archive and bulk-submission tasks run the code of the checkout whose service queued them, every task starts through the runner's own `taskcenter/wrap.py`, and a running service imports some modules only on first use; all of them read your half-finished files. Tell the owner, and work in a separate worktree when you can.

### Sandbox

For a check that needs a real service, use temporary folders and a private state directory. Real data only with the owner's permission, through a `--data-root` they approve.

```bash
SANDBOX=$(mktemp -d)
mkdir -p "$SANDBOX/tmp" "$SANDBOX/data"
export TMPDIR="$SANDBOX/tmp" HISTOPILOT_STATE_DIR="$SANDBOX/state" HISTOPILOT_TASK_CENTER_AUTOSTART=0
.venv/bin/histopilot serve --workspace "$SANDBOX/workspace" --data-root "$SANDBOX/data" \
  --port "$PORT" --no-runner --no-browser   # PORT: a free port that no one else uses
# Second terminal, same three exports: run queued work in the foreground; Ctrl+C stops it.
.venv/bin/histopilot runner run --state-dir "$SANDBOX/state"
.venv/bin/histopilot runner status --state-dir "$SANDBOX/state"
```

- The private `TMPDIR` keeps the sandbox runner out of the user's lease registry; otherwise the two runners count each other's reservations.
- `histopilot serve` still reads `~/.histopilot/config.toml` when it exists; the explicit `--workspace` and `--data-root` override its storage settings. Pass `--config` to use a sandbox file.
- The service serves the existing `histopilot/static/` bundle and only warns if it is stale.
- Stop both processes and delete `$SANDBOX` when you are done.

### Privacy

The GitHub remote is public.

- Tracked files, tests, fixtures, screenshots and commit messages must not contain study data or results: no study or cohort names or sizes, real slide or patient identifiers, performance numbers from real studies, reader or consensus details, or paths from a real machine.
- `/.local/` and the private docs listed in `.gitignore` stay private: never copy, quote or summarize them into tracked files.
- Use placeholders such as `/path/to/slides`, and the synthetic BLCA demo for examples and screenshots.
- Check `git diff` for these before you hand a change over.
