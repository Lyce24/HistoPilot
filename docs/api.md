# Local HTTP API

The HistoPilot service exposes a JSON API under `/api/v1`. The React UI and the `histopilot` CLI both use it. It serves a single user on a loopback address; see [deployment](deployment.md#local-api-protections). There is no OpenAPI or Swagger endpoint.

This page lists every route group and documents the ones that scripts most often call. For what a route's work means scientifically, see the [user guide](user-guide.md) and [methods](methods.md). For how queued work runs, see the [Task Center](task-center.md).

## Session and conventions

Obtain a token with `GET /api/v1/session`, then send it as the `X-HistoPilot-Token` header on every other request. `GET /health` and `GET /session` do not need it. Tokens belong to one service process; get a new one after a restart. Requests also pass Host, Origin and `Sec-Fetch-Site` checks: the Host must be a loopback name, and cross-site browser requests are refused.

A scoped agent token (`hpt_…`) goes in the same header and reaches one project; see [AI agent access](#ai-agent-access). A service started with `--login` answers `/session` only to a signed-in browser or to a request that carries the sign-in secret as `X-HistoPilot-Login`; otherwise it returns 401 `LOGIN_REQUIRED`.

| Convention | Meaning |
| --- | --- |
| `{identity}` | The project ID. Project routes below are written relative to `/api/v1/projects/{identity}`. |
| Strict request bodies | Command bodies reject unknown fields. |
| Errors | Error responses carry `detail`, a stable `code` and, when a review blocked the request, its `findings`. [Error codes](error-codes.md) lists every code with its kind. |
| `expectedRevision` | Draft updates and previews name the revision they read. A stale revision returns 409; reload, review and retry. |
| `operationId` | Publications, launches and Task Center actions take an operation ID. Retrying the same ID replays the original result; reusing it with a different request is rejected (`OPERATION_CONFLICT`). Creating a project or a draft takes an optional one: a retry returns the record the first request created. |
| `previewHash` | Freeze requests repeat the hash of the preview the user reviewed. If inputs changed since, the freeze is refused. |
| `versionLabel` | Freezing a dataset, feature source, feature bundle, Targets & splits version, cohort or development batch requires `{tag, note?}`: a tag of 1–80 characters and an optional note. A tag names one version of its kind per project, without regard to case. |

## Route map

| Prefix (under `/api/v1`) | Purpose |
| --- | --- |
| `/health`, `/session`, `/version`, `/templates`, `/system`, `/system/compute` | Service status, session token, version and features, the service-owned templates, workspace and runtime readiness, live compute measurements |
| `/filesystem/roots`, `/filesystem/list`, `/filesystem/directories` | Browse allowed roots and create a folder |
| `/projects`, `/projects/open`, `/projects/{identity}` | Create, open, read and configure projects |
| `…/drafts`, `…/imports`, `…/datasets` | Drafts, CSV/XLSX import, frozen datasets, record queries, version labels |
| `…/protocols/explore`, `…/target-splits` | Cohort exploration and Targets & splits |
| `…/configurations` | Frozen configurations of any kind |
| `…/features`, `…/feature-bundles`, `…/feature-packs`, `…/extractions` | Feature attachment, bundles, packing and TRIDENT extraction |
| `…/mil-experiments` | Development batches: preview, freeze, launch, resume, cancel, results, histories, OOF exports |
| `…/model-experiments` | Experiments: inputs, batch plans, freeze, submit, results |
| `…/predictors` | Predictor registry, builds and refits |
| `…/evaluation-cohorts`, `…/drafts/{id}/evaluation-preview`, `…/evaluation-freeze` | Cohorts of Apply models, labeled or unlabeled |
| `…/evaluation-runs` | Runs of predictors on cohorts, single and in batches, with performance breakdowns, case review, prediction summaries and exports |
| `…/clinical-analyses` | Clinical utility reports |
| `…/reference-standards` | Labels attached to a frozen cohort after it was frozen |
| `…/interpretations` | Attention studies, slide gallery, slide and patch images |
| `…/morphology` | Visual QC geometry and slide images |
| `…/datasets/{dataset_id}/slide-reviews` | Per-slide review notes |
| `…/cleanup` | Workspace cleanup: archive, Trash and restore |
| `…/operations` | Study backups (archives) and source folder relinking |
| `/task-center` | The machine-level task queue |
| `/access`, `/tokens`, `/agent-requests`, `…/ai-exposure` | AI agent access: a caller's own access, scoped tokens, agents' requests waiting for approval, and each project's exposure level |

## Service and projects

| Method and path | Behaviour |
| --- | --- |
| `GET /health` | Minimal service status and version. Its legacy `executionEnabled` field is always `false`; runtime readiness is in `GET /system`. |
| `GET /session` | The session token and the service's `scientificCapabilities`. |
| `GET /version` | The package version, `apiVersion`, `contractVersion`, the checkout's `gitRevision`, the workspace schema, the templates' version and the `features` this service offers clients. The CLI compares it with its own. |
| `GET /templates` | The service-owned starting specs, presets, couplings and defaults that the browser builds new records from; the CLI and agent tools read the same values from their own copy of `histopilot/templates.py`. |
| `GET /system` | Workspace, storage and package diagnostics, plus worker and TRIDENT readiness and whether tmux is available to host the Task Center runner. `control.torchImported` reports whether the service process has imported Torch, which it never should. |
| `GET /system/compute` | Current CPU, RAM, GPU and VRAM measurements. |
| `GET /filesystem/roots?purpose=source` | Allowed roots. `purpose=source` (the default) lists only configured data roots; `purpose=storage` adds the application workspace. |
| `GET /filesystem/list?path=…&purpose=…` | A bounded listing inside those roots. Symlinks are resolved before the root check. Long listings report `truncated: true`. |
| `POST /filesystem/directories` | Create a folder inside an allowed root (201). Never replaces an existing entry. |
| `GET /projects` | Recent projects with their availability, the BLCA demo, and `defaultStoragePath`. |
| `POST /projects` | Create a project in an exact new or empty folder (201). An optional `operationId` makes a retry return the same project. |
| `POST /projects/open` | Register an existing project folder by `{path}` and return its summary. |
| `GET /projects/{identity}/workspace` | The project's summary, saved setup and sources. The demo ID `blca-demo-v1` returns the demo. |
| `PATCH /projects/{identity}` | Replace the optional planning `config`. Send `expectedConfig` to detect concurrent edits. |
| `POST /projects/{identity}/sources` | Register a read-only `data`, `slides` or `features` folder (201). Nothing is scanned or copied. |
| `GET /projects/{identity}/storage` | Scientific store schema, journal mode, counts and publication operation states. |

Create a project:

```json
{
  "name": "Example study",
  "storagePath": "/path/to/histopilot-workspace/example-study",
  "slidePath": "/path/to/slides",
  "config": {"task": "binary_classification", "folds": 5, "seed": 42}
}
```

Only `name` and `storagePath` are required. The storage folder must be new or empty, with an existing parent inside the workspace or a data root. Optional `dataPath`, `slidePath` and `featurePath` must be existing folders under data roots. `config` holds planning choices only (`task`, `targetColumn`, `positiveLabel`, `seed`, `folds` 2–10, `encoderId`, `milId`); it creates no target, split or run.

## Drafts, imports and datasets

Project routes, relative to `/api/v1/projects/{identity}`:

| Method and path | Behaviour |
| --- | --- |
| `GET, POST drafts` | List drafts, or save an unvalidated `import` or `experiment` draft (201). An optional `operationId` makes a retry return the same draft. |
| `GET, PATCH drafts/{draft_id}` | Read a draft, or replace its name and payload when `expectedRevision` matches. Frozen drafts cannot change. Experiment records use the typed `model-experiments` routes instead. |
| `POST imports/inspect` | Inspect a table by `{source: {path, sheet?}}` or an uploaded `{filename, contentBase64, sheet?}`: sheets, columns, a SHA-256 fingerprint and per-column examples, distinct and missing counts. |
| `POST imports/{draft_id}/preview` | `{expectedRevision}`: full reconciliation, dictionary, findings, counts and `previewHash`. |
| `POST imports/{draft_id}/freeze` | `{expectedRevision, previewHash, operationId, versionLabel}`: reread the sources and publish an immutable dataset (201). |
| `GET datasets`, `GET datasets/{dataset_id}` | Published datasets and one dataset's manifest and artifact fingerprints. Partial publications never appear. |
| `GET datasets/{dataset_id}/records?offset=0&limit=200` | Canonical records, up to 1,000 per page. |
| `POST datasets/{dataset_id}/query` | Filter, compare and search records with pagination; the same population feeds records, summaries and cross-tabs. |
| `PUT datasets/{dataset_id}/label`, `PUT configurations/{configuration_id}/label` | Rename a frozen version: `{tag, note, expectedRevision}`. Labels never change scientific content or hashes. |

A draft payload is intent only. Saving paths, labels or a readiness flag never reads files or validates a dataset. Dataset publication happens only through import preview and freeze, which reread the source; the server never accepts browser-built frozen records.

### Patient identity

`ImportSpec.patientIdFallback` is `"unresolved"` (default) or `"slide_id"`. The UI asks **Patient ID unresolved, fallback to Slide ID** when included slides lack patient IDs. Main-table IDs and patient crosswalks are resolved first; only missing patient IDs are then filled with the slide ID. Each record keeps `patientIdSource`:

| Value | Meaning |
| --- | --- |
| `source` | Patient ID supplied in the main table |
| `crosswalk` | Patient ID linked through a slide-to-patient table |
| `unresolved` | No patient ID, and fallback not chosen |
| `slide_fallback` | Fallback chosen: the patient ID equals the slide ID, one group per slide |

Import summaries report `verifiedPatientCount`, `fallbackSlideCount` and `unlinkedSlideCount`; `mappedPatientCount` counts every non-null grouping ID, including fallbacks. A fallback that collides with a supplied patient ID blocks with `PATIENT_ID_FALLBACK_COLLISION`, and `PATIENT_ID_SLIDE_FALLBACK` warns that slides of one unknown patient may land in different sets. Patient-level scoring and patient bootstrap intervals refuse fallback groups.

## Targets and splits

A target/split draft is an `experiment` draft whose payload is `{type: "target-split", spec: TargetSplitSpec}` (`histopilot/schemas/target_splits.py`). The spec holds `datasetId`, `splitUnit` (`slide` or `patient`), `eligibility`, `split` (method `random`, `rules` or `imported` with its settings), `target` and an optional `testTarget` (`null` for pure inference). See [split units and grouping](methods.md#split-units-and-grouping).

| Method and path | Behaviour |
| --- | --- |
| `POST protocols/explore` | Read-only eligibility and raw target counts for a frozen dataset, and live development partition counts for a version-4 split. Accepts an unfinished spec. |
| `POST target-splits/partition-preview` | Live training/testing counts and distributions for an unsaved spec. Target problems are reported in place. |
| `POST target-splits/{draft_id}/preview` | `{expectedRevision}`: exact memberships, distributions, findings, `canFreeze` and `previewHash`. |
| `POST target-splits/{draft_id}/freeze` | `{expectedRevision, previewHash, operationId, versionLabel}`: publish the version (201), then derive its testing cohort. The response carries `testCohort`; if the cohort could not be derived, it adds `testCohortError` instead of failing. |
| `GET target-splits/{configuration_id}` | The frozen version, `evaluationCohortId` and `testCohort: {required, id, state}`. `state` is `failed`, with an `error`, when the last attempt to derive the cohort (at freeze or a retry) failed. Reading never creates the cohort. |
| `POST target-splits/{configuration_id}/test-cohort` | Derive the testing set's cohort, idempotently: labeled, or unlabeled without a testing target. A success clears a saved failure; a failure replaces it. |
| `GET configurations?kind=…`, `GET configurations/{configuration_id}` | Frozen configurations, filtered by kind (`target-split`, `evaluation-cohort`, `feature`, `feature-bundle`, `experiment-setup` or the historical `protocol`), and one checksum-verified envelope. |

Historical protocol records from before Targets & splits (split versions 1–3) stay readable, but new work accepts only version 4. `protocols/explore` still accepts the old `rules` and `splitMode` fields; `rules` is ignored, and a non-empty split of another version returns eligibility and target counts with `INVALID_STRATEGY_CONFIG` and `partitions: null`.

## Features and extraction

| Method and path | Behaviour |
| --- | --- |
| `POST features/preview`, `POST features/freeze` | Attach existing HDF5 features: exact slide matching and a header/coverage report, then publish the binding (with `versionLabel`). `datasetId` is optional for a dataset-independent store. |
| `GET features/{feature_id}/validation`, `GET, PUT features/{feature_id}/pack-selection` | Validation evidence and the default pack for a feature source. |
| `GET feature-bundles`, `POST feature-bundles/preview`, `POST feature-bundles/freeze`, `GET feature-bundles/{bundle_id}` | Verified feature bundles. |
| `GET, POST feature-packs`, `POST feature-packs/preview`, `GET feature-packs/{job_id}`, `POST feature-packs/{job_id}/cancel`, `GET feature-packs/artifacts/{artifact_id}` | Validation and packing jobs, run as Task Center tasks, and their pack artifacts. |
| `GET extractions/catalog` | Encoders, options, the default output folder and the TRIDENT runtime, including `searchedRoots` and `otherCheckouts` when no checkout is found. |
| `POST extractions/preview` | The resolved command and slide list, findings, `estimatedBytes` and `availableBytes`. Errors such as `SLIDE_READER_UNAVAILABLE` or `INSUFFICIENT_SPACE` block `canRun`; warnings such as `LOW_DISK_SPACE` or `SINGLE_GPU_TASK` do not. |
| `POST extractions` | Submit a previewed extraction (201) as an extraction task plus a dependent validation task. |
| `GET extractions`, `GET extractions/{job_id}` | Jobs with `executor`, `task`, `tasks.extraction`, `tasks.validation` and `ownerKey`. |
| `POST extractions/{job_id}/cancel`, `POST extractions/{job_id}/resume` | Cancel, or requeue TRIDENT on the same output. Finished slides are skipped. Jobs from before the Task Center return 409 `CREATED_BEFORE_TASK_CENTER`; continue those through a new preview on the same output folder. |

## Experiments and development batches

| Method and path | Behaviour |
| --- | --- |
| `GET model-experiments?state=active&summary=…`, `POST model-experiments` | List (`state` is `active`, `archived`, `trashed` or `all`) or create an experiment (201). A new experiment states `setupVersion`: `1` for an experiment design, `null` for the legacy form; omitting it returns 422 `SETUP_VERSION_REQUIRED`. |
| `GET model-experiments/headlines` | One seed-mean headline per batch, for the list page. |
| `GET, PATCH model-experiments/{experiment_id}` | Read or update an experiment's editable fields, including its whole list of batch plans under `expectedRevision`. |
| `POST model-experiments/{experiment_id}/setup-inputs`, `/freeze-setup` | Save and check setup inputs, then freeze the setup (201). |
| `POST model-experiments/{experiment_id}/submit` | Start the frozen experiment (202): enqueue its fold, collection and predictor tasks. |
| `GET model-experiments/{experiment_id}/results` | The Results summary: per fold, per seed, seed average with intervals, seed ensemble and batch comparisons. See [methods](methods.md#cross-validated-results). |
| `POST model-experiments/{experiment_id}/predictors/resume`, `/predictors/cancel` | Resume or cancel automatic predictor work (202). |
| `POST mil-experiments/preview`, `GET mil-experiments/runtime` | Model input compatibility and the training runtime probe. |
| `GET mil-experiments/clinical-fields?protocolId=` | Dataset columns a batch may use as clinical inputs: owner, type, suggested kind, and why a field is refused (the target, the testing target, or a field that defines the split). |
| `POST mil-experiments/batches/preview`, `/batches/freeze`, `GET mil-experiments/batches` | Development batches. A batch may declare a `comparison` (reference arm and primary metric); review blocks arms that differ in more than the model and its inputs. |
| `GET mil-experiments/batches/{batch_id}/execution`, `/results`, `/runs/{run_id}/history`, `/resources/history` | Progress, results, one run's epoch history and recorded resource history. |
| `GET mil-experiments/batches/{batch_id}/oof/{candidate_id}/{training_seed}/{split_seed}/{unit}.csv` | Out-of-fold predictions of one configuration and seed pair, per slide or patient. |
| `POST mil-experiments/batches/{batch_id}/launch`, `/resume`, `/cancel` | Batch controls (202). An experiment's batches are launched by `submit`; use `resume` to continue one. |

## Predictors and runs

The routes keep their earlier names: an `evaluation-cohort` is a cohort of Apply models and an `evaluation-run` is a run of one predictor on one cohort, labeled or not.

| Method and path | Behaviour |
| --- | --- |
| `GET predictors`, `GET predictors/choices`, `GET predictors/{predictor_id}` | Frozen predictors and the groups that can produce one. |
| `POST predictors/preview`, `POST predictors/freeze`, `POST predictors/builds/preview`, `POST predictors/builds`, `GET predictors/builds/{operation_id}` | Build ensemble predictors, singly or in bulk. |
| `GET predictors/seed-ensembles?experiment_id=`, `POST predictors/seed-ensembles/preview`, `POST predictors/seed-ensembles` | Configurations whose seed groups can pool into one seed-ensemble predictor, and its review and publication (`experimentId`, `batchId`, `candidateId`, `name`; freezing adds `previewHash` and `operationId`). Nothing trains. |
| `GET, POST predictors/refits`, `GET predictors/refits/{refit_id}/execution`, `POST …/launch`, `…/resume`, `…/cancel`, `…/publish` | Refit plans, their execution and publication. |
| `GET evaluation-cohorts`, `GET evaluation-cohorts/{configuration_id}` | Cohorts. Created through `drafts/{draft_id}/evaluation-preview` and `evaluation-freeze`. |
| `GET, POST evaluation-runs`, `POST evaluation-runs/preview`, `GET evaluation-runs/{evaluation_id}` | Runs of one predictor on one cohort. Runs on labeled cohorts saved now carry `labelSource: "cohort"`: their job never receives labels, and a completed run's `execution.result.metrics` is scored by the service from the cohort's frozen labels (`metricsError` says why when it cannot be). |
| `GET evaluation-runs/{evaluation_id}/execution`, `POST …/launch`, `…/resume`, `…/cancel`, `GET …/artifacts/{filename}` | Run control and artifacts. Worker files are served byte for byte. For label-blind scored runs, `metrics.json` and the slide and patient prediction tables come from the service, with each row's label, development-patient flag and `scored` state. |
| `GET evaluation-runs/bulk`, `POST evaluation-runs/bulk/preview`, `POST evaluation-runs/bulk`, `GET evaluation-runs/bulk/{batch_id}`, `POST …/cancel` | Apply many predictors to one cohort as a batch, one run each. The reviewed predictor list is fixed at submission. `predictorIds` need `scope: "selected"` stated, and a request's `inference` object must hold every setting (`INFERENCE_SETTINGS_INCOMPLETE`); `"predictor"` applies each predictor's own patient aggregation and threshold. |
| `GET reference-standards?cohort_id=&include_inactive=`, `POST reference-standards/preview`, `POST reference-standards`, `GET reference-standards/{reference_id}` | Reference standards. Selection: `cohortId`, `name`, `datasetIds` (matched to the cohort by slide ID), `field` (a data dictionary key of every dataset), `classes` and `labels` (source value → class). The preview lists the column's values over the cohort with their counts, and findings: `REFERENCE_FIELD_UNKNOWN`, `REFERENCE_DUPLICATE_SLIDES` and `REFERENCE_NO_LABELS` block saving; unmatched, unmapped and conflicting-patient counts are warnings. Save takes the preview's `previewHash` and an `operationId`. Lists omit per-slide memberships. |
| `POST evaluation-runs/{evaluation_id}/scores` | A completed run's metrics. Body: `referenceId`, or null for the cohort's own labels. A reference of another cohort returns `REFERENCE_COHORT_MISMATCH` and one with other classes `REFERENCE_CLASSES_MISMATCH` (409). |
| `GET evaluation-runs/{evaluation_id}/reference-standards/{reference_id}/{filename}` | `metrics.json`, `slide-predictions.csv` or `patient-predictions.csv` scored against a reference, in the form of the run's own downloads. |
| `POST evaluation-runs/{evaluation_id}/recalibration` | Calibration of a scored run as predicted and after recalibration. Body: `unit` and optional `referenceId`. The map (Platt for binary targets, temperature for multiclass) is fitted on the predictor's verified out-of-fold development predictions. Runs without labels return `RECALIBRATION_REQUIRES_LABELS`, and predictors without development predictions `RECALIBRATION_UNAVAILABLE` (409). |
| `POST evaluation-runs/{evaluation_id}/agreement` | Agreement between the run's decisions and every label source of its cohort, pair by pair. Body: `unit`. Returns the sources (a source that cannot label the run gives its `reason`), and for each pair the count, percent agreement, Cohen's κ, linear weighted κ (three or more classes), disagreements and the cross-tabulation. |
| `POST evaluation-runs/{evaluation_id}/performance/breakdown` | Performance by subgroup of a scored run. Body: `unit` (`selected`, `slide` or `patient`), `attribute`, a key of the frozen data dictionary, and optional `referenceId`. Returns the run's overall metrics and the same metrics within each value, over exactly the records the run scores; the 50 largest groups are listed and the rest pooled as `other`. Runs on unlabeled cohorts need a `referenceId`, otherwise `PERFORMANCE_REQUIRES_LABELS` (409); an unknown attribute returns `PERFORMANCE_ATTRIBUTE_INVALID` (422). |
| `POST evaluation-runs/compare` | Paired comparison of two scored runs. Runs on unlabeled cohorts return `COMPARISON_REQUIRES_LABELS`. |
| `POST evaluation-runs/{evaluation_id}/cases/query`, `…/cases/export` | Case review and its CSV export. An optional `referenceId` takes the labels from a reference standard; the export's label column then names it. |

An unlabeled cohort has `"purpose": "inference"` and `"target": null`. Runs on it use the same routes; they carry `purpose: inference`, read no labels and compute no metrics. Their artifacts are `predictions.json`, `summary.json`, `slide-predictions.csv` and `patient-predictions.csv`; there is no `metrics.json`.

| Method and path | Behaviour |
| --- | --- |
| `POST evaluation-runs/{evaluation_id}/inference/summary` | Label-free summary of any run's predictions. Body: `unit` (`selected`, `slide` or `patient`), optional `attribute` and optional `comparisonId` (another run on the same cohort). Returns predicted-class counts, confidence and margin histograms, a binary threshold sweep, fold-member agreement, the development-patient split, an attribute cross-tab and run agreement with Cohen's κ. |
| `POST evaluation-runs/{evaluation_id}/inference/export` | CSV with one row per slide or patient: predicted class, probabilities, confidence, margin, member agreement, `Development_patient`, the chosen frozen attributes (`null` for all, `[]` for none) and the predictions SHA-256. |
| `POST evaluation-runs/{evaluation_id}/attention` | Queue attention maps for 1–32 cohort slides through the run's frozen features (202). Requires `operationId`; completed or running studies are reused. |

`cases/query` accepts `sort` (`confidence_desc`, `confidence_asc`, `margin_asc`, `agreement_asc`), `minConfidence`, `maxConfidence`, `maxMargin`, `memberDisagreement`, `developmentPatients` (`all`, `shared`, `new`), outcome and class filters, attribute filters, search and paging of up to 100 rows.

## Clinical utility and interpretation

| Method and path | Behaviour |
| --- | --- |
| `GET clinical-analyses`, `POST clinical-analyses/preview`, `POST clinical-analyses`, `GET clinical-analyses/{analysis_id}`, `GET …/artifacts/{filename}` | Clinical utility reports from completed scored runs. An optional `referenceId` takes the outcomes from a reference standard of the run's cohort. |
| `GET interpretations`, `GET interpretations/sources`, `GET interpretations/datasets` | Attention studies and their selectable inputs. |
| `POST interpretations/gallery`, `GET interpretations/gallery/thumbnail` | Browse a dataset's slide folder. |
| `POST interpretations/visualize` | Prepare attention for selected slides (202), reusing completed and running studies. |
| `POST interpretations/preview`, `POST interpretations`, `GET interpretations/{interpretation_id}`, `…/execution`, `POST …/launch`, `…/resume`, `…/cancel`, `GET …/artifacts/{filename}` | One attention study and its execution. |
| `POST interpretations/slide-inspection`, `GET interpretations/{interpretation_id}/slides/{slide_id}/thumbnail`, `…/region`, `…/attention`, `…/attention/top`, `…/patches/{patch_index}/image` | Slide views, attention arrays, top patches and patch crops. |

## Slide viewing

| Method and path | Behaviour |
| --- | --- |
| `GET morphology/quality?datasetId=…&slideId=…` | Level-0 geometry and a `sourceFingerprint` (64 lowercase hex characters) of the resolved slide file. Optional `featureBundleId` adds patch coverage. |
| `GET morphology/image?datasetId=…&slideId=…&sourceFingerprint=…&max_size=1024&x=…&y=…&width=…&height=…` | A lossless PNG. Coordinates are level-0 pixels; give all four or none (for an overview). `max_size` is 64–2048, default 1024. |
| `GET morphology/patch-region?…&featureBundleId=…&patchIndex=…` | One feature patch's exact level-0 bounds, clipped to the slide. |
| `GET morphology/patch?…&featureBundleId=…&patchIndex=…` | That patch as a PNG, at most 512 pixels on its longer side. |
| `GET morphology/slides`, `POST morphology/index`, `POST morphology/neighbors` | Slide listing, a feature index and nearest-neighbour search. |
| `GET, PUT datasets/{dataset_id}/slide-reviews/{slide_id}` | Per-slide review notes. |

Send the fingerprint from `quality` with every image request. If the file changed, the route returns 409 `MORPHOLOGY_SLIDE_CHANGED`, so images from different file versions never mix. Native readers run in at most two isolated child processes, with a 20-second deadline per operation. Reader errors are `SLIDE_VIEWER_UNAVAILABLE` or `SLIDE_READER_BUSY` (503), `SLIDE_READER_TIMEOUT` (504) and `SLIDE_READER_FAILED` (422). An SDPC file is never decoded as a full raster.

## Cleanup and study backups

| Method and path | Behaviour |
| --- | --- |
| `GET cleanup`, `POST cleanup/preview`, `POST cleanup/apply`, `POST cleanup/cancel` | The cleanup inventory; preview an archive, Trash or restore selection with its dependencies and review hash; apply it; or cancel the jobs of a record (202). |
| `GET operations`, `GET operations/sources`, `POST operations/sources/relink` | Operations inventory, registered source folders and their availability, and relinking a moved folder. |
| `GET, POST operations/archives`, `GET operations/archives/{job_id}`, `POST …/cancel`, `POST …/retry` | Export, verify and restore study archives as Task Center tasks. Retry requeues the same task. |

## Task Center

These routes are machine-level, not project-scoped: `/api/v1/task-center/…`. Reads use only the task store and never take a project lock. Every action body requires `{operationId}`, so a repeated request is applied once. See [Task Center](task-center.md) for the states and rules.

| Method and path | Behaviour |
| --- | --- |
| `GET summary` | Runner state, capacity, task counts, time estimate, foreign leases and failures of the last 24 hours. |
| `GET snapshot` | Summary, running tasks and live owners in one read. The page polls this. |
| `GET rollup` | One stage's run status. Scope with `owner`, `ownerKind` + `ownerId`, `recordKind` + `recordId`, `recordIds` or `batchIds` (comma-separated), `project` (optionally with `kinds`), or nothing for the whole machine. Returns `state` (`not-started`, `queued`, `running`, `held`, `stopping`, `attention`, `runner-stopped`, `completed`, `cancelled`), counts, progress, queue position, waiting reason, estimate, last failure and a deep link. |
| `GET history` | Finished tasks grouped by owner, newest first: `project`, `kind`, `state`, `limit` (1–200, default 25), `offset`. |
| `GET tasks` | Tasks filtered by `state`, `owner`, `project`, `kind`; `limit` 1–2000 (default 200); `offset` adds paging and `hasMore`. |
| `GET tasks/{task_id}` | One task: command, working folder, filtered environment, paths, progress, measurements, labels, dependencies, attempts, events, log tail and a plain-language `failure`. |
| `GET tasks/{task_id}/log` | The whole log, streamed; `download=true` returns a file. `offset` and `limit` (at most 4 MiB) read one part, with the log's size in `X-HistoPilot-Log-Size` and the next offset in `X-HistoPilot-Log-Next-Offset`. |
| `POST tasks/{task_id}/cancel`, `POST tasks/{task_id}/retry` | Cancel or retry one task. |
| `GET owners?scope=live`, `GET owners/{key}` | Owners (experiments, batches and records), live or all, and one owner's counts, position, estimate, waiting reason and allowed actions. |
| `POST owners/{key}/{action}` | `hold`, `release`, `stop` (stop and hold), `cancel`, `retry`, or `move` with `position` `top`, `up`, `down` or `bottom`. |
| `GET capacity`, `PUT capacity` | Settings, effective limits and the parallelism suggestion. `PUT` accepts `parallelGpuTasks` (1–16), per-GPU `gpuSlots`, `cpuTaskSlots` (1–128), `paused`, `autoResume` and `defaults` (`cpuThreadsPerRun` 1–32, `dataLoaderWorkers` 0–16). |
| `POST runner/start`, `POST runner/restart` | Start the runner, or restart it. |

## AI agent access

These routes are machine-level unless noted. See [AI agents](agents.md) for what the levels, scopes and approvals mean.

| Method and path | Behaviour |
| --- | --- |
| `GET access` | The caller's own access: `kind: "session"` with every scope, or for a scoped token its ID, name, project, scopes, the project's exposure level and the token's expiry. |
| `GET projects/{identity}/ai-exposure`, `PUT projects/{identity}/ai-exposure` | A project's level, `none`, `metadata` or `full`, and a change to it: `{level, expectedLevel}`; a stale `expectedLevel` returns 409 `EXPOSURE_CONFLICT`. |
| `GET tokens?project=`, `POST tokens`, `POST tokens/{token_id}/revoke` | Scoped tokens, never their secrets; create one with `{projectId, scopes, name, days}` (201, the secret appears once); revoke one. A project at `none` takes no token (`TOKEN_EXPOSURE_REQUIRED`). |
| `GET agent-requests?project=`, `GET agent-requests/{request_id}` | Changes agents asked for, and their state: `pending`, `approving` while a person replays one, then `approved`, `failed` or `unknown`; `declined`; or `expired` when still pending after 24 hours. |
| `POST agent-requests/{request_id}/claim` | Take a pending request for one replay: `pending` becomes `approving`. A request that is not pending returns 409 `AGENT_REQUEST_SETTLED`, so two people approving at once replay it only once. |
| `POST agent-requests/{request_id}/resolve` | Record how it ended: `{outcome, status?}`. `declined` applies to a pending request. `approved` (the replay succeeded), `failed` (the service refused the replay, so nothing changed) and `unknown` (no answer came) apply to a claimed one; otherwise 409 `AGENT_REQUEST_NOT_CLAIMED`. `status` records the replay's HTTP status. A settled request returns 409 `AGENT_REQUEST_SETTLED`. |

A scoped token reaches read routes, and with `preview` scope preview routes, of its own project only; the Task Center's reads are narrowed to its project's work. Every route's class is in `histopilot/api/route_classes.py`. Refusals are 403 `SCOPE_MISSING`, `PROJECT_OUT_OF_SCOPE`, `EXPOSURE_NONE` or `EXPOSURE_METADATA`; an unknown, expired or revoked token gets 401 `TOKEN_INVALID`. A commit from a token with `commit` scope runs nothing: it returns 202 `CONFIRMATION_PENDING` with a `requestId`. On a `metadata` project, JSON answers are redacted and files, images, logs and single cases are refused. Only the session reaches tokens, exposure levels and decisions on requests.
