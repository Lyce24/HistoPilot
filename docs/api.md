# Local API foundation

The implemented API serves **folder-backed experiment setup, project-local scientific storage, and an explicit synthetic demo**. CSV/XLSX import, identifier/attribute mapping, immutable publication, grouped target/split validation and existing-feature header attachment are implemented. Full tensor validation, GPU jobs and WSI tiles remain future work. React and the CLI share this service and its canonical model-run experiment schema.

All paths below are under `/api/v1`. The service validates Host/Origin and browser fetch context. Obtain a token from `GET /session`, then send it as `X-HistoPilot-Token` on protected requests. The health and session endpoints do not require that header. Tokens are local to the running service process; clients reacquire a token after a restart.

| Method / path | Behavior |
| --- | --- |
| `GET /health` | Minimal service/version status; `executionEnabled: false`. |
| `GET /session` | Establish a local-session token. |
| `GET /projects` | Recent experiment summaries plus `synthetic-v1`; returns `defaultStoragePath` and availability for each folder. |
| `POST /projects` | Create named experiment setup in an exact new or empty server folder; HTTP 201. |
| `POST /projects/open` | Validate and register an existing folder's `histopilot-project.json`; return its summary. |
| `GET /projects/{id}/workspace` | Selected experiment's latest dataset counts, scientific summary, saved setup/sources, and model registry; `synthetic-v1` returns the demo. |
| `PATCH /projects/{id}` | Replace a local experiment's optional configuration and persist its descriptor. |
| `POST /projects/{id}/sources` | Add a read-only `data`, `slides`, or `features` directory reference; HTTP 201. |
| `GET /projects/{id}/storage` | Project-local schema/journal mode, draft/dataset counts, and publication operation states. |
| `POST /projects/{id}/drafts` | Save an unvalidated import/experiment draft; HTTP 201. |
| `GET /projects/{id}/drafts` | Project-local drafts under `drafts`. |
| `GET /projects/{id}/drafts/{draftId}` | Read one draft and its current revision. |
| `PATCH /projects/{id}/drafts/{draftId}` | Replace draft name/payload only if required `expectedRevision` matches; stale or frozen drafts return 409. |
| `GET /projects/{id}/datasets` | Published internal snapshots under `datasets`; partial publication is never listed as a dataset. |
| `GET /projects/{id}/datasets/{datasetId}` | Read a published snapshot's manifest and artifact fingerprints. |
| `GET /workspace` | Legacy synthetic seed, saved demo cohorts/drafts, and source-directory references. |
| `GET /workspace/export` | Downloadable JSON snapshot with schema version and synthetic/non-executable markers. |
| `GET /models/encoders` | Server registry under `encoders`; entries describe planned encoder choices. |
| `GET /models/mil` | Server registry under `milModels`; no availability guarantee. |
| `POST /cohorts` | Validate filters and persist canonical synthetic membership. |
| `POST /experiments` | Validate selections against a stored cohort and registry; persist one draft per unique model pair. |
| `GET /experiments/{id}/manifest` | Return that draft's canonical `ExperimentSpec`. |
| `DELETE /experiments/{id}` | Remove an unexecuted draft only. |
| `GET /filesystem/roots?purpose=source` | Configured source roots by default; `purpose=storage` includes the application workspace and data roots. |
| `GET /filesystem/list?path=...&purpose=source` | Bounded listing within the corresponding resolved roots; accepts `purpose=storage` for experiment folders. |
| `POST /sources` | Validate and store a directory reference without importing/copying data. |
| `GET /system` | Workspace/storage/control-service context and package metadata diagnostics. |
| `GET /jobs` | Empty job list and `executionEnabled: false`. |
| `POST /jobs` | HTTP 501; execution is not implemented and no job is submitted. |
| `GET /jobs/events` | Reserved SSE progress endpoint; HTTP 501 until worker events are implemented. |

Unknown fields on command request schemas are rejected. This is a narrow command surface: there is no whole-workspace replacement endpoint and no arbitrary file-content or WSI upload route. Interactive OpenAPI/Swagger endpoints are not exposed by this local service.

## Create and reopen experiment setup

The UI's top-level **experiment** uses `/projects`; `/experiments` continues to mean individual model-run drafts in the synthetic workflow. Creating setup opens the normal Overview/Dataset navigation without populating it with demo records.

```json
{
  "name": "Bladder",
  "storagePath": "/path/to/histopilot-workspace/bladder",
  "slidePath": "/mnt/pathology/bladder/slides",
  "config": {
    "task": "binary_classification",
    "targetColumn": "Binary WHO 2022",
    "positiveLabel": "1",
    "seed": 42,
    "folds": 5
  }
}
```

Only `name` and `storagePath` are required. The exact destination must be empty or new with an existing parent, within the application workspace or a configured data root. Optional `dataPath`, `slidePath`, and `featurePath` are existing directories under source roots. Optional `description` and `config` may be omitted. Configuration accepts binary/multiclass `task`, `targetColumn`, `positiveLabel`, integer `seed` (0–4294967295), `folds` (2–10), and registered `encoderId`/`milId`. These are planning choices; they do not create a scientific target, split, or execution plan.

Creation writes `histopilot-project.json` in the selected folder and adds a recent entry to SQLite. Reopen with `POST /projects/open` and `{"path":"/path/to/histopilot-workspace/bladder"}`, then load the returned ID through `/projects/{id}/workspace`. The descriptor is authoritative; missing or invalid folders produce errors, and the recent list retains unavailable entries. The UI's `?experiment=<id>#overview` link restores registered context after refresh.

Use `PATCH /projects/{id}` with `{"config": {...}}` to replace the full optional configuration; `{ "config": {} }` clears it. The **MIL experiments** page exposes these settings. Add later source folders with `POST /projects/{id}/sources`, for example `{"path":"/mnt/pathology/bladder/features","role":"features"}`. Saved sources remain `readOnly: true` and `importStatus: "not-imported"`.

## Project-local scientific storage

Creating or opening a local project initializes `histopilot-state.sqlite` in the chosen folder and reconciles interrupted snapshot publication. Existing version-1 setup descriptors remain unchanged when opened. The central registry is not the authority for these drafts or snapshots; a new registry can reopen the same folder and read them. The workspace response includes a `scientificStorage` summary, and the Dataset UI reads project-local versions and import drafts.

Create an unvalidated draft with `POST /projects/{id}/drafts`:

```json
{
  "kind": "import",
  "name": "Bladder mapping draft",
  "payload": {"notes": "Map De ID to Slide_ID; patient mapping remains unresolved"}
}
```

The response includes a server-generated `id`, owning `projectId`, `revision`, `status`, timestamps and saved payload. Supported kinds are `import` and `experiment`. Payloads are draft intent only: saving paths, label values, or a claimed readiness flag never reads files, validates a dataset, creates a split, or authorizes execution.

Update with `PATCH /projects/{id}/drafts/{draftId}`:

```json
{
  "expectedRevision": 1,
  "name": "Bladder mapping draft",
  "payload": {"notes": "Revised mapping choices"}
}
```

Name/payload replacement is atomic and increments the revision. The client must reload after a conflict, review the latest value and intentionally retry. Missing or non-integer `expectedRevision` is rejected; there is no implicit last-write-wins path for scientific drafts. Existing setup configuration remains a separate descriptor command without draft-revision semantics.

Storage errors return `detail` plus a stable `code`. They distinguish busy project writers, stale revisions, incompatible storage, unsafe paths and artifact problems. Project-scoped draft/dataset lookup never searches another project. The synthetic demo has no local scientific store.

Dataset publication uses an internal storage boundary reached through the validated import workflow. The browser sends saved draft revision and preview hash; the server rereads sources and derives records before publishing. It never accepts browser-authored frozen records. Unresolved patient identity can be retained at import. Protocol assignment requires supplied patient identities or the explicit `patientIdFallback: "slide_id"` import choice described below; fallback is recorded separately and does not verify patient independence.

## Save a synthetic cohort

```json
{
  "datasetId": "crc-demo-v1",
  "specimenType": "Primary",
  "msi": "MSS",
  "braf": "WT"
}
```

The server validates the dataset and categorical values, computes patient/slide membership from stored records, and rejects an empty selection. `Any` is the no-filter value; `all` is normalized to `Any`. Repeating the same canonical request returns the same snapshot identity. It does not generate a new scientific split; the fixed demo split remains illustrative.

The response contains a server-assigned `id`, filters, `patientIds`, `slideIds`, dataset/split references, and creation metadata. Use that returned ID when creating an experiment.

## Save experiment drafts

```json
{
  "cohortId": "<returned-cohort-id>",
  "pairs": ["uni2:abmil"],
  "seeds": [42, 43, 44],
  "folds": 5,
  "aggregation": "mean"
}
```

The server verifies the stored cohort, encoder/MIL names, registered example feature set, seed values, fold bounds, and aggregation. It canonicalizes duplicate pair/seed selections and returns `{ "drafts": [...] }`. Each draft pins its cohort snapshot and contains a validated manifest. Neither model compatibility with real weights nor real feature coverage is established by these synthetic checks.

The canonical schema is [`ExperimentSpec`](../histopilot/contracts/experiment.py), exported as [JSON Schema](../examples/experiment.schema.json). It represents one encoder/MIL pair and explicit dataset/cohort/split/feature IDs, binary target convention, seeds, folds, and aggregation. GUI draft creation and `histopilot run ... --validate-only` both use it. It is a first version with shape validation; real artifact resolution and execution planning remain future work.

## Register a server directory

```json
{"path": "/mnt/pathology/crc"}
```

The absolute directory must exist and resolve within an explicitly configured root, including any symlinks. The result is a read-only directory reference marked `not-imported`. Listings are bounded and may report `truncated: true`; file content is never returned by this endpoint. With no data roots configured, the picker has no directories to browse.

Future additions should preserve this intent-based boundary: real imports, splits, preflight, execution plans, worker progress/SSE, and artifact/tile reads must validate their own domain references and permissions rather than accept browser-authored scientific state.

## Real import, exploration and protocols (P0.1/P0.2)

All routes below use `/projects/{id}` and the same session boundary. A saved import payload is `{type: "dataset-import", spec: ImportSpec}` with draft kind `import`; a protocol payload is `{type: "analysis-protocol", spec: ProtocolSpec}` with kind `experiment`. The strict intent schemas live in `histopilot/schemas/imports.py`, `protocols.py` and `features.py`.

| Method / relative path | Behavior |
| --- | --- |
| `POST /imports/inspect` | `{source: {path, sheet?}}` or `{source: {filename, contentBase64, sheet?}}`; column/sheet preview, SHA-256 source fingerprint and `columnSummaries` per header: up to four most frequent raw `examples`, `distinctCount` and `missingCount` across the complete table. Frequency ties follow source encounter order. Only null/empty cells count as missing here; declared missing tokens are applied during import preview. |
| `POST /imports/{draftId}/preview` | `{expectedRevision}`; complete bounded reconciliation, dictionary, findings, counts and `previewHash`. |
| `POST /imports/{draftId}/freeze` | `{expectedRevision, previewHash, operationId}`; reread/validate and publish immutable dataset; HTTP 201. |
| `GET /datasets/{datasetId}/records?offset=0&limit=200` | Canonical records page with `total`, `offset`, `limit`. |
| `POST /datasets/{datasetId}/query` | Field/compare/search/category filters plus pagination; identical server-side population for records, summary, distributions and cross-tab. |
| `POST /features/preview` | `{datasetId,path,encoderId?,fileSuffix?,idSuffix?,recursive?}`; exact matching, HDF5 header/coverage report. |
| `POST /features/freeze` | Same feature intent plus `{previewHash,operationId}`; recheck and publish immutable header binding. |
| `POST /protocols/explore` | `{datasetId,targetField?,eligibility?,rules?,splitMode?,split?}`; read-only full-population target/cohort counts, direct and group-expanded rule matches, sample rows and findings. The optional partial `split` object identifies version-2 strategy behavior without requiring complete controls. No saved draft, label mapping or feature binding required. |
| `POST /protocols/{draftId}/preview` | `{expectedRevision}`; targets, patient identities, filters, exact seed/fold memberships and feasibility findings. |
| `POST /protocols/{draftId}/freeze` | `{expectedRevision,previewHash,operationId}`; regenerate/validate and atomically freeze protocol and draft. |
| `GET /configurations?kind=protocol` | Frozen protocol configurations; `kind=feature` selects feature bindings. |
| `GET /configurations/{configurationId}` | Checksum-verified immutable configuration envelope. |
| `GET /protocols/{configurationId}/preflight` | Verify dataset and selected feature file availability/headers against the frozen binding; report pending full validation, `executionReady:false`. |
| `POST /jobs` (project-scoped) | `{protocolId}`; 422 on input blockers, otherwise 501 because execution is unavailable. No job is submitted. |

Configuration envelopes contain `id`, `projectId`, `contentHash`, `manifest`, `createdAt`. The manifest distinguishes `kind:protocol` from `kind:feature` and pins `datasetId`. Protocol manifests retain the full spec, algorithm version, exact memberships, counts and findings. Publication retries use the same operation ID; changed content with that ID is rejected. See [implementation notes](p0-import-protocol-implementation.md) for limits and semantics.

The frozen input-preflight response declares `scope: "protocol-and-feature-headers"`, `protocolReady`, `headerInputsReady`, and `fullFeatureValidationComplete: false`. `scientificReady` is the compatibility flag for that stated limited scope; it never overrides `executionReady: false` or substitutes for full feature validation.

### Explicit patient-ID fallback

`ImportSpec.patientIdFallback` accepts `"unresolved"` (default) or `"slide_id"`. The UI asks **Patient ID unresolved, fallback to Slide ID** when included slides have missing patient IDs; Continue records the latter choice, and Go back returns to mapping. API clients opt in by saving `patientIdFallback: "slide_id"` in the import draft spec before preview and freeze. A saved opt-in is part of the scientific mapping and immutable dataset provenance.

The importer resolves main-table IDs and patient crosswalk/attribute joins first. It then fills only missing `patientId` values with the row's `slideId`. Fallback identifiers are never used to join a patient attribute table. New canonical records include `patientIdSource`:

| Value | Meaning |
| --- | --- |
| `source` | Patient ID supplied in the main table. |
| `crosswalk` | Patient ID linked through a slide-to-patient crosswalk. |
| `unresolved` | No patient ID supplied and fallback not selected. |
| `slide_fallback` | Explicit fallback: `patientId` equals `slideId`, representing one slide group. |

Import summaries return `verifiedPatientCount`, `fallbackSlideCount` and `unlinkedSlideCount`. The compatibility field `mappedPatientCount` counts all non-null grouping IDs, including fallback. Here, "verified" distinguishes supplied patient linkage from fallback; the service does not independently authenticate a clinical registry. A fallback-ID collision with a supplied patient ID returns the blocking finding `PATIENT_ID_FALLBACK_COLLISION`. The warning `PATIENT_ID_SLIDE_FALLBACK` records that slides from the same unknown patient could appear in different sets.

Frozen records and mappings preserve this distinction across reloads and service registries. Older frozen records without `patientIdSource` continue to treat a non-null patient ID as supplied linkage. Revising an old dataset to acknowledge fallback creates a new dataset version; it does not rewrite existing records or automatically transfer a feature binding to another dataset ID.

### Live target, cohort and rule exploration

Example request to `POST /projects/{id}/protocols/explore`, using a real frozen dataset ID in place of the placeholder:

```json
{
  "datasetId": "<frozen-dataset-id>",
  "targetField": "WHO 2022",
  "eligibility": [],
  "rules": {
    "train": [],
    "val": [],
    "test": [{"field": "WHO 1973", "op": "eq", "value": "2"}]
  },
  "splitMode": "rules"
}
```

Only `datasetId` is required. `targetField` is optional; eligibility and all three rule arrays default to empty. `splitMode` accepts `rules` (default), `kfold`, `holdout` or `imported`. Conditions use the same strict `Condition` schema as final protocol validation: `field`, `op` and a typed `value`. Numeric comparisons require JSON numbers; membership requires a scalar array; `exists` requires a boolean; regex requires a bounded pattern string. Unknown fields, invalid regex, nonnumeric values encountered by a numeric comparison, overlapping group rules and exhausted regex time budgets produce findings. All conditions within one rule must hold on the same slide.

Response fields:

| Field | Meaning |
| --- | --- |
| `datasetId`, `splitMode` | Echo the selected frozen dataset and assignment strategy. |
| `valid` | Whether this exploration has no blocking findings; not a freeze or execution-readiness flag. |
| `dataset` | Counts and sample rows from the complete dataset before eligibility. |
| `cohort` | Counts and sample rows after eligibility, before label-mapping exclusions; null if eligibility cannot be evaluated. |
| `target` | Optional `{field, values: [{value, slides}], distinctCount}` for raw labels in the eligible cohort. Up to 20 most frequent values are returned; `distinctCount` covers all observed values, including null. |
| `partitions` | `train`, `val`, `test`, each with `selection`, `directMatches` and `expanded`; null if grouping/rules cannot be safely evaluated. |
| `unassigned` | Counts and samples of remaining groups; generated/imported modes leave this pool for final assignment. |
| `findings` | `{severity, code, message}` entries explaining invalid inputs or acknowledged fallback. |

Every count/sample object (`dataset`, `cohort`, partition `directMatches`/`expanded`, `unassigned`) has this shape:

```text
totalSlides         Complete-population slide count
patientCount        Distinct supplied patient IDs; excludes fallback IDs
fallbackSlideCount  Slides explicitly marked slide_fallback
groupCount          Supplied patient groups plus fallback slide groups
unlinkedSlideCount  Slides without a usable grouping identity
sample              Up to five rows sorted by Slide_ID:
                    {slideId, patientId, patientIdSource?, attributes}
```

`directMatches` counts eligible slides that satisfy a partition's conditions. `expanded` includes all eligible slides in the matching groups; it never brings back slides excluded by eligibility. `selection` is `rules` for an explicit condition set, `remaining` for default training in rules mode, or `none` when there is no fixed selection. Default training has no direct rule matches; its expanded count is the actual remaining training cohort.

An invalid eligibility expression returns null cohort/partition counts instead of unfiltered or previous totals. If only partition rules fail, the valid eligible-cohort counts remain available while partition counts are cleared. Explicit training rules that leave groups unassigned return the evaluated counts plus `UNASSIGNED_RULE_GROUPS`, and final freezing remains blocked. Unacknowledged null patient IDs allow dataset/cohort counts but block group-based partition feedback; confirmed fallback groups remain countable with a warning.

The endpoint reads checksum-verified frozen records and never creates or changes drafts or configurations. It shares the final protocol evaluator and cumulative regex budget. Its counts precede missing/unmapped-label exclusion policies and all feature/constraint checks; **Preview & preflight** remains the authoritative final included population. Field examples and declared formats shown beside UI conditions come from the frozen dictionary and dataset query route, independent of the edited rule.

### Version-1 rule-based and generated assignments

`ProtocolSpec.split.mode` additionally accepts `"rules"`. In that mode:

- Test and validation rules select whole eligible groups. Empty validation rules create no validation set.
- Empty training rules assign every eligible group outside test and validation to training.
- Explicit training rules assign only matching groups. Any remaining unassigned groups block with `UNASSIGNED_RULE_GROUPS`; groups are not silently excluded.
- Overlapping train/validation/test group selections block. A configured rule that matches nothing also blocks.
- Each selected seed produces one assignment at fold `0`. Multiple seeds reuse the same rule-based memberships with `RULE_ASSIGNMENTS_REUSED`; fold count and generation ratios do not generate additional partitions.
- Training is required for minimum constraints. Validation and test are required when their rules are configured. The compatibility fields `minPatientsPerClass` and `minPatientsPerPartition` count grouping units, including explicitly acknowledged fallback slides; the UI calls these minimum groups.

The UI defaults new protocols to **Choose sets with rules — train is the remainder**. Selecting generated k-fold, generated holdout or imported assignments retains their existing semantics: fixed rules reserve groups and the remaining cohort follows that strategy. The backend `SplitSpec` default remains `kfold` for existing API clients that omit `mode`.

API clients using rules mode can omit `folds` and `ratios`; the existing defaults satisfy schema validation. An explicitly supplied `folds` still has the shared 2–10 bound, although rules mode emits only fold `0`. Non-default holdout ratios outside holdout mode remain a blocking `UNUSED_HOLDOUT_RATIOS` finding.

Final protocol summaries expose `includedPatients` (supplied patients only), `includedGroups`, `fallbackSlideCount`, `unlinkedSlideCount` and a `grouping` value of `patient` or `patient_with_slide_fallback`. Partition summaries expose `patients`, `groups`, `fallbackSlides`, `slides` and class counts over groups. Membership rows retain `patientIdSource` when available. A fallback warning persists in the frozen protocol; no patient-leakage guarantee is asserted for unknown patient identities. These changes prepare immutable protocols and do not enable job execution.

### Version-2 cross-validation strategies

The current UI creates `split.version: 2` protocols. Version 1 remains the API default for older clients and retains its existing semantics. In version 2, `mode` is one of:

| Mode | Meaning |
| --- | --- |
| `kfold` | Rotate reported test folds; reserve separate internal early-stop validation. |
| `monte_carlo` | Repeat independent test/development sampling with derived repeat seeds. |
| `leave_one_domain_out` | Hold out each selected site/cohort in turn; all early-stop data comes from other domains. |
| `nested_kfold` | Outer reported test folds, inner tuning folds, and separate early-stop subsets. |
| `held_out` | One fixed test allocation per seed, using fractions, rules, or a predefined partition column. |

Example split object:

```json
{
  "version": 2,
  "mode": "kfold",
  "folds": 5,
  "seeds": [42],
  "stratify": true,
  "validationFraction": 0.2
}
```

`validationFraction` is the early-stopping fraction of the available development/training pool after the reported test allocation, not a fraction of the entire dataset. Modern fraction-based modes use `testFraction` for the whole-cohort test allocation. All assignments use patient groups; acknowledged Slide_ID fallbacks retain their source markers and warnings.

| Field | Use |
| --- | --- |
| `folds` | K-fold count; 2–10. |
| `seeds` | Distinct reproducible base seeds; up to 10. |
| `stratify` | Balance generated allocations by mapped target class; defaults true. |
| `validationFraction` | Automatic early-stop fraction of the remaining training pool; defaults 0.2. |
| `testFraction` | Test fraction for Monte Carlo and fraction-based held-out; defaults 0.2. |
| `repeats` | Monte Carlo repetitions per seed; 1–100, default 5. |
| `domainField` | Required site/cohort attribute for domain CV. |
| `domainPolicy` | `all` or `selected`; selected values are listed in `heldOutDomains`. |
| `heldOutDomains` | Domains to rotate as reported test; empty with the `all` policy. |
| `outerFolds`, `innerFolds` | Nested fold counts; 2–10, defaults 5 and 3. |
| `heldOutSource` | `fractions`, `rules`, or `imported`; applies to `held_out`. |
| `rules` | Explicit train/val/test conditions for held-out rules only. Empty training rules use the complement; empty validation rules trigger automatic early-stop sampling from training. |
| `imported` | Explicit partition-column mapping for held-out imported mode. Fold columns are not used here. A supplied validation set is preserved; otherwise validation is sampled from training. |

Version-2 CV modes reject fixed partition rules. Use held-out validation when fixed test/validation/training conditions or existing partition labels define the experiment. A held-out test set is required. The legacy `ratios` object is used only for version-1 holdout behavior.

Each modern membership retains `seed`, `fold`, `slideId`, `patientId`, `patientIdSource` when present, `label`, and `partition`, plus `planId` and `phase`. Plans may additionally include `repeat`, `outerFold`, `innerFold`, or `domain`. Partition-count rows carry the same plan identifiers, train/val/test counts, and a `tune` count for nested inner plans.

`phase` distinguishes ordinary `evaluation`, nested `inner`, and nested `outer` plans. Inner plans use `train`, `val`, and `tune`; the outer test groups do not appear in them. Outer plans use `train`, `val`, and reported `test`. Early-stop `val` and inner model-selection `tune` remain distinct.

Modern summaries include strategy/version information, evaluation and inner plan counts, and planned test/OOF coverage. These are assignment statistics, not generated predictions. Exact plans and role memberships are part of the preview hash and frozen configuration.

The exploration endpoint accepts a partial `split` object so eligible-cohort counts remain available while strategy settings are incomplete. Generated assignments are available from the authoritative saved-draft preview after target mapping. The exploration response does not present fixed-rule counts as final generated CV assignments.

See [split strategy behavior](split-strategies.md) for percentage examples, nesting, checks, and the boundary with MIL experiment settings.

### Version-3 explicit training and test pools

New UI drafts use `split.version: 3`. Every strategy requires `pools` to define training and final test groups before generating CV plans:

```json
{
  "version": 3,
  "mode": "kfold",
  "folds": 5,
  "seeds": [42],
  "validationFraction": 0.15,
  "pools": {
    "source": "rules",
    "trainSelection": "rules",
    "validationSource": "training_fraction",
    "rules": {
      "train": [{"field": "Partition", "op": "eq", "value": "train"}],
      "test": [{"field": "Partition", "op": "eq", "value": "test"}],
      "val": []
    }
  }
}
```

`pools.source` is `rules` or `imported`. Rule mode requires training and test conditions; explicitly selecting `trainSelection: "remaining"` uses the complement of test and fixed validation. Imported mode requires `pools.imported.partitionField` and an explicit `partitionLabels` map. Fold columns are not source-pool definitions. Version-3 top-level `rules`/`imported` are unused and must be empty/absent.

`pools.validationSource` is `training_fraction` (default) or `fixed`. Version-3 protocols default `validationFraction` to 0.15; explicitly saved percentages and the version-1/version-2 default of 0.2 are preserved. Fixed validation requires nonempty validation conditions or actual mapped validation groups. With automatic validation, imported rows mapped to `val` block until the user chooses fixed validation or explicitly remaps those rows. Validation groups never become CV assessment or final test groups.

CV plans carry `pool: "training"` and retain their evaluation/inner/outer phases. Each seed also gets a `phase: "final"`, `pool: "external_test"` plan using the selected training pool and reserved test pool. Held-out mode generates only final plans. Summaries include `poolCounts`, `finalPlanCount`, and CV-only `evaluationPlanCount`/OOF coverage. The exploration endpoint reports source-pool counts before target-label exclusions and generated CV assignments.

New UI target fields start unconfigured; saved drafts may contain unfinished settings, while preview continues to require a valid target contract. Source-value suggestions do not alter that backend validation.

Version-1 and version-2 serialized specs and hashes remain unchanged. See [split strategies](split-strategies.md) for the current UI and validation behavior.
