# Error codes

Every error the local service returns has a JSON body `{detail, code}`: `detail` says what went wrong for a person, `code` names it for a program. A commit refused over review findings also sends `findings`, each `{code, message, severity}` plus `field` when one field is at fault. For a request that fails schema validation, `detail` is the list of rejected fields.

A code's kind groups codes by what the caller should do next, and the CLI turns each kind into an exit code: see [exit codes](cli-contract.md#exit-codes). A code missing from this page takes its kind from its HTTP status by the rules there.

`scripts/error_codes.py` writes this page from the raise sites in `histopilot/` and the registry in `histopilot/api/error_codes.py`. Edit those, not this page.

| Kind | Next move |
| --- | --- |
| `invalid` | Fix the command or spec. |
| `refused` | Change the request or the state first; resent unchanged, it fails again. |
| `conflict` | Reload or preview again, then retry. |
| `not-found` | Check the ID or tag. |
| `unavailable` | Start the service, fix --url, or sign in. |
| `internal` | Report it with the output. |

## invalid

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `ARTIFACT_PATH_INVALID` | invalid | 422 | Artifact names must be safe relative paths. |
| `ATTRIBUTE_KEY_REQUIRED` | invalid | 422 | Map long source headers to explicit attribute keys of at most 128 characters. |
| `CASE_COMPARISON_MISMATCH` | invalid | 409 | Compare evaluations of the same frozen cohort, target, and patient aggregation. |
| `CASE_COMPARISON_REQUIRED` | invalid | 422 | Select a second evaluation to review disagreements. |
| `CASE_REVIEW_FILTER_INVALID` | invalid | 422 | Choose a class in this evaluation. |
| `CLINICAL_FIELD_UNKNOWN` | invalid | 422 | … is not a numeric or categorical column of the frozen dataset. |
| `CLINICAL_PATIENT_UNAVAILABLE` | invalid | 409 | This evaluation has no patient grouping identities. Select slide-level analysis. |
| `CLINICAL_POSITIVE_CLASS_INVALID` | invalid | 422 | Choose a class for one-versus-rest analysis. |
| `COMPARISON_CONTEXT_MISMATCH` | invalid | 409 | Choose the same frozen cohort, target, and patient aggregation for paired comparison. |
| `COMPARISON_PATIENT_MISMATCH` | invalid | 409 | The compared runs do not predict the same patients. |
| `DATASET_KIND_INVALID` | invalid | 422 | This record is not an imported dataset. |
| `EVALUATION_AGGREGATION_MISMATCH` | invalid | 409 | Patient aggregation must preserve the frozen predictor's scoring rule. |
| `EVALUATION_EXTRACTION_MISMATCH` | invalid | 409 | Development and test extraction settings differ. |
| `EVALUATION_FEATURE_BUNDLE_AMBIGUOUS` | invalid | 409 | More than one compatible test feature inventory is available. Select the test feature bundle to evaluate. |
| `EVALUATION_FEATURE_CONTRACT_MISMATCH` | invalid | 409 | Test features must preserve the verified representation, encoder, dimensions, and dtype used by the predictor. |
| `EVALUATION_PREDICTOR_MISMATCH` | invalid | 409 | The cohort must use this predictor's exact development protocol, target encoding, and development feature bundle. |
| `EVALUATION_THRESHOLD_MISMATCH` | invalid | 409 | Use the decision threshold frozen in the training recipe for every external cohort. |
| `EXPERIMENT_DATASET_MISMATCH` | invalid | 422 | The target split belongs to a different dataset. |
| `EXPERIMENT_DUPLICATE_BATCH` | invalid | 422 | Two batch plans are identical. Remove the duplicate or give the new batch its own name and settings. |
| `FEATURE_LAYOUT_AMBIGUOUS` | invalid | 422 | Multiple TRIDENT feature sets found; choose an encoder and its specific feature folder: … |
| `FEATURE_LAYOUT_NOT_FOUND` | invalid | 422 | No TRIDENT …&lt;encoder&gt; directory matches the selected encoder. |
| `FOLDER_NAME_INVALID` | invalid | 422 | Folder names and paths must contain valid Unicode text. |
| `FOLDER_PARENT_INVALID` | invalid | 422 | Choose a parent folder without '..' path components. |
| `IMPORT_DRAFT_REQUIRED` | invalid | 422 | Use a dataset-import draft for this operation. |
| `IMPORT_SPEC_INVALID` | invalid | 422 | Review the import source and mappings. … |
| `INFERENCE_ATTRIBUTE_INVALID` | invalid | 422 | Choose an attribute from the frozen data dictionary. |
| `INFERENCE_COMPARISON_INVALID` | invalid | 422 | Choose a different run to compare. |
| `INFERENCE_COMPARISON_MISMATCH` | invalid | 409 | Compare runs on the same frozen cohort, target, prediction unit and patient aggregation. |
| `INFERENCE_SETTINGS_INCOMPLETE` | invalid | 422 | Write every inference setting; missing: …. |
| `INFERENCE_SLIDE_UNAVAILABLE` | invalid | 422 | Choose slides predicted by this run. |
| `INTERPRETATION_DUPLICATE_SLIDE` | invalid | 422 | Select each slide once. |
| `INTERPRETATION_ENCODER_MISMATCH` | invalid | 422 | Features require an explicit encoder identity matching the frozen predictor. |
| `INTERPRETATION_INPUT_INVALID` | invalid | 422 | Attention inputs fail validation. |
| `INTERPRETATION_LINEAGE_MISMATCH` | invalid | 409 | Clinical analysis must belong to the selected predictor. |
| `INTERPRETATION_MEMBER_INVALID` | invalid | 422 | Choose the ensemble mean or an existing model member. |
| `INTERPRETATION_PACK_MISMATCH` | invalid | 422 | Select a pack included in this frozen feature bundle. |
| `INTERPRETATION_PATCH_INVALID` | invalid | 422 | Choose an existing patch index. |
| `INTERPRETATION_SLIDE_UNAVAILABLE` | invalid | 422 | The slide is not in the selected slide folder. |
| `INTERPRETATION_VIEW_INVALID` | invalid | 422 | Provide all four viewport bounds. |
| `INVALID_ARTIFACT` | invalid | 422 | Materialized artifacts must be bytes. |
| `INVALID_BATCH` | invalid | 422 | Select a development batch. |
| `INVALID_CACHE` | invalid | 409, 422 | Select a directory for the cache parent. |
| `INVALID_CHECKPOINT` | invalid | 422 | Select a regular checkpoint file. |
| `INVALID_CONFIGURATION` | invalid | 422 | Unsupported scientific configuration kind. |
| `INVALID_COORDS` | invalid | 422 | Coordinates must be in a subdirectory of this run's output. |
| `INVALID_DATASET` | invalid | 409, 422 | Select a frozen imported dataset. |
| `INVALID_DOCUMENT` | invalid | 422 | Supply a JSON object containing finite JSON values. |
| `INVALID_DRAFT_KIND` | invalid | 422 | Draft kind must be import or experiment. |
| `INVALID_EVALUATION_DRAFT` | invalid | 422 | Select a test-cohort draft. |
| `INVALID_EVALUATION_SPEC` | invalid | 422 | Invalid test-cohort settings: … |
| `INVALID_EXPERIMENT` | invalid | 422 | Choose a model development experiment. |
| `INVALID_EXPERIMENT_DRAFT` | invalid | 422 | Invalid experiment draft type. |
| `INVALID_EXPERIMENT_REFERENCE` | invalid | 422 | The draft names conflicting experiment revisions. |
| `INVALID_FEATURE_BUNDLE` | invalid | 422 | Choose a frozen feature bundle. |
| `INVALID_FEATURE_SET` | invalid | 422 | Select a feature inventory. |
| `INVALID_INFERENCE_COHORT` | invalid | 409 | Inference cohorts must be unlabeled. |
| `INVALID_INPUT` | invalid | 422 | A value is missing, too long or malformed. |
| `INVALID_LIFECYCLE_CHANGE` | invalid | 422 | Supply valid lifecycle changes, operation identity and revision. |
| `INVALID_OUTPUT` | invalid | 422 | Choose a dedicated TRIDENT output directory. |
| `INVALID_PARENT` | invalid | 422 | A parentId must identify an existing frozen dataset. |
| `INVALID_PATH` | invalid | 403, 422 | Use an absolute path without parent traversal. |
| `INVALID_PREDICTOR` | invalid | 422 | Select a frozen predictor. |
| `INVALID_PROTOCOL` | invalid | 422 | Choose the experiment's frozen split. |
| `INVALID_PROTOCOL_DRAFT` | invalid | 422 | Select an analysis-protocol experiment draft. |
| `INVALID_PROTOCOL_SPEC` | invalid | 422 | The protocol specification is invalid: … |
| `INVALID_RESOURCE_TYPE` | invalid | 422 | Labels belong to a frozen dataset or configuration. |
| `INVALID_REVIEW_COHORT` | invalid | 409 | Slide review predictions require shared patient identifiers. |
| `INVALID_REVIEW_PAGE` | invalid | 422 | Choose a valid review page. |
| `INVALID_REVISION` | invalid | 422 | Supply a positive draft revision. |
| `INVALID_SLIDES` | invalid | 422 | Choose a slide folder or a slide list. |
| `INVALID_SLIDE_LIST` | invalid | 422 | Invalid slide selection: … |
| `INVALID_TARGET_SPLIT` | invalid | 422 | Choose frozen targets and train/test populations. |
| `INVALID_TARGET_SPLIT_DRAFT` | invalid | 422 | Select a target/split draft. |
| `INVALID_TARGET_SPLIT_SPEC` | invalid | 422 | Invalid target/split: … |
| `INVALID_TRAINING_RECIPE` | invalid | 422 | A resolved training recipe is invalid: … |
| `INVALID_TRAINING_SPLIT` | invalid | 422 | Training design must use a development-only split. |
| `INVALID_TRIDENT_OPTIONS` | invalid | 422 | The TRIDENT extraction options fail validation. |
| `INVALID_VERSION_LABEL` | invalid | 422 | Supply a nonempty version tag of at most 80 characters and a commit note of at most 2000 characters. |
| `MAPPING_COLUMN_MISSING` | invalid | 422 | One or more selected columns are absent from the main table. |
| `MORPHOLOGY_DATASET_MISMATCH` | invalid | 422 | This feature bundle is scoped to a different frozen dataset. |
| `MORPHOLOGY_REGION_INVALID` | invalid | 422 | Provide every viewport coordinate. |
| `PACK_FEATURE_MISMATCH` | invalid | 422 | This pack was verified for a different feature version. |
| `PATH_INVALID` | invalid | 400 | Choose an absolute server metadata file path. |
| `PATH_NOT_ABSOLUTE` | invalid | 400 | Select an absolute path within a configured data root. |
| `PATH_NOT_DIRECTORY` | invalid | 400 | Select a directory; file contents are not served. |
| `PATIENT_ANALYSIS_DISABLED` | invalid | 409 | Patient analysis is disabled for slide-level experiments. Choose slide review. |
| `PERFORMANCE_ATTRIBUTE_INVALID` | invalid | 422 | Choose an attribute from the frozen data dictionary. |
| `PREDICTOR_CANDIDATE_MISSING` | invalid | 422 | Select a candidate and its frozen seeds. |
| `PREDICTOR_CHECKPOINT_INVALID` | invalid | 409 | Select a training checkpoint. |
| `PREDICTOR_EXPERIMENT_MISMATCH` | invalid | 409 | This batch belongs to another experiment. |
| `PROJECT_MIL_MODEL_UNSUPPORTED` | invalid | 422 | Choose a supported MIL model: …. |
| `REFERENCE_CLASSES_MISMATCH` | invalid | 409 | This reference standard's classes differ from the run's classes. |
| `REFERENCE_COHORT_INVALID` | invalid | 422 | Choose a frozen cohort. |
| `REFERENCE_COHORT_MISMATCH` | invalid | 409 | This reference standard labels another cohort. |
| `REFIT_METHOD_REQUIRED` | invalid | 422 | Select the refit build method. |
| `REFIT_RESOURCES_INVALID` | invalid | 422 | A refit trains one model on at most one GPU. |
| `REQUEST_INVALID` | invalid | 422 | The body, path or query does not match the route's schema. `detail` lists each rejected field. |
| `REVIEW_CONTEXT_INVALID` | invalid | 422 | Choose a saved model evaluation. |
| `SETUP_VERSION_REQUIRED` | invalid | 422 | State setupVersion: 1 for a new experiment; null keeps the legacy record form. |
| `SHEET_INVALID` | invalid | 422 | CSV files do not have worksheets. |
| `SLIDE_ROOT_REQUIRED` | invalid | 422 | A slide list needs the slide folder its paths are relative to. |
| `SLIDE_VIEW_INVALID` | invalid | 422 | Patch image size must be 64–1024 pixels. |
| `TABLE_EMPTY` | invalid | 422 | The table has no header row. |
| `TABLE_ENCODING` | invalid | 422 | CSV metadata must use UTF-8 encoding. |
| `TABLE_FILE_REQUIRED` | invalid | 422 | Select a regular CSV or XLSX metadata file. |
| `TABLE_FORMAT_UNSUPPORTED` | invalid | 422 | Select a CSV or XLSX metadata file. |
| `TABLE_HEADERS_DUPLICATE` | invalid | 422 | Duplicate column headers must be resolved before mapping. |
| `TABLE_HEADERS_INVALID` | invalid | 422 | Use one nonempty header per column, up to 128 columns. |
| `TABLE_INVALID` | invalid | 422 | The XLSX workbook could not be parsed. |
| `TABLE_ROW_WIDTH` | invalid | 422 | Row … does not match the header width. |
| `TARGET_CONTRACT_MISMATCH` | invalid | 409 | Test targets must match the selected model's task, prediction unit, class order, and positive class. |
| `TARGET_SPLIT_DATASET_MISMATCH` | invalid | 422 | Targets & Splits must belong to the selected dataset. |
| `TARGET_SPLIT_TARGET_MISMATCH` | invalid | 422 | Use the target frozen in Targets & Splits. |
| `TARGET_SPLIT_UNIT_MISMATCH` | invalid | 422 | Use the slide or patient split unit frozen in Targets & Splits. |
| `TASK_ACTION_INVALID` | invalid | 422 | Unknown task action: …. |
| `TASK_CENTER_FILTER_INVALID` | invalid | 422 | Unknown task state filter: …. |
| `TASK_CENTER_SETTINGS_INVALID` | invalid | 422 | Task Center settings must be an object. |
| `TASK_DEPENDENCY_UNKNOWN` | invalid | 422 | Task … depends on unknown task …. |
| `TASK_INVALID` | invalid | 422 | A task specification is invalid. |
| `TOKEN_LIFETIME_INVALID` | invalid | 422 | A token lasts 1 to … days. |
| `TOKEN_SCOPE_INVALID` | invalid | 422 | Unknown scope …; choose from …. |
| `TRAINING_FEATURE_KIND_MISMATCH` | invalid | 422 | This bundle holds … features. Choose one of: …. |
| `TRAINING_METRIC_UNAVAILABLE` | invalid | 422 | The selection metric is unavailable for this target. |
| `TRAINING_MODEL_UNSUPPORTED` | invalid | 422 | Choose one of: …. |
| `TRAINING_OOF_UNIT_INVALID` | invalid | 422 | Choose patient or slide predictions. |
| `TRAINING_RECIPE_UNAVAILABLE` | invalid | 422 | The training recipe does not fit this target or data. |
| `TRAINING_RUNS_REQUIRED` | invalid | 422 | Choose at least one run to cancel. |
| `UNKNOWN_ATTRIBUTE` | invalid | 422 | Unknown attribute: …. |
| `UPLOAD_INVALID` | invalid | 422 | The uploaded file is not valid base64. |

## refused

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `AGENT_REQUEST_NOT_CLAIMED` | refused | 409 | This agent request is …; claim it before replaying it. |
| `AGENT_REQUEST_SETTLED` | refused | 409 | This agent request is already …. |
| `AGREEMENT_UNAVAILABLE` | refused | 409 | Agreement cannot be computed for this run's predictions. |
| `ARTIFACT_CORRUPT` | refused | 409 | A dataset artifact failed checksum validation. |
| `ARTIFACT_LIMIT` | refused | 413 | Too many artifacts in one publication. |
| `BATCH_HAS_FROZEN_PREDICTOR` | refused | 409 | This batch supplies a frozen predictor. |
| `BATCH_PREFLIGHT_BLOCKED` | refused | 409, 422 | Resolve batch findings before freezing. |
| `BATCH_TOO_LARGE` | refused | 422 | Limit a batch to 512 configurations. |
| `CASE_REVIEW_EVIDENCE_INVALID` | refused | 409 | A reviewed slide resolves to more than one frozen dataset. |
| `CASE_REVIEW_LIMIT` | refused | 413 | This selection contains too many slides to review at once. Narrow the filters or choose a smaller slide-level page. |
| `CLEANUP_BLOCKED` | refused | 409 | Cleanup is blocked. Resolve the listed dependencies or active jobs first. |
| `CLEANUP_EVIDENCE_INVALID` | refused | 409 | Job evidence cannot be read safely; cleanup is blocked until it is repaired. |
| `CLEANUP_LIMIT` | refused | 413 | Too many records to review cleanup safely. |
| `CLEANUP_PROCESS_UNKNOWN` | refused | 409 | Job process identity is invalid; cleanup is blocked. |
| `CLINICAL_COVERAGE_MISSING` | refused | 422 | Clinical values must cover every frozen selected slide. |
| `CLINICAL_EVIDENCE_INVALID` | refused | 409 | Saved predictions must contain a list of records. |
| `CLINICAL_FIELDS_MISSING` | refused | 422 | Map the required clinical fields in the frozen dataset: … |
| `CLINICAL_IDENTITY_AMBIGUOUS` | refused | 422 | Clinical slide identity occurs in multiple datasets. |
| `CLINICAL_INPUTS_CHANGED` | refused | 409 | Clinical schema or frozen values changed. |
| `CLINICAL_NO_LABELED_OUTCOMES` | refused | 409 | No labeled predictions are available for this unit. Clinical statistics require observed outcomes. |
| `CLINICAL_PATIENT_IDENTITIES_UNVERIFIED` | refused | 409 | Patient-level clinical analysis requires verified patient identities. |
| `CLINICAL_TARGET_LEAKAGE` | refused | 422 | The field cannot be a clinical input, for example the target. |
| `CLINICAL_VALUES_INVALID` | refused | 422 | Frozen clinical values cannot be used for the selected fields. |
| `COMPARISON_EVIDENCE_CHANGED` | refused | 409 | Evaluation references or targets changed. |
| `COMPARISON_EVIDENCE_INVALID` | refused | 409 | Saved predictions of a compared run cannot be read. |
| `COMPARISON_REQUIRES_LABELS` | refused | 409 | A run on an unlabeled cohort has no labels to compare patients by. Compare its predictions case by case instead. |
| `COMPUTE_ACTIVE` | refused | 409 | This compute task is already queued or running. |
| `COMPUTE_PLAN_CHANGED` | refused | 409 | The saved execution plan changed. |
| `COMPUTE_RESOURCES_UNAVAILABLE` | refused | 409 | The requested CPU or RAM reservation exceeds this workstation's capacity. |
| `COMPUTE_RESULT_CHANGED` | refused | 409 | Compute result evidence changed. |
| `COMPUTE_RESUME_REQUIRED` | refused | 409 | Resume an existing job; launch a new job only once. |
| `COMPUTE_STATE_INVALID` | refused | 409 | Compute state is invalid. |
| `CONFLICTING_PATIENT_LABELS` | refused | 409 | A test patient has conflicting labels across slides. |
| `CREATED_BEFORE_TASK_CENTER` | refused | 409 | Created before the Task Center, so it can no longer run. Clone it, or preview it again, to run it in the Task Center. |
| `DATASET_SLIDE_FOLDER_UNAVAILABLE` | refused | 422 | These features are scoped to a slide store, not a frozen dataset. Attach them to a dataset to open slides for interpretation. |
| `DEMO_PROJECT_READ_ONLY` | refused | 409 | The BLCA demo is read only and has no local project folder. |
| `DEMO_STORE_UNAVAILABLE` | refused | 409 | The synthetic demo has no local scientific store. |
| `DEVELOPMENT_FEATURE_BINDING_MISMATCH` | refused | 409 | The development bundle differs from the feature version pinned by this protocol. |
| `DEVELOPMENT_FEATURE_COVERAGE` | refused | 409 | The selected development bundle is missing … development slide IDs. |
| `DEVELOPMENT_MEMBERSHIPS_MISSING` | refused | 409 | The development protocol has no frozen membership evidence. |
| `DEVELOPMENT_PATIENT_OVERLAP` | refused | 409 | … selected patient IDs occur in model development. |
| `DEVELOPMENT_SLIDE_OVERLAP` | refused | 409 | … selected slide IDs occur in model development. |
| `DEVELOPMENT_SLIDE_SOURCE_OVERLAP` | refused | 409 | … selected slides refer to source files used in model development under different slide IDs. |
| `DIRECTORY_UNREADABLE` | refused | 403 | The selected directory cannot be listed. |
| `DOCUMENT_TOO_LARGE` | refused | 413 | The metadata document exceeds the supported size. |
| `DRAFT_FROZEN` | refused | 409 | The draft or record is frozen. Copy it to make changes. |
| `DUPLICATE_FEATURE_ID` | refused | 409 | The … feature inventory contains duplicate slide IDs. |
| `DUPLICATE_SLIDE_ID` | refused | 409 | Selected test slides have duplicate slide identifiers across the selected datasets. |
| `DUPLICATE_TEST_SLIDE_SOURCE` | refused | 409 | … selected slide IDs refer to shared source files. Keep one record per physical slide to avoid counting the same slide more than once. |
| `EMPTY_TARGET_SPLIT_PARTITION` | refused | 422 | Targets & Splits has no … slides. |
| `EMPTY_TEST_COHORT` | refused | 409 | The test-cohort conditions select no slides. |
| `ENCODER_IDENTITY_REQUIRED` | refused | 409 | Distinct development and test feature sources require an explicit encoder identity on both inventories. |
| `ENCODER_MISMATCH` | refused | 409 | The test feature encoder differs from the development encoder. |
| `EVALUATION_BATCH_BLOCKED` | refused | 422 | Choose compatible active predictors and remove deleted selections before running this batch. |
| `EVALUATION_BATCH_CHANGED` | refused | 409 | A batch member identity changed. |
| `EVALUATION_BATCH_LIMIT` | refused | 422 | Select at most … predictors in one evaluation batch. |
| `EVALUATION_COVERAGE_CHANGED` | refused | 409 | The exact test feature membership is incomplete. |
| `EVALUATION_DEVELOPMENT_OVERLAP` | refused | 409 | Inference would predict slides or patients used in this predictor's development. |
| `EVALUATION_ENCODER_UNVERIFIABLE` | refused | 409 | Distinct feature inventories require explicit matching encoder identities. |
| `EVALUATION_EVIDENCE_CHANGED` | refused | 409 | The evaluation no longer matches its frozen predictor and cohort. |
| `EVALUATION_EVIDENCE_INVALID` | refused | 409 | The saved prediction evidence is invalid. |
| `EVALUATION_EXTRACTION_UNVERIFIABLE` | refused | 409 | Extraction provenance is available for only one feature source. Attach matching extraction provenance to both sources. |
| `EVALUATION_FEATURE_BUNDLE_REQUIRED` | refused | 409 | No current feature bundle covers every selected test slide with this model's representation, encoder, dimensions, and dtype. |
| `EVALUATION_INPUTS_CHANGED` | refused | 409 | The evaluation's frozen inputs changed. Create and review a new one. |
| `EVALUATION_PACK_CHANGED` | refused | 409 | The selected test pack changed. |
| `EVALUATION_PREFLIGHT_BLOCKED` | refused | 422 | Resolve the test-cohort findings before freezing. |
| `EVALUATION_RECORD_LIMIT` | refused | 413 | A cohort supports at most 50,000 slide records. |
| `EVALUATION_RESULT_CHANGED` | refused | 409 | Evaluation result provenance changed. |
| `EXPERIMENT_ACTIVE` | refused | 409 | This experiment coordinator already exists. |
| `EXPERIMENT_ALREADY_FROZEN` | refused | 409 | This configuration already has a seed ensemble. Restore it to reuse its identity. |
| `EXPERIMENT_ALREADY_SUBMITTED` | refused | 409 | This experiment is already submitted. Retry its original submission or resume an unfinished batch. |
| `EXPERIMENT_ARCHIVED` | refused | 409 | Restore this experiment to Active before resuming predictors. |
| `EXPERIMENT_BATCHES_REQUIRED` | refused | 422 | Save at least one batch plan before freezing setup. |
| `EXPERIMENT_BATCH_INACTIVE` | refused | 409 | Batch '…' already exists in the … state with these exact settings. |
| `EXPERIMENT_BATCH_INPUTS_MISMATCH` | refused | 409 | Existing batches use different experiment inputs. |
| `EXPERIMENT_CONFIGURATION_LOCKED` | refused | 409 | This experiment's design is frozen or started. Copy the experiment to adjust inputs or batches. |
| `EXPERIMENT_COPY_INVALID` | refused | 422 | An old batch draft is incomplete. Complete its recipe before copying. |
| `EXPERIMENT_FINISHED` | refused | 409 | This experiment is finished. Copy it to start new runs. |
| `EXPERIMENT_FOLDS_INCOMPLETE` | refused | 409 | Resume this experiment's unfinished fold batch before retrying predictors. |
| `EXPERIMENT_INPUTS_INVALID` | refused | 422 | Resolve experiment input compatibility: … |
| `EXPERIMENT_INPUTS_REQUIRED` | refused | 422 | Choose inputs before saving batch plans. |
| `EXPERIMENT_OWNER_CHANGED` | refused | 409 | The draft names conflicting experiment owners. |
| `EXPERIMENT_PREDICTORS_LEGACY` | refused | 409 | This historical experiment has no automatic predictor plan. Build its predictors explicitly, or copy it into a new experiment. |
| `EXPERIMENT_PREDICTORS_LOCKED` | refused | 409 | This experiment's predictor work is closed. |
| `EXPERIMENT_PREDICTOR_PLAN_CHANGED` | refused | 409 | The submitted per-batch predictor policy version or choices changed. |
| `EXPERIMENT_PREDICTOR_POLICY_LOCKED` | refused | 409 | This batch is outside the submitted predictor plan. |
| `EXPERIMENT_REFIT_STOPPED` | refused | 409 | Refit stopped. Resume predictor creation to continue. |
| `EXPERIMENT_RUNTIME_CHANGED` | refused | 409 | Training code or the Python environment changed after experiment review. |
| `EXPERIMENT_SETUP_CHANGED` | refused | 409 | The frozen setup does not belong to this experiment. |
| `EXPERIMENT_SETUP_FROZEN` | refused | 409 | This setup is already frozen. Copy it to make changes. |
| `EXPERIMENT_SETUP_INPUTS_REQUIRED` | refused | 409, 422 | Choose the dataset, targets, features and training design in the experiment's inputs. |
| `EXPERIMENT_SETUP_REQUIRED` | refused | 409 | Freeze the experiment's design before starting it. |
| `EXPERIMENT_SUBMISSION_REQUIRED` | refused | 409 | Finish starting the experiment before creating predictors. |
| `EXPERIMENT_TOO_LARGE` | refused | 422 | Limit copied experiments to 100 batch plans. |
| `EXPERIMENT_TYPED_ENDPOINT_REQUIRED` | refused | 409 | Manage model experiments through their experiment record. |
| `EXPOSURE_METADATA` | refused | 403 | Files, images and logs of a project shared at the metadata level stay with the service. |
| `EXPOSURE_NONE` | refused | 403 | Agents see nothing of this project: its AI-exposure level is none. |
| `EXTRACTION_ACTIVE` | refused | 409 | This extraction is still running in the Task Center. |
| `EXTRACTION_CORRUPT` | refused | 409 | Extraction metadata is invalid. |
| `EXTRACTION_INVALID` | refused | 422 | Resolve extraction preflight findings before starting. |
| `EXTRACTION_LIMIT` | refused | 413 | Too many extraction records. |
| `FALLBACK_PATIENT_ID_COLLISION` | refused | 409 | A Slide_ID fallback collides with another grouping identity or differs from its slide ID. Revise the dataset before assigning partitions. |
| `FEATURES_INVALID` | refused | 422 | Resolve feature inspection errors before freezing. |
| `FEATURE_BUNDLE_INVALID` | refused | 422 | Resolve all verification findings before freezing this bundle. |
| `FEATURE_DIMENSION_MISMATCH` | refused | 409 | Test feature dimensions must match the development features. |
| `FEATURE_DTYPE_MISMATCH` | refused | 409 | Test feature dtype must match the single dtype used by the development features. |
| `FEATURE_KIND_MISMATCH` | refused | 409 | Test features must use the same representation as development: patch embeddings or one embedding per slide. |
| `FEATURE_SCAN_LIMIT` | refused | 413 | Feature discovery exceeded its limit; select a smaller folder. |
| `FEATURE_SOURCE_CHANGED` | refused | 409, 422 | The frozen feature inventory changed. |
| `FEATURE_VERIFICATION_CHANGED` | refused | 409, 422 | Frozen feature verification is unavailable or changed. |
| `FOLDER_CREATE_FAILED` | refused | 400 | The folder could not be created in this location. |
| `FOLDER_CREATION_UNSUPPORTED` | refused | 501 | Folder creation requires POSIX filesystem support. |
| `FOLDER_EXISTS` | refused | 409 | A file or folder with this name already exists. Choose another name. |
| `FOLDER_NO_SPACE` | refused | 507 | There is not enough space to create a folder here. |
| `FOLDER_PARENT_SYMLINK` | refused | 403 | Create folders in the actual parent location, not through symbolic links. |
| `FOLDER_PERMISSION_DENIED` | refused | 403 | HistoPilot does not have permission to create a folder here, or the filesystem is read-only. |
| `IDENTIFIER_TARGET` | refused | 409 | Identifiers and partition fields cannot serve as target labels. |
| `IMPORT_BLOCKED` | refused | 422 | Resolve the blocking import findings before freezing. |
| `IMPORT_TOO_LARGE` | refused | 413 | Joined records exceed the supported artifact size; reduce the selected scalar fields or source population. |
| `INFERENCE_SLIDE_IMAGE_MISSING` | refused | 422 | Slide … has no linked image in its frozen dataset. |
| `INTERPRETATION_BATCH_LIMIT` | refused | 422 | Batch review reached its 45-second limit. Completed submissions are retained; retry the remaining slides in a smaller batch. |
| `INTERPRETATION_CLINICAL_DATA_REQUIRED` | refused | 422 | Combined attention needs a frozen dataset with clinical covariates. |
| `INTERPRETATION_DATASET_FOLDER_UNAVAILABLE` | refused | 422 | The frozen dataset has no usable slide folder. Link it in Datasets, then freeze the feature bundle again. |
| `INTERPRETATION_ENCODER_MISSING` | refused | 422 | The bundle must record its feature encoder identity. |
| `INTERPRETATION_FEATURE_MISMATCH` | refused | 422 | This representation does not match the predictor's encoder, dimensions and dtype. Select a compatible bundle or pack. |
| `INTERPRETATION_FOLDER_INVALID` | refused | 403 | Select a slide folder inside a configured data root, without symbolic links. |
| `INTERPRETATION_FOLDER_UNREADABLE` | refused | 403 | A slide subfolder could not be read. Check permissions or select a readable folder. |
| `INTERPRETATION_LEGACY_VIEW_LIMIT` | refused | 413 | This legacy JSON attention map exceeds the 64 MiB viewer limit. |
| `INTERPRETATION_MEMORY_INSUFFICIENT` | refused | 422 | Whole-bag attention exceeds this RAM reservation. Increase RAM per run; patches are never silently sampled. |
| `INTERPRETATION_MODEL_UNSUPPORTED` | refused | 422 | Clinical-only predictors have no image attention. |
| `INTERPRETATION_PATH_INVALID` | refused | 403 | Select a regular local file inside a configured data root, without symbolic links. |
| `INTERPRETATION_PREDICTOR_INVALID` | refused | 422 | Predictor membership is incomplete or unsupported. |
| `INTERPRETATION_RECEIPT_INVALID` | refused | 409 | Visualization receipt is inconsistent. |
| `INTERPRETATION_RESULT_CHANGED` | refused | 409 | Attention output changed. |
| `INTERPRETATION_SCAN_LIMIT` | refused | 413 | Slide discovery exceeded its bounded scan. Select a smaller slide folder. |
| `INVALID_ATTRIBUTE` | refused | 422 | A numeric attribute contains invalid or unsupported values. |
| `INVALID_FILTER_VALUE` | refused | 409 | A numeric filter encountered a nonnumeric value. |
| `INVALID_REGEX` | refused | 409 | A regular expression is invalid. |
| `INVALID_SLIDE` | refused | 403 | Slide input is outside configured source roots. |
| `INVALID_SOURCE_EXTRACTION` | refused | 422 | The linked extraction cannot be verified: … |
| `LEGACY_DEVELOPMENT_PROTOCOL` | refused | 409 | This targets & splits version predates development-only training designs. |
| `LEGACY_EXPERIMENT_READ_ONLY` | refused | 409 | Historical experiments are read-only. Create a new experiment to continue. |
| `LEGACY_PROTOCOL_SPLIT` | refused | 422 | This protocol was created before development-only splits; frozen versions stay readable, but it can no longer be previewed or frozen. |
| `LIFECYCLE_CORRUPT` | refused | 409 | Workspace lifecycle metadata is missing. |
| `LIFECYCLE_LIMIT` | refused | 409, 413 | Workspace cleanup metadata reached a size or revision limit. |
| `METADATA_LIMIT` | refused | 413 | A metadata file exceeds its size limit. |
| `MIL_BAG_PLANNING_INVALID` | refused | 422 | MIL bags cannot be planned for these features. |
| `MISSING_PATIENT_ID` | refused | 409 | Some slides have unresolved Patient_ID. Revise the dataset to map patients or explicitly confirm Slide_ID fallback. |
| `MISSING_TARGET_LABEL` | refused | 409 | Selected slides have missing or unmapped target labels. Map them, explicitly exclude them, or keep them unlabeled. |
| `MISSING_TEST_FEATURES` | refused | 409 | … selected test slide IDs have no features in the selected bundle. |
| `MISSING_TEST_PACK_SLIDES` | refused | 409 | … selected test slide IDs are absent from the selected pack index. |
| `MORPHOLOGY_DATASET_INVALID` | refused | 422 | Legacy frozen dataset records are invalid. |
| `MORPHOLOGY_INDEX_LIMIT` | refused | 413 | This exploration exceeds its bounded index. Reduce slides or patches per slide. |
| `MORPHOLOGY_INVALID` | refused | 404, 413, 422 | Features, coordinates or slides cannot be explored as requested. |
| `MORPHOLOGY_PATCH_FEATURES_REQUIRED` | refused | 422 | Patch exploration requires patch embeddings with coordinates. Use original-image review for a slide-embedding bundle. |
| `MORPHOLOGY_SCOPE_REQUIRED` | refused | 422 | The feature inventory has no explicit dataset or store scope. |
| `MORPHOLOGY_SOURCE_CHANGED` | refused | 409 | Feature verification changed. Revalidate the selected bundle. |
| `NESTED_SELECTION_REQUIRED` | refused | 422 | Nested CV requires a separate search and selected refit inside each outer fold. Batch planning for that dependency is not connected yet. |
| `OPERATION_CONFLICT` | refused | 409 | This operation ID was already used for a different request. |
| `OUTPUT_IMMUTABLE` | refused | 422 | An extraction output cannot be inside an existing feature pack. |
| `PACKING_CORRUPT` | refused | 409 | Packing metadata is invalid. |
| `PACKING_INVALID` | refused | 422 | Resolve preview findings before starting. |
| `PACKING_LIMIT` | refused | 413 | Too many packing jobs. |
| `PACKING_UNSAFE_LOCK` | refused | 403 | Packing lock storage is unsafe. |
| `PACK_BINDING_CHANGED` | refused | 409, 422 | An included pack no longer matches its frozen identity. |
| `PACK_NOT_CURRENT` | refused | 409 | The pack or source files changed. Verify this existing pack before selecting it. |
| `PACK_NOT_IN_BUNDLE` | refused | 409 | Select a verified pack included in the test feature bundle. |
| `PACK_SOURCE_CHANGED` | refused | 422 | The pack's files changed since verification. Verify the pack again. |
| `PACK_VERIFICATION_REFRESH_REQUIRED` | refused | 422 | Verify this existing pack once to record file freshness before selecting it. |
| `PACK_VERIFICATION_REQUIRED` | refused | 409, 422 | Every included pack must have current full verification against these exact features. |
| `PATH_OUTSIDE_ROOTS` | refused | 403 | The selected path is outside configured data roots. |
| `PATH_REFUSED` | refused | 403 | Feature folders must not traverse symbolic links. |
| `PATH_UNRESOLVABLE` | refused | 403 | The selected directory cannot be resolved. |
| `PATIENT_AGGREGATION_UNSUPPORTED` | refused | 409 | Choose mean probabilities or mean logits to match the frozen predictor's patient scoring rule. |
| `PERFORMANCE_REQUIRES_LABELS` | refused | 409 | This run predicts an unlabeled cohort. Score it against a reference standard to break its performance down. |
| `PORTABILITY_CHECKSUM_MISMATCH` | refused | 409 | Checksum verification failed for …. |
| `PORTABILITY_INVALID` | refused | 403, 404, 409, 413, 422 | An archive, restore or relink request, or the archive itself, is invalid. |
| `PREDICTOR_BUILD_BLOCKED` | refused | 409 | This build item is blocked. |
| `PREDICTOR_BUILD_RECEIPT_INVALID` | refused | 409 | The durable predictor build receipt changed or is malformed. |
| `PREDICTOR_CHECKPOINT_CHANGED` | refused | 409 | A selected checkpoint is missing, empty, too large, or changed while being read. |
| `PREDICTOR_CLINICAL_PROVENANCE_CHANGED` | refused | 409 | Clinical preprocessing differs from its training patients. |
| `PREDICTOR_EVIDENCE_INVALID` | refused | 409 | Training evidence must be a regular file inside its own run directory. |
| `PREDICTOR_FEATURES_INVALID` | refused | 409 | Feature representation provenance is incomplete. |
| `PREDICTOR_FEATURE_KIND_MISMATCH` | refused | 422 | The trained model requires … features, but its frozen bundle holds … features. Train a compatible model in a new batch. |
| `PREDICTOR_FOLDS_INCOMPLETE` | refused | 409 | The candidate does not contain every frozen fold. |
| `PREDICTOR_INACTIVE` | refused | 409 | Restore this predictor to Active before adding it to a new evaluation batch. |
| `PREDICTOR_MODEL_UNSUPPORTED` | refused | 422 | Predictors are built from models of a trainable development design, from: …. |
| `PREDICTOR_PROCESS_UNKNOWN` | refused | 409 | Cannot confirm whether a selected training process has stopped. |
| `PREDICTOR_PROVENANCE_CHANGED` | refused | 409 | Seed groups of one configuration differ in their frozen training provenance. |
| `PREDICTOR_RECORD_INACTIVE` | refused | 409 | This exact configuration, seed group and method already exists in Archive or Trash. Restore that record explicitly before building. |
| `PREDICTOR_REFIT_REQUIRED` | refused | 422 | Create and run a refit plan, then publish its completed model. |
| `PREDICTOR_SELECTION_CHANGED` | refused | 409 | Configuration selection differs from the frozen batch. |
| `PREDICTOR_VALIDATION_SELECTION_REQUIRED` | refused | 409 | This batch promotes the configuration selected by its frozen validation metric after all configurations finish. |
| `PROJECT_ALREADY_REGISTERED` | refused | 409 | This project ID is already registered at another folder. |
| `PROJECT_DESCRIPTOR_INVALID` | refused | 422 | The project descriptor is invalid or uses an unsupported format. |
| `PROJECT_DESCRIPTOR_TOO_LARGE` | refused | 422 | The project descriptor exceeds the supported size. |
| `PROJECT_DESCRIPTOR_UNREADABLE` | refused | 403 | The project descriptor cannot be read. |
| `PROJECT_DESCRIPTOR_UNSAFE` | refused | 403, 422 | The project descriptor must not be a symbolic link. |
| `PROJECT_EXISTS` | refused | 409 | A project already exists in this folder. Load it instead. |
| `PROJECT_FOLDER_INACCESSIBLE` | refused | 403 | The project folder cannot be created or inspected. |
| `PROJECT_FOLDER_NOT_EMPTY` | refused | 409 | The selected folder is not empty. Choose a new or empty folder, or load its existing project. |
| `PROJECT_FOLDER_REASSIGNED` | refused | 409 | The folder now belongs to a different project. |
| `PROJECT_FOLDER_UNWRITABLE` | refused | 403 | The project folder cannot be written. Choose a writable location. |
| `PROJECT_ID_MISMATCH` | refused | 409 | This scientific database belongs to a different experiment. |
| `PROJECT_OUT_OF_SCOPE` | refused | 403 | This token reaches one project only: …. |
| `PROJECT_SOURCE_LIMIT` | refused | 422 | The project has reached the maximum of 1000 source folders. |
| `PROTOCOL_BUNDLE_CHANGED` | refused | 409, 422 | The feature bundle no longer matches the version saved in this older protocol. |
| `PROTOCOL_BUNDLE_MISMATCH` | refused | 409, 422 | This older protocol pins another feature bundle. Use that bundle or create a dataset-only protocol revision. |
| `PROTOCOL_PREFLIGHT_BLOCKED` | refused | 409 | Resolve all blocking protocol preflight findings before freezing. |
| `PROTOCOL_RECORD_LIMIT` | refused | 413 | This protocol supports at most 50,000 slide records. |
| `PUBLICATION_CONFLICT` | refused | 409 | The dataset destination is occupied by unregistered artifacts. |
| `RECALIBRATION_REQUIRES_LABELS` | refused | 409 | Recalibration is checked against labels. Add a reference standard to this run's cohort first. |
| `RECALIBRATION_UNAVAILABLE` | refused | 409 | This predictor records no development predictions to fit a map on. |
| `RECORD_TRASHED` | refused | 409 | The record, or one it uses, is in Trash. Restore it first. |
| `REFERENCE_DUPLICATE_SLIDES` | refused | 422 | The chosen datasets list a cohort slide more than once, so its label is ambiguous. |
| `REFERENCE_FIELD_UNKNOWN` | refused | 422 | Column '…' is not in every chosen dataset's data dictionary. |
| `REFERENCE_NO_LABELS` | refused | 422 | No cohort slide receives a label. Map the column's values to the classes. |
| `REFIT_BAG_PLANNING_INVALID` | refused | 422 | MIL bags cannot be planned for this refit. |
| `REFIT_BEST_EPOCH_UNAVAILABLE` | refused | 409 | This fold has no verifiable best checkpoint epoch. |
| `REFIT_CLINICAL_PROVENANCE_CHANGED` | refused | 409 | Refit clinical preprocessing differs from development patients. |
| `REFIT_EPOCH_INVALID` | refused | 409 | nnMIL refit requires its selected checkpoint epoch. |
| `REFIT_EVIDENCE_CHANGED` | refused | 409 | The saved refit plan's reviewed development evidence changed. |
| `REFIT_MEMBERSHIP_INVALID` | refused | 409 | Unknown development partition. |
| `REFIT_PROVENANCE_CHANGED` | refused | 409 | Refit output does not match the reviewed plan. |
| `REFIT_TEST_LEAKAGE` | refused | 409 | Test rows cannot enter a refit. |
| `REGEX_TIMEOUT` | refused | 409 | Regular-expression filtering exceeded its time budget. |
| `REVIEW_DATASET_INVALID` | refused | 409 | Frozen dataset records are invalid. |
| `REVIEW_REQUIRES_SLIDE_TARGET` | refused | 409 | Review predictions require a slide-level development target. |
| `REVISION_LIMIT` | refused | 409 | The version label revision limit was reached. |
| `RUN_NOT_SCORED` | refused | 409 | This run has no scores against its cohort's labels. |
| `SCAN_LIMIT` | refused | 413 | The source tree exceeds the bounded scan limit; choose a smaller root. |
| `SCOPE_MISSING` | refused | 403 | This token's scopes (…) do not reach this route. |
| `SEED_ENSEMBLE_SINGLE_GROUP` | refused | 422 | A seed ensemble pools two or more seed groups; this configuration has one. Use its fold ensemble instead. |
| `SEED_ENSEMBLE_TOO_LARGE` | refused | 422 | A seed ensemble holds at most … fold models. |
| `SESSION_NOT_FOR_TOKENS` | refused | 403 | An agent token cannot fetch the service's session; it reaches only its own project. |
| `SHARED_PATIENT_NAMESPACE` | refused | 409 | The same dataset must use shared patient identifiers. |
| `SLIDE_FORMAT_UNSUPPORTED` | refused | 422 | The whole-slide region cannot be read. |
| `SLIDE_GEOMETRY_INVALID` | refused | 422 | The slide dimensions are invalid. |
| `SLIDE_RASTER_TOO_LARGE` | refused | 422 | This image requires a pyramidal slide reader; raster decoding is bounded to 32 megapixels. |
| `SLIDE_READER_FAILED` | refused | 422 | The slide decoder could not read this slide. Retry the view; the HistoPilot service is still available. |
| `SLIDE_REVIEW_CORRUPT` | refused | 409 | Saved slide review failed validation. |
| `SLIDE_REVIEW_LIMIT` | refused | 413 | This review has reached its history storage limit. |
| `SLIDE_VIEWER_UNAVAILABLE` | refused | 503 | Slide viewing needs optional imaging libraries that are missing. |
| `SLIDE_VIEW_TOO_LARGE` | refused | 422 | This pyramid level is too large for a bounded view. Zoom into a smaller region. |
| `SPLIT_UNIT_MISMATCH` | refused | 409 | The cohort must use the predictor's slide or patient split unit. |
| `STALE_FEATURE_BUNDLE` | refused | 409 | The … feature bundle is stale or no longer verified. |
| `STALE_TEST_PACK` | refused | 409 | The selected test pack is no longer current. |
| `STORAGE_CORRUPT` | refused | 409 | Stored project data failed validation. |
| `STORAGE_DATABASE_INVALID` | refused | 409 | The scientific database is unavailable or invalid. |
| `STORAGE_DATABASE_MISSING` | refused | 409 | Scientific files exist but their database is missing or empty. Restore the project database. |
| `STORAGE_LOCK_FAILED` | refused | 403 | The workspace lifecycle lock is unavailable. |
| `STORAGE_READ_FAILED` | refused | 403 | A managed storage path cannot be inspected. |
| `STORAGE_SCHEMA_UNSUPPORTED` | refused | 409 | The scientific database version or format is unsupported. |
| `STORAGE_SYNC_FAILED` | refused | 409 | The project filesystem cannot synchronize directory changes. |
| `STORAGE_UNSAFE_PATH` | refused | 403 | Managed storage cannot traverse symbolic links. |
| `STORAGE_UNSUPPORTED` | refused | 409 | Project storage needs POSIX file locking and directory sync. |
| `STORAGE_WRITE_FAILED` | refused | 403 | Project storage cannot be written. |
| `TABLE_CELL_LIMIT` | refused | 413 | A metadata cell exceeds the supported size. |
| `TABLE_COLUMN_LIMIT` | refused | 413 | The worksheet exceeds 128 columns. |
| `TABLE_ROW_LIMIT` | refused | 413 | The worksheet exceeds 20,000 data rows. |
| `TABLE_TOO_LARGE` | refused | 413 | Metadata files must be at most 16 MiB. |
| `TARGET_LABEL_MAPPING_MISMATCH` | refused | 409 | The same label field in the same dataset must preserve development label mappings. |
| `TARGET_SPLIT_BLOCKED` | refused | 422 | Resolve blocking target/split findings before freezing. |
| `TARGET_SPLIT_PATIENT_LEAKAGE` | refused | 422 | Frozen training and testing share patient identities. |
| `TARGET_SPLIT_SLIDE_LEAKAGE` | refused | 422 | Frozen training and testing must contain unique slides. |
| `TARGET_SPLIT_SOURCE_CHANGED` | refused | 409 | Frozen partition slides are missing from the dataset. |
| `TARGET_TESTING_BLOCKED` | refused | 422 | The testing cohort cannot be prepared: … |
| `TASK_ACTIVE` | refused | 409 | This task is already queued or running. |
| `TASK_CENTER_STATE_UNSAFE` | refused | 403 | The Task Center state directory … belongs to another user; set HISTOPILOT_STATE_DIR to a directory of your own. |
| `TASK_CENTER_STORE_NEWER` | refused | 409 | The Task Center store was created by a newer HistoPilot. Update this checkout before using it. |
| `TASK_CENTER_STORE_UNSUPPORTED` | refused | 503 | The Task Center store requires SQLite write-ahead logging on a local disk. |
| `TASK_OWNER_OTHER_WORKSPACE` | refused | 409 | This task belongs to a project of another HistoPilot workspace. Manage it from the workspace that submitted it. |
| `TEST_PACK_UNAVAILABLE` | refused | 409 | The selected pack cannot be verified: … |
| `TOKEN_EXPOSURE_REQUIRED` | refused | 409 | Agents see nothing of this project: its AI-exposure level is none. |
| `TRAINING_ACTIVE` | refused | 409 | Tasks of this batch are still queued or running in the Task Center. |
| `TRAINING_ALREADY_LAUNCHED` | refused | 409 | This batch was already launched. Resume its unfinished runs or clone a new batch. |
| `TRAINING_BATCH_INVALID` | refused | 409 | The frozen batch identity is invalid. |
| `TRAINING_FEATURES_INVALID` | refused | 409 | All development slides need one compatible feature dimension. |
| `TRAINING_GPU_UNAVAILABLE` | refused | 409 | A requested GPU is unavailable. |
| `TRAINING_INPUTS_STALE` | refused | 409 | Training inputs are no longer current: … |
| `TRAINING_LOCATION_CHANGED` | refused | 409 | The saved execution belongs to a different output location or training session. |
| `TRAINING_NOT_LAUNCHED` | refused | 409 | This batch has not been launched. |
| `TRAINING_NOT_RESUMABLE` | refused | 409 | Only interrupted, failed, or cancelled batches can be resumed. |
| `TRAINING_OOF_INVALID` | refused | 409 | A saved training identity is invalid. |
| `TRAINING_PARTITION_MISSING` | refused | 409 | Every fold requires training, validation, and development assessment rows. |
| `TRAINING_PLAN_CHANGED` | refused | 409 | The saved execution plan changed after launch; it cannot be resumed. |
| `TRAINING_PROCESS_UNKNOWN` | refused | 409 | Worker process identity is invalid. |
| `TRAINING_REGISTRY_UNSAFE` | refused | 403 | Training resource registry is unsafe. |
| `TRAINING_RESOURCES_UNAVAILABLE` | refused | 409 | CPU threads plus overlapping training and validation data workers exceed the available CPU capacity per run. |
| `TRAINING_RUNTIME_CHANGED` | refused | 409 | The compute environment changed; restore its original package versions. |
| `TRAINING_RUNTIME_UNAVAILABLE` | refused | 409 | The optional training runtime is unavailable. |
| `TRAINING_RUN_INVALID` | refused | 409 | The frozen run identity is invalid. |
| `TRAINING_SPLITS_CHANGED` | refused | 409 | Frozen split plans do not match this batch. |
| `TRAINING_SPLIT_BLOCKED` | refused | 422 | Training design is not feasible: … |
| `TRAINING_SPLIT_UNSUPPORTED` | refused | 422 | Training needs a development-only training design. |
| `TRAINING_STATE_INVALID` | refused | 409 | Invalid training metadata. |
| `TRAINING_TASK_UNSUPPORTED` | refused | 422 | MIL training supports binary and multiclass classification. |
| `UNKNOWN_FIELD` | refused | 409 | The field '…' is not in the frozen dataset dictionary. |
| `UNKNOWN_TARGET_FIELD` | refused | 409 | Select an available target label field. |
| `UNMAPPED_TARGET_LABEL` | refused | 409 | Selected slides have missing or unmapped target labels. Map them, explicitly exclude them, or keep them unlabeled. |
| `UPLOAD_TOO_LARGE` | refused | 413 | Inline metadata uploads must be at most 256 KiB; use a server file path for larger tables. |
| `VALIDATION_SELECTION_UNAVAILABLE` | refused | 409 | Complete valid validation scores are required for configuration selection. |
| `VERSION_LABEL_MISMATCH` | refused | 409 | These scientific contents are already saved as … with the tag …. |
| `VERSION_TAG_CONFLICT` | refused | 409 | This tag is already used or reserved for another … version in this project. Choose a different tag. |

## conflict

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `CLEANUP_PREVIEW_STALE` | conflict | 409 | Workspace records or job status changed. Review cleanup again. |
| `COMPUTE_LAUNCH_UNCERTAIN` | conflict | 409 | Launch acknowledgement was lost. Check execution status before retrying. |
| `EVALUATION_COHORT_STALE` | conflict | 409 | The test dataset or target membership changed. Review and freeze the cohort again. |
| `EVALUATION_NOT_COMPLETED` | conflict | 409 | Evaluation results are available after the job finishes. |
| `EXPERIMENT_PREDICTORS_LAUNCH_UNCERTAIN` | conflict | 409 | Predictor launch acknowledgement was lost. Inspect its status before retrying. |
| `EXPOSURE_CONFLICT` | conflict | 409 | The AI-exposure level is now …; review it before changing it. |
| `FEATURE_JOB_BUSY` | conflict | 409 | This feature version already has an active job for this action. |
| `FOLDER_PARENT_CHANGED` | conflict | 409 | The parent folder changed. Browse it again before creating a folder. |
| `INTERPRETATION_FOLDER_CHANGED` | conflict | 409 | A slide folder changed during discovery. Browse it again. |
| `INTERPRETATION_INPUTS_CHANGED` | conflict | 409 | The slide changed while its thumbnail was rendered. |
| `INTERPRETATION_NOT_COMPLETED` | conflict | 409 | Attention is available after execution completes. |
| `INTERPRETATION_SOURCE_MISMATCH` | conflict | 409 | The source selection changed during visualization. |
| `LIFECYCLE_REVISION_CONFLICT` | conflict | 409 | Workspace records changed. Review cleanup again before applying it. |
| `MORPHOLOGY_BUSY` | conflict | 409 | Another morphology index is being prepared. Try again shortly. |
| `MORPHOLOGY_INDEX_EXPIRED` | conflict | 409 | This temporary index expired. Build the projection again. |
| `MORPHOLOGY_SLIDE_CHANGED` | conflict | 409 | The slide file changed since it was prepared. Reopen the slide. |
| `OUTPUT_BUSY` | conflict | 409 | Another job is using this output folder. Retry after it finishes. |
| `PORTABILITY_ACTIVE_JOBS` | conflict | 409 | Finish or cancel active jobs before exporting a consistent project. |
| `PREDICTOR_RUN_INCOMPLETE` | conflict | 409 | Every selected fold must have finished and stopped writing. |
| `PREVIEW_STALE` | conflict | 409 | Inputs changed since the preview. Preview again, then send its new hash. |
| `PROJECT_BUSY` | conflict | 409 | Another operation is writing this project. Retry after it finishes. |
| `PROJECT_CONFIG_CONFLICT` | conflict | 409 | Project settings changed in another tab. Your edits have not been saved. Reload the saved settings before making further changes. |
| `PUBLICATION_INTERRUPTED` | conflict | 403 | Dataset publication was interrupted; reopen to recover. |
| `REFIT_INCOMPLETE` | conflict | 409 | The refit must complete and stop before publication. |
| `REVISION_CONFLICT` | conflict | 409 | The record changed since it was read. Reload it, then retry. |
| `RUN_NOT_COMPLETED` | conflict | 409 | Only a completed run has predictions to score. |
| `SLIDE_READER_BUSY` | conflict | 503 | Slide images are still being prepared. Retry this view shortly. |
| `SLIDE_READER_TIMEOUT` | conflict | 504 | The slide reader timed out. Retry a smaller view or check the slide file. |
| `SLIDE_REVIEW_CONFLICT` | conflict | 409 | This review changed in another tab. Your edits are preserved; reload the saved review before merging. |
| `SLIDE_SOURCE_CHANGED` | conflict | 409 | The slide changed or is no longer available. Reopen it before viewing. |
| `SOURCE_CHANGED` | conflict | 409 | The metadata file changed while being read. |
| `SOURCE_RELINK_CONFLICT` | conflict | 409 | The source changed in another tab. Refresh before relinking. |
| `TASK_CENTER_CONFLICT` | conflict | 409 | The Task Center store rejected a conflicting change: … |
| `TASK_CENTER_UNAVAILABLE` | conflict | 503 | The Task Center store cannot be reached or updated right now. |
| `TASK_RETRY_BUSY` | conflict | 409 | Wait for the running tasks of this batch to finish before retrying it. |
| `TRAINING_CLEANUP_FAILED` | conflict | 409 | Cancellation was requested for every orphan worker, but some workers could not be confirmed stopped. Their reservations are retained. |
| `TRAINING_OOF_INCOMPLETE` | conflict | 409 | Every fold in this seed group must finish before export. |

## not-found

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `AGENT_REQUEST_NOT_FOUND` | not-found | 404 | This agent request does not exist. |
| `API_ENDPOINT_UNKNOWN` | not-found | 404 | No API route has this path. |
| `API_METHOD_NOT_ALLOWED` | not-found | 405 | The API route does not accept this HTTP method. |
| `ARTIFACT_NOT_FOUND` | not-found | 404 | The requested artifact is not declared in this dataset. |
| `CLINICAL_ANALYSIS_NOT_FOUND` | not-found | 404 | Clinical analysis not found. |
| `CLINICAL_ARTIFACT_NOT_FOUND` | not-found | 404 | Clinical artifact not found. |
| `COMPUTE_NOT_FOUND` | not-found | 404 | The compute record does not exist. |
| `CONFIGURATION_NOT_FOUND` | not-found | 404 | The frozen configuration does not exist. |
| `DATASET_NOT_FOUND` | not-found | 404 | The frozen dataset does not exist. |
| `DIRECTORY_NOT_FOUND` | not-found | 404 | The selected directory does not exist. |
| `DRAFT_NOT_FOUND` | not-found | 404 | The scientific draft does not exist. |
| `EVALUATION_ARTIFACT_NOT_FOUND` | not-found | 404 | Evaluation artifact not found. |
| `EVALUATION_BATCH_NOT_FOUND` | not-found | 404 | Evaluation batch not found. |
| `EVALUATION_NOT_FOUND` | not-found | 404 | Test cohort not found. |
| `EXPERIMENT_NOT_FOUND` | not-found | 404 | This experiment does not exist in this project. |
| `EXTRACTION_NOT_FOUND` | not-found | 404 | Extraction not found. |
| `FEATURE_BUNDLE_NOT_FOUND` | not-found | 404 | Feature bundle not found. |
| `FOLDER_PARENT_NOT_FOUND` | not-found | 404 | The parent folder no longer exists. Browse to an existing folder. |
| `INTERPRETATION_ARTIFACT_NOT_FOUND` | not-found | 404 | Attention artifact not found. |
| `INTERPRETATION_NOT_FOUND` | not-found | 404 | Interpretation not found. |
| `INTERPRETATION_PATCH_NOT_FOUND` | not-found | 404 | Patch not found on this slide. |
| `INTERPRETATION_SLIDE_NOT_FOUND` | not-found | 404 | Selected slide not found. |
| `JOB_NOT_FOUND` | not-found | 404 | Select an existing compute job or launched training batch. |
| `MODEL_EVALUATION_NOT_FOUND` | not-found | 404 | Evaluation record not found. |
| `PACKING_NOT_FOUND` | not-found | 404 | Packing job not found. |
| `PACK_NOT_FOUND` | not-found | 404 | The existing pack folder is unavailable. |
| `PATH_NOT_FOUND` | not-found | 404 | The metadata file cannot be resolved. |
| `PORTABILITY_NOT_FOUND` | not-found | 404 | Archive operation not found. |
| `PREDICTOR_BUILD_NOT_FOUND` | not-found | 404 | Predictor build receipt not found. |
| `PREDICTOR_NOT_FOUND` | not-found | 404 | Predictor not found. |
| `PROJECT_DESCRIPTOR_NOT_FOUND` | not-found | 404 | This folder has no histopilot-project.json. Select an existing HistoPilot project. |
| `PROJECT_NOT_FOUND` | not-found | 404 | The project does not exist. Load its folder first. |
| `REFERENCE_NOT_FOUND` | not-found | 404 | Reference standard not found. |
| `REFIT_NOT_FOUND` | not-found | 404 | Refit plan not found. |
| `REVIEW_SLIDE_NOT_FOUND` | not-found | 404 | This slide is not in the frozen dataset. |
| `TASK_LOG_NOT_FOUND` | not-found | 404 | This task has no log. |
| `TASK_NOT_FOUND` | not-found | 404 | This task does not exist. |
| `TASK_OWNER_NOT_FOUND` | not-found | 404 | This task owner does not exist. |
| `TOKEN_NOT_FOUND` | not-found | 404 | This access token does not exist. |
| `TRAINING_OOF_NOT_FOUND` | not-found | 404 | This configuration and seed group is not in the batch. |
| `TRAINING_RUN_NOT_FOUND` | not-found | 404 | Run … is not part of this batch. |

## unavailable

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `BROWSER_CONTEXT_REFUSED` | unavailable | 403 | Unrecognized browser request context. |
| `CROSS_SITE_REFUSED` | unavailable | 403 | Cross-site browser requests are not permitted. |
| `HOST_REFUSED` | unavailable | 400 | Unrecognized local service Host. |
| `LOGIN_REQUIRED` | unavailable | 401 | This service requires sign-in: open the link it printed when it started, or run `histopilot login url` as the same user. |
| `ORIGIN_REFUSED` | unavailable | 403 | This browser Origin is not permitted. |
| `ORIGIN_REQUIRED` | unavailable | 403 | Same-site browser requests require an approved Origin. |
| `SESSION_TOKEN_REQUIRED` | unavailable | 401 | A valid local session token is required. |
| `TOKEN_INVALID` | unavailable | 401 | This access token is unknown, expired or revoked. |

## internal

| Code | Kind | HTTP status | Meaning |
| --- | --- | --- | --- |
| `COMPUTE_LAUNCH_FAILED` | internal | 409 | Cannot launch the compute worker: … |
| `INTERNAL_ERROR` | internal | 500 | HistoPilot hit an unexpected error. The server log has the details. |
| `TASK_ACTION_FAILED` | internal | 503 | The owning record could not be updated: … |
| `TRAINING_LAUNCH_FAILED` | internal | 409 | Training tasks could not be queued: … |

## Deprecated codes

The service still returns these names in some places. Treat each one as the code that replaces it.

| Code | Kind | HTTP status | Replaced by |
| --- | --- | --- | --- |
| `STALE_PREVIEW` | conflict | 409 | `PREVIEW_STALE` |
