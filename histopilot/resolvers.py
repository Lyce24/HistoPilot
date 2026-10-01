"""The browser's default choices, as pure functions over the service's own records.

Each function ports one browser helper, named in its docstring, and reads the same JSON the
browser reads. `web/src/lib/resolverCases.json` holds cases that `tests/test_resolvers.py`
and `web/src/lib/resolvers.test.ts` both run, so the CLI, agents and the browser propose
the same predictors, cohorts, targets, arms and feature specs. Nothing here reads the
service: callers such as `histopilot.client.resolve` fetch the records first.
"""

from copy import deepcopy

from histopilot import templates
from histopilot.models import catalog

# Limits and vocabulary the service's request models and services share with the browser's
# ports; this module imports neither pydantic nor the service.
MAX_APPLY_PREDICTORS = 256
MAX_COMPARISON_ARMS = 8
APPLY_METHODS = ("seed_ensemble", "both", "ensemble", "refit")
ARM_INPUTS = ("image", "multimodal", "clinical")
# Unlabeled cohorts that receive predictions only; ``review`` is the earlier name of the same
# cohort type, kept for stored records.
INFERENCE_PURPOSES = ("inference", "review")


def given(value, default):
    """JavaScript's `value ?? default`: only a missing value takes the default."""
    return default if value is None else value


def _manifest(record) -> dict:
    return (record or {}).get("manifest") or {}


def _method(predictor) -> str:
    """A predictor's method; records saved before methods existed are fold ensembles."""
    return given(_manifest(predictor).get("method"), "ensemble")


def _finding(code: str, message: str, severity: str = "error") -> dict:
    return {"code": code, "message": message, "severity": severity}


# Apply models ------------------------------------------------------------------------------


def experiment_predictors(predictors: list, experiment_ids, method: str) -> list:
    """`experimentPredictors` (web/src/lib/evaluationSelection.ts): the experiments' active
    predictors of one method. `both` is every fold ensemble and refit, never a seed ensemble."""
    sources = set(experiment_ids)
    chosen = []
    for item in predictors:
        manifest = _manifest(item)
        if item.get("lifecycleState") != "active" or manifest.get("experimentId") not in sources:
            continue
        if method == "both":
            fits = manifest.get("method") != "seed_ensemble"
        else:
            fits = _method(item) == method
        if fits:
            chosen.append(item)
    return chosen


def default_method(predictors: list, experiment_ids, linked_predictor: str | None = None) -> str:
    """`defaultEvaluationMethod` (web/src/lib/evaluationSelection.ts): a linked predictor
    keeps its own method; otherwise seed ensembles when any is ready, since they are the
    deployable form of the Results headline; otherwise fold ensembles and refits."""
    if linked_predictor:
        return _method(
            next((item for item in predictors if item.get("id") == linked_predictor), None)
        )
    sources = set(experiment_ids)
    ready = any(
        item.get("lifecycleState") == "active"
        and _manifest(item).get("experimentId") in sources
        and _method(item) == "seed_ensemble"
        for item in predictors
    )
    return "seed_ensemble" if ready else "both"


def reserved_testing_cohort(protocol_ids, protocols: list, cohorts: list) -> dict | None:
    """`reservedTestingCohort` (web/src/lib/applyModels.ts): the current cohort that Targets &
    splits derived for the protocols the predictors were trained on, or None."""
    wanted = set(protocol_ids)
    splits = set()
    for item in protocols:
        if item.get("id") not in wanted:
            continue
        manifest = _manifest(item)
        split = (manifest.get("sourceTargetSplit") or {}).get("id")
        if split is None:
            split = (manifest.get("spec") or {}).get("sourceTargetSplitId")
        if split:
            splits.add(split)
    for cohort in cohorts:
        spec = _manifest(cohort).get("spec") or {}
        if (
            cohort.get("current") is not False
            and given(spec.get("sourceTargetSplitId"), "") in splits
        ):
            return cohort
    return None


def cohort_labeled(cohort) -> bool:
    """`cohortLabeled`: an unlabeled cohort (purpose `inference`, formerly `review`) is
    predicted only."""
    return (_manifest(cohort).get("spec") or {}).get("purpose") not in INFERENCE_PURPOSES


def initial_inputs(cohort=None) -> dict:
    """`initialEvaluationInputs` (web/src/components/EvaluationInputSettings.tsx): the
    cohort's feature bundle, patient identity and inference settings over the apply
    defaults. A predictor always applies its own frozen aggregation and threshold."""
    spec = _manifest(cohort).get("spec") or {}
    apply = templates.APPLY["inference"]
    return {
        "featureBundleId": given(spec.get("featureBundleId"), ""),
        "patientIdentifiers": given(spec.get("patientIdentifiers"), "shared"),
        "inference": deepcopy(
            {
                **apply,
                **(spec.get("inference") or {}),
                "patientAggregation": apply["patientAggregation"],
                "decisionThreshold": apply["decisionThreshold"],
            }
        ),
    }


def execution_selection(inputs: dict) -> dict:
    """`evaluationExecutionSelection`: the feature bundle only when one is named."""
    return {
        **({"featureBundleId": inputs["featureBundleId"]} if inputs["featureBundleId"] else {}),
        "inference": inputs["inference"],
        "patientIdentifiers": inputs["patientIdentifiers"],
    }


def apply_selection(
    predictors: list,
    protocols: list,
    cohorts: list,
    experiment_ids,
    *,
    method: str | None = None,
    cohort_id: str | None = None,
    linked_predictor: str | None = None,
    name: str = "",
) -> dict:
    """What Apply models proposes (web/src/components/BulkEvaluationRunner.tsx): the method,
    the ready predictors, the testing cohort their development reserved, and the cohort's
    inference settings. Findings say why the selection cannot be reviewed yet."""
    experiment_ids = list(dict.fromkeys(experiment_ids))
    chosen_method = method or default_method(predictors, experiment_ids, linked_predictor)
    candidates = experiment_predictors(predictors, experiment_ids, chosen_method)
    protocol_ids = list(
        dict.fromkeys(
            ((_manifest(item).get("inputs") or {}).get("protocol") or {}).get("id")
            for item in candidates
        )
    )
    reserved = reserved_testing_cohort(protocol_ids, protocols, cohorts)
    chosen_cohort = cohort_id or (reserved["id"] if reserved else "")
    cohort = next((item for item in cohorts if item.get("id") == chosen_cohort), None)
    eligible = list(dict.fromkeys(item["id"] for item in candidates))
    if linked_predictor:
        selected = [linked_predictor] if linked_predictor in eligible else []
    else:
        selected = eligible
    inputs = initial_inputs(cohort)
    prefixes = templates.APPLY["namePrefix"]
    labeled = cohort is None or cohort_labeled(cohort)
    selection = {
        "cohortId": chosen_cohort,
        "scope": "selected",
        "predictorIds": selected,
        **execution_selection(inputs),
        "namePrefix": name.strip() or prefixes["labeled" if labeled else "unlabeled"],
    }
    ready = {"seed_ensemble": 0, "ensemble": 0, "refit": 0}
    for item in candidates:
        ready[_method(item)] = ready.get(_method(item), 0) + 1

    findings = []
    if not experiment_ids:
        findings.append(_finding("EXPERIMENTS_REQUIRED", "Choose one or more experiments."))
    elif linked_predictor and not selected:
        findings.append(
            _finding(
                "PREDICTOR_UNAVAILABLE",
                f"Predictor {linked_predictor} is not an active predictor of these experiments.",
            )
        )
    elif not candidates:
        hint = (
            " Build their seed ensembles first (`histopilot predictor seed-ensemble`), or "
            "choose another method."
            if chosen_method == "seed_ensemble"
            else ""
        )
        findings.append(
            _finding(
                "NO_READY_PREDICTORS",
                "The selected experiments have no ready predictors for this method." + hint,
            )
        )
    if len(selected) > MAX_APPLY_PREDICTORS:
        findings.append(
            _finding(
                "TOO_MANY_PREDICTORS",
                f"This selection has {len(selected)} predictors; one batch takes at most "
                f"{MAX_APPLY_PREDICTORS}. Choose fewer experiments or another method.",
            )
        )
    if cohort is None:
        findings.append(
            _finding(
                "COHORT_REQUIRED",
                f"No cohort {chosen_cohort} in this project."
                if chosen_cohort
                else "Choose a cohort: these predictors' development reserved no testing cohort.",
            )
        )
    elif cohort.get("current") is False:
        findings.append(
            _finding(
                "COHORT_NEEDS_VERIFICATION",
                "This cohort's inputs changed since it was frozen; the review will check it.",
                "warning",
            )
        )
    return {
        "selection": selection,
        "method": chosen_method,
        "methodSource": "chosen" if method else "linked" if linked_predictor else "default",
        "cohortSource": "chosen" if cohort_id else "reserved" if reserved else None,
        "cohortLabeled": None if cohort is None else cohort_labeled(cohort),
        "ready": ready,
        "findings": findings,
    }


def short_record_id(identity: str) -> str:
    """`shortRecordId` (web/src/lib/recordLabels.ts): a long ID keeps its two ends."""
    return f"{identity[:12]}…{identity[-10:]}" if len(identity) > 28 else identity


def predictor_method_label(method) -> str:
    """`predictorMethodLabel` (web/src/api/predictors.ts)."""
    if method == "refit":
        return "Refit"
    return "Seed ensemble" if method == "seed_ensemble" else "Fold ensemble"


def predictor_seed_label(manifest: dict) -> str:
    """`predictorSeedLabel` (web/src/api/predictors.ts): "Train 42 / split 42" for one seed
    group; the seed counts and models for a seed ensemble."""
    if manifest.get("trainingSeeds") is not None:
        training = len(manifest["trainingSeeds"])
        splits = manifest.get("splitSeeds")
        split = 1 if splits is None else len(splits)
        models = len(manifest.get("checkpoints") or [])
        return (
            f"{training} training × {split} split seed{'' if split == 1 else 's'} · {models} models"
        )
    return f"Train {manifest.get('trainingSeed')} / split {manifest.get('splitSeed')}"


def predictor_configuration_label(manifest: dict) -> str:
    """`predictorConfigurationLabel` (web/src/lib/predictorGroups.ts)."""
    short = short_record_id(str(manifest.get("candidateId") or ""))
    return f"Configuration {given(manifest.get('candidateNumber'), short)}"


def batch_names(experiments: list):
    """`batchNameLookup` (web/src/lib/applyModels.ts): ``name(batch_id)``, the batch's name
    from the experiments that list it, else a short form of its ID."""
    names = {
        batch.get("id"): batch.get("name")
        for experiment in experiments
        for batch in experiment.get("batches") or []
    }
    return lambda batch_id: given(names.get(batch_id), f"Batch {short_record_id(str(batch_id))}")


def model_description(manifest: dict, batch_name) -> str:
    """`modelDescription` (web/src/lib/applyModels.ts): "ABMIL baseline · Configuration 1 ·
    Seed ensemble · 3 training × 1 split seed · 15 models". Predictor names omit their batch,
    so two models' predictors of one experiment, configuration and seeds share a name; the
    batch tells them apart."""
    return " · ".join(
        [
            batch_name(manifest.get("batchId")),
            predictor_configuration_label(manifest),
            predictor_method_label(manifest.get("method")),
            predictor_seed_label(manifest),
        ]
    )


def configuration_choice(
    choices: list, predictors: list, experiment: dict, batch_id: str, candidate_id: str
) -> dict:
    """`configurationChoice` (web/src/lib/applyModels.ts), behind "Apply this configuration":
    a built seed ensemble, a seed ensemble to build from verified checkpoints (nothing
    trains), the configuration's single fold ensemble, or a choice among several."""
    choice = next(
        (
            item
            for item in choices
            if item.get("batchId") == batch_id
            and item.get("candidateId") == candidate_id
            and (item.get("seedGroups") or 0) > 1
        ),
        None,
    )
    active = {item.get("id") for item in predictors if item.get("lifecycleState") == "active"}
    folds = [
        item["id"]
        for item in predictors
        if item.get("lifecycleState") == "active"
        and _manifest(item).get("experimentId") == experiment.get("id")
        and _manifest(item).get("batchId") == batch_id
        and _manifest(item).get("candidateId") == candidate_id
        and _method(item) == "ensemble"
    ]
    seeds = ""
    if choice:
        split = len(choice.get("splitSeeds") or [])
        seeds = (
            f"{len(choice.get('trainingSeeds') or [])} training × {split} split "
            f"{'seed' if split == 1 else 'seeds'} · {choice.get('members')} fold models"
        )
    existing = (choice or {}).get("existingPredictorId")
    if existing and existing in active:
        return {
            "action": "apply",
            "predictorId": existing,
            "detail": f"Its seed ensemble is built: {seeds}.",
        }
    if choice and choice.get("eligible"):
        number = choice.get("candidateNumber")
        name = f"{experiment.get('name')} · configuration {number} · seed ensemble"
        return {
            "action": "build-seed-ensemble",
            "selection": {
                "experimentId": choice.get("experimentId"),
                "batchId": choice.get("batchId"),
                "candidateId": choice.get("candidateId"),
                "name": name[:120],
            },
            "detail": f"Its seed ensemble ({seeds}) is built from the verified fold "
            "checkpoints; nothing trains.",
        }
    if choice is None and len(folds) == 1:
        return {"action": "apply", "predictorId": folds[0], "detail": "Its fold ensemble is ready."}
    reason = (choice or {}).get("reason")
    if reason is None:
        reason = (
            f"{len(folds)} fold ensembles are ready, one per seed group; choose among them in "
            "Apply models."
            if folds
            else "No predictor is ready for this configuration yet."
        )
    return {"action": "choose" if folds else "none", "predictorIds": folds, "detail": reason}


# Targets and labels ------------------------------------------------------------------------


def infer_target(values, truncated: bool = False) -> dict:
    """`inferTargetSettings` (web/src/lib/protocol.ts): the field's distinct non-blank values
    as classes, two for binary and more for multiclass. Nothing is inferred from a truncated
    value list, and value order never names the positive class."""
    classes = (
        []
        if truncated
        else list(dict.fromkeys(value for value in values if value is not None and value.strip()))
    )
    task = (
        "binary_classification"
        if len(classes) == 2
        else "multiclass_classification"
        if len(classes) > 2
        else ""
    )
    return {
        "task": task,
        "classes": classes,
        "labels": {value: value for value in classes},
        "positiveClass": None,
    }


def training_target(spec: dict, target: dict) -> dict:
    """`targetSplitTrainingTarget` (web/src/api/targetSplits.ts): a separate testing target
    keeps the training target's task, unit, classes and positive class."""
    changed = {"target": target}
    if spec.get("testTarget"):
        changed["testTarget"] = {
            **spec["testTarget"],
            "task": target.get("task"),
            "unit": target.get("unit"),
            "classes": target.get("classes"),
            "positiveClass": target.get("positiveClass"),
        }
    return changed


def partition_request(spec: dict) -> dict:
    """`targetSplitPartitionRequest(spec, undefined, true)`: the raw values of the target
    fields in each partition, without mapping them to classes."""
    testing = spec["target"] if "testTarget" not in spec else spec["testTarget"]
    fields = {"train": spec["target"].get("field") or None}
    if testing is not None and testing.get("field"):
        fields["test"] = testing["field"]
    request = {"datasetId": spec.get("datasetId")}
    if spec.get("splitUnit"):
        request["splitUnit"] = spec["splitUnit"]
    for key in ("eligibility", "split"):
        if spec.get(key) is not None:
            request[key] = spec[key]
    request["targetFields"] = {key: value for key, value in fields.items() if value}
    if "testTarget" in spec and spec["testTarget"] is None:
        request["testTarget"] = None
    return request


def reference_fits(standard: dict, cohort_id: str, classes) -> bool:
    """`referenceFits` (web/src/api/references.ts): a reference standard for the run's cohort
    with exactly the run's classes."""
    manifest = _manifest(standard)
    return (
        standard.get("lifecycleState") != "trashed"
        and manifest.get("cohortId") == cohort_id
        and sorted(manifest.get("classes") or []) == sorted(classes)
    )


def run_labeled(run) -> bool:
    """`runLabeled`: a run on an unlabeled cohort has no labels of its own."""
    return _manifest(run).get("purpose") not in INFERENCE_PURPOSES


def label_sources(run: dict, standards: list) -> list[dict]:
    """`labelSources` (web/src/components/ApplyRunDetail.tsx): the cohort's own labels for a
    labeled run, then every fitting active reference standard by name."""
    manifest = _manifest(run)
    classes = (manifest.get("target") or {}).get("classes") or []
    fitting = sorted(
        (
            item
            for item in standards
            if reference_fits(item, manifest.get("cohortId"), classes)
            and given(item.get("lifecycleState"), "active") == "active"
        ),
        key=lambda item: (_manifest(item).get("name") or "", item.get("id") or ""),
    )
    own = [{"id": None, "name": "Cohort labels"}] if run_labeled(run) else []
    return own + [{"id": item["id"], "name": _manifest(item).get("name")} for item in fitting]


def label_source(sources: list[dict]) -> dict | None:
    """`labelSource`: the labels a run is scored against when none is named, the first
    source, which is the cohort's own labels for a labeled run."""
    return sources[0] if sources else None


# Controlled comparisons --------------------------------------------------------------------


def clinical_only_recipe(recipe: dict) -> dict:
    """`clinicalOnlyRecipe` (web/src/components/ClinicalInputFields.tsx)."""
    return {**recipe, **deepcopy(templates.COUPLING["clinicalOnly"])}


def matched_input_recipes(recipe: dict) -> list[dict]:
    """`matchedInputRecipes`: image only, clinical only and clinical + image."""
    if not recipe.get("clinicalFields"):
        raise ValueError("Select clinical fields before creating matched arms.")
    return [
        {**recipe, "inputMode": "image", "clinicalFields": []},
        clinical_only_recipe(recipe),
        {**recipe, "inputMode": "multimodal"},
    ]


def with_arm_model(base: dict, model: str) -> dict:
    """`withArmModel` (web/src/components/ControlledComparison.tsx): the base recipe with
    another model. Only settings the model owns change, so the comparison measures it alone."""
    if model == base.get("model"):
        return base
    reference = templates.recipe()
    arm = templates.COUPLING["comparisonArm"]
    if model == "nnmil":
        own = deepcopy(arm["nnmil"])
    else:
        own = deepcopy(arm["nnmilOff"])
        if model == "abmil":
            own.update(
                attentionDim=reference["attentionDim"], gatedAttention=reference["gatedAttention"]
            )
    slide = (
        deepcopy(templates.COUPLING["modelSelect"]["slide"])
        if catalog.feature_kind(model) == "slide"
        and catalog.feature_kind(base.get("model")) != "slide"
        else {}
    )
    return {**base, "model": model, **own, **slide}


def ablation_arms(base: dict, models, inputs) -> list[dict]:
    """`ablationArms`: the base recipe first, as the reference, then every chosen model ×
    input; one clinical-only arm, since it reads no image whatever the model."""
    base_input = given(base.get("inputMode"), "image")
    clinical = bool(base.get("clinicalFields"))
    arms = [base]
    for input_mode in ("image", "multimodal"):
        if input_mode not in inputs or (input_mode == "multimodal" and not clinical):
            continue
        for model in models:
            if model == base.get("model") and input_mode == base_input:
                continue
            arm_base = with_arm_model(base, model)
            variants = (
                matched_input_recipes(arm_base)
                if clinical
                else [{**arm_base, "inputMode": "image", "clinicalFields": []}]
            )
            arms.append(
                next(arm for arm in variants if given(arm.get("inputMode"), "image") == input_mode)
            )
    if "clinical" in inputs and clinical and base_input != "clinical":
        arms.append(clinical_only_recipe(base))
    return arms


def comparison_batch(batch: dict, arms: list[dict], primary_metric: str = "auroc") -> dict:
    """`comparisonFromArms` then `batchSpecification`: the arms as explicit configurations,
    every one built, the first the reference."""
    if not 2 <= len(arms) <= MAX_COMPARISON_ARMS:
        raise ValueError(
            f"A controlled comparison has 2 to {MAX_COMPARISON_ARMS} arms; these choices make "
            f"{len(arms)}."
        )
    return {
        **batch,
        "mode": "explicit",
        "configurations": arms,
        "candidateSelection": "all",
        "comparison": {"reference": 1, "primaryMetric": primary_metric},
    }


# Features ----------------------------------------------------------------------------------


def extraction_feature_input(job: dict) -> dict | None:
    """`extractionFeatureInput` (web/src/lib/featureSource.ts): the features a succeeded
    extraction wrote, or None before it succeeds."""
    result = job.get("result") or {}
    layout = given(result.get("outputLayout"), job.get("outputLayout")) or {}
    path = given(result.get("featurePath"), result.get("featureDirectory"))
    if job.get("state") != "succeeded" or not path:
        return None
    spec = job.get("spec") or {}
    options = spec.get("options") or {}
    encoder = options.get(
        "slide_encoder" if layout.get("featureKind") == "slide" else "patch_encoder"
    )
    return {
        "datasetId": spec.get("datasetId"),
        "path": path,
        "encoderId": str(encoder or "") or None,
        "featureKind": given(layout.get("featureKind"), "patch"),
        "sourceExtractionJobId": job.get("id"),
    }


def feature_spec_from_extraction(job: dict) -> dict | None:
    """`editFeatureSource(starter, extractionFeatureUpdate(input))` (web/src/lib/
    featureSource.ts): the features starter for an extraction's outputs, read exactly as
    written: that folder only, no slide list."""
    source = extraction_feature_input(job)
    if source is None:
        return None
    return {
        **templates.spec_starter("features"),
        **source,
        "slideList": None,
        "slideListPath": None,
        "layout": "auto",
        "fileSuffix": ".h5",
        "recursive": False,
        "idSuffix": "",
        "coordinatesPath": None,
    }
