"""Persist development batches without reassigning any frozen protocol membership."""

from itertools import product

from pydantic import ValidationError

from histopilot.application.feature_bundles import FeatureBundleService, _hash
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.model_experiments import ModelExperimentService, input_snapshot
from histopilot.application.protocols import FilterEvaluator, ProtocolService
from histopilot.models import catalog
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.schemas.nnmil import resolve_nnmil_recipe
from histopilot.schemas.training_controls import (
    sampling_memberships,
    validate_selection_metric,
    validate_training_controls,
)
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

MAX_CONFIGURATIONS = 512
MAX_RUNS = 20000


def expand_recipes(spec: DevelopmentBatchSpec) -> list[dict]:
    if spec.mode == "explicit":
        recipes = [item.model_dump() for item in spec.configurations]
    elif spec.mode == "grid":
        grid = spec.grid
        if (
            len(grid.learningRates) * len(grid.weightDecays) * len(grid.maxEpochs)
            > MAX_CONFIGURATIONS
        ):
            raise StorageError("Limit a batch to 512 configurations.", "BATCH_TOO_LARGE", 422)
        recipes = [
            {**spec.recipe.model_dump(), "learningRate": lr, "weightDecay": wd, "maxEpochs": epochs}
            for lr, wd, epochs in product(grid.learningRates, grid.weightDecays, grid.maxEpochs)
        ]
    else:
        recipes = [spec.recipe.model_dump()]
    try:
        recipes = [TrainingRecipe.model_validate(recipe, context={"legacy": True}).model_dump()
                   for recipe in recipes]
    except ValidationError as error:
        raise StorageError(
            f"A resolved training recipe is invalid: {error}", "INVALID_TRAINING_RECIPE", 422
        ) from error
    # Duplicate explicit rows are one scientific configuration, never extra repetitions.
    return list({_hash(recipe): recipe for recipe in recipes}.values())


def _plan_metadata(row):
    metadata = {
        key: row[key]
        for key in ("planId", "seed", "fold", "phase", "outerFold", "innerFold", "repeat", "domain")
        if key in row
    }
    metadata.setdefault("planId", f"seed:{row.get('seed', 0)}/fold:{row.get('fold')}")
    return metadata


def development_plans(protocol: dict) -> list[dict]:
    plans = {}
    for row in protocol.get("memberships", []):
        if row.get("phase") == "final" or row.get("pool") == "external_test":
            continue
        metadata = _plan_metadata(row)
        key = _hash(metadata)
        if key not in plans:
            plans[key] = {"id": key, **metadata, "slideIds": set(), "partitions": {}}
        plan = plans[key]
        plan["slideIds"].add(row["slideId"])
        role = row["partition"]
        plan["partitions"][role] = plan["partitions"].get(role, 0) + 1
    return [
        {**{k: v for k, v in item.items() if k != "slideIds"}, "slideCount": len(item["slideIds"])}
        for item in plans.values()
    ]


class DevelopmentService:
    def __init__(self, store, filesystem):
        self.store = store
        self.filesystem = filesystem

    def list(self):
        return {"items": self.store.list_configurations("mil-batch"), "executionImplemented": False}

    def preview(self, spec: DevelopmentBatchSpec):
        with lifecycle_guard(self.store.folder):
            return self._preview(spec)

    def _training_control_findings(
        self, recipes, protocol, plans, *, selection_metric=None, feature_kind="patch"
    ):
        # New batches carry an explicit configuration-selection metric. Validate
        # their standard recipes too; historical requests retain their old shape.
        experimental = [
            (number, recipe)
            for number, recipe in enumerate(recipes, 1)
            if selection_metric is not None
            or recipe.get("analysis") is not None
            or recipe.get("lossType", "ce") != "ce"
            or recipe.get("classWeighting", "none") != "none"
            or recipe.get("classWeights") is not None
            or recipe.get("samplingStrategy", "slide_uniform") != "slide_uniform"
            or recipe.get("classWeightedSampling", False)
            or recipe.get("minValidationPositives") is not None
            or recipe.get("fixedEpochBudget") is not None
            or recipe.get("model", "abmil").lower() != "abmil"
            or (recipe.get("inputMode", "image") != "clinical"
                and catalog.feature_kind(recipe.get("model")) != feature_kind)
        ]
        if not experimental or not plans:
            return []
        groups = {plan["id"]: [] for plan in plans}
        for row in protocol["memberships"]:
            if row.get("phase") == "final" or row.get("pool") == "external_test":
                continue
            identity = _hash(_plan_metadata(row))
            if identity in groups:
                groups[identity].append(row)
        columns = {
            recipe.get("cohortColumn", "cohort")
            for _, recipe in experimental
            if recipe.get("samplingStrategy") in {"cohort_balanced", "cohort_label_balanced"}
        }
        cohort_values = {}
        if columns:
            _, _, dataset_rows = ProtocolService(self.store, self.filesystem)._load_dataset(
                protocol["datasetId"]
            )
            selected = {row["slideId"] for rows in groups.values() for row in rows}
            cohort_values = {
                column: {
                    row["slideId"]: FilterEvaluator.field(row, column)
                    for row in dataset_rows
                    if row["slideId"] in selected
                }
                for column in sorted(columns)
            }
        findings = {}

        def record(severity, code, message, number, plan_id):
            key = severity, code, message
            item = findings.setdefault(key, {"configurations": set(), "plans": set()})
            item["configurations"].add(number)
            item["plans"].add(plan_id)

        for number, recipe in experimental:
            for plan_id, rows in groups.items():
                if not catalog.is_supported(recipe.get("model")):
                    record(
                        "error",
                        "TRAINING_MODEL_UNSUPPORTED",
                        f"Choose one of: {catalog.choices()}.",
                        number,
                        plan_id,
                    )
                    continue
                if (recipe.get("inputMode", "image") != "clinical"
                        and catalog.feature_kind(recipe.get("model")) != feature_kind):
                    record(
                        "error",
                        "TRAINING_FEATURE_KIND_MISMATCH",
                        f"This bundle holds {feature_kind} features. Choose one of: "
                        f"{catalog.choices(feature_kind)}.",
                        number,
                        plan_id,
                    )
                    continue
                try:
                    decision = validate_training_controls(
                        recipe,
                        protocol["spec"]["target"],
                        sampling_memberships(rows, recipe, cohort_values),
                    )
                    validate_selection_metric(selection_metric, protocol["spec"]["target"], rows)
                except ValueError as error:
                    code = (
                        "TRAINING_METRIC_UNAVAILABLE"
                        if "Validation AUROC" in str(error)
                        else "TRAINING_RECIPE_UNAVAILABLE"
                    )
                    record("error", code, str(error), number, plan_id)
                    continue
                if decision:
                    message = (
                        f"Train for {decision['epochs']} epochs and use the final epoch checkpoint."
                    )
                    if decision["reason"] == "insufficient_validation_positive_patients":
                        noun = "patient" if decision["positivePatients"] == 1 else "patients"
                        message = (
                            f"Validation has {decision['positivePatients']} positive {noun} "
                            f"(minimum {decision['minimumPositivePatients']}). " + message
                        )
                    record("warning", "FIXED_EPOCH_BUDGET", message, number, plan_id)
        return [
            {
                "severity": severity,
                "code": code,
                "message": (
                    f"{message} Affected configurations: {len(affected['configurations'])}; "
                    f"folds: {len(affected['plans'])}."
                ),
            }
            for (severity, code, message), affected in findings.items()
        ]

    def _preview(self, spec: DevelopmentBatchSpec, *, experiment_record=None):
        # A submitted experiment may finish publishing its previously reviewed
        # recipes. Only the experiment service supplies this saved owner snapshot.
        experiment = experiment_record
        if spec.experimentId and experiment is None:
            experiment = ModelExperimentService(self.store, self.filesystem).require_editable(
                spec.experimentId, spec.experimentRevision
            )
            if spec.experimentName != experiment["name"]:
                raise StorageError(
                    "The experiment name changed. Reload it before reviewing this batch.",
                    "REVISION_CONFLICT",
                    409,
                )
        binding = MILInputService(self.store, self.filesystem).preview(spec.inputs)
        findings = list(binding["findings"])
        recipes = expand_recipes(spec)
        plans, candidates, runs = [], [], []
        protocol = None
        if binding["canPlan"]:
            protocol = self.store.get_configuration(spec.inputs.protocolId)["manifest"]
            plans = development_plans(protocol)
            from histopilot.application.clinical_inputs import development_clinical_values
            from histopilot.clinical_features import clinical_fields, fit_clinical_preprocessor

            try:
                clinical_values = development_clinical_values(
                    self.store, self.filesystem, protocol, recipes
                )
                for recipe in recipes:
                    if clinical_fields(recipe):
                        for split in plans:
                            fitting = [row for row in protocol["memberships"]
                                       if row["partition"] == "train"
                                       and _hash(_plan_metadata(row)) == split["id"]]
                            fit_clinical_preprocessor(fitting, clinical_values, clinical_fields(recipe))
            except (StorageError, ValueError) as error:
                findings.append({"severity": "error", "code": getattr(error, "code", "CLINICAL_INPUTS_INVALID"),
                                 "message": str(error)})
            if protocol["spec"]["target"]["unit"] == "patient" and any(
                row.get("patientIdSource") == "slide_fallback"
                for row in protocol["memberships"]
            ):
                findings.append({
                    "severity": "error", "code": "VERIFIED_PATIENTS_REQUIRED",
                    "message": "Patient targets require verified patient IDs, including when confidence intervals are disabled. Map patient identities before creating this batch.",
                })
            if protocol["spec"].get("split", {}).get("version", 1) != 4:
                findings.append(
                    {
                        "severity": "error",
                        "code": "LEGACY_DEVELOPMENT_PROTOCOL",
                        "message": "Create a development-only protocol revision in Stage 2 before planning new batches. Existing protocols and batches remain available.",
                    }
                )
            if not plans:
                findings.append(
                    {
                        "severity": "error",
                        "code": "NO_DEVELOPMENT_PLANS",
                        "message": "The protocol has no development split memberships.",
                    }
                )
            if protocol["spec"].get("split", {}).get("mode") == "nested_kfold":
                findings.append(
                    {
                        "severity": "error",
                        "code": "NESTED_SELECTION_REQUIRED",
                        "message": "Nested CV requires a separate search and selected refit inside each outer fold. Batch planning for that dependency is not connected yet.",
                    }
                )
            elif (
                protocol["spec"].get("split", {}).get("version") == 4
                and protocol["spec"]["split"].get("mode") != "kfold"
            ):
                findings.append(
                    {
                        "severity": "error",
                        "code": "TRAINING_SPLIT_UNSUPPORTED",
                        "message": "Training currently supports development-only k-fold protocols. Create a k-fold protocol revision and select it before saving this training batch. This protocol remains available for reviewing its study design.",
                    }
                )
            if not any(item["severity"] == "error" for item in findings):
                findings.extend(self._training_control_findings(
                    recipes, protocol, plans,
                    selection_metric=spec.selectionMetric
                    if spec.candidateSelection == "best_validation" else None,
                    feature_kind=binding.get("featureKind", "patch"),
                ))
        total = len(recipes) * len(spec.trainingSeeds) * len(plans)
        if total > MAX_RUNS:
            findings.append(
                {
                    "severity": "error",
                    "code": "BATCH_TOO_LARGE",
                    "message": "Limit a batch to 20,000 training runs.",
                }
            )
        if spec.resources.runsPerGpu > 1:
            findings.append(
                {
                    "severity": "warning",
                    "code": "GPU_SHARING",
                    "message": "GPU sharing requires a measured memory and throughput profile. Resource requests do not guarantee that models fit.",
                }
            )
        if not any(item["severity"] == "error" for item in findings):
            for index, recipe in enumerate(recipes):
                candidate_id = "candidate-" + _hash(
                    {"inputs": spec.inputs.model_dump(), "recipe": recipe}
                )
                candidates.append({"id": candidate_id, "number": index + 1, "recipe": recipe})
                for seed, plan in product(spec.trainingSeeds, plans):
                    intent = {
                        "candidateId": candidate_id,
                        "trainingSeed": seed,
                        "splitPlanId": plan["id"],
                    }
                    runs.append({"id": "run-" + _hash(intent), **intent, "status": "planned"})
        nnmil_planning = []
        if candidates and any(
            item["recipe"].get("model", "abmil").lower() == "nnmil"
            or item["recipe"].get("bagSizeMode") == "training_median"
            for item in candidates
        ):
            feature = self.store.get_configuration(binding["featureSetId"])
            files = {row["slideId"]: row for row in feature["manifest"]["files"]}
            groups = {plan["id"]: [] for plan in plans}
            for row in protocol["memberships"]:
                identity = _hash(_plan_metadata(row))
                if identity in groups:
                    groups[identity].append(row)
            try:
                for candidate in candidates:
                    for plan_id, rows in groups.items():
                        _, resolution = resolve_nnmil_recipe(candidate["recipe"], rows, files)
                        if resolution:
                            nnmil_planning.append({
                                "candidateId": candidate["id"], "splitPlanId": plan_id,
                                **resolution,
                            })
            except ValueError as error:
                findings.append({"severity": "error", "code": "MIL_BAG_PLANNING_INVALID",
                                 "message": str(error)})
                candidates, runs, nnmil_planning = [], [], []
        manifest = {
            "kind": "mil-batch",
            "version": 1,
            "datasetId": protocol["datasetId"] if protocol else "",
            "spec": spec.model_dump(),
            "resolvedInputs": binding,
            "configurations": candidates,
            "splitPlans": plans,
            "runs": runs,
            **({"nnmilPlanning": nnmil_planning} if nnmil_planning else {}),
            "summary": {
                "configurationCount": len(recipes),
                "trainingSeedCount": len(spec.trainingSeeds),
                "splitPlanCount": len(plans),
                "runCount": total,
            },
            "executionImplemented": False,
        }
        if experiment:
            manifest["experiment"] = {
                "id": experiment["id"],
                "revision": experiment["revision"],
                "name": experiment["name"],
                "notes": experiment["payload"].get("notes", ""),
                "tags": experiment["payload"].get("tags", []),
            }
            if binding["canPlan"]:
                manifest["inputSnapshot"] = {
                    **input_snapshot(self.store, spec.inputs),
                    "resolvedInputs": binding,
                    "configurations": candidates,
                    "trainingSeeds": list(spec.trainingSeeds),
                    "resources": spec.resources.model_dump(),
                }
        return {
            "canFreeze": not any(item["severity"] == "error" for item in findings),
            "findings": findings,
            "previewHash": _hash(manifest),
            **manifest,
        }

    def freeze(self, spec, preview_hash, operation_id, version_label):
        with lifecycle_guard(self.store.folder):
            return self._freeze(spec, preview_hash, operation_id, version_label)

    def _freeze(self, spec, preview_hash, operation_id, version_label, *, experiment_record=None):
        prior = self.store.configuration_publication(operation_id)
        before_publish = None
        if prior:
            manifest = prior["manifest"]
            if (
                manifest.get("kind") != "mil-batch"
                or manifest.get("previewHash") != preview_hash
                or manifest.get("spec") != spec.model_dump()
            ):
                raise StorageError(
                    "This operation ID belongs to another batch.", "OPERATION_CONFLICT", 409
                )
        else:
            preview = self._preview(spec, experiment_record=experiment_record)
            if not preview["canFreeze"]:
                raise StorageError(
                    "Resolve batch findings before freezing.", "BATCH_PREFLIGHT_BLOCKED", 409
                )
            if preview["previewHash"] != preview_hash:
                raise StorageError(
                    "Batch inputs changed. Review the batch again.", "STALE_PREVIEW", 409
                )
            manifest = {k: v for k, v in preview.items() if k not in {"canFreeze", "findings"}}
            bundles = FeatureBundleService(self.store, self.filesystem)
            bundle = bundles.get(spec.inputs.featureBundleId)
            if not bundle["current"]:
                raise StorageError(
                    "Feature verification changed. Review the batch again.", "STALE_PREVIEW", 409
                )
            feature = self.store.get_configuration(bundle["manifest"]["feature"]["id"])
            before_publish = bundles._freshness_guard(feature, bundle["manifest"])
        return self.store.publish_configuration(
            manifest=manifest,
            operation_id=operation_id,
            version_label=version_label,
            before_publish=before_publish,
        )
