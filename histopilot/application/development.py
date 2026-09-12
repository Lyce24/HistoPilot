"""Persist development batches without reassigning any frozen protocol membership."""

from itertools import product

from pydantic import ValidationError

from histopilot.application.feature_bundles import FeatureBundleService, _hash
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.model_experiments import ModelExperimentService, input_snapshot
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
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
        recipes = [TrainingRecipe.model_validate(recipe).model_dump() for recipe in recipes]
    except ValidationError as error:
        raise StorageError(
            f"A resolved training recipe is invalid: {error}", "INVALID_TRAINING_RECIPE", 422
        ) from error
    # Duplicate explicit rows are one scientific configuration, never extra repetitions.
    return list({_hash(recipe): recipe for recipe in recipes}.values())


def development_plans(protocol: dict) -> list[dict]:
    plans = {}
    for row in protocol.get("memberships", []):
        if row.get("phase") == "final" or row.get("pool") == "external_test":
            continue
        metadata = {
            key: row[key]
            for key in (
                "planId",
                "seed",
                "fold",
                "phase",
                "outerFold",
                "innerFold",
                "repeat",
                "domain",
            )
            if key in row
        }
        metadata.setdefault("planId", f"seed:{row.get('seed', 0)}/fold:{row.get('fold')}")
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
        manifest = {
            "kind": "mil-batch",
            "version": 1,
            "datasetId": protocol["datasetId"] if protocol else "",
            "spec": spec.model_dump(),
            "resolvedInputs": binding,
            "configurations": candidates,
            "splitPlans": plans,
            "runs": runs,
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
