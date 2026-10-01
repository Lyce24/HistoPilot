"""Experiments from a design file, the way the browser builds them.

A design holds the experiment's name, notes, tags and predictor policy; its inputs (dataset,
Targets & splits, feature bundle and the version 4 training design); and its batches. The
steps: create the experiment and save its inputs and batch plans (all preview class, so
nothing is frozen), preview every plan, freeze the setup, then start it (both commits).
"""

import hashlib
import json

from histopilot import resolvers, templates

from . import specs
from .api import Client
from .errors import ClientError
from .resources import experiment, experiment_path, project_path

KIND = "experiment"
# The experiment's own values; the rest of a design is its inputs and batches.
VALUE_FIELDS = ("name", "notes", "tags", "predictorPolicy")
# Filled by the service from the experiment itself; a design never states them.
FILLED = ("experimentId", "experimentRevision", "experimentName", "inputs")
INPUT_FIELDS = (
    "datasetId",
    "targetSplitId",
    "featureBundleId",
    "loadingPolicy",
    "packArtifactId",
    "trainingSplit",
)


def complete_recipe(recipe: dict) -> dict:
    """Every recipe field written out: the service's defaults under the given values."""
    from histopilot.schemas.development import TrainingRecipe

    return TrainingRecipe.model_validate(recipe).model_dump(mode="json")


def template(
    preset: str = templates.DEFAULT_PRESET,
    *,
    name: str = "New experiment",
    compare_models=(),
    compare_inputs=(),
    clinical_fields=(),
) -> dict:
    """A complete starting design: every science field is explicit. Models, inputs or
    clinical fields to compare make its batch a controlled comparison (see `comparison`)."""
    if preset not in templates.PRESETS:
        raise specs.spec_error(f"Choose a preset from: {', '.join(templates.PRESETS)}.")
    batch = templates.batch(preset, inputs={}, experiment_name=name)
    batch["recipe"] = complete_recipe(batch["recipe"])
    batch["batchName"] = batch["batchName"] or "Baseline"
    if compare_models or compare_inputs or clinical_fields:
        batch = comparison(batch, compare_models, compare_inputs, clinical_fields)
    described = templates.describe()
    return {
        "name": name,
        "notes": "",
        "tags": [],
        "predictorPolicy": described["predictorPolicy"],
        "inputs": {
            "datasetId": "@dataset-tag",
            "targetSplitId": "@targets-tag",
            "featureBundleId": "configuration-…",
            "loadingPolicy": "auto",
            "packArtifactId": None,
            "trainingSplit": described["starters"]["trainingSplit"],
        },
        "batches": [
            {"id": preset, **{key: value for key, value in batch.items() if key not in FILLED}}
        ],
    }


def comparison(batch: dict, models=(), inputs=(), clinical_fields=()) -> dict:
    """The batch as a controlled comparison, built as the batch editor builds one: its recipe
    first, as the reference, then each model and input, changing only what that model owns.

    ``models`` are catalog names such as `mean_pool`; ``inputs`` are image, multimodal or
    clinical; ``clinical_fields`` are `FIELD:numeric` or `FIELD:categorical`, which make the
    reference arm clinical plus image."""
    from histopilot.models import catalog

    for model in models:
        if not catalog.is_supported(model):
            raise specs.spec_error(
                f"Unknown model {model!r}; choose from: {', '.join(catalog.CATALOG)}."
            )
    for value in inputs:
        if value not in resolvers.ARM_INPUTS:
            raise specs.spec_error(f"Choose inputs from: {', '.join(resolvers.ARM_INPUTS)}.")
    recipe = dict(batch["recipe"])
    if clinical_fields:
        chosen = []
        for item in clinical_fields:
            field, _, kind = item.rpartition(":")
            if not field or kind not in ("numeric", "categorical"):
                raise specs.spec_error(
                    f"Write a clinical field as FIELD:numeric or FIELD:categorical, not {item!r}."
                )
            chosen.append({"field": field, "kind": kind})
        recipe.update(inputMode="multimodal", clinicalFields=chosen)
    try:
        arms = resolvers.ablation_arms(recipe, models or [recipe["model"]], inputs or ["image"])
        arms = [complete_recipe(arm) for arm in arms]
        return resolvers.comparison_batch({**batch, "recipe": arms[0]}, arms)
    except ValueError as error:
        raise specs.spec_error(str(error)) from error


def prepare(design: dict, resolve=None) -> tuple[dict, list[dict]]:
    """A design checked in the contract's order for spec files: its shape and explicit
    science fields, then `@tag` names when ``resolve(record kind, name) -> id`` is given,
    then the service's request models. Returns the design and one note per resolved tag."""
    _check_explicit(design)
    notes: list[dict] = []
    if resolve is not None:
        design, notes = specs.resolve_tags(design, resolve)
    return _validate(design), notes


def _check_explicit(design: dict) -> None:
    unknown = set(design) - {*VALUE_FIELDS, "inputs", "batches"}
    if unknown:
        raise specs.spec_error(f"Unknown design keys: {', '.join(sorted(unknown))}.")
    inputs = design.get("inputs")
    if not isinstance(inputs, dict):
        raise specs.spec_error(
            "A design needs its inputs: dataset, targets, features and training design."
        )
    if not isinstance(inputs.get("trainingSplit"), dict):
        raise specs.spec_error(
            "Write out inputs.trainingSplit; `histopilot experiment template` does."
        )
    specs.check_explicit("experiment-inputs", inputs, prefix="inputs.")
    _check_batch_fields(design.get("batches"))


def _check_batch_fields(batches) -> None:
    if not isinstance(batches, list) or not batches:
        raise specs.spec_error("A design needs at least one batch.")
    required = set(templates.DEFAULT_RECIPE)
    seen = set()
    for index, batch in enumerate(batches):
        prefix = f"batches.{index}"
        if not isinstance(batch, dict) or not batch.get("id"):
            raise specs.spec_error(f"{prefix} needs an id, such as `baseline`.")
        if batch["id"] in seen:
            raise specs.spec_error(f"Two batches share the id {batch['id']!r}.")
        seen.add(batch["id"])
        stated = [key for key in FILLED if key in batch]
        if stated:
            raise specs.spec_error(
                f"{prefix} states {', '.join(stated)}; the experiment fills them."
            )
        # A design writes out every recipe field, not only those the service lacks defaults for.
        missing = [key for key in specs.EXPLICIT["batch"] if key not in batch]
        missing += [f"recipe.{key}" for key in sorted(required - set(batch.get("recipe") or {}))]
        if missing:
            raise specs.fields_required(missing, prefix=f"{prefix}.")


def _validate(design: dict) -> dict:
    from pydantic import ValidationError

    from histopilot.schemas.model_experiments import (
        ConfigureModelExperimentSetup,
        ExperimentValues,
    )

    found: list[dict] = []

    def validate(model, value, prefix):
        try:
            model.model_validate(value)
        except ValidationError as error:
            found.extend(specs.findings(error, prefix))

    values = {key: design[key] for key in VALUE_FIELDS if design.get(key) is not None}
    validate(ExperimentValues, values, "experiment")
    validate(ConfigureModelExperimentSetup, {**design["inputs"], "expectedRevision": 1}, "inputs")
    if not found:
        found = _batch_findings(design["batches"], name=design.get("name") or "x")
    if found:
        raise specs.invalid("experiment design", found)
    return design


def check_batches(batches, *, name: str = "x") -> list:
    """Validate a design's batches: ids, every recipe field, and the batch request model."""
    _check_batch_fields(batches)
    found = _batch_findings(batches, name=name)
    if found:
        raise specs.invalid("experiment design", found)
    return batches


def _batch_findings(batches: list, *, name: str) -> list[dict]:
    from pydantic import ValidationError

    from histopilot.schemas.development import DevelopmentBatchSpec

    placeholder = {"protocolId": "configuration-placeholder", "featureBundleId": "placeholder"}
    found: list[dict] = []
    for index, batch in enumerate(batches):
        spec = {key: value for key, value in batch.items() if key != "id"}
        try:
            DevelopmentBatchSpec.model_validate(
                {**spec, "experimentName": name, "inputs": placeholder}
            )
        except ValidationError as error:
            found.extend(specs.findings(error, f"batches.{index}"))
    return found


def design_hash(design: dict) -> str:
    """The design's science: inputs and batches, without names or notes."""
    material = {"inputs": design.get("inputs"), "batches": design.get("batches")}
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def export(record: dict) -> dict:
    """The design of an existing experiment, ready to create again."""
    setup = record.get("setupDesign") or {}
    inputs = record.get("inputs") or {}
    return {
        "name": record.get("name"),
        "notes": record.get("notes") or "",
        "tags": record.get("tags") or [],
        "predictorPolicy": record.get("predictorPolicy"),
        "inputs": {
            "datasetId": setup.get("datasetId"),
            "targetSplitId": setup.get("targetSplitId"),
            "featureBundleId": inputs.get("featureBundleId"),
            "loadingPolicy": inputs.get("loadingPolicy", "auto"),
            "packArtifactId": inputs.get("packArtifactId"),
            "trainingSplit": setup.get("trainingSplit"),
        },
        "batches": [_design_batch(plan) for plan in record.get("batchPlans") or []],
    }


def _values(design: dict) -> dict:
    return {key: design[key] for key in VALUE_FIELDS if key in design}


def create(client: Client, project: str, design: dict) -> dict:
    """Create the experiment and save its inputs and batch plans; nothing is frozen.

    Rerunning with the same design replays the creation and skips steps already done.
    """
    record = client.operation(
        "POST",
        experiment_path(project),
        {**_values(design), "setupVersion": 1},
        prefix="experiment-create",
    )
    return _save(client, project, record, design)


def revise(client: Client, project: str, identity: str, design: dict) -> dict:
    """Save a design over an experiment whose setup is not frozen: its name, notes, tags,
    predictor policy, inputs and batch plans. Nothing is frozen, and no other draft is made."""
    record = experiment(client, project, identity)
    if record.get("frozenSetupId"):
        raise ClientError(
            "This experiment's setup is frozen; create a new experiment from the design instead.",
            code="EXPERIMENT_SETUP_FROZEN",
            kind="refused",
        )
    values = _values(design)
    if {key: record.get(key) for key in values} != values:
        record = client.request(
            "PATCH",
            experiment_path(project, identity),
            body={"expectedRevision": record["revision"], **values},
        )
    return _save(client, project, record, design)


def _save(client: Client, project: str, record: dict, design: dict) -> dict:
    """Save the design's inputs and batch plans where they differ from the record's."""
    wanted = {key: design["inputs"].get(key) for key in INPUT_FIELDS}
    if export(record)["inputs"] != wanted:
        record = client.request(
            "POST",
            f"{experiment_path(project, record['id'])}/setup-inputs",
            body={"expectedRevision": record["revision"], **wanted},
        )
    plans = [plan_for(record, batch) for batch in design["batches"]]
    if export(record)["batches"] != [_design_batch(plan) for plan in plans]:
        record = save_plans(client, project, record, plans)
    return record


def _design_batch(plan: dict) -> dict:
    """A plan as a design's batch: its id and spec without the fields the experiment fills."""
    return {
        "id": plan["id"],
        **{key: value for key, value in plan["spec"].items() if key not in FILLED},
    }


def preview(client: Client, project: str, identity: str) -> dict:
    """Each batch plan's review, and whether the setup can be frozen."""
    record = experiment(client, project, identity)
    if record.get("frozenSetupId"):
        # The service previews only the plans of a setup that can still change.
        raise ClientError(
            "This experiment's setup is already frozen; start it, or create a new experiment "
            "from its design (`experiment export-design`).",
            code="EXPERIMENT_SETUP_FROZEN",
            kind="refused",
        )
    plans = []
    for plan in record.get("batchPlans") or []:
        review = client.request(
            "POST", f"{project_path(project)}/mil-experiments/batches/preview", body=plan["spec"]
        )
        plans.append(
            {
                "planId": plan["id"],
                "canFreeze": bool(review.get("canFreeze")),
                "previewHash": review.get("previewHash"),
                "findings": review.get("findings") or [],
                "runCount": (review.get("summary") or {}).get("runCount"),
            }
        )
    findings = [{**item, "planId": plan["planId"]} for plan in plans for item in plan["findings"]]
    return {
        "experimentId": identity,
        "revision": record["revision"],
        "previewHash": preview_hash(identity, record["revision"], plans),
        "designHash": design_hash(export(record)),
        "canFreeze": bool(plans) and all(plan["canFreeze"] for plan in plans),
        "plans": plans,
        "findings": findings,
    }


def preview_hash(identity: str, revision, plans: list[dict]) -> str:
    """What `experiment freeze --preview-hash` confirms: the experiment, its revision and each
    batch plan's own preview hash. An agent's redacted view and a person's give the same one,
    so a person can freeze exactly what an agent previewed."""
    material = {
        "experimentId": identity,
        "revision": revision,
        "plans": [[plan["planId"], plan["previewHash"], plan["canFreeze"]] for plan in plans],
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def plan_for(record: dict, batch: dict) -> dict:
    """A design's batch as a plan of this experiment, which fills in its own fields."""
    return {
        "id": batch["id"],
        "spec": {
            **{key: value for key, value in batch.items() if key != "id"},
            "experimentId": record["id"],
            "experimentRevision": record["revision"],
            "experimentName": record["name"],
            "inputs": record["inputs"],
        },
    }


def save_plans(client: Client, project: str, record: dict, plans: list[dict]) -> dict:
    """Replace an unfrozen experiment's batch plans, under the revision that was read."""
    return client.request(
        "PATCH",
        experiment_path(project, record["id"]),
        body={"expectedRevision": record["revision"], "name": record["name"], "batchPlans": plans},
    )


def freeze(client, project, identity, *, revision, operation_id=None) -> dict:
    return client.operation(
        "POST",
        f"{experiment_path(project, identity)}/freeze-setup",
        {"expectedRevision": revision},
        prefix="experiment-freeze",
        operation_id=operation_id,
    )


def start(client, project, identity, *, revision, operation_id=None) -> dict:
    return client.operation(
        "POST",
        f"{experiment_path(project, identity)}/submit",
        {"expectedRevision": revision},
        prefix="experiment-start",
        operation_id=operation_id,
    )
