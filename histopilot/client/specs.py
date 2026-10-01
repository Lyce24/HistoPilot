"""Spec files: YAML or JSON with `kind` and `specVersion`, checked before anything is sent.

See docs/cli-contract.md#spec-files. The other keys are the fields of the service's request
model for the kind. A spec must write out every field whose service default differs from
the browser's choice, so leaving one out can never change the science silently.
"""

import importlib
import json
from pathlib import Path

from .errors import ClientError

SPEC_VERSION = 1

# kind → (module, request model). An experiment design is three parts: its own values
# (`experiment`), its `inputs` (`experiment-inputs`) and its `batches` (each a `batch`).
MODELS = {
    "import": ("histopilot.schemas.imports", "ImportSpec"),
    "targets": ("histopilot.schemas.target_splits", "TargetSplitSpec"),
    "features": ("histopilot.schemas.features", "FeatureSpec"),
    "feature-bundle": ("histopilot.schemas.feature_bundles", "FeatureBundleSpec"),
    "feature-pack": ("histopilot.schemas.feature_packs", "FeaturePackSpec"),
    "extraction": ("histopilot.schemas.extractions", "ExtractionSpec"),
    "experiment": ("histopilot.schemas.model_experiments", "ExperimentValues"),
    "experiment-inputs": ("histopilot.schemas.model_experiments", "ConfigureModelExperimentSetup"),
    "batch": ("histopilot.schemas.development", "DevelopmentBatchSpec"),
    "cohort": ("histopilot.schemas.evaluations", "EvaluationSpec"),
    "apply": ("histopilot.schemas.bulk_evaluations", "BulkEvaluationSelection"),
    "reference": ("histopilot.schemas.references", "ReferenceSelection"),
    "clinical-analysis": ("histopilot.schemas.clinical", "ClinicalSelection"),
    "interpretation": ("histopilot.schemas.interpretation", "InterpretationSelection"),
    "seed-ensemble": ("histopilot.schemas.predictors", "SeedEnsembleSelection"),
}
# What a kind's schema leaves unsaid about its spec file.
SCHEMA_NOTES = {
    "experiment": "An experiment design holds these values plus `inputs`, whose fields are "
    "`histopilot schema experiment-inputs` without expectedRevision, and `batches`, each a "
    "`histopilot schema batch` with an `id` and without the fields the experiment fills "
    "(experimentId, experimentRevision, experimentName, inputs).",
}

# Fields whose service default differs from the browser's choice (the legacy defaults that
# keep old records meaningful). A spec that leaves one out is refused.
EXPLICIT = {
    "import": ["includeMissingSlides", "recursive", "patientIdFallback"],
    "targets": ["splitUnit", "target.unit"],
    "features": ["recursive"],
    "feature-pack": ["action"],
    "cohort": ["target.unit", "inference"],
    "apply": ["scope", "inference", "patientIdentifiers"],
    "experiment-inputs": ["trainingSplit.version", "trainingSplit.pools"],
    "batch": ["predictorPolicy", "recipe"],
    "interpretation": ["resources"],
}
WHY = {
    "splitUnit": "the service would split by patient",
    "target.unit": "the service would score patients",
    "includeMissingSlides": "the service would drop rows without slide files",
    "recursive": "the service would not search subfolders",
    "patientIdFallback": "patient grouping needs an explicit choice",
    "action": "the service would pack instead of validating",
    "inference": "a partial or missing inference object falls back to mean aggregation and a "
    "0.5 threshold",
    "scope": "the service would apply every frozen predictor in the project",
    "patientIdentifiers": "patient identity across cohorts needs an explicit choice",
    "trainingSplit.version": "older split versions are refused for new experiments",
    "trainingSplit.pools": "a version 4 design needs its pools",
    "predictorPolicy": "the batch would inherit no predictor policy",
    "recipe": "an omitted recipe field takes a legacy default",
    "resources": "the service would use 2 CPU threads instead of 4",
}
# ID fields that may name a tagged version as `@tag`, by the record kind that holds tags.
# Tags are unique only within one kind, so a configuration field names its kind.
TAGGED_FIELDS = {
    "datasetId": "dataset",
    "datasetIds": "dataset",
    "targetSplitId": "targets",
    "featureSetId": "features",
    "featureBundleId": "bundle",
    "developmentFeatureBundleId": "bundle",
    "protocolId": "configuration:protocol",
    "cohortId": "cohort",
    "referenceId": "configuration:reference-standard",
}


def spec_error(message: str, found=()) -> ClientError:
    """SPEC_INVALID: a spec that cannot be sent as written."""
    return ClientError(message, code="SPEC_INVALID", kind="invalid", findings=list(found))


def model(kind: str, models: dict = MODELS):
    """The request model of a kind in ``models``."""
    if kind not in models:
        raise ClientError(
            f"Unknown spec kind {kind!r}. Choose one of: {', '.join(models)}.",
            code="SPEC_KIND_UNKNOWN",
            kind="invalid",
        )
    module, name = models[kind]
    return getattr(importlib.import_module(module), name)


def schema(kind: str, models: dict = MODELS) -> dict:
    """A kind's JSON Schema, with what it leaves unsaid about the spec file."""
    document = model(kind, models).model_json_schema()
    if kind in SCHEMA_NOTES:
        document["description"] = SCHEMA_NOTES[kind]
    return document


def read(path: Path, kind: str) -> dict:
    """The spec body from a YAML or JSON file, after its `kind` and `specVersion`."""
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as error:
        raise ClientError(
            f"Cannot read {path}: {error.strerror or error}", code="SPEC_UNREADABLE", kind="invalid"
        ) from error
    if str(path).endswith(".json"):
        try:
            document = json.loads(text)
        except ValueError as error:
            raise spec_error(f"{path} is not valid JSON: {error}") from error
    else:
        import yaml

        try:
            document = yaml.safe_load(text)
        except (yaml.YAMLError, ValueError) as error:  # an impossible date raises ValueError
            raise spec_error(f"{path} is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        raise spec_error(f"{path} must hold one mapping of fields.")
    body = dict(document)
    declared, version = body.pop("kind", None), body.pop("specVersion", None)
    if declared != kind:
        raise ClientError(
            f"{path} declares kind {declared!r}; this command reads kind {kind!r}.",
            code="SPEC_KIND_MISMATCH",
            kind="invalid",
        )
    if version != SPEC_VERSION:
        raise ClientError(
            f"{path} declares specVersion {version!r}; this CLI reads specVersion {SPEC_VERSION}.",
            code="SPEC_VERSION_UNSUPPORTED",
            kind="invalid",
        )
    return body


def _present(body: dict, dotted: str):
    value = body
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def prepare(kind: str, body: dict, resolve=None) -> tuple[dict, list[dict]]:
    """A spec checked in the contract's order: the explicit science fields, then `@tag` names
    when ``resolve(record kind, name) -> id`` is given, then the kind's request model. Returns
    the body with its tags replaced, and one note per tag."""
    body = check_explicit(kind, body)
    notes: list[dict] = []
    if resolve is not None:
        body, notes = resolve_tags(body, resolve)
    return validate(kind, body), notes


def check(kind: str, body: dict) -> dict:
    """Explicit science fields, then the kind's request model; ``body`` is returned as is."""
    return prepare(kind, body)[0]


def check_explicit(kind: str, body: dict, *, prefix: str = "") -> dict:
    """Refuse a spec that leaves out a field whose service default differs from the browser's;
    ``prefix`` places the fields inside a larger document."""
    missing = []
    for field in EXPLICIT.get(kind, []):
        if (
            field.startswith("target.")
            and isinstance(body, dict)
            and body.get("target") is None
            and "target" in body
        ):
            continue  # an unlabeled cohort has no target
        if not _present(body, field):
            missing.append(field)
    if missing:
        raise fields_required(missing, prefix=prefix)
    return body


def fields_required(missing: list[str], *, prefix: str = "") -> ClientError:
    """SPEC_FIELD_REQUIRED naming each omitted field, and why its default would not do."""

    def why(field: str) -> str:
        reason = WHY.get(field) or WHY.get(field.split(".")[0])
        return reason or "the service default would apply"

    shown = [prefix + field for field in missing]
    return ClientError(
        "Write out every field whose service default differs from the browser's: "
        + ", ".join(shown[:6])
        + (" …" if len(shown) > 6 else "")
        + ". `histopilot template` writes them all.",
        code="SPEC_FIELD_REQUIRED",
        kind="invalid",
        findings=[
            {
                "code": "SPEC_FIELD_REQUIRED",
                "message": f"Omitted, {why(field)}.",
                "severity": "error",
                "field": prefix + field,
            }
            for field in missing
        ],
    )


def findings(error, prefix: str = "") -> list[dict]:
    """One finding per field a pydantic ValidationError names."""
    return [
        {
            "code": item["type"],
            "message": item["msg"],
            "severity": "error",
            "field": ".".join([*([prefix] if prefix else []), *(str(p) for p in item["loc"])])
            or None,
        }
        for item in error.errors()
    ]


def invalid(what: str, found: list[dict]) -> ClientError:
    shown = "; ".join(f"{item['field']}: {item['message']}" for item in found[:3])
    return spec_error(f"The {what} is invalid: {shown}", found)


def validate(kind: str, body: dict) -> dict:
    """Validate ``body`` against the kind's request model, one finding per field."""
    from pydantic import ValidationError

    try:
        model(kind).model_validate(body)
    except ValidationError as error:
        raise invalid(f"{kind} spec", findings(error)) from error
    return body


def resolve_tags(body, resolve) -> tuple[object, list[dict]]:
    """Replace `@tag` values of known ID fields using ``resolve(record kind, name) -> id``;
    returns the body and one note per replaced tag, naming the ID it stands for."""
    notes: list[dict] = []

    def named(field: str, noun: str, value):
        if not (isinstance(value, str) and value.startswith("@")):
            return value
        identity = resolve(noun, value)
        notes.append(
            {
                "code": "TAG_RESOLVED",
                "message": f"{field}: {value} is {identity}.",
                "severity": "info",
                "field": field,
            }
        )
        return identity

    def walk(value):
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return value
        resolved = {}
        for key, item in value.items():
            noun = TAGGED_FIELDS.get(key)
            if noun and isinstance(item, list):
                resolved[key] = [named(key, noun, entry) for entry in item]
            elif noun:
                resolved[key] = named(key, noun, item)
            else:
                resolved[key] = walk(item)
        return resolved

    return walk(body), notes


def render(kind: str, body: dict, *, note: str | None = None) -> str:
    """A spec file: YAML with `kind` and `specVersion` first."""
    import yaml

    header = "".join(f"# {line}\n" for line in (note or "").splitlines())
    document = {"kind": kind, "specVersion": SPEC_VERSION, **body}
    return header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)
