"""Project records by kind: listing, reading, `@tag` names and verified downloads.

One table describes where each kind lives in the service, so the CLI's nouns and the
agent tools read records the same way.
"""

import hashlib
from dataclasses import dataclass, field

from histopilot import resolvers

from .api import Client, segment
from .errors import ClientError, usage_error
from .resources import experiments, listed_items, project_path
from .states import run_state


@dataclass(frozen=True)
class Kind:
    noun: str
    help: str
    list_path: str
    list_key: str
    show_path: str
    query: dict = field(default_factory=dict)
    # Versions carry a unique personal tag per kind and project: `@tag` names one.
    tagged: bool = False
    # Records that run get a `runState`.
    runs: bool = False
    # The list takes include_inactive=true for archived and trashed records.
    inactive: bool = False


KINDS = {
    kind.noun: kind
    for kind in (
        Kind("dataset", "Frozen datasets.", "/datasets", "datasets", "/datasets/{id}", tagged=True),
        Kind(
            "targets",
            "Frozen Targets & splits versions.",
            "/configurations",
            "configurations",
            "/target-splits/{id}",
            query={"kind": "target-split"},
            tagged=True,
        ),
        Kind(
            "features",
            "Attached feature sources.",
            "/configurations",
            "configurations",
            "/configurations/{id}",
            query={"kind": "feature"},
            tagged=True,
        ),
        Kind(
            "bundle",
            "Verified feature bundles.",
            "/feature-bundles",
            "items",
            "/feature-bundles/{id}",
            tagged=True,
        ),
        Kind(
            "pack",
            "Feature validation and packing jobs.",
            "/feature-packs",
            "jobs",
            "/feature-packs/{id}",
            runs=True,
        ),
        Kind(
            "extraction",
            "TRIDENT feature extraction jobs.",
            "/extractions",
            "jobs",
            "/extractions/{id}",
            runs=True,
        ),
        Kind(
            "batch",
            "Development batches: the training runs of an experiment.",
            "/mil-experiments/batches",
            "items",
            "/mil-experiments/batches/{id}/execution",
            tagged=True,
            runs=True,
        ),
        Kind(
            "predictor",
            "Frozen predictors: fold ensembles, seed ensembles and refits.",
            "/predictors",
            "items",
            "/predictors/{id}",
            inactive=True,
        ),
        Kind(
            "refit",
            "Refit plans and their execution.",
            "/predictors/refits",
            "items",
            "/predictors/refits/{id}/execution",
            runs=True,
            inactive=True,
        ),
        Kind(
            "cohort",
            "Cohorts that predictors are applied to, labeled or not.",
            "/evaluation-cohorts",
            "items",
            "/evaluation-cohorts/{id}",
            tagged=True,
        ),
        Kind(
            "run",
            "Runs of one predictor on one cohort.",
            "/evaluation-runs",
            "items",
            "/evaluation-runs/{id}",
            runs=True,
            inactive=True,
        ),
        Kind(
            "apply",
            "Batches of runs from Apply models.",
            "/evaluation-runs/bulk",
            "items",
            "/evaluation-runs/bulk/{id}",
            runs=True,
            inactive=True,
        ),
        Kind(
            "reference",
            "Reference standards: labels attached to a cohort after it was frozen.",
            "/reference-standards",
            "items",
            "/reference-standards/{id}",
            inactive=True,
        ),
        Kind(
            "analysis",
            "Clinical utility analyses.",
            "/clinical-analyses",
            "items",
            "/clinical-analyses/{id}",
            inactive=True,
        ),
        Kind(
            "interpretation",
            "Attention studies.",
            "/interpretations",
            "items",
            "/interpretations/{id}",
            runs=True,
            inactive=True,
        ),
        Kind("draft", "Unfrozen drafts.", "/drafts", "drafts", "/drafts/{id}"),
        Kind(
            "configuration",
            "Frozen configurations of any kind.",
            "/configurations",
            "configurations",
            "/configurations/{id}",
            tagged=True,
        ),
    )
}


def status_of(record: dict):
    """The record's own status, wherever its kind keeps it."""
    execution = record.get("execution") if isinstance(record.get("execution"), dict) else {}
    for value in (record.get("status"), execution.get("status"), record.get("state")):
        if isinstance(value, str) and value not in ("active", "archived", "trashed"):
            return value
    return None


def _presented(kind: Kind, record: dict) -> dict:
    return {**record, "runState": run_state(status_of(record))} if kind.runs else record


# A batch's execution as its list row carries it: what `status_of` and a table read.
BATCH_EXECUTION = ("status", "runCounts", "updatedAt", "finishedAt")


def _with_executions(listed, items: list[dict]) -> list[dict]:
    """Batch rows with their execution's status: the list answer carries executions beside
    the frozen batches, and a batch that never launched has none."""
    executions = {
        row.get("batchId"): row
        for row in (listed.get("executions") if isinstance(listed, dict) else None) or []
        if isinstance(row, dict)
    }
    joined = []
    for item in items:
        execution = executions.get(item.get("id"))
        if execution and "execution" not in item:
            item = {**item, "execution": {key: execution.get(key) for key in BATCH_EXECUTION}}
        joined.append(item)
    return joined


def experiment_of(noun: str, row: dict):
    """The experiment ID a batch, predictor or run belongs to."""
    manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
    if noun == "batch":
        return (manifest.get("experiment") or {}).get("id") or (manifest.get("spec") or {}).get(
            "experimentId"
        )
    return manifest.get("experimentId")


def records(
    client: Client,
    project: str,
    noun: str,
    *,
    inactive: bool = False,
    query=None,
    describe: bool = False,
) -> list[dict]:
    kind = KINDS[noun]
    parameters = {**kind.query, **(query or {})}
    if inactive and kind.inactive:
        parameters["include_inactive"] = True
    listed = client.get(project_path(project) + kind.list_path, query=parameters or None)
    items = listed_items(listed, kind.list_key)
    if noun == "batch":
        items = _with_executions(listed, items)
    items = [_presented(kind, item) for item in items]
    return described(client, project, noun, items) if describe else items


# Kinds whose rows `described` gives a `model`: the predictor that made them.
DESCRIBED = ("predictor", "run")


def described(client: Client, project: str, noun: str, rows: list[dict]) -> list[dict]:
    """Predictors and runs with `model`, the predictor described as the browser's Runs table
    describes it ("nnMIL · Configuration 1 · Fold ensemble · Train 42 / split 42"), with its
    experiment, batch, architecture and method. Predictor names omit their batch, so two
    models' predictors of one configuration and seeds read alike without it."""
    if noun not in DESCRIBED or not rows:
        return rows
    try:
        listed = experiments(client, project, state="all")
        known = records(client, project, "predictor", inactive=True) if noun == "run" else []
    except ClientError:
        # A description helps read the rows; it never stops the command that lists them.
        return rows
    batch_name = resolvers.batch_names(listed)
    experiment_names = {item.get("id"): item.get("name") for item in listed}

    def model(manifest: dict) -> dict:
        experiment = manifest.get("experimentId")
        return {
            "experiment": (manifest.get("experiment") or {}).get("name")
            or experiment_names.get(experiment),
            "experimentId": experiment,
            "batch": batch_name(manifest.get("batchId")),
            "batchId": manifest.get("batchId"),
            "architecture": (manifest.get("recipe") or {}).get("model"),
            "method": manifest.get("method") or "ensemble",
            "description": resolvers.model_description(manifest, batch_name),
        }

    if noun == "predictor":
        return [{**row, "model": model(row.get("manifest") or {})} for row in rows]
    wanted = {(row.get("manifest") or {}).get("predictorId") for row in rows}
    predictors = {
        item["id"]: item.get("manifest") or {} for item in known if item.get("id") in wanted
    }
    shown = []
    for row in rows:
        identity = (row.get("manifest") or {}).get("predictorId")
        manifest = predictors.get(identity)
        shown.append(
            {
                **row,
                "model": {**model(manifest), "predictorId": identity}
                if manifest is not None
                else {"predictorId": identity, "description": "Predictor unavailable"},
            }
        )
    return shown


def resolver(client: Client, project: str):
    """``resolve(noun, name) -> id`` for spec files: an ID as it is, a `@tag` to its ID."""
    return lambda noun, name: resolve(client, project, noun, name)


def resolve(client: Client, project: str, noun: str, name: str) -> str:
    """An ID for ``name``: the name itself, or the record whose version tag is ``@tag``.

    Tags are unique within one kind of record, so ``configuration:KIND`` (such as
    ``configuration:protocol``) looks among the configurations of that kind only."""
    if not name.startswith("@"):
        return name
    noun, _, configuration = noun.partition(":")
    kind = KINDS[noun]
    if not kind.tagged:
        raise usage_error(f"{noun.capitalize()} records have no version tags; name one by its ID.")
    tag, shown = name[1:], configuration or noun
    # The service keeps tags unique without regard to case.
    query = {"kind": configuration} if configuration else None
    matches = [
        item["id"]
        for item in records(client, project, noun, inactive=True, query=query)
        if ((item.get("versionLabel") or {}).get("tag") or "").casefold() == tag.casefold()
    ]
    if not matches:
        raise ClientError(
            f"No {shown} version is tagged {tag!r} in this project.",
            code="TAG_NOT_FOUND",
            kind="not-found",
        )
    if len(matches) > 1:
        raise ClientError(
            f"The tag {tag!r} names {len(matches)} {shown} versions: {', '.join(matches)}.",
            code="TAG_AMBIGUOUS",
            kind="refused",
        )
    return matches[0]


def existing(client: Client, project: str, noun: str, name: str) -> str:
    """The ID of a record that is there: a filter on an unknown one is refused (not found)
    instead of answering with an empty list."""
    identity = resolve(client, project, noun, name)
    if not any(
        item.get("id") == identity for item in records(client, project, noun, inactive=True)
    ):
        raise ClientError(
            f"No {noun} in this project has the ID {identity!r}.",
            code="RECORD_NOT_FOUND",
            kind="not-found",
        )
    return identity


def record(client: Client, project: str, noun: str, name: str, *, describe: bool = False) -> dict:
    kind = KINDS[noun]
    identity = resolve(client, project, noun, name)
    path = kind.show_path.replace("{id}", segment(identity))
    found = client.get(project_path(project) + path)
    if found is None:
        # A batch that never launched has no execution to show: its list row stands in.
        found = next(
            (item for item in records(client, project, noun) if item.get("id") == identity),
            {"id": identity},
        )
    found = _presented(kind, found)
    return described(client, project, noun, [found])[0] if describe else found


def recorded_sha256(document: dict, filename: str) -> str | None:
    """The SHA-256 the service recorded for an artifact, where the record keeps one."""
    execution = document.get("execution") if isinstance(document.get("execution"), dict) else {}
    result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
    for artifacts in (result.get("artifacts"), document.get("artifacts")):
        entry = artifacts.get(filename) if isinstance(artifacts, dict) else None
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
            return entry["sha256"]
    return None


# A label-blind run's scored files are computed by the service on each read (with the
# cohort's labels joined), so no worker hash describes them.
_DERIVED = {"metrics.json", "slide-predictions.csv", "patient-predictions.csv"}


def derived_download(document: dict, filename: str) -> bool:
    manifest = document.get("manifest") if isinstance(document.get("manifest"), dict) else {}
    source = document.get("labelSource") or manifest.get("labelSource")
    return source == "cohort" and filename in _DERIVED


def download(client: Client, path: str, *, expected: str | None) -> tuple[bytes, dict]:
    """Bytes from ``path``, checked against ``expected`` when the service recorded a hash."""
    content = client.get(path, accept="bytes")
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ClientError(
            "The downloaded file does not match the SHA-256 the service recorded.",
            code="DOWNLOAD_MISMATCH",
            kind="internal",
            data={"expected": expected, "received": digest},
        )
    return content, {"bytes": len(content), "sha256": digest, "verified": expected is not None}
