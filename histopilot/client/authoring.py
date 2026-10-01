"""Author records from spec files: save a draft or a selection, preview it, then commit.

Drafts (Targets & splits, cohorts, dataset imports) are saved, previewed by revision and
frozen with a version tag. Selections (every other kind in `SELECTIONS`) are previewed and
then committed with the preview's hash: feature sources and bundles are frozen with a tag,
the rest saved or queued. Saving a draft and every preview are preview class; freezing,
saving and queueing are commits.
"""

from dataclasses import dataclass

from .api import Client, segment
from .resources import project_path


@dataclass(frozen=True)
class Draft:
    draft_kind: str
    type: str
    preview: str
    freeze: str
    gate: str = "canFreeze"


@dataclass(frozen=True)
class Selection:
    preview: str
    commit: str
    gate: str
    echo: tuple[str, ...] = ()
    prefix: str = "save"
    # Frozen under a version tag, like drafts.
    versioned: bool = False


DRAFTS = {
    "targets": Draft(
        "experiment", "target-split", "/target-splits/{id}/preview", "/target-splits/{id}/freeze"
    ),
    "cohort": Draft(
        "experiment",
        "evaluation-cohort",
        "/drafts/{id}/evaluation-preview",
        "/drafts/{id}/evaluation-freeze",
    ),
    "import": Draft("import", "dataset-import", "/imports/{id}/preview", "/imports/{id}/freeze"),
}

SELECTIONS = {
    "apply": Selection(
        "/evaluation-runs/bulk/preview",
        "/evaluation-runs/bulk",
        "canRun",
        echo=("reviewedPredictorIds",),
        prefix="apply",
    ),
    "reference": Selection(
        "/reference-standards/preview", "/reference-standards", "canSave", prefix="reference"
    ),
    "clinical-analysis": Selection(
        "/clinical-analyses/preview", "/clinical-analyses", "canSave", prefix="analysis"
    ),
    "seed-ensemble": Selection(
        "/predictors/seed-ensembles/preview",
        "/predictors/seed-ensembles",
        "canFreeze",
        prefix="seed-ensemble",
    ),
    "features": Selection(
        "/features/preview", "/features/freeze", "canFreeze", prefix="features", versioned=True
    ),
    "feature-bundle": Selection(
        "/feature-bundles/preview",
        "/feature-bundles/freeze",
        "canFreeze",
        prefix="bundle",
        versioned=True,
    ),
    "feature-pack": Selection(
        "/feature-packs/preview", "/feature-packs", "canRun", prefix="packing"
    ),
    "extraction": Selection("/extractions/preview", "/extractions", "canRun", prefix="extraction"),
    "interpretation": Selection(
        "/interpretations/preview", "/interpretations", "canSave", prefix="interpretation"
    ),
}


def review(preview: dict, gate: str) -> list[dict]:
    """The preview's findings, plus one naming the closed gate when no error explains it."""
    found = list(preview.get("findings") or [])
    if preview.get(gate, True) is False and not any(
        item.get("severity") == "error" for item in found
    ):
        found.append(
            {
                "code": "PREVIEW_NOT_READY",
                "message": f"The service's review says {gate} is false.",
                "severity": "error",
            }
        )
    return found


def blocking(preview: dict, gate: str) -> list[dict]:
    """The findings that stop a commit."""
    return [item for item in review(preview, gate) if item.get("severity") == "error"]


# Drafts ----------------------------------------------------------------------------------


def save_draft(client: Client, project: str, noun: str, name: str, spec: dict) -> dict:
    """Save a new draft. Its operation ID is journaled, so a rerun after a lost answer
    returns the same draft instead of a second one."""
    kind = DRAFTS[noun]
    return client.operation(
        "POST",
        f"{project_path(project)}/drafts",
        {"kind": kind.draft_kind, "name": name, "payload": {"type": kind.type, "spec": spec}},
        prefix=f"{noun}-draft",
    )


def preview_draft(client: Client, project: str, noun: str, draft: dict) -> dict:
    path = DRAFTS[noun].preview.replace("{id}", segment(draft["id"]))
    return client.request(
        "POST", project_path(project) + path, body={"expectedRevision": draft["revision"]}
    )


def freeze_draft(
    client: Client,
    project: str,
    noun: str,
    draft: dict,
    preview: dict,
    *,
    tag: str,
    note: str = "",
    operation_id: str | None = None,
) -> dict:
    path = DRAFTS[noun].freeze.replace("{id}", segment(draft["id"]))
    body = {
        "expectedRevision": draft["revision"],
        "previewHash": preview["previewHash"],
        "versionLabel": {"tag": tag, "note": note},
    }
    return client.operation(
        "POST",
        project_path(project) + path,
        body,
        prefix=f"{noun}-freeze",
        operation_id=operation_id,
    )


# Selections ------------------------------------------------------------------------------


def preview_selection(client: Client, project: str, noun: str, selection: dict) -> dict:
    return client.request("POST", project_path(project) + SELECTIONS[noun].preview, body=selection)


def commit_selection(
    client: Client,
    project: str,
    noun: str,
    selection: dict,
    preview: dict,
    *,
    operation_id: str | None = None,
    label: dict | None = None,
) -> dict:
    kind = SELECTIONS[noun]
    body = {**selection, "previewHash": preview["previewHash"]}
    for key in kind.echo:
        body[key] = preview[key]
    if kind.versioned:
        body["versionLabel"] = label
    return client.operation(
        "POST",
        project_path(project) + kind.commit,
        body,
        prefix=kind.prefix,
        operation_id=operation_id,
    )
