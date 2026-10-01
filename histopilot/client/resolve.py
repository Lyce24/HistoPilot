"""The browser's default choices, read from the service.

Each function fetches the records one resolver in `histopilot.resolvers` needs, from the
routes the browser reads them from, and returns what the browser would propose. Nothing here
changes the service; building a seed ensemble is left to the caller's commit.
"""

from histopilot import resolvers

from . import resources
from .errors import ClientError, usage_error
from .records import record, records, resolve
from .resources import project_path


def _info(code: str, message: str) -> dict:
    return {"code": code, "message": message, "severity": "info"}


# Apply models ------------------------------------------------------------------------------


def apply(
    client,
    project: str,
    experiments=(),
    *,
    method: str | None = None,
    cohort: str | None = None,
    predictor: str | None = None,
    name: str = "",
) -> dict:
    """Apply models' proposal: `resolvers.apply_selection` over this project's predictors,
    protocols and cohorts, with notes that say where each choice came from."""
    if method is not None and method not in resolvers.APPLY_METHODS:
        raise usage_error(f"Choose a method from: {', '.join(resolvers.APPLY_METHODS)}.")
    predictors = records(client, project, "predictor", inactive=True)
    experiment_ids = list(experiments)
    if predictor and not experiment_ids:
        # A linked predictor implies its experiment, as the browser's link does.
        linked = next((item for item in predictors if item.get("id") == predictor), None)
        if linked is None:
            raise ClientError(
                f"No predictor {predictor} in this project.",
                code="PREDICTOR_NOT_FOUND",
                kind="not-found",
            )
        experiment_ids = [(linked.get("manifest") or {}).get("experimentId")]
    protocols = client.get(
        f"{project_path(project)}/configurations", query={"kind": "protocol"}
    ).get("configurations")
    cohorts = records(client, project, "cohort")
    cohort_id = resolve(client, project, "configuration", cohort) if cohort else None
    resolution = resolvers.apply_selection(
        predictors,
        protocols or [],
        cohorts,
        experiment_ids,
        method=method,
        cohort_id=cohort_id,
        linked_predictor=predictor,
        name=name,
    )
    return {**resolution, "notes": _apply_notes(resolution)}


def _apply_notes(resolution: dict) -> list[dict]:
    method, ready = resolution["method"], resolution["ready"]
    why = {
        "chosen": "as chosen",
        "linked": "the linked predictor's own",
        "default": "seed ensembles are ready"
        if method == "seed_ensemble"
        else "no seed ensemble is ready",
    }[resolution["methodSource"]]
    notes = [
        _info("APPLY_METHOD", f"Method {method}: {why}."),
        _info(
            "APPLY_PREDICTORS",
            f"{len(resolution['selection']['predictorIds'])} predictors selected; ready for this "
            f"method: {ready['seed_ensemble']} seed ensembles, {ready['ensemble']} fold "
            f"ensembles, {ready['refit']} refits.",
        ),
    ]
    if resolution["cohortSource"] == "reserved":
        notes.append(
            _info(
                "APPLY_COHORT",
                f"Cohort {resolution['selection']['cohortId']}: the testing set these "
                "predictors' development reserved.",
            )
        )
    if resolution["cohortLabeled"] is False:
        notes.append(
            _info("APPLY_UNLABELED", "The cohort is unlabeled: predictions only, no metrics.")
        )
    return notes


def configuration(client, project: str, experiment: str, batch: str, candidate: str) -> dict:
    """What "Apply this configuration" does for one configuration of an experiment, its batch
    named by ID or name and the configuration by ID or number (`configuration_of`)."""
    found = resources.experiment(client, project, experiment)
    batch, candidate = configuration_of(found, batch, candidate)
    choices = client.get(
        f"{project_path(project)}/predictors/seed-ensembles", query={"experiment_id": found["id"]}
    ).get("items")
    predictors = records(client, project, "predictor", inactive=True)
    return resolvers.configuration_choice(
        choices or [], predictors, {"id": found["id"], "name": found.get("name")}, batch, candidate
    )


def configuration_of(experiment: dict, batch: str, candidate: str) -> tuple[str, str]:
    """The batch and configuration IDs of an experiment's configuration: the batch named by
    its ID or by its name (without regard to case), the configuration by its ID or by its
    number in the batch, as Results numbers them. A name or number the experiment does not
    have is refused with the ones it has, never read as "not ready". A record that does not
    list its batches, or a batch its configurations, is taken at its word."""
    if not isinstance(experiment.get("batches"), list):
        return batch, candidate
    batches = experiment["batches"]
    chosen = next((row for row in batches if row.get("id") == batch), None)
    if chosen is None:
        named = [row for row in batches if (row.get("name") or "").casefold() == batch.casefold()]
        if len(named) > 1:
            raise ClientError(
                f"{len(named)} batches of this experiment are named {batch!r}: "
                f"{', '.join(row['id'] for row in named)}. Name one by its ID.",
                code="BATCH_AMBIGUOUS",
                kind="refused",
            )
        chosen = named[0] if named else None
    if chosen is None:
        offered = ", ".join(f"{row.get('name')} ({row.get('id')})" for row in batches)
        raise ClientError(
            f"Experiment {experiment.get('name')!r} has no batch {batch!r}; "
            f"its batches: {offered or 'none'}.",
            code="BATCH_NOT_IN_EXPERIMENT",
            kind="not-found",
        )
    rows = (chosen.get("manifest") or {}).get("configurations")
    if not isinstance(rows, list):
        return chosen["id"], candidate
    match = next(
        (row for row in rows if row.get("id") == candidate or str(row.get("number")) == candidate),
        None,
    )
    if match is None:
        offered = ", ".join(f"{row.get('number')} ({row.get('id')})" for row in rows)
        raise ClientError(
            f"Batch {chosen.get('name')!r} has no configuration {candidate!r}; "
            f"its configurations: {offered or 'none'}.",
            code="CONFIGURATION_NOT_IN_BATCH",
            kind="not-found",
        )
    return chosen["id"], match["id"]


# Targets and labels ------------------------------------------------------------------------


def target(client, project: str, spec: dict, field: str) -> dict:
    """Targets & splits' target choice: FIELD's training values as the target's classes.

    ``spec`` names its records by ID. Returns the spec's new `target` (and `testTarget`)."""
    if not isinstance(spec.get("target"), dict):
        raise ClientError(
            "The spec has no target to fill in; `histopilot targets template` writes one.",
            code="SPEC_INVALID",
            kind="invalid",
        )
    working = dict(spec)
    blank = {**working["target"], "field": field, **resolvers.infer_target([])}
    working.update(resolvers.training_target(working, blank))
    result = client.request(
        "POST",
        f"{project_path(project)}/target-splits/partition-preview",
        body=resolvers.partition_request(working),
    )
    if not result.get("valid"):
        errors = [item for item in result.get("findings") or [] if item.get("severity") == "error"]
        raise ClientError(
            errors[0]["message"]
            if errors
            else "Review the partition conditions before choosing a target.",
            code="TARGET_VALUES_UNREADABLE",
            kind="refused",
            findings=errors,
        )
    distribution = ((result.get("partitions") or {}).get("train") or {}).get("target") or {}
    values = distribution.get("values") or []
    inferred = resolvers.infer_target(
        [item.get("value") for item in values],
        (distribution.get("distinctCount") or 0) > len(values),
    )
    return resolvers.training_target(working, {**working["target"], **inferred})


def _sources(client, project: str, found: dict) -> list[dict]:
    cohort = (found.get("manifest") or {}).get("cohortId")
    standards = records(
        client, project, "reference", inactive=True, query={"cohort_id": cohort} if cohort else None
    )
    return resolvers.label_sources(found, standards)


def label_sources(client, project: str, run: str) -> list[dict]:
    """The labels a run can be scored against, its cohort's own first when it has them."""
    return _sources(client, project, record(client, project, "run", run))


def default_reference(client, project: str, run: str) -> tuple[str | None, dict | None]:
    """The reference a run is scored against when none is named: None for the cohort's own
    labels, else the first fitting reference standard (with a note), as in the browser."""
    found = record(client, project, "run", run)
    if resolvers.run_labeled(found):
        return None, None
    source = resolvers.label_source(_sources(client, project, found))
    if source is None:
        return None, None
    return source["id"], _info(
        "REFERENCE_DEFAULT",
        f"Scored against the reference standard {source['name']!r} ({source['id']}), the first "
        "that fits this unlabeled run; pass --reference to choose another.",
    )


# The roadmap -------------------------------------------------------------------------------

ROADMAP_READS = {
    # evidence key: (record kind, with archived and trashed records), as the roadmap page reads
    "drafts": ("draft", False),
    "datasets": ("dataset", False),
    "targetSplits": ("targets", False),
    "features": ("features", False),
    "bundles": ("bundle", False),
    "extractions": ("extraction", False),
    "evaluationCohorts": ("cohort", False),
    "predictors": ("predictor", True),
    "modelEvaluations": ("run", True),
    "clinicalAnalyses": ("analysis", False),
    "interpretations": ("interpretation", False),
}


def roadmap(client, project: str) -> dict:
    """Each stage's status, evidence and blockers, and the suggested next stage."""
    from histopilot import roadmap as stages_of

    base = project_path(project)
    evidence = {
        key: records(client, project, noun, inactive=inactive)
        for key, (noun, inactive) in ROADMAP_READS.items()
    }
    evidence["setups"] = (
        client.get(f"{base}/configurations", query={"kind": "experiment-setup"}).get(
            "configurations"
        )
        or []
    )
    batches = client.get(f"{base}/mil-experiments/batches")
    evidence["batches"] = batches.get("items") or []
    evidence["executions"] = batches.get("executions") or []
    workspace = client.get(f"{base}/workspace")
    stages = stages_of.build(workspace, evidence)
    chosen = stages_of.suggested(stages)
    return {
        "stages": [{**stage, "step": stages_of.step_of(stage["id"])} for stage in stages],
        "next": chosen["id"] if chosen else None,
    }


# Features ----------------------------------------------------------------------------------


def features_from_extraction(client, project: str, job: str) -> dict:
    """A features spec for a succeeded extraction's outputs, read exactly as written."""
    found = record(client, project, "extraction", job)
    built = resolvers.feature_spec_from_extraction(found)
    if built is None:
        raise ClientError(
            f"Extraction {job} has no features to attach yet (state {found.get('state')}).",
            code="EXTRACTION_NOT_READY",
            kind="refused",
        )
    return built
