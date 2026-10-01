"""Task-level tools for AI agents, over the same client library as the CLI.

Each tool is one plain function of an `AgentSession`. The MCP server offers a tool only when
the session's token allows its route class and the project's AI-exposure level allows its
data; the service checks every call again. Nothing here commits without a person: a commit
tool files a request that a person approves (see docs/agents.md).
"""

import json
from dataclasses import dataclass
from typing import Any

from histopilot import resolvers, templates
from histopilot.client import Client, ClientError, records, resolve, resources, runs, specs
from histopilot.client import authoring as prepare
from histopilot.client import experiments as designs
from histopilot.client.paging import page
from histopilot.client.states import SETTLED
from histopilot.client.wait import wait as wait_until
from histopilot.exports import RESULT_METRICS

MAX_ITEMS = 200
WAIT_LIMIT_SECONDS = 600
# Characters of compact JSON in one answer: well inside what MCP clients show inline.
RESULT_BUDGET = 40_000
# Characters of a long text an answer keeps when it must shrink.
TEXT_KEPT = 2_000
# Omitted paths an answer lists before it only counts the rest.
OMITTED_SHOWN = 50
# In a summary, nested details this small stay; larger ones are named in `omitted`.
SUMMARY_NESTED = 600
SUMMARY_MANIFEST = 1_500
DETAILS = ("summary", "full")


@dataclass
class AgentSession:
    client: Client
    access: dict

    @property
    def project(self) -> str:
        return self.access["projectId"]

    @property
    def scopes(self) -> set[str]:
        return set(self.access.get("scopes") or [])

    @property
    def exposure(self) -> str:
        return self.access.get("exposure") or "none"

    @property
    def project_name(self) -> str:
        return self.access.get("projectName") or self.project


def _resolver(session: AgentSession):
    """``resolve(noun, name) -> id`` for the `@tag` names in specs an agent passes."""
    return records.resolver(session.client, session.project)


def bounded(value: Any, limit: int = MAX_ITEMS) -> Any:
    """Long lists are cut to ``limit`` items with a marker, so one answer stays readable."""
    if isinstance(value, list):
        kept = [bounded(item, limit) for item in value[:limit]]
        if len(value) > limit:
            kept.append({"truncated": len(value) - limit, "hint": "Ask for a page with offset."})
        return kept
    if isinstance(value, dict):
        return {key: bounded(item, limit) for key, item in value.items()}
    return value


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def fit(value: Any, budget: int = RESULT_BUDGET) -> Any:
    """Shrink an answer until its compact JSON fits ``budget``, and say what was cut.

    Lists are cut first; then the largest parts are shortened, or left out with `_omitted`
    naming their paths. Every step makes the answer smaller, so this always ends, and an
    answer too large even in outline comes back as its size alone."""
    if len(compact(value)) <= budget:
        return value
    for limit in (50, 20, 10, 5):
        shrunk = bounded(value, limit)
        if len(compact(shrunk)) <= budget:
            return _noted(shrunk, f"Lists were cut to {limit} items to fit one answer.", [])
    shrunk = bounded(value, 5)
    omitted: list[str] = []

    def size() -> int:
        return len(compact(_noted(shrunk, OMITTED_NOTE, omitted)))

    current = size()
    while current > budget:
        path = _largest(shrunk, budget)
        if not path:
            break
        parent = shrunk
        for part in path[:-1]:
            parent = parent[part]
        item = parent[path[-1]]
        if isinstance(item, str) and len(item) > 2 * TEXT_KEPT:
            parent[path[-1]] = item[:TEXT_KEPT] + f"… [{len(item) - TEXT_KEPT} more characters]"
        else:
            parent[path[-1]] = {"omitted": f"{len(compact(item))} characters"}
            omitted.append(".".join(str(part) for part in path))
        smaller = size()
        if smaller >= current:
            break
        current = smaller
    if current > budget:
        return {
            "_note": f"This answer is {current} characters, too large for one answer even in "
            "outline. Ask for less: a smaller page, a summary, or one part of a record.",
        }
    return _noted(shrunk, OMITTED_NOTE if omitted else SHORTENED_NOTE, omitted)


SHORTENED_NOTE = "Long texts were shortened to fit one answer; each says how much was cut."
OMITTED_NOTE = (
    "Parts were left out to fit one answer; `_omitted` lists where they were. Ask for less: a "
    "smaller page, a summary, or one part of a record with get_record and a path."
)


def _largest(value: Any, budget: int) -> list:
    """The path to the part to shrink next: the largest child, descending while it is still
    a large container, so the outline of the answer survives."""
    path: list = []
    while isinstance(value, (dict, list)):
        children = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        sized = [(len(compact(item)), key, item) for key, item in children]
        if not sized:
            break
        size, key, item = max(sized, key=lambda entry: entry[0])
        if size < 200:
            break
        path.append(key)
        if not (isinstance(item, (dict, list)) and item and size > budget // 2):
            return path
        value = item
    return path


def _noted(value: Any, note: str, omitted: list[str]) -> Any:
    shown = omitted[:OMITTED_SHOWN]
    if len(omitted) > OMITTED_SHOWN:
        shown.append(f"… and {len(omitted) - OMITTED_SHOWN} more")
    extra = {"_note": note, **({"_omitted": shown} if omitted else {})}
    if isinstance(value, dict):
        return {**extra, **value}
    return {**extra, "items": value}


def summary(record: dict) -> dict:
    """A record's own fields and its small details; larger parts are named in `omitted`, for
    `get_record` to read in full."""
    kept: dict = {}
    omitted: list[str] = []
    for key, value in record.items():
        if not isinstance(value, (dict, list)) or len(compact(value)) <= SUMMARY_NESTED:
            kept[key] = value
        elif key == "manifest" and isinstance(value, dict):
            kept[key] = {
                name: item
                for name, item in value.items()
                if not isinstance(item, (dict, list)) or len(compact(item)) <= SUMMARY_MANIFEST
            }
            omitted += [f"manifest.{name}" for name in value if name not in kept[key]]
        else:
            omitted.append(key)
    if omitted:
        kept["omitted"] = omitted
    return kept


def pick(value: Any, path: str) -> Any:
    """The part of a record at a dotted path such as `manifest.coverage` or `batches.0`."""
    for part in path.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            keys = list(value)[:40] if isinstance(value, dict) else f"0 to {len(value) - 1}"
            raise ValueError(f"No {part!r} at this point of the record; it has: {keys}.")
    return value


def _detail(detail: str) -> str:
    if detail not in DETAILS:
        raise ValueError(f"Choose detail from: {', '.join(DETAILS)}.")
    return detail


# Reads ---------------------------------------------------------------------------------

LISTABLE = ("experiment", *records.KINDS)


def status(session: AgentSession) -> dict:
    """The service, the one project this token reaches (its ID and name), the token's scopes
    and expiry, the project's exposure level as it is now, and the runner and task counts."""
    data = resources.status(session.client)
    access = data["access"] or {}
    return {
        "version": data["version"],
        "serviceRevision": data["serviceRevision"],
        "project": access.get("projectId"),
        "projectName": access.get("projectName"),
        "scopes": sorted(access.get("scopes") or []),
        "exposure": access.get("exposure") or "none",
        "tokenExpiresAt": access.get("expiresAt"),
        "runner": data["runner"],
        "tasks": data["tasks"],
    }


def roadmap(session: AgentSession) -> dict:
    """The project's stages as its roadmap page shows them: status, evidence, what each still
    needs, and the suggested next stage."""
    data = resolve.roadmap(session.client, session.project)
    keep = ("id", "step", "shortTitle", "status", "evidence", "blockers", "optional")
    return {
        "stages": [{key: stage.get(key) for key in keep} for stage in data["stages"]],
        "next": data["next"],
    }


def list_records(
    session: AgentSession,
    kind: str,
    limit: int = 50,
    offset: int = 0,
    include_inactive: bool = False,
    detail: str = "summary",
    experiment: str | None = None,
    cohort: str | None = None,
) -> dict:
    """One page of records of a kind: experiment, dataset, targets, run, predictor, …

    Each record is summarized: its own fields and small details, with larger parts named in
    `omitted`. Runs and predictors carry `model`: the predictor that made them, described as
    the browser describes it ("nnMIL · Configuration 1 · Fold ensemble · Train 42 / split
    42"), with its experiment, batch and architecture. Predictor and run names omit the
    batch, so read `model` to tell two models apart. `experiment` (an ID or a name) keeps one
    experiment's batches, predictors or runs; `cohort` (an ID or `@tag`) keeps the runs on
    one cohort. Read one record in full with `get_record`; `detail="full"` lists them whole."""
    client, project = session.client, session.project
    if experiment is not None and kind not in ("batch", "predictor", "run"):
        raise ValueError("`experiment` keeps batches, predictors or runs only.")
    if cohort is not None and kind != "run":
        raise ValueError("`cohort` keeps runs only.")
    if kind == "experiment":
        items = resources.experiments(
            client, project, state="all" if include_inactive else "active"
        )
    elif kind in records.KINDS:
        items = records.records(
            client, project, kind, inactive=include_inactive, describe=kind in records.DESCRIBED
        )
    else:
        raise ValueError(f"Unknown kind {kind!r}; choose one of: {', '.join(LISTABLE)}.")
    if experiment is not None:
        wanted = resources.experiment_id(client, project, experiment, existing=True)
        items = [item for item in items if records.experiment_of(kind, item) == wanted]
    if cohort is not None:
        wanted = records.existing(client, project, "cohort", cohort)
        items = [item for item in items if (item.get("manifest") or {}).get("cohortId") == wanted]
    rows, bounds = page(items, offset=offset, limit=min(limit, MAX_ITEMS))
    if _detail(detail) == "summary":
        rows = [summary(row) if isinstance(row, dict) else row for row in rows]
    return {"items": bounded(rows), "page": bounds}


def get_record(session: AgentSession, kind: str, identity: str, path: str | None = None) -> dict:
    """One record by ID, by `@tag` for tagged versions, or an experiment by its name. A long
    record is shortened to fit one answer, naming each omitted part's path; `path` (such as
    `manifest.coverage`) reads just that part. Runs and predictors carry `model`, as in
    `list_records`."""
    client, project = session.client, session.project
    if kind == "experiment":
        record = resources.experiment(
            client, project, resources.experiment_id(client, project, identity)
        )
    elif kind in records.KINDS:
        record = records.record(client, project, kind, identity, describe=kind in records.DESCRIBED)
    else:
        raise ValueError(f"Unknown kind {kind!r}; choose one of: {', '.join(LISTABLE)}.")
    if path:
        value = bounded(pick(record, path))
        for part in reversed(path.split(".")):
            value = {part: value}
        return value
    return bounded(record)


def template(
    session: AgentSession,
    kind: str,
    preset: str | None = None,
    compare_models: list[str] | None = None,
    compare_inputs: list[str] | None = None,
    clinical_fields: list[str] | None = None,
) -> dict:
    """A starting spec from HistoPilot's templates, with every science field written out.
    For an experiment, `preset` is one of the batch presets (default baseline), and
    `compare_models` (names from `models`), `compare_inputs` (image, multimodal, clinical) or
    `clinical_fields` (FIELD:numeric or FIELD:categorical) make its batch a controlled
    comparison against the preset's recipe."""
    if kind == "experiment":
        return designs.template(
            preset or templates.DEFAULT_PRESET,
            compare_models=compare_models or (),
            compare_inputs=compare_inputs or (),
            clinical_fields=clinical_fields or (),
        )
    if preset or compare_models or compare_inputs or clinical_fields:
        raise ValueError("Presets and comparisons are for experiment templates only.")
    if kind not in templates.SPEC_STARTERS:
        raise ValueError(
            f"No template for {kind!r}; choose one of: experiment, "
            f"{', '.join(templates.SPEC_STARTERS)}."
        )
    return templates.spec_starter(kind)


def experiment_results(session: AgentSession, experiment: str, detail: str = "summary") -> dict:
    """Cross-validated results of an experiment, named by ID or name. The summary gives each
    configuration's seed-averaged metrics with their intervals, per-class results averaged
    over the training seeds, the seed ensemble (the mean of the seeds' out-of-fold
    probabilities) with its intervals, every seed's out-of-fold result and the paired
    comparisons between batches; `detail="full"` adds every test fold of every seed."""
    client, project = session.client, session.project
    identity = resources.experiment_id(client, project, experiment)
    results = resources.experiment_results(client, project, identity)
    if _detail(detail) == "full":
        return bounded(results)
    return _results_summary(results)


CONFIGURATION_FIELDS = (
    "candidateId",
    "number",
    "model",
    "inputMode",
    "seedCount",
    "plannedSeedCount",
    "foldCount",
    "plannedFoldCount",
    "complete",
    "selected",
    "validationScore",
)


def _results_summary(results: dict) -> dict:
    batches = []
    for batch in results.get("batches") or []:
        configurations = []
        for item in batch.get("configurations") or []:
            intervals = item.get("intervals") or {}
            seeds = [
                {
                    "trainingSeed": seed.get("trainingSeed"),
                    "splitSeed": seed.get("splitSeed"),
                    "complete": seed.get("complete"),
                    "oof": {
                        key: (seed.get("oof") or {}).get(key)
                        for key in ("count", *RESULT_METRICS, "confusionMatrix")
                    },
                }
                for group in item.get("splitSeeds") or []
                for seed in group.get("seeds") or []
            ]
            configurations.append(
                {
                    **{key: item.get(key) for key in CONFIGURATION_FIELDS},
                    "seedAverage": item.get("seedAverage"),
                    # Each class's metrics averaged over the seeds; the seeds' own stay out.
                    **({"perClass": item["perClass"]} if item.get("perClass") else {}),
                    "intervals": {
                        key: intervals.get(key) for key in ("unit", "units", "resamples", "note")
                    }
                    | {
                        "seedAverage": (intervals.get("seedAverage") or {}).get("intervals"),
                        "ensemble": (intervals.get("ensemble") or {}).get("intervals"),
                    },
                    "ensemble": item.get("ensemble"),
                    "seeds": seeds,
                }
            )
        batches.append(
            {
                key: batch.get(key)
                for key in (
                    "batchId",
                    "name",
                    "state",
                    "status",
                    "progress",
                    "selection",
                    "selectedCandidateId",
                )
            }
            | {"configurations": configurations}
        )
    return {
        key: results.get(key)
        for key in (
            "experimentId",
            "target",
            "design",
            "policy",
            "primaryMetric",
            "comparisons",
            "findings",
        )
    } | {
        "batches": batches,
        "detail": "summary; detail='full' adds every test fold of every seed",
    }


RUN_ANALYSES = ("metrics", "agreement", "recalibration", "subgroups", "summary")


def run_analysis(
    session: AgentSession,
    run: str,
    analysis: str,
    unit: str = "selected",
    attribute: str | None = None,
    reference: str | None = None,
    comparison: str | None = None,
) -> dict:
    """A run's metrics, agreement, recalibration, subgroup performance or label-free summary.

    `summary` works for any run, labeled or not: what it predicted, how sure it was, how its
    members agreed, and `model`, the predictor that made it. With `comparison`, another run
    on the same cohort, the summary adds how often the two runs' decisions agree (kappa, and
    for three or more classes a weighted kappa reading the class order as a scale) and their
    cross-table: how two models applied to one cohort compare without labels. `agreement`
    compares a run with its cohort's label sources instead."""
    client, project = session.client, session.project
    if comparison is not None and analysis != "summary":
        raise ValueError("`comparison` belongs to the summary analysis.")
    if analysis == "metrics":
        result = runs.metrics(client, project, run, reference=reference)
    elif analysis == "agreement":
        result = runs.agreement(client, project, run, unit=unit)
    elif analysis == "recalibration":
        result = runs.recalibration(client, project, run, unit=unit, reference=reference)
    elif analysis == "subgroups":
        if not attribute:
            raise ValueError("Subgroups need an attribute: a data dictionary key.")
        result = runs.subgroups(
            client, project, run, attribute=attribute, unit=unit, reference=reference
        )
    elif analysis == "summary":
        result = runs.with_models(
            client,
            project,
            runs.inference_summary(
                client, project, run, unit=unit, attribute=attribute, comparison=comparison
            ),
        )
    else:
        raise ValueError(f"Choose analysis from: {', '.join(RUN_ANALYSES)}.")
    return bounded(result)


def cases(
    session: AgentSession,
    run: str,
    outcome: str = "all",
    sort: str = "confidence_desc",
    limit: int = 30,
    offset: int = 0,
    comparison: str | None = None,
) -> dict:
    """Case review of a run: errors, the least confident, disagreements. With `comparison`,
    another run on the same cohort, each case carries that run's call; `outcome=
    "disagreement"` keeps the cases where the two runs disagree."""
    query = {"outcome": outcome, "sort": sort, "limit": min(limit, 100), "offset": offset}
    if comparison is not None:
        query["comparisonId"] = comparison
    return bounded(runs.cases(session.client, session.project, run, query))


def tasks(
    session: AgentSession, state: str | None = None, limit: int = 50, offset: int = 0
) -> dict:
    """This project's Task Center tasks, queue order for live work, each summarized; `task`
    reads one in full, with its failure explanation."""
    rows, bounds = resources.tasks(
        session.client,
        state=state,
        project=session.project,
        limit=min(limit, MAX_ITEMS),
        offset=offset,
    )
    return {"items": bounded([_task_row(row) for row in rows]), "page": bounds}


# What triage needs from each task in a list; `task` reads everything else.
TASK_FIELDS = (
    "id",
    "kind",
    "title",
    "state",
    "runState",
    "attempt",
    "queuePosition",
    "waitingReason",
    "progress",
    "error",
    "failure",
    "exit",
    "startedAt",
    "finishedAt",
    "updatedAt",
)


def _task_row(task: dict) -> dict:
    row = {key: task[key] for key in TASK_FIELDS if key in task}
    owner = task.get("owner") or {}
    row["owner"] = {key: owner[key] for key in ("key", "kind", "title") if key in owner}
    return row


def task(session: AgentSession, identity: str) -> dict:
    """One task: state, waiting reason and a plain-language failure explanation."""
    return bounded(resources.task(session.client, identity))


WAITABLE = ("experiment", "task", *[noun for noun, kind in records.KINDS.items() if kind.runs])


def wait(session: AgentSession, kind: str, identity: str, timeout_seconds: int = 300) -> dict:
    """Wait until work settles, at most ten minutes per call; returns its runState.

    Exits early when the Task Center runner is stopped, since queued work cannot start.
    """
    if kind not in WAITABLE:
        raise ValueError(f"Choose kind from: {', '.join(WAITABLE)}.")
    client = session.client
    if kind == "experiment":
        identity = resources.experiment_id(client, session.project, identity)
        read = lambda: resources.experiment(client, session.project, identity)  # noqa: E731
    elif kind == "task":
        read = lambda: resources.task(client, identity)  # noqa: E731
    else:
        read = lambda: records.record(client, session.project, kind, identity)  # noqa: E731
    try:
        with client.time_limit(min(max(timeout_seconds, 1), WAIT_LIMIT_SECONDS)):
            final = wait_until(client, read)
    except ClientError as error:
        if error.kind not in ("timeout", "work-failed"):
            raise
        state = (error.data or {}).get("runState") if isinstance(error.data, dict) else None
        return {
            "settled": state in SETTLED,
            "outcome": error.code,
            "runState": state,
            "message": error.message,
        }
    return {"settled": True, "outcome": "succeeded", "runState": final.get("runState")}


# Proposals: the browser's default choices -------------------------------------------------


def propose_apply(
    session: AgentSession,
    experiment_ids: list[str],
    method: str | None = None,
    cohort: str | None = None,
    predictor: str | None = None,
) -> dict:
    """What Apply models proposes for experiments named by ID or name: the method (seed
    ensembles when ready), every ready predictor, the testing cohort their development
    reserved and its inference settings. The spec goes to `apply_preview` as is; notes say
    where each choice came from."""
    resolution = resolve.apply(
        session.client,
        session.project,
        [resources.experiment_id(session.client, session.project, item) for item in experiment_ids],
        method=method,
        cohort=cohort,
        predictor=predictor,
    )
    return bounded(
        {
            "spec": resolution["selection"],
            "notes": resolution["notes"],
            "findings": resolution["findings"],
        }
    )


def apply_configuration(session: AgentSession, experiment: str, batch: str, candidate: str) -> dict:
    """What "Apply this configuration" does for one configuration: its seed ensemble or single
    fold ensemble to apply, a seed ensemble a person must build first (nothing trains), or a
    choice among the fold ensembles of its seed groups. Name the batch by its ID or name and
    the configuration by its ID or number, as Results shows them."""
    client, project = session.client, session.project
    identity = resources.experiment_id(client, project, experiment)
    return resolve.configuration(client, project, identity, batch, candidate)


def label_sources(session: AgentSession, run: str) -> dict:
    """The labels a run can be scored against, and the default used when none is named: the
    cohort's own labels, else the first fitting reference standard."""
    sources = resolve.label_sources(session.client, session.project, run)
    chosen = resolvers.label_source(sources)
    return {"sources": sources, "default": chosen["id"] if chosen else None}


def target_from_field(session: AgentSession, spec: dict, field: str) -> dict:
    """A Targets & splits spec's training target, read from a field's training values: the
    classes, task and labels. The positive class is left for a person to choose. `notes`
    names the ID each `@tag` stands for."""
    resolved, notes = specs.resolve_tags(spec, _resolver(session))
    target = resolve.target(session.client, session.project, resolved, field)
    return bounded({**target, "notes": notes})


def features_from_extraction(session: AgentSession, extraction: str) -> dict:
    """A features spec for a succeeded extraction's outputs, read exactly as written."""
    return resolve.features_from_extraction(session.client, session.project, extraction)


def models(session: AgentSession) -> dict:
    """The models HistoPilot trains: each name, what it is, which features it reads (patch
    or slide) and whether it has attention to show."""
    from histopilot.models import catalog

    keep = ("name", "label", "summary", "featureKind", "supportsAttention")
    return {"models": [{key: item[key] for key in keep} for item in catalog.describe()]}


def comparison_arms(
    session: AgentSession,
    recipe: dict,
    models: list[str],
    inputs: list[str] | None = None,
    clinical_fields: list[str] | None = None,
) -> dict:
    """A controlled comparison over a recipe, such as an existing experiment's, as the batch
    editor builds one: the recipe first as the reference, then each model (names from
    `models`) and input (image, multimodal, clinical), changing only what that model owns.
    Use the result as a batch's explicit configurations."""
    return bounded(
        designs.comparison({"recipe": recipe}, models, inputs or (), clinical_fields or ())
    )


# Previews ------------------------------------------------------------------------------


def preview_design(session: AgentSession, design: dict, experiment: str | None = None) -> dict:
    """Save a design as an experiment draft and preview every batch plan. To change the
    design, pass the `experiment` a previous call returned: that draft is revised, and no
    other is made. `@tag` names are read as the IDs `notes` names.

    Nothing is frozen. Hand the experiment ID and the returned top-level previewHash to a
    person, who freezes it with `histopilot experiment freeze EXPERIMENT --preview-hash
    HASH` and starts it."""
    checked, notes = designs.prepare(design, _resolver(session))
    if experiment:
        identity = resources.experiment_id(session.client, session.project, experiment)
        record = designs.revise(session.client, session.project, identity, checked)
    else:
        record = designs.create(session.client, session.project, checked)
    review = designs.preview(session.client, session.project, record["id"])
    return bounded({"experimentId": record["id"], **review, "notes": notes})


def apply_preview(session: AgentSession, spec: dict) -> dict:
    """Preview which runs applying predictors to a cohort would queue; nothing runs.
    `@tag` names are read as the IDs `notes` names."""
    checked, notes = specs.prepare("apply", spec, _resolver(session))
    preview = prepare.preview_selection(session.client, session.project, "apply", checked)
    return bounded({**preview, "notes": notes})


# Requests a person approves ----------------------------------------------------------------

TASK_ACTIONS = ("cancel", "retry")


def _filed(send) -> dict:
    """Send a commit. The service parks a scoped token's commit until a person approves it."""
    try:
        result = send()
    except ClientError as error:
        if error.code != "CONFIRMATION_PENDING":
            raise
        request = (error.data or {}).get("requestId")
        return {
            "state": "pending",
            "requestId": request,
            "expiresAt": (error.data or {}).get("expiresAt"),
            "message": "Nothing has changed yet. A person must approve this request: "
            f"`histopilot confirm show {request}`, then `histopilot confirm approve {request}`, "
            "or the AI agent access panel on the Study backups & sources page. Do not report "
            "it as done.",
        }
    return {"state": "applied", "result": bounded(result)}


def request_freeze(session: AgentSession, experiment: str, revision: int) -> dict:
    """Needs a person's approval: ask to freeze an experiment's setup at the revision you
    previewed. The service refuses the approval if the experiment changed since."""
    identity = resources.experiment_id(session.client, session.project, experiment)
    return _filed(
        lambda: designs.freeze(session.client, session.project, identity, revision=revision)
    )


def request_start(session: AgentSession, experiment: str, revision: int) -> dict:
    """Needs a person's approval: ask to start training a frozen experiment's batches."""
    identity = resources.experiment_id(session.client, session.project, experiment)
    return _filed(
        lambda: designs.start(session.client, session.project, identity, revision=revision)
    )


def request_apply(session: AgentSession, spec: dict, preview_hash: str) -> dict:
    """Needs a person's approval: ask to queue the runs of an apply spec. `preview_hash` is
    the one `apply_preview` returned; if the preview changed since, nothing is filed."""
    spec, _ = specs.prepare("apply", spec, _resolver(session))
    client, project = session.client, session.project
    preview = prepare.preview_selection(client, project, "apply", spec)
    if preview.get("previewHash") != preview_hash:
        raise ClientError(
            "The preview changed since you reviewed it; preview again.",
            code="PREVIEW_CHANGED",
            kind="conflict",
        )
    blocking = prepare.blocking(preview, prepare.SELECTIONS["apply"].gate)
    if blocking:
        raise ClientError(
            "The preview has blocking findings; nothing was filed.",
            code="PREVIEW_BLOCKED",
            kind="refused",
            findings=blocking,
        )
    return _filed(lambda: prepare.commit_selection(client, project, "apply", spec, preview))


def request_task_action(session: AgentSession, identity: str, action: str) -> dict:
    """Needs a person's approval: ask to cancel or retry one of this project's tasks."""
    if action not in TASK_ACTIONS:
        raise ValueError(f"Choose action from: {', '.join(TASK_ACTIONS)}.")
    return _filed(lambda: resources.task_action(session.client, identity, action))
