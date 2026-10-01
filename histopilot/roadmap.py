"""A project's roadmap: each stage's status, evidence and blockers, as its roadmap page shows.

`build` ports `buildRoadmap` (web/src/lib/roadmap.ts) over the same records the page reads,
so `histopilot project roadmap` and an agent see the status a person sees. The cases in
`web/src/lib/roadmapCases.json` run against both, in `tests/test_roadmap.py` and
`web/src/lib/roadmap.cases.test.ts`.
"""

from histopilot.models import catalog
from histopilot.resolvers import INFERENCE_PURPOSES

MODULES = [
    {
        "id": "dataset",
        "title": "Datasets",
        "shortTitle": "Datasets",
        "phase": "prepare",
        "description": "Import slide and patient information, map annotations, and save a "
        "reusable dataset.",
        "prerequisites": [],
    },
    {
        "id": "features",
        "title": "Slide features",
        "shortTitle": "Slide features",
        "phase": "prepare",
        "description": "Register or extract slide features and save a verified bundle. Prepare "
        "this alongside Targets & splits.",
        "prerequisites": ["dataset"],
    },
    {
        "id": "cohort",
        "title": "Targets & splits",
        "shortTitle": "Targets & splits",
        "phase": "prepare",
        "description": "Choose the prediction target and split dataset records into fixed "
        "training and testing sets.",
        "prerequisites": ["dataset"],
    },
    {
        "id": "experiments",
        "title": "Experiments",
        "shortTitle": "Experiments",
        "phase": "develop",
        "description": "Design training on frozen targets, splits and features: folds, "
        "validation and hyperparameters. Freeze the design, start it, and review its results "
        "and predictors.",
        "prerequisites": ["cohort", "features"],
    },
    {
        "id": "apply",
        "title": "Apply models",
        "shortTitle": "Apply models",
        "phase": "evaluate",
        "description": "Apply ready predictors to a cohort. A labeled cohort is scored, by "
        "subgroup and for clinical utility; an unlabeled one gets predictions only.",
        "prerequisites": ["experiments"],
    },
    {
        "id": "interpretation",
        "title": "Model interpretation",
        "shortTitle": "Model interpretation",
        "phase": "insights",
        "description": "Load trained model weights with a dataset and its feature bundle, then "
        "review attention overlays, top patches and predicted labels. Needs no runs from Apply "
        "models.",
        "prerequisites": ["dataset", "features", "experiments"],
        "optional": True,
    },
]

# The numbered steps; stages worked on side by side share one.
STEPS = [
    {"id": "datasets", "step": "01", "title": "Datasets", "modules": ["dataset"]},
    {
        "id": "prepare",
        "step": "02",
        "title": "Prepare in parallel",
        "modules": ["features", "cohort"],
    },
    {"id": "develop", "step": "03", "title": "Experiments", "modules": ["experiments"]},
    {"id": "apply", "step": "04", "title": "Apply models", "modules": ["apply"]},
    {"id": "interpret", "step": "05", "title": "Interpretation", "modules": ["interpretation"]},
]

EVIDENCE_KEYS = (
    "drafts",
    "datasets",
    "targetSplits",
    "setups",
    "features",
    "bundles",
    "extractions",
    "batches",
    "executions",
    "evaluationCohorts",
    "predictors",
    "modelEvaluations",
    "clinicalAnalyses",
    "interpretations",
)
COMPATIBILITY = (
    "Freeze targets and splits and a current feature bundle, then check their training "
    "coverage in an experiment’s inputs."
)


def _s(count: int, one: str = "", many: str = "s") -> str:
    return one if count == 1 else many


def _get(record, *keys):
    value = record
    for key in keys:
        value = (value or {}).get(key)
    return value


def _progress(complete: int, draft: int, completed: str, drafted: str, empty: str) -> dict:
    if complete > 0:
        return {
            "status": "complete",
            "artifactCount": complete,
            "evidence": f"{complete} {completed}{_s(complete)}",
        }
    if draft > 0:
        return {
            "status": "draft",
            "artifactCount": draft,
            "evidence": f"{draft} {drafted}{_s(draft)}",
        }
    return {"status": "not-started", "artifactCount": 0, "evidence": empty}


def extraction_active(job) -> bool:
    return _get(job, "state") in ("queued", "starting", "running", "cancelling")


def training_active(execution) -> bool:
    return _get(execution, "status") in ("queued", "running")


def _extraction_evidence(jobs: list) -> str | None:
    active = [job for job in jobs if extraction_active(job)]
    if active:
        job, current = active[0], active[0].get("progress")
        if job.get("state") == "queued":
            label = "Queued"
        else:
            label = _get(current, "label")
            if label is None:
                label = "Stopping extraction" if job.get("state") == "cancelling" else "Preparing"
        count = ""
        if (
            current
            and current.get("completed") is not None
            and current.get("total") is not None
            and current["total"] > 0
            and job.get("state") != "queued"
        ):
            where = "latest batch" if current.get("scope") == "batch" else "stage"
            count = f" · {current['completed']}/{current['total']} {current.get('unit')} in {where}"
        return f"{len(active)} extraction{_s(len(active))} in progress · {label}{count}"
    if not jobs:
        return None
    latest = jobs[0]
    state = latest.get("state")
    outcome = "completed" if state == "succeeded" else state
    next_step = "review outputs and freeze a bundle" if state == "succeeded" else "review run"
    return f"{len(jobs)} extraction run{_s(len(jobs))} · latest {outcome} · {next_step}"


def _bundle_ready(bundle: dict) -> bool:
    manifest = bundle.get("manifest") or {}
    return bool(
        bundle.get("current")
        and not any(item.get("severity") == "error" for item in bundle.get("findings") or [])
        and _get(manifest, "feature", "validation", "tensorValidationComplete")
        and all(
            _get(pack, "validation", "tensorValidationComplete")
            for pack in manifest.get("packs") or []
        )
    )


def _inference_run(run) -> bool:
    return _get(run, "manifest", "purpose") in INFERENCE_PURPOSES


def _apply_progress(runs: list, cohorts: int, analyses: int) -> dict:
    completed = [run for run in runs if _get(run, "execution", "status") == "completed"]
    if not completed:
        if runs:
            return {
                "status": "draft",
                "artifactCount": len(runs),
                "evidence": f"{len(runs)} run{_s(len(runs))} not completed yet",
            }
        if cohorts:
            return {
                "status": "draft",
                "artifactCount": cohorts,
                "evidence": f"{cohorts} prepared cohort{_s(cohorts)} · no runs yet",
            }
        return {
            "status": "not-started",
            "artifactCount": 0,
            "evidence": "No model applied to a cohort",
        }
    predicted = sum(1 for run in completed if _inference_run(run))
    scored = len(completed) - predicted
    kinds = ", ".join(
        part
        for part in (
            f"{scored} scored" if scored else "",
            f"{predicted} predictions only" if predicted else "",
        )
        if part
    )
    clinical = f" · {analyses} clinical {_s(analyses, 'analysis', 'analyses')}" if analyses else ""
    return {
        "status": "complete",
        "artifactCount": len(completed),
        "evidence": f"{len(completed)} completed run{_s(len(completed))} · {kinds}{clinical}",
    }


def completed_batches(batches: list, executions: list) -> list:
    """`completedDevelopmentBatches`: only a complete execution of every planned run counts."""
    by_batch = {execution.get("batchId"): execution for execution in executions}
    done = []
    for batch in batches:
        execution = by_batch.get(batch.get("id"))
        planned = _get(batch, "manifest", "runs") or []
        expected = _get(batch, "manifest", "summary", "runCount") or 0
        counts = (execution or {}).get("runCounts") or {}
        runs = (execution or {}).get("runs") or []
        if (
            execution is None
            or execution.get("status") != "completed"
            or expected < 1
            or len(planned) != expected
            or counts.get("total") != expected
            or counts.get("completed") != expected
            or len(runs) != expected
            or any(item.get("severity") == "error" for item in execution.get("findings") or [])
        ):
            continue
        finished = {run.get("id") for run in runs if run.get("status") == "completed"}
        if len(finished) == expected and all(run.get("id") in finished for run in planned):
            done.append(batch)
    return done


def build(workspace: dict, evidence: dict | None = None) -> list[dict]:
    """Every stage with its status (`not-started`, `draft`, `complete`), evidence, blockers
    and whether it can be opened. Progress counts saved records, never an open form."""
    demo = workspace.get("demoPipeline")
    if workspace.get("mode") == "synthetic-demo" and demo:
        stages = []
        for module in MODULES:
            count = sum(
                1 for record in demo.get("records") or [] if record.get("module") == module["id"]
            )
            stages.append(
                {
                    **module,
                    "status": "draft",
                    "unlocked": True,
                    "blockers": [],
                    "artifactCount": count,
                    "evidence": f"{count} illustrative {_s(count, 'record', 'records')} · "
                    "synthetic walkthrough"
                    if count
                    else "Illustrative workflow explanation",
                }
            )
        return stages
    saved = {key: list((evidence or {}).get(key) or []) for key in EVIDENCE_KEYS}
    dataset_ids = {item.get("id") for item in saved["datasets"]}
    ready_bundles = [item for item in saved["bundles"] if _bundle_ready(item)]

    def drafts_of(*types):
        return [item for item in saved["drafts"] if _get(item, "payload", "type") in types]

    import_drafts = drafts_of("dataset-import")
    protocol_drafts = drafts_of("target-split")
    target_splits = [
        item for item in saved["targetSplits"] if _get(item, "manifest", "datasetId") in dataset_ids
    ]
    model_drafts = drafts_of("model-experiment", "mil-experiment", "development-batch")

    states = {}
    states["dataset"] = _progress(
        len(saved["datasets"]),
        len(import_drafts),
        "frozen dataset",
        "saved import draft",
        "No frozen dataset or saved import",
    )
    states["cohort"] = _progress(
        len(target_splits),
        len(protocol_drafts) + len(saved["targetSplits"]) - len(target_splits),
        "frozen target and split",
        "saved target draft",
        "No frozen targets and splits",
    )
    states["features"] = _progress(
        len(ready_bundles),
        len(saved["features"]) + len(saved["bundles"]) - len(ready_bundles),
        "verified frozen bundle",
        "saved feature artifact",
        "No saved feature source or frozen bundle",
    )
    if states["features"]["status"] == "draft":
        states["features"]["evidence"] += " · complete bundle verification"
    extraction = _extraction_evidence(saved["extractions"])
    if extraction:
        existing = states["features"]
        complete = existing["status"] == "complete"
        states["features"] = {
            "status": "complete" if complete else "draft",
            "artifactCount": existing["artifactCount"]
            if complete
            else existing["artifactCount"] + len(saved["extractions"]),
            "evidence": extraction
            if existing["status"] == "not-started"
            else f"{existing['evidence']} · {extraction}",
        }
    designs = sum(1 for item in model_drafts if item.get("status") != "frozen")
    states["experiments"] = _progress(
        0,
        designs + len(saved["setups"]) + len(saved["batches"]),
        "",
        "saved experiment design",
        "No experiment designed yet",
    )
    if saved["executions"]:
        finished = completed_batches(saved["batches"], saved["executions"])
        done = sum(_get(item, "runCounts", "completed") or 0 for item in saved["executions"])
        total = sum(_get(item, "runCounts", "total") or 0 for item in saved["executions"])
        active = sum(1 for item in saved["executions"] if training_active(item))
        prefix = (
            f"{len(finished)} completed development batch{_s(len(finished), '', 'es')} · "
            if finished
            else ""
        )
        states["experiments"] = {
            "status": "complete" if finished else "draft",
            "artifactCount": max(len(saved["batches"]), len(saved["executions"])),
            "evidence": f"{prefix}{done}/{total} training runs completed"
            + (f" · {active} active batch{_s(active, '', 'es')}" if active else ""),
        }
    predictors = [item for item in saved["predictors"] if item.get("lifecycleState") != "trashed"]
    runs = [item for item in saved["modelEvaluations"] if item.get("lifecycleState") != "trashed"]
    if predictors:
        published = f"{len(predictors)} ready predictor{_s(len(predictors))}"
        current = states["experiments"]
        states["experiments"] = {
            "status": "complete",
            "artifactCount": max(current["artifactCount"], len(predictors)),
            "evidence": f"{current['evidence']} · {published}"
            if current["artifactCount"]
            else published,
        }
    cohort_drafts = len(drafts_of("evaluation-cohort"))
    analyses = [
        item for item in saved["clinicalAnalyses"] if item.get("lifecycleState") != "trashed"
    ]
    states["apply"] = _apply_progress(
        runs, len(saved["evaluationCohorts"]) + cohort_drafts, len(analyses)
    )
    maps = [item for item in saved["interpretations"] if item.get("lifecycleState") != "trashed"]
    completed_maps = sum(1 for item in maps if _get(item, "execution", "status") == "completed")
    states["interpretation"] = _progress(
        completed_maps,
        len(maps) - completed_maps,
        "completed attention map",
        "saved interpretation plan",
        "No attention overlay generated",
    )

    compatible = bool(target_splits) and bool(ready_bundles)
    retained = {
        "cohort": bool(saved["targetSplits"] or protocol_drafts),
        "features": bool(saved["features"] or saved["bundles"] or saved["extractions"]),
        "experiments": bool(saved["batches"] or saved["setups"] or model_drafts or predictors),
        "apply": bool(runs),
    }

    def blocked(needed: str, module: str) -> bool:
        if needed == "experiments" and module == "apply":
            return not predictors
        if needed == "experiments" and module == "interpretation":
            return not any(
                catalog.supports_attention(_get(item, "manifest", "recipe", "model"), None)
                for item in predictors
            )
        return states[needed]["status"] != "complete"

    stages = []
    for module in MODULES:
        kept = retained.get(module["id"]) is True
        blockers = (
            []
            if kept
            else [item for item in module["prerequisites"] if blocked(item, module["id"])]
        )
        issue = (
            COMPATIBILITY
            if module["id"] == "experiments" and not kept and not blockers and not compatible
            else None
        )
        if issue:
            blockers.append("features")
        stages.append(
            {
                **module,
                **states[module["id"]],
                "blockers": blockers,
                # Every stage is a registry: it opens to inspect or recover saved records.
                "unlocked": True,
                "compatibilityIssue": issue,
                "retainedWork": kept,
            }
        )
    return stages


def suggested(stages: list[dict]) -> dict | None:
    """`suggestedRoadmapModule`: the first required stage that is open, unblocked and not
    complete."""
    return next(
        (
            stage
            for stage in stages
            if not stage.get("optional")
            and stage["status"] != "complete"
            and stage["unlocked"]
            and not stage["blockers"]
        ),
        None,
    )


def step_of(stage: str) -> str | None:
    return next((item["step"] for item in STEPS if stage in item["modules"]), None)
