"""What each service route does: the basis for confirmation, retries and access scopes.

Every route under `/api/v1` has exactly one class; see docs/cli-contract.md#route-classes.
A test fails when a route is missing here or an entry names a route that no longer exists,
so a new route is classified on purpose rather than by default. Stdlib only: the CLI
imports this table without building the service.
"""

import re
from functools import cache

READ = "read"  # Changes nothing. Caches and housekeeping do not count.
PREVIEW = "preview"  # Prepares a change: preview checks and draft saves.
COMMIT = "commit"  # Changes one project for real.
ADMIN = "admin"  # Reaches beyond one project, or widens what a project can reach.
CLASSES = (READ, PREVIEW, COMMIT, ADMIN)

API_PREFIX = "/api/v1"
PROJECT_PREFIX = "/projects/{identity}"

# Scoped agent tokens (docs/agents.md): one project, scopes weakest first, each including the
# ones before it. No token reaches admin routes.
TOKEN_PREFIX = "hpt_"
TOKEN_SCOPES = (READ, PREVIEW, COMMIT)
DEFAULT_TOKEN_SCOPES = (READ, PREVIEW)
TOKEN_DAYS = 7
MAX_TOKEN_DAYS = 90
# How much of a project agent tokens may see, least first.
EXPOSURE_LEVELS = ("none", "metadata", "full")

# Paths are relative to /api/v1; {P} stands for /projects/{identity}. A route with
# several actions takes its strongest: archive restore writes a new study folder.
_TABLE = {
    # Service, filesystem and projects
    ("GET", "/access"): READ,
    ("GET", "/agent-requests"): READ,
    ("GET", "/agent-requests/{request_id}"): READ,
    ("POST", "/agent-requests/{request_id}/claim"): ADMIN,
    ("POST", "/agent-requests/{request_id}/resolve"): ADMIN,
    ("POST", "/filesystem/directories"): ADMIN,
    ("GET", "/filesystem/list"): READ,
    ("GET", "/filesystem/roots"): READ,
    ("GET", "/health"): READ,
    ("GET", "/projects"): READ,
    ("POST", "/projects"): ADMIN,
    ("POST", "/projects/open"): ADMIN,
    ("GET", "/session"): READ,
    ("GET", "/system"): READ,
    ("GET", "/system/compute"): READ,
    ("GET", "/templates"): READ,
    ("GET", "/tokens"): READ,
    ("POST", "/tokens"): ADMIN,
    ("POST", "/tokens/{token_id}/revoke"): ADMIN,
    ("GET", "/version"): READ,
    # Project
    ("PATCH", "{P}"): COMMIT,
    # Widening what agents may see is an admin change.
    ("GET", "{P}/ai-exposure"): READ,
    ("PUT", "{P}/ai-exposure"): ADMIN,
    ("POST", "{P}/sources"): ADMIN,
    ("GET", "{P}/storage"): READ,
    ("GET", "{P}/workspace"): READ,
    # Drafts, imports and datasets
    ("GET", "{P}/configurations"): READ,
    ("GET", "{P}/configurations/{configuration_id}"): READ,
    ("PUT", "{P}/configurations/{configuration_id}/label"): COMMIT,
    ("GET", "{P}/datasets"): READ,
    ("GET", "{P}/datasets/{dataset_id}"): READ,
    ("PUT", "{P}/datasets/{dataset_id}/label"): COMMIT,
    ("POST", "{P}/datasets/{dataset_id}/query"): READ,
    ("GET", "{P}/datasets/{dataset_id}/records"): READ,
    ("GET", "{P}/datasets/{dataset_id}/slide-reviews"): READ,
    ("GET", "{P}/datasets/{dataset_id}/slide-reviews/{slide_id}"): READ,
    ("PUT", "{P}/datasets/{dataset_id}/slide-reviews/{slide_id}"): COMMIT,
    ("GET", "{P}/drafts"): READ,
    ("POST", "{P}/drafts"): PREVIEW,
    ("GET", "{P}/drafts/{draft_id}"): READ,
    ("PATCH", "{P}/drafts/{draft_id}"): PREVIEW,
    ("POST", "{P}/drafts/{draft_id}/evaluation-freeze"): COMMIT,
    ("POST", "{P}/drafts/{draft_id}/evaluation-preview"): PREVIEW,
    ("POST", "{P}/imports/inspect"): READ,
    ("POST", "{P}/imports/{draft_id}/freeze"): COMMIT,
    ("POST", "{P}/imports/{draft_id}/preview"): PREVIEW,
    # Targets and splits
    ("POST", "{P}/protocols/explore"): READ,
    ("POST", "{P}/target-splits/partition-preview"): PREVIEW,
    ("GET", "{P}/target-splits/{configuration_id}"): READ,
    ("POST", "{P}/target-splits/{configuration_id}/test-cohort"): COMMIT,
    ("POST", "{P}/target-splits/{draft_id}/freeze"): COMMIT,
    ("POST", "{P}/target-splits/{draft_id}/preview"): PREVIEW,
    # Features and extraction
    ("GET", "{P}/extractions"): READ,
    ("POST", "{P}/extractions"): COMMIT,
    ("GET", "{P}/extractions/catalog"): READ,
    ("POST", "{P}/extractions/preview"): PREVIEW,
    ("GET", "{P}/extractions/{job_id}"): READ,
    ("POST", "{P}/extractions/{job_id}/cancel"): COMMIT,
    ("POST", "{P}/extractions/{job_id}/resume"): COMMIT,
    ("GET", "{P}/feature-bundles"): READ,
    ("POST", "{P}/feature-bundles/freeze"): COMMIT,
    ("POST", "{P}/feature-bundles/preview"): PREVIEW,
    ("GET", "{P}/feature-bundles/{bundle_id}"): READ,
    ("GET", "{P}/feature-packs"): READ,
    ("POST", "{P}/feature-packs"): COMMIT,
    ("GET", "{P}/feature-packs/artifacts/{artifact_id}"): READ,
    ("POST", "{P}/feature-packs/preview"): PREVIEW,
    ("GET", "{P}/feature-packs/{job_id}"): READ,
    ("POST", "{P}/feature-packs/{job_id}/cancel"): COMMIT,
    ("POST", "{P}/features/freeze"): COMMIT,
    ("POST", "{P}/features/preview"): PREVIEW,
    ("GET", "{P}/features/{feature_id}/pack-selection"): READ,
    ("PUT", "{P}/features/{feature_id}/pack-selection"): COMMIT,
    ("GET", "{P}/features/{feature_id}/validation"): READ,
    # Experiments and development batches
    ("GET", "{P}/mil-experiments/batches"): READ,
    ("POST", "{P}/mil-experiments/batches/freeze"): COMMIT,
    ("POST", "{P}/mil-experiments/batches/preview"): PREVIEW,
    ("POST", "{P}/mil-experiments/batches/{batch_id}/cancel"): COMMIT,
    ("GET", "{P}/mil-experiments/batches/{batch_id}/execution"): READ,
    ("POST", "{P}/mil-experiments/batches/{batch_id}/launch"): COMMIT,
    (
        "GET",
        "{P}/mil-experiments/batches/{batch_id}/oof/{candidate_id}/{training_seed}/{split_seed}/{unit}.csv",
    ): READ,
    ("GET", "{P}/mil-experiments/batches/{batch_id}/resources/history"): READ,
    ("GET", "{P}/mil-experiments/batches/{batch_id}/results"): READ,
    ("POST", "{P}/mil-experiments/batches/{batch_id}/resume"): COMMIT,
    ("GET", "{P}/mil-experiments/batches/{batch_id}/runs/{run_id}/history"): READ,
    ("GET", "{P}/mil-experiments/clinical-fields"): READ,
    ("POST", "{P}/mil-experiments/preview"): PREVIEW,
    ("GET", "{P}/mil-experiments/runtime"): READ,
    ("GET", "{P}/model-experiments"): READ,
    ("POST", "{P}/model-experiments"): PREVIEW,
    ("GET", "{P}/model-experiments/headlines"): READ,
    ("GET", "{P}/model-experiments/{experiment_id}"): READ,
    ("PATCH", "{P}/model-experiments/{experiment_id}"): PREVIEW,
    ("POST", "{P}/model-experiments/{experiment_id}/freeze-setup"): COMMIT,
    ("POST", "{P}/model-experiments/{experiment_id}/predictors/cancel"): COMMIT,
    ("POST", "{P}/model-experiments/{experiment_id}/predictors/resume"): COMMIT,
    ("GET", "{P}/model-experiments/{experiment_id}/results"): READ,
    ("POST", "{P}/model-experiments/{experiment_id}/setup-inputs"): PREVIEW,
    ("POST", "{P}/model-experiments/{experiment_id}/submit"): COMMIT,
    # Predictors
    ("GET", "{P}/predictors"): READ,
    ("POST", "{P}/predictors/builds"): COMMIT,
    ("POST", "{P}/predictors/builds/preview"): PREVIEW,
    ("GET", "{P}/predictors/builds/{operation_id}"): READ,
    ("GET", "{P}/predictors/choices"): READ,
    ("POST", "{P}/predictors/freeze"): COMMIT,
    ("POST", "{P}/predictors/preview"): PREVIEW,
    ("GET", "{P}/predictors/refits"): READ,
    ("POST", "{P}/predictors/refits"): COMMIT,
    ("POST", "{P}/predictors/refits/{refit_id}/cancel"): COMMIT,
    ("GET", "{P}/predictors/refits/{refit_id}/execution"): READ,
    ("POST", "{P}/predictors/refits/{refit_id}/launch"): COMMIT,
    ("POST", "{P}/predictors/refits/{refit_id}/publish"): COMMIT,
    ("POST", "{P}/predictors/refits/{refit_id}/resume"): COMMIT,
    ("GET", "{P}/predictors/seed-ensembles"): READ,
    ("POST", "{P}/predictors/seed-ensembles"): COMMIT,
    ("POST", "{P}/predictors/seed-ensembles/preview"): PREVIEW,
    ("GET", "{P}/predictors/{predictor_id}"): READ,
    # Cohorts and runs
    ("GET", "{P}/evaluation-cohorts"): READ,
    ("GET", "{P}/evaluation-cohorts/{configuration_id}"): READ,
    ("GET", "{P}/evaluation-runs"): READ,
    ("POST", "{P}/evaluation-runs"): COMMIT,
    ("GET", "{P}/evaluation-runs/bulk"): READ,
    ("POST", "{P}/evaluation-runs/bulk"): COMMIT,
    ("POST", "{P}/evaluation-runs/bulk/preview"): PREVIEW,
    ("GET", "{P}/evaluation-runs/bulk/{batch_id}"): READ,
    ("POST", "{P}/evaluation-runs/bulk/{batch_id}/cancel"): COMMIT,
    ("POST", "{P}/evaluation-runs/compare"): READ,
    ("POST", "{P}/evaluation-runs/preview"): PREVIEW,
    ("GET", "{P}/evaluation-runs/{evaluation_id}"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/agreement"): READ,
    ("GET", "{P}/evaluation-runs/{evaluation_id}/artifacts/{filename}"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/attention"): COMMIT,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/cancel"): COMMIT,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/cases/export"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/cases/query"): READ,
    ("GET", "{P}/evaluation-runs/{evaluation_id}/execution"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/inference/export"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/inference/summary"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/launch"): COMMIT,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/performance/breakdown"): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/recalibration"): READ,
    (
        "GET",
        "{P}/evaluation-runs/{evaluation_id}/reference-standards/{reference_id}/{filename}",
    ): READ,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/resume"): COMMIT,
    ("POST", "{P}/evaluation-runs/{evaluation_id}/scores"): READ,
    ("GET", "{P}/reference-standards"): READ,
    ("POST", "{P}/reference-standards"): COMMIT,
    ("POST", "{P}/reference-standards/preview"): PREVIEW,
    ("GET", "{P}/reference-standards/{reference_id}"): READ,
    # Clinical analyses and interpretation
    ("GET", "{P}/clinical-analyses"): READ,
    ("POST", "{P}/clinical-analyses"): COMMIT,
    ("POST", "{P}/clinical-analyses/preview"): PREVIEW,
    ("GET", "{P}/clinical-analyses/{analysis_id}"): READ,
    ("GET", "{P}/clinical-analyses/{analysis_id}/artifacts/{filename}"): READ,
    ("GET", "{P}/interpretations"): READ,
    ("POST", "{P}/interpretations"): COMMIT,
    ("GET", "{P}/interpretations/datasets"): READ,
    ("POST", "{P}/interpretations/gallery"): READ,
    ("GET", "{P}/interpretations/gallery/thumbnail"): READ,
    ("POST", "{P}/interpretations/preview"): PREVIEW,
    ("POST", "{P}/interpretations/slide-inspection"): READ,
    ("GET", "{P}/interpretations/sources"): READ,
    ("POST", "{P}/interpretations/visualize"): COMMIT,
    ("GET", "{P}/interpretations/{interpretation_id}"): READ,
    ("GET", "{P}/interpretations/{interpretation_id}/artifacts/{filename}"): READ,
    ("POST", "{P}/interpretations/{interpretation_id}/cancel"): COMMIT,
    ("GET", "{P}/interpretations/{interpretation_id}/execution"): READ,
    ("POST", "{P}/interpretations/{interpretation_id}/launch"): COMMIT,
    ("POST", "{P}/interpretations/{interpretation_id}/resume"): COMMIT,
    ("GET", "{P}/interpretations/{interpretation_id}/slides/{slide_id}/attention"): READ,
    ("GET", "{P}/interpretations/{interpretation_id}/slides/{slide_id}/attention/top"): READ,
    (
        "GET",
        "{P}/interpretations/{interpretation_id}/slides/{slide_id}/patches/{patch_index}/image",
    ): READ,
    ("GET", "{P}/interpretations/{interpretation_id}/slides/{slide_id}/region"): READ,
    ("GET", "{P}/interpretations/{interpretation_id}/slides/{slide_id}/thumbnail"): READ,
    # Slide viewing
    ("GET", "{P}/morphology/image"): READ,
    ("POST", "{P}/morphology/index"): READ,
    ("POST", "{P}/morphology/neighbors"): READ,
    ("GET", "{P}/morphology/patch"): READ,
    ("GET", "{P}/morphology/patch-region"): READ,
    ("GET", "{P}/morphology/quality"): READ,
    ("GET", "{P}/morphology/slides"): READ,
    # Cleanup and study backups
    ("GET", "{P}/cleanup"): READ,
    ("POST", "{P}/cleanup/apply"): COMMIT,
    ("POST", "{P}/cleanup/cancel"): COMMIT,
    ("POST", "{P}/cleanup/preview"): PREVIEW,
    ("GET", "{P}/operations"): READ,
    ("GET", "{P}/operations/archives"): READ,
    ("POST", "{P}/operations/archives"): ADMIN,
    ("GET", "{P}/operations/archives/{job_id}"): READ,
    ("POST", "{P}/operations/archives/{job_id}/cancel"): ADMIN,
    ("POST", "{P}/operations/archives/{job_id}/retry"): ADMIN,
    ("GET", "{P}/operations/sources"): READ,
    ("POST", "{P}/operations/sources/relink"): ADMIN,
    # Task Center (machine-wide)
    ("GET", "/task-center/capacity"): READ,
    ("PUT", "/task-center/capacity"): ADMIN,
    ("GET", "/task-center/history"): READ,
    ("GET", "/task-center/owners"): READ,
    ("GET", "/task-center/owners/{key}"): READ,
    ("POST", "/task-center/owners/{key}/{action}"): COMMIT,
    ("GET", "/task-center/rollup"): READ,
    ("POST", "/task-center/runner/restart"): ADMIN,
    ("POST", "/task-center/runner/start"): ADMIN,
    ("GET", "/task-center/snapshot"): READ,
    ("GET", "/task-center/summary"): READ,
    ("GET", "/task-center/tasks"): READ,
    ("GET", "/task-center/tasks/{task_id}"): READ,
    ("GET", "/task-center/tasks/{task_id}/log"): READ,
    ("POST", "/task-center/tasks/{task_id}/{action}"): COMMIT,
}


def _expand(path: str) -> str:
    return API_PREFIX + (PROJECT_PREFIX + path[3:] if path.startswith("{P}") else path)


ROUTE_CLASSES: dict[tuple[str, str], str] = {
    (method, _expand(path)): route_class for (method, path), route_class in _TABLE.items()
}


@cache
def _patterns() -> list[tuple[int, str, re.Pattern, str, str]]:
    compiled = []
    for (method, template), route_class_ in ROUTE_CLASSES.items():
        parts = re.split(r"(\{[^}]+\})", template)
        pattern = "".join("[^/]+?" if part.startswith("{") else re.escape(part) for part in parts)
        # Literal routes win over parameterised ones, as FastAPI's order does for these.
        compiled.append(
            (len(parts) // 2, method, re.compile(f"^{pattern}$"), route_class_, template)
        )
    return sorted(compiled, key=lambda row: row[0])


def _match(method: str, path: str):
    path = path.split("?", 1)[0].split("#", 1)[0]
    if len(path) > 1:
        path = path.rstrip("/")
    method = method.upper()
    for _, candidate, pattern, route_class_, template in _patterns():
        if candidate == method and pattern.match(path):
            return route_class_, template
    return None, None


def route_template(method: str, path: str) -> str | None:
    """The route a concrete request matches, as the service declares it; None if unknown."""
    return _match(method, path)[1]


def route_class(method: str, path: str) -> str | None:
    """The class of a concrete request, e.g. PATCH /api/v1/projects/p/drafts/d.

    None for a route this table does not know; callers treat that as commit.
    """
    return _match(method, path)[0]
