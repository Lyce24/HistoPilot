"""Read-only runtime advice for a reviewed draft; never changes a running plan."""

from datetime import UTC, datetime

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.development import DevelopmentService, _hash, _plan_metadata
from histopilot.application.runtime_advisor import recommend_runtime
from histopilot.application.runtime_evidence import observed_runtime_runs, read_active_leases
from histopilot.application.runtime_workload import workload_for_recipe
from histopilot.storage.project_lock import StorageError
from histopilot.workers.training_process import gpu_snapshot, host_snapshot


def runtime_recommendation(store, filesystem, spec):
    """Use current capacity, eligible feature headers, and compatible saved peaks."""
    preview = DevelopmentService(store, filesystem).preview(spec)
    timestamp = datetime.now(UTC).isoformat()
    if not preview["canFreeze"]:
        return {
            "version": 1,
            "generatedAt": timestamp,
            "applicable": False,
            "basis": "unavailable",
            "resources": None,
            "summary": "Complete the batch inputs before estimating runtime settings.",
            "memory": {"perRunGpuGb": None, "perRunRamGb": None, "observedRuns": 0},
            "limits": {
                "cpuConcurrency": 0,
                "ramConcurrency": 0,
                "gpuConcurrency": None,
                "additionalRunsNow": None,
            },
            "evidence": [],
            "findings": preview["findings"],
        }
    try:
        runtime = dict(training_runtime())
        # The environment probe is cached; resource availability must be read now.
        runtime.update(host=host_snapshot(), **gpu_snapshot())
    except (OSError, ValueError, RuntimeError):
        result = recommend_runtime({}, [], [], spec.resources.gpuIds, 1)
        return {**result, "generatedAt": timestamp}
    binding = preview["resolvedInputs"]
    protocol = store.get_configuration(spec.inputs.protocolId)
    bundle = store.get_configuration(spec.inputs.featureBundleId)
    feature = store.get_configuration(binding["featureSetId"])
    files = {row["slideId"]: row for row in feature["manifest"]["files"]}
    groups = {row["id"]: [] for row in preview["splitPlans"]}
    for row in protocol["manifest"]["memberships"]:
        identity = _hash(_plan_metadata(row))
        if identity in groups:
            groups[identity].append(row)
    workloads = {}
    try:
        for candidate in preview["configurations"]:
            for rows in groups.values():
                item = workload_for_recipe(
                    candidate["recipe"],
                    rows,
                    files,
                    protocol["contentHash"],
                    bundle["contentHash"],
                    binding["resolvedLoadingPolicy"],
                    len(protocol["manifest"]["spec"]["target"]["classes"]),
                )
                # Evaluate the largest requirements across all candidate folds.
                if item["key"] in workloads:
                    prior = workloads[item["key"]]
                    for key in ("trainingPatches", "evaluationPatches", "sourcePatches"):
                        item[key] = max(item[key], prior[key])
                item["runtimeVersions"] = runtime.get("versions", {})
                item["runtimePython"] = runtime.get("python")
                workloads[item["key"]] = item
    except (KeyError, TypeError, ValueError) as error:
        raise StorageError(str(error), "RUNTIME_WORKLOAD_INVALID", 422) from error
    profiles = list(workloads.values())
    observations = observed_runtime_runs(store, profiles)
    result = recommend_runtime(
        runtime,
        profiles,
        observations,
        spec.resources.gpuIds,
        preview["summary"]["runCount"],
        active_leases=read_active_leases(),
    )
    return {
        **result,
        "generatedAt": timestamp,
        "hardware": {**runtime["host"], "gpus": runtime.get("gpus", [])},
    }
