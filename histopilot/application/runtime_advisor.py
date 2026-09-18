"""Pure, conservative advice from resource snapshots and matching run evidence.

This is a capacity starting point, not a throughput benchmark. No CUDA imports,
feature reads, process launches, lease mutations, or changes to frozen batches.
All memory units follow the existing runtime API: binary GiB named ``Gb``.
"""

import math

from histopilot.models.catalog import NAMES as _MODELS

_GIB = 1024**3


def _number(value, *, positive=False):
    return (
        type(value) in {int, float}
        and (value > 0 if positive else value >= 0)
        and value <= 2**31
        and math.isfinite(value)
    )


def _integer(value, *, minimum=1):
    return type(value) is int and minimum <= value <= 2**31 - 1


def _finding(code, message, severity="warning"):
    return {"code": code, "message": message, "severity": severity}


def _empty(message, code="RUNTIME_ADVICE_UNAVAILABLE", *, findings=()):
    return {
        "version": 1,
        "applicable": False,
        "basis": "unavailable",
        "resources": None,
        "summary": message,
        "memory": {"perRunGpuGb": None, "perRunRamGb": None, "observedRuns": 0},
        "limits": {
            "cpuConcurrency": 0,
            "ramConcurrency": 0,
            "gpuConcurrency": None,
            "additionalRunsNow": 0,
        },
        "evidence": [],
        "findings": [*findings, _finding(code, message, "error")],
    }


def _valid_workload(row):
    return (
        isinstance(row, dict)
        and isinstance(row.get("key"), str)
        and bool(row["key"])
        and isinstance(row.get("model"), str)
        and row.get("model") in _MODELS
        and all(
            _integer(row.get(name))
            for name in (
                "featureDimension",
                "trainingPatches",
                "evaluationPatches",
                "batchSize",
                "evalBatchSize",
                "embedDim",
                "attentionDim",
                "numFcLayers",
            )
        )
        and isinstance(row.get("precision"), str)
        and row.get("precision") in {"32-true", "16-mixed", "bf16-mixed"}
        and type(row.get("gradientCheckpointing")) is bool
        and isinstance(row.get("loadingPolicy"), str)
        and row.get("loadingPolicy") in {"native", "mmap"}
        and ("sourcePatches" not in row or _integer(row["sourcePatches"]))
    )


def _gpu_estimate(row, *, evaluation_only=False):
    """Count full-dimensional bags and live activations, including full-bag eval.

    Even AMP keeps the source bag and stable pooling in float32. nnMIL evaluates
    its feature windows sequentially, so their number is not a memory multiplier.
    Checkpointing receives no speculative discount before actual measurements.
    """
    dim, embed, attention = row["featureDimension"], row["embedDim"], row["attentionDim"]
    activation_bytes = 4 if row["precision"] == "32-true" else 2
    if row["model"] == "nnmil":
        train_width = 8 * attention
        eval_width = 4 * attention
        parameters = 2 * dim * attention
    elif row["model"] == "abmil":
        train_width = 4 * embed * row["numFcLayers"] + 8 * attention
        eval_width = 2 * embed + 4 * attention
        parameters = dim * embed + embed * embed * row["numFcLayers"] + 2 * embed * attention
    else:
        train_width = 4 * embed * row["numFcLayers"]
        eval_width = 2 * embed
        parameters = dim * embed + embed * embed * row["numFcLayers"]
    train = row["batchSize"] * row["trainingPatches"] * (8 * dim + activation_bytes * train_width)
    evaluation = (
        row["evalBatchSize"] * row["evaluationPatches"] * (8 * dim + activation_bytes * eval_width)
    )
    # Adam state, gradients, allocator fragmentation, context and kernels.
    largest = evaluation if evaluation_only else max(train, evaluation)
    return 1.0 + 1.25 * (largest + parameters * 20) / _GIB


def _ram_estimate(workloads, workers, *, cpu_only=False):
    estimates = [8.0]
    for row in workloads:
        train = row["batchSize"] * row["trainingPatches"] * row["featureDimension"] * 4 / _GIB
        evaluation = (
            row["evalBatchSize"] * row["evaluationPatches"] * row["featureDimension"] * 4 / _GIB
        )
        # Conservative transport/collation buffers for two persistent pools.
        # This is an allowance, not the loader's configured prefetch factor.
        buffers = (2 * workers + 2) * (train + evaluation)
        source_patches = row.get(
            "sourcePatches", max(row["trainingPatches"], row["evaluationPatches"])
        )
        source = source_patches * row["featureDimension"] * 4 / _GIB
        # Native PT/NPY readers may materialize a source bag before sampling.
        native = 2 * max(1, workers) * source if row["loadingPolicy"] == "native" else 0
        # CPU fitting keeps the model, gradients, optimizer state and live
        # activations in host RAM too. Budget float32 conservatively, regardless
        # of the requested precision, and omit the GPU-context allowance.
        computation = _gpu_estimate({**row, "precision": "32-true"}) - 1.0 if cpu_only else 0.0
        estimates.append(4.0 + buffers + native + computation)
    return float(math.ceil(max(estimates)))


def _matching_observations(row, gpu, observations):
    matches = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        if (
            observation.get("key") != row["key"]
            or not gpu.get("uuid")
            or observation.get("gpuUuid") != gpu["uuid"]
            or not isinstance(observation.get("stage"), str)
            or observation.get("stage") not in {"fit", "assessment"}
            or not _integer(observation.get("epoch"), minimum=0)
            or (observation["stage"] == "fit" and observation["epoch"] < 1)
            or not _number(observation.get("peakReservedGpuGb"), positive=True)
            or observation["peakReservedGpuGb"] > gpu["totalMemoryGb"]
            or not all(
                _integer(observation.get(name)) for name in ("trainingPatches", "evaluationPatches")
            )
            or not all(
                isinstance(observation.get(name), str) and observation[name]
                for name in ("batchId", "runId")
            )
        ):
            continue
        scale = max(
            1.0,
            row["trainingPatches"] / observation["trainingPatches"],
            row["evaluationPatches"] / observation["evaluationPatches"],
        )
        # Nearby folds may have different training medians. Larger changes need
        # a fresh profile; never extrapolate a small-bag peak across a huge bag.
        if scale <= 1.25:
            matches.append((observation, scale))
    return matches


def recommend_runtime(
    runtime: dict,
    workloads: list[dict],
    observations: list[dict],
    gpu_ids: list[int],
    run_count: int,
    active_leases: list[dict] | None = None,
) -> dict:
    """Suggest editable resources; existing batches and reservations stay frozen."""
    if not isinstance(runtime, dict) or runtime.get("available") is not True:
        return _empty("The training runtime is unavailable; no resource settings were suggested.")
    host = runtime.get("host", {})
    if not isinstance(host, dict) or not (
        _integer(host.get("cpuCount"))
        and _number(host.get("totalRamGb"), positive=True)
        and _number(host.get("availableRamGb"))
        and host["availableRamGb"] <= host["totalRamGb"]
    ):
        return _empty("CPU count and valid available RAM telemetry are required.")
    if (
        not isinstance(gpu_ids, list)
        or any(not _integer(index, minimum=0) or index > 127 for index in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
        or not _integer(run_count)
        or not isinstance(workloads, list)
        or not all(_valid_workload(row) for row in workloads)
        or not isinstance(observations, list)
    ):
        return _empty("Valid workload dimensions, run count and device selection are required.")
    leases_known = active_leases is not None
    leases = [] if active_leases is None else active_leases
    if not isinstance(leases, list) or any(
        not isinstance(lease, dict)
        or not _integer(lease.get("cpus"))
        or not _number(lease.get("ramGb"), positive=True)
        or not _number(lease.get("rssGb", 0))
        or not _integer(lease.get("runsPerGpu"))
        or (lease.get("gpu") is not None and not _integer(lease["gpu"], minimum=0))
        for lease in leases
    ):
        return _empty("Active resource reservations could not be validated.")

    findings = [
        _finding(
            "RUNTIME_ADVICE_STARTING_POINT",
            "This is a conservative capacity starting point, not a benchmarked throughput optimum. "
            "Compare complete epoch times before increasing concurrency or data workers.",
            "info",
        )
    ]
    physical = host.get("physicalCpuCount")
    physical = min(physical, host["cpuCount"]) if _integer(physical) else host["cpuCount"]
    cpu_reserve = min(2, max(0, physical // 8))
    cpu_budget = max(0, host["cpuCount"] - cpu_reserve - sum(row["cpus"] for row in leases))
    ram_reserve = min(16.0, max(2.0, host["totalRamGb"] * 0.05))
    unused_reservations = sum(max(0, row["ramGb"] - row.get("rssGb", 0)) for row in leases)
    ram_budget = max(0.0, host["availableRamGb"] - ram_reserve - unused_reservations)
    workers, threads = (2 if gpu_ids else 0), min(2, cpu_budget)
    while workers and (
        threads + 2 * workers > cpu_budget
        or _ram_estimate(workloads, workers, cpu_only=not gpu_ids) > ram_budget
    ):
        workers -= 1
    if threads == 0:
        return _empty("No CPU capacity remains after active reservations.", findings=findings)
    ram_per_run = _ram_estimate(workloads, workers, cpu_only=not gpu_ids)
    cpu_limit = cpu_budget // (threads + 2 * workers)
    ram_limit = int(ram_budget // ram_per_run)
    if not cpu_limit or not ram_limit:
        return _empty(
            "Available CPU or RAM cannot safely accommodate one run with feature buffers.",
            "RUNTIME_ADVICE_NO_CAPACITY",
            findings=findings,
        )

    usable_gpus, device_limits, memory_needs, device_additional, evidence = [], [], [], [], []
    all_measured = bool(workloads and gpu_ids)
    if gpu_ids and (runtime.get("cudaAvailable") is not True):
        return _empty(
            "The selected GPUs are unavailable to the training runtime.", findings=findings
        )
    reported_gpus = runtime.get("gpus", [])
    if not isinstance(reported_gpus, list):
        reported_gpus = []
    for index in gpu_ids:
        devices = [
            gpu for gpu in reported_gpus if isinstance(gpu, dict) and gpu.get("index") == index
        ]
        gpu = devices[0] if len(devices) == 1 else {}
        if not (
            _integer(runtime.get("gpuCount"), minimum=0)
            and index < runtime["gpuCount"]
            and _number(gpu.get("totalMemoryGb"), positive=True)
            and _number(gpu.get("freeMemoryGb"))
            and gpu["freeMemoryGb"] <= gpu["totalMemoryGb"]
        ):
            findings.append(
                _finding(
                    "RUNTIME_ADVICE_GPU_UNKNOWN",
                    f"GPU {index} was excluded because its available VRAM is unknown.",
                )
            )
            continue
        measured, assessed, needs, device_evidence = bool(workloads), bool(workloads), [], []
        for row in workloads:
            matches = _matching_observations(row, gpu, observations)
            if matches:
                peak = max(item["peakReservedGpuGb"] * scale for item, scale in matches)
                completed_assessment = any(
                    item["stage"] == "assessment" for item, _scale in matches
                )
                required = peak * 1.3 + 0.5
                if not completed_assessment:
                    # Assessment may contain a larger eligible slide than the
                    # fitting/validation stage has loaded so far.
                    required = max(required, _gpu_estimate(row, evaluation_only=True))
                needs.append(required)
                device_evidence.extend(item for item, _scale in matches)
                assessed = assessed and completed_assessment
            else:
                needs.append(_gpu_estimate(row))
                measured = assessed = False
        per_run = max(needs, default=4.0)
        reserve = max(2.0, gpu["totalMemoryGb"] * 0.10)
        free_slots = int(max(0, gpu["freeMemoryGb"] - reserve) // per_run)
        cap = 4 if assessed else (3 if measured else 1)
        limit = min(free_slots, cap, run_count)
        if not limit:
            findings.append(
                _finding(
                    "RUNTIME_ADVICE_GPU_BUSY",
                    f"GPU {index} was excluded: current free VRAM cannot cover one run plus headroom.",
                )
            )
            continue
        all_measured = all_measured and measured
        same_gpu = [lease for lease in leases if lease["gpu"] == index]
        lease_limit = min([limit, *(lease["runsPerGpu"] for lease in same_gpu)])
        usable_gpus.append(index)
        device_limits.append(limit)
        memory_needs.append(per_run)
        device_additional.append(max(0, lease_limit - len(same_gpu)))
        evidence.extend(device_evidence)
    if gpu_ids and not usable_gpus:
        return _empty(
            "No selected GPU has verified capacity for this workload right now.",
            "RUNTIME_ADVICE_NO_GPU_CAPACITY",
            findings=findings,
        )

    # The scheduler has a uniform runs-per-GPU policy. The smallest device must
    # remain safe even when it receives the largest requested candidate.
    runs_per_gpu = min(device_limits) if usable_gpus else 1
    gpu_limit = len(usable_gpus) * runs_per_gpu if usable_gpus else None
    concurrency = min(run_count, 128, cpu_limit, ram_limit, gpu_limit or 1)
    runs_per_gpu = min(runs_per_gpu, concurrency)
    additional = min(concurrency, sum(device_additional)) if usable_gpus else concurrency
    if usable_gpus:
        # Reapply the final uniform policy to prevent heterogeneous-device or
        # host limits from inflating the number of immediately available slots.
        additional = min(
            concurrency,
            sum(
                max(
                    0,
                    min(
                        [
                            runs_per_gpu,
                            *(lease["runsPerGpu"] for lease in leases if lease["gpu"] == index),
                        ]
                    )
                    - sum(lease["gpu"] == index for lease in leases),
                )
                for index in usable_gpus
            ),
        )
    basis = "measured" if all_measured else ("estimated" if workloads else "hardware_only")
    if basis != "measured":
        findings.append(
            _finding(
                "RUNTIME_ADVICE_UNPROFILED",
                "Matching completed-epoch memory evidence does not cover every requested workload and GPU. "
                "Unprofiled devices are limited to one run; estimates cannot guarantee that every bag fits.",
            )
        )
    if leases and additional < concurrency:
        findings.append(
            _finding(
                "RUNTIME_ADVICE_RESERVATIONS_BLOCKING",
                "Existing runs limit additional launches now. Their frozen runs-per-GPU settings stay unchanged; "
                "apply this suggestion to a new batch after those reservations finish.",
            )
        )
    if not leases_known:
        additional = None
        findings.append(
            _finding(
                "RUNTIME_ADVICE_RESERVATIONS_UNKNOWN",
                "Active reservations could not be read. Additional capacity now is unknown; "
                "the scheduler must check reservations before any launch.",
            )
        )
    findings.append(
        _finding(
            "RUNTIME_ADVICE_RESOURCE_BUDGET",
            f"Reserved {cpu_reserve} host CPU slots and {ram_reserve:g} GiB RAM for other work. "
            f"Each run reserves {threads + 2 * workers} CPU slots (threads + two persistent loader pools) "
            f"and {ram_per_run:g} GiB RAM including feature buffers.",
            "info",
        )
    )
    unique_evidence = {
        (row["batchId"], row["runId"], row["gpuUuid"], row["key"], row["stage"], row["epoch"]): row
        for row in evidence
    }
    return {
        "version": 1,
        "applicable": True,
        "basis": basis,
        "resources": {
            "maxConcurrentRuns": concurrency,
            "gpuIds": usable_gpus,
            "runsPerGpu": runs_per_gpu,
            "cpuThreadsPerRun": threads,
            "dataLoaderWorkers": workers,
            "ramGbPerRun": ram_per_run,
        },
        "summary": (
            f"Start with {concurrency} concurrent run{'s' if concurrency != 1 else ''}, "
            f"{threads} CPU threads per run and {workers} workers per loader. "
            f"Additional runs permitted now: {additional if additional is not None else 'unknown'}."
        ),
        "memory": {
            "perRunGpuGb": round(max(memory_needs), 2) if memory_needs else None,
            "perRunRamGb": ram_per_run,
            "observedRuns": len({(row["batchId"], row["runId"]) for row in evidence}),
        },
        "limits": {
            "cpuConcurrency": cpu_limit,
            "ramConcurrency": ram_limit,
            "gpuConcurrency": gpu_limit,
            "additionalRunsNow": additional,
        },
        "evidence": list(unique_evidence.values()),
        "findings": findings,
    }
