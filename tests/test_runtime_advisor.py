"""Capacity advice stays conservative across scientific recipes and live leases."""

import copy
import json

import pytest

from histopilot.application.runtime_advisor import recommend_runtime
from histopilot.schemas.development import ResourcePolicy


def runtime(**host):
    return {
        "available": True,
        "cudaAvailable": True,
        "gpuCount": 1,
        "host": {
            "cpuCount": 36,
            "physicalCpuCount": 18,
            "totalRamGb": 192.0,
            "availableRamGb": 173.0,
            **host,
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "gpu-a",
                "totalMemoryGb": 24.0,
                "freeMemoryGb": 19.6,
                "usedMemoryGb": 4.4,
            }
        ],
    }


def workload(**changes):
    return {
        "key": "crc-kras-nnmil",
        "model": "nnmil",
        "featureDimension": 1536,
        "trainingPatches": 6233,
        "evaluationPatches": 80000,
        "batchSize": 32,
        "evalBatchSize": 1,
        "precision": "32-true",
        "embedDim": 512,
        "attentionDim": 256,
        "numFcLayers": 1,
        "gradientCheckpointing": False,
        "loadingPolicy": "mmap",
        **changes,
    }


def observation(**changes):
    return {
        "key": "crc-kras-nnmil",
        "gpuUuid": "gpu-a",
        "trainingPatches": 6233,
        "evaluationPatches": 80000,
        "peakReservedGpuGb": 3.39,
        "stage": "fit",
        "epoch": 26,
        "batchId": "batch-one",
        "runId": "fold-one",
        **changes,
    }


def recommend(*, host=None, workloads=None, observations=None, ids=None, count=5, leases=None):
    return recommend_runtime(
        host or runtime(),
        [workload()] if workloads is None else workloads,
        [observation()] if observations is None else observations,
        [0] if ids is None else ids,
        count,
        [] if leases is None else leases,
    )


def codes(result):
    return {row["code"] for row in result["findings"]}


def test_running_crc_kras_profile_proposes_three_without_changing_one_run_lease():
    snapshot = runtime()
    leases = [{"gpu": 0, "cpus": 6, "ramGb": 16, "rssGb": 8, "runsPerGpu": 1}]
    original = copy.deepcopy((snapshot, leases))
    result = recommend(host=snapshot, leases=leases)
    assert result["basis"] == "measured"
    assert result["resources"]["maxConcurrentRuns"] == 3
    assert result["resources"]["runsPerGpu"] == 3
    assert result["resources"]["cpuThreadsPerRun"] == 2
    assert result["resources"]["dataLoaderWorkers"] == 2
    assert result["limits"]["additionalRunsNow"] == 0
    assert result["memory"]["perRunGpuGb"] > 3.39
    assert "RUNTIME_ADVICE_RESERVATIONS_BLOCKING" in codes(result)
    assert (snapshot, leases) == original
    ResourcePolicy.model_validate(result["resources"])


def test_completed_assessment_can_allow_four_but_live_fit_is_capped_at_three():
    snapshot = runtime()
    snapshot["gpus"][0].update(freeMemoryGb=23.7, usedMemoryGb=0.3)
    assert recommend(host=snapshot)["resources"]["runsPerGpu"] == 3
    result = recommend(host=snapshot, observations=[observation(stage="assessment")])
    assert result["resources"]["runsPerGpu"] == 4
    assert result["resources"]["maxConcurrentRuns"] == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"key": "other-cohort"},
        {"gpuUuid": "gpu-b"},
        {"epoch": 0},
        {"peakReservedGpuGb": float("nan")},
        {"peakReservedGpuGb": float("inf")},
        {"peakReservedGpuGb": 0},
        {"peakReservedGpuGb": 25},
        {"stage": "startup"},
        {"trainingPatches": 0},
        {"evaluationPatches": None},
    ],
)
def test_unmatched_or_invalid_evidence_never_unlocks_gpu_sharing(changes):
    result = recommend(observations=[observation(**changes)])
    assert result["applicable"]
    assert result["basis"] == "estimated"
    assert result["resources"]["runsPerGpu"] == 1
    assert result["evidence"] == []
    json.dumps(result, allow_nan=False)


def test_nearby_fold_larger_cap_scales_peak_up_and_large_change_requires_profile():
    original = recommend()
    nearby = recommend(workloads=[workload(trainingPatches=7000)])
    assert nearby["basis"] == "measured"
    assert nearby["memory"]["perRunGpuGb"] > original["memory"]["perRunGpuGb"]
    larger = recommend(workloads=[workload(trainingPatches=15000)])
    assert larger["basis"] == "estimated"
    assert larger["resources"]["runsPerGpu"] == 1


def test_smaller_patch_counts_never_reduce_the_observed_peak():
    smaller = recommend(workloads=[workload(trainingPatches=1000, evaluationPatches=10000)])
    assert smaller["memory"]["perRunGpuGb"] == recommend()["memory"]["perRunGpuGb"]


def test_all_candidate_recipes_require_matching_evidence():
    result = recommend(workloads=[workload(), workload(key="new-architecture", model="abmil")])
    assert result["basis"] == "estimated"
    assert result["resources"]["runsPerGpu"] == 1
    assert result["memory"]["observedRuns"] == 1


def test_same_recipe_all_fold_caps_must_fit_matched_evidence():
    result = recommend(workloads=[workload(), workload(trainingPatches=20000)])
    assert result["basis"] == "estimated"
    assert result["resources"]["runsPerGpu"] == 1


def test_full_bag_evaluation_can_be_the_limiting_gpu_allocation():
    normal = recommend(observations=[])
    huge = recommend(workloads=[workload(evaluationPatches=2000000)], observations=[])
    assert normal["applicable"]
    assert not huge["applicable"]
    assert huge["resources"] is None


def test_unprofiled_small_gpu_rejects_large_training_minibatch():
    snapshot = runtime()
    snapshot["gpus"][0].update(totalMemoryGb=8, freeMemoryGb=7.8, usedMemoryGb=0.2)
    result = recommend(host=snapshot, workloads=[workload(batchSize=128)], observations=[])
    assert not result["applicable"]


def test_mixed_precision_reduces_estimated_activations_but_not_float32_input_storage():
    fp32 = recommend(observations=[])
    amp = recommend(workloads=[workload(precision="bf16-mixed")], observations=[])
    assert fp32["memory"]["perRunGpuGb"] > amp["memory"]["perRunGpuGb"]
    assert amp["memory"]["perRunGpuGb"] > fp32["memory"]["perRunGpuGb"] / 2


def test_heterogeneous_gpu_policy_respects_smallest_and_requires_per_device_profile():
    snapshot = runtime()
    snapshot["gpuCount"] = 2
    snapshot["gpus"].append(
        {
            "index": 1,
            "uuid": "gpu-b",
            "totalMemoryGb": 12.0,
            "freeMemoryGb": 11.8,
            "usedMemoryGb": 0.2,
        }
    )
    unprofiled = recommend(host=snapshot, ids=[0, 1])
    assert unprofiled["basis"] == "estimated"
    assert unprofiled["resources"]["gpuIds"] == [0, 1]
    assert unprofiled["resources"]["runsPerGpu"] == 1
    assert unprofiled["resources"]["maxConcurrentRuns"] == 2
    measured = recommend(
        host=snapshot,
        ids=[0, 1],
        observations=[
            observation(),
            observation(gpuUuid="gpu-b", runId="fold-two"),
        ],
    )
    assert measured["basis"] == "measured"
    assert measured["resources"]["runsPerGpu"] == 1  # 9.8 GiB cannot fit two 4.907-GiB runs.


def test_explicit_cpu_selection_never_switches_to_available_gpu():
    result = recommend(ids=[])
    assert result["resources"]["gpuIds"] == []
    assert result["resources"]["maxConcurrentRuns"] == 1
    assert result["resources"]["dataLoaderWorkers"] == 0
    assert result["memory"]["perRunGpuGb"] is None
    assert result["limits"]["gpuConcurrency"] is None
    assert result["evidence"] == []


def test_missing_one_gpu_telemetry_excludes_that_device_without_cpu_fallback():
    snapshot = runtime()
    snapshot["gpuCount"] = 2
    snapshot["gpus"].append({"index": 1, "totalMemoryGb": 24, "freeMemoryGb": None})
    result = recommend(host=snapshot, ids=[0, 1])
    assert result["resources"]["gpuIds"] == [0]
    assert "RUNTIME_ADVICE_GPU_UNKNOWN" in codes(result)
    unavailable = recommend(host=snapshot, ids=[1])
    assert unavailable["resources"] is None
    assert not unavailable["applicable"]


@pytest.mark.parametrize("free", [None, float("nan"), float("inf"), -1, 25, 0, 1])
def test_invalid_unknown_or_busy_gpu_does_not_receive_a_policy(free):
    snapshot = runtime()
    snapshot["gpus"][0]["freeMemoryGb"] = free
    result = recommend(host=snapshot)
    assert not result["applicable"]
    assert result["resources"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"availableRamGb": 0},
        {"availableRamGb": float("nan")},
        {"availableRamGb": float("inf")},
        {"availableRamGb": 200},
        {"totalRamGb": 0},
        {"cpuCount": 0},
        {"cpuCount": None},
        {"cpuCount": True},
    ],
)
def test_missing_or_insufficient_host_resources_never_round_up_to_one(changes):
    result = recommend(host=runtime(**changes))
    assert not result["applicable"]
    assert result["resources"] is None
    json.dumps(result, allow_nan=False)


def test_small_cpu_host_reduces_workers_and_threads_without_oversubscription():
    result = recommend(host=runtime(cpuCount=1, physicalCpuCount=1))
    assert result["applicable"]
    assert result["resources"]["cpuThreadsPerRun"] == 1
    assert result["resources"]["dataLoaderWorkers"] == 0
    assert result["resources"]["maxConcurrentRuns"] == 1


def test_native_large_bags_increase_ram_and_low_available_ram_reduces_workers():
    packed = recommend(workloads=[workload(loadingPolicy="mmap")])
    native = recommend(workloads=[workload(loadingPolicy="native")])
    assert native["memory"]["perRunRamGb"] > packed["memory"]["perRunRamGb"]
    low = recommend(host=runtime(totalRamGb=32, availableRamGb=14))
    assert low["applicable"]
    assert low["resources"]["dataLoaderWorkers"] < 2
    assert low["resources"]["ramGbPerRun"] <= 12


def test_ram_reservations_subtract_only_nonresident_headroom():
    reservation = {"gpu": None, "cpus": 1, "ramGb": 20, "rssGb": 19, "runsPerGpu": 1}
    mostly_resident = recommend(host=runtime(availableRamGb=30), ids=[], leases=[reservation])
    not_resident = recommend(
        host=runtime(availableRamGb=30),
        ids=[],
        leases=[
            {
                **reservation,
                "rssGb": 0,
            }
        ],
    )
    assert mostly_resident["applicable"]
    assert not not_resident["applicable"]


def test_existing_shared_lease_limits_additional_count_and_duplicate_evidence_count():
    leases = [{"gpu": 0, "cpus": 6, "ramGb": 16, "rssGb": 10, "runsPerGpu": 3}]
    result = recommend(leases=leases, observations=[observation(), observation()])
    assert result["limits"]["additionalRunsNow"] == 2
    assert result["memory"]["observedRuns"] == 1
    assert len(result["evidence"]) == 1


def test_unavailable_runtime_and_invalid_workload_return_no_recommendation():
    snapshot = runtime()
    snapshot["available"] = False
    assert not recommend(host=snapshot)["applicable"]
    assert not recommend(workloads=[workload(model="unknown")])["applicable"]
    assert not recommend(workloads=[workload(featureDimension=0)])["applicable"]
    assert not recommend(count=0)["applicable"]


def test_hardware_only_is_explicit_and_does_not_enable_gpu_sharing():
    result = recommend(workloads=[], observations=[])
    assert result["applicable"]
    assert result["basis"] == "hardware_only"
    assert result["resources"]["runsPerGpu"] == 1
    assert result["memory"]["observedRuns"] == 0


def test_run_count_bounds_recommendation_and_policy_validates():
    result = recommend(count=1)
    assert result["resources"]["maxConcurrentRuns"] == 1
    assert result["resources"]["runsPerGpu"] == 1
    ResourcePolicy.model_validate(result["resources"])


def test_unreadable_lease_registry_does_not_claim_free_scheduler_capacity():
    result = recommend_runtime(runtime(), [workload()], [observation()], [0], 5, None)
    assert result["applicable"]
    assert result["limits"]["additionalRunsNow"] is None
    assert "RUNTIME_ADVICE_RESERVATIONS_UNKNOWN" in codes(result)


def test_native_reader_budgets_source_bag_even_when_both_training_and_eval_are_capped():
    capped = workload(trainingPatches=1024, evaluationPatches=1024, loadingPolicy="native")
    normal = recommend(workloads=[capped], observations=[])
    source = recommend(workloads=[{**capped, "sourcePatches": 1000000}], observations=[])
    assert source["memory"]["perRunRamGb"] > normal["memory"]["perRunRamGb"]


def test_fitting_peak_cannot_certify_an_unseen_oversized_assessment_bag():
    result = recommend(
        workloads=[workload(evaluationPatches=2000000)],
        observations=[observation(evaluationPatches=2000000)],
    )
    assert not result["applicable"]
    assert result["resources"] is None


def test_cpu_ram_includes_large_model_optimizer_and_activations_even_with_tiny_bags():
    large_model = workload(
        key="deep-wide-abmil",
        model="abmil",
        featureDimension=1024,
        embedDim=8192,
        numFcLayers=8,
        attentionDim=384,
        trainingPatches=16,
        evaluationPatches=16,
        batchSize=1,
    )
    cpu = recommend(workloads=[large_model], ids=[], observations=[])
    gpu = recommend(workloads=[large_model], observations=[])
    assert cpu["applicable"] and gpu["applicable"]
    assert gpu["resources"]["ramGbPerRun"] == 8
    assert cpu["resources"]["ramGbPerRun"] > 16
    insufficient = recommend(
        host=runtime(totalRamGb=32, availableRamGb=14),
        workloads=[large_model],
        ids=[],
        observations=[],
    )
    assert not insufficient["applicable"]
    assert insufficient["resources"] is None
