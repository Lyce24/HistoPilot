"""Progress uses outer slide counters and supervisor truth, never raw log success guesses."""

import json
from datetime import UTC, datetime

import pytest

from histopilot.adapters.trident.progress import build_progress

START = "2026-09-10T06:20:18.721038+00:00"
NOW = "2026-09-10T06:23:19.830526+00:00"


def job(state="running", **options):
    return {
        "state": state,
        "slideCount": 138,
        "createdAt": START,
        "spec": {"options": {"task": "all", **options}},
    }


# Reduced real cancelled job frames, including download bars and concatenated exit.
REAL_CANCELLED = (
    f"[{START}] Starting TRIDENT worker\n"
    "[MAIN] Found 138 slides. Processing 138 pending slides (0 skipped).\n"
    "\rFetching 1 files: 100%|██████████| 1/1 [00:00<00:00, 481.88it/s]\n"
    "\rSegmenting tissue: 4%|▎ | 5/138 [02:45<1:16:41, 34.60s/it, Segmenting <name=T102>]"
    "\rSegmenting tissue: 4%|▎ | 5/138 [02:45<1:16:41, 34.60s/it, Segmenting <name=T103>]"
    f"[{NOW}] TRIDENT cancelled (exit -15)\n"
)


def test_real_cancelled_job_keeps_last_slide_counts_and_freezes_elapsed():
    progress = build_progress(job("cancelled"), REAL_CANCELLED, now="2026-09-11T12:00:00Z")
    assert progress["stage"] == "segmentation"
    assert (progress["completed"], progress["total"], progress["percent"]) == (5, 138, 3.62)
    assert progress["currentSlide"] == "T103"
    assert progress["elapsedSeconds"] == pytest.approx(181.109488)
    assert progress["stageElapsedSeconds"] == 165
    assert progress["etaSeconds"] is None
    assert progress["stages"][1]["status"] == "stopped"
    assert progress["stages"][2]["status"] == "pending"


def test_running_slide_progress_ignores_download_and_nested_patch_bars():
    logs = (
        "Extracting patch features from coords in 20x_256px_0px_overlap: 25%|██ | 2/8 "
        "[01:00<03:00, 30.00s/it, Extracting features from slide.1.svs]\n"
        "\rFetching 4 files: 100%|███| 4/4 [00:00<00:00, 800.00it/s]\n"
        "\r 90%|██████| 90/100 [00:09<00:01, 10.00it/s]\n"
    )
    progress = build_progress(job(task="feat"), logs, now=NOW)
    assert progress["stage"] == "patch_features"
    assert (progress["completed"], progress["total"]) == (2, 8)
    assert progress["currentSlide"] == "slide.1.svs"
    assert progress["etaSeconds"] == 180
    assert progress["ratePerSecond"] == pytest.approx(1 / 30)
    assert progress["scope"] == "stage"


@pytest.mark.parametrize(
    "bar,elapsed,eta,rate",
    [
        ("[00:19<00:19, 19.00s/it, Segmenting <name=slide.1>]", 19, 19, 1 / 19),
        ("[1:02:03<2:03:04, 1.25it/s]", 3723, 7384, 1.25),
        ("[00:00<?, ?it/s]", 0, None, None),
    ],
)
def test_progress_time_formats_and_rates(bar, elapsed, eta, rate):
    progress = build_progress(job(task="seg"), "Segmenting tissue: 50%|▍| 1/2 " + bar, now=NOW)
    assert progress["stageElapsedSeconds"] == elapsed
    assert progress["etaSeconds"] == eta
    assert (
        progress["ratePerSecond"] == pytest.approx(rate)
        if rate is not None
        else progress["ratePerSecond"] is None
    )
    assert progress["percent"] == 50


def test_missing_start_and_unrecognized_logs_leave_counts_unknown():
    progress = build_progress(
        {"state": "starting"},
        "Importing dependencies\nFetching 1 files: 100%| | 1/1 [00:00<00:00]",
        now=NOW,
    )
    assert progress["stage"] == "preparing"
    assert progress["completed"] is None
    assert progress["total"] is None
    assert progress["percent"] is None
    assert progress["elapsedSeconds"] is None
    assert progress["etaSeconds"] is None
    assert progress["label"] == "Loading model files"


def test_completed_stage_then_model_download_prepares_next_stage():
    logs = (
        "Saving tissue coordinates to 20x_256px_0px_overlap: 100%|██| 8/8 [00:20<00:00, 2.5s/it]\n"
        "Fetching 4 files: 0%| | 0/4 [00:00<?, ?it/s]\n"
    )
    progress = build_progress(job(), logs, now=NOW)
    assert progress["stage"] == "patch_features"
    assert progress["label"] == "Loading model files"
    assert progress["percent"] is None
    statuses = {stage["id"]: stage["status"] for stage in progress["stages"]}
    assert statuses["coordinates"] == "complete"
    assert statuses["patch_features"] == "active"
    assert statuses["validation"] == "pending"


def test_full_stage_bar_does_not_assert_job_success():
    logs = "Extracting patch features from coords in coords: 100%|█| 8/8 [00:08<00:00, 1it/s]"
    progress = build_progress(job(task="feat"), logs, now=NOW)
    assert progress["stage"] == "patch_features"
    assert progress["label"] != "Extraction complete"
    assert progress["stages"][-1]["status"] == "pending"
    assert progress["etaSeconds"] is None


@pytest.mark.parametrize("options", [{"gpus": [0, 1]}, {"gpus": [-1, -1]}, {"wsi_cache": "/cache"}])
def test_independent_worker_counts_are_never_aggregated(options):
    logs = (
        "Segmenting tissue: 50%| | 2/4 [00:20<00:20, 10s/it]\n"
        "Extracting patch features from coords in coords: 100%| | 3/3 [00:30<00:00, 10s/it]\n"
        "Segmenting tissue: 25%| | 1/4 [00:10<00:30, 10s/it]\n"
    )
    progress = build_progress(job(**options), logs, now=NOW)
    assert progress["scope"] == "batch"
    assert progress["stage"] == "segmentation"
    assert (progress["completed"], progress["total"]) == (1, 4)
    assert progress["percent"] == 25
    assert all(
        stage["status"] != "complete" for stage in progress["stages"] if stage["id"] != "preparing"
    )
    assert any("not overall" in text for text in progress["warnings"])


def test_repeated_positive_gpu_ids_do_not_imply_multiple_workers():
    progress = build_progress(
        job(gpus=[0, 0]), "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]", now=NOW
    )
    assert progress["scope"] == "stage"


def test_new_cache_batch_resets_previous_local_progress():
    logs = (
        "Extracting patch features from coords in coords: 100%| | 4/4 [00:20<00:00, 5s/it]\n"
        "[CONSUMER] Processing batch 2: /cache/batch2\n"
    )
    progress = build_progress(job(wsi_cache="/cache"), logs, now=NOW)
    assert progress["stage"] == "preparing"
    assert progress["percent"] is None
    assert not all(stage["status"] == "complete" for stage in progress["stages"])


def test_slide_encoder_pipeline_includes_automatic_patch_features():
    logs = "Extracting slide features using titan: 50%| | 1/2 [00:04<00:04, 4s/it, Extracting slide features for slide.1.svs]"
    progress = build_progress(job(task="feat", slide_encoder="titan"), logs, now=NOW)
    assert [stage["id"] for stage in progress["stages"]] == [
        "preparing",
        "patch_features",
        "slide_features",
        "validation",
    ]
    assert progress["stage"] == "slide_features"
    assert progress["stages"][1]["status"] == "complete"


def test_validation_phase_has_authoritative_inspection_counts():
    logs = (
        f"[{START}] Starting Artifact validation worker\n"
        "[validation] Inspected 16/138 slides: 14 complete, 2 invalid or missing, 0 unvalidated.\n"
    )
    progress = build_progress(job(), logs, now=NOW)
    assert progress["stage"] == "validation"
    assert (progress["completed"], progress["total"]) == (16, 138)
    assert progress["etaSeconds"] is None
    assert all(stage["status"] == "complete" for stage in progress["stages"][:-1])


def test_success_uses_verified_coverage_instead_of_old_partial_bar():
    document = job("succeeded")
    document["result"] = {"completedSlides": 138, "missingSlides": 0, "finishedAt": NOW}
    progress = build_progress(
        document, "Segmenting tissue: 10%| | 1/10 [00:10<01:30, 10s/it]", now=NOW
    )
    assert progress["stage"] == "validation"
    assert progress["percent"] == 100
    assert progress["completed"] == 138
    assert all(stage["status"] == "complete" for stage in progress["stages"])


def test_failed_validation_reports_usable_coverage_without_eta():
    document = job("failed")
    document["result"] = {"completedSlides": 130, "missingSlides": 8, "finishedAt": NOW}
    progress = build_progress(document, "[validation] Inspected 138/138 slides:", now=NOW)
    assert progress["completed"] == 130
    assert progress["total"] == 138
    assert "validated outputs" in progress["detail"]
    assert progress["etaSeconds"] is None
    assert progress["stages"][-1]["status"] == "stopped"


def test_terminal_coverage_does_not_keep_unrelated_segmentation_rate():
    document = job("failed")
    document["result"] = {"completedSlides": 130, "missingSlides": 8, "finishedAt": NOW}
    progress = build_progress(
        document, "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]", now=NOW
    )
    assert progress["stage"] == "validation"
    assert progress["ratePerSecond"] is None
    assert progress["stageElapsedSeconds"] is None


def test_stale_log_eta_is_suppressed_but_elapsed_keeps_advancing():
    document = job()
    document["_logUpdatedAt"] = START
    progress = build_progress(
        document, "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]", now=NOW
    )
    assert progress["elapsedSeconds"] == pytest.approx(181.109488)
    assert progress["etaSeconds"] is None
    assert any("fresh progress" in text for text in progress["warnings"])


def test_long_slides_keep_eta_for_two_estimated_iterations():
    document = job()
    document["_logUpdatedAt"] = START
    progress = build_progress(
        document, "Segmenting tissue: 50%| | 1/2 [10:00<10:00, 600s/it]", now=NOW
    )
    assert progress["etaSeconds"] == 600


def test_interrupted_job_freezes_elapsed_at_last_log_not_stale_job_metadata():
    document = job("interrupted")
    document["updatedAt"] = START
    document["_logUpdatedAt"] = NOW
    progress = build_progress(
        document, "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]", now="2026-09-12T00:00:00Z"
    )
    assert progress["elapsedSeconds"] == pytest.approx(181.109488)
    assert progress["etaSeconds"] is None


def test_cancelled_legacy_job_without_completion_stamp_uses_last_log_time():
    document = job("cancelled")
    document["updatedAt"] = START
    document["_logUpdatedAt"] = NOW
    progress = build_progress(
        document,
        "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]",
        now="2026-09-12T00:00:00Z",
    )
    assert progress["elapsedSeconds"] == pytest.approx(181.109488)


def test_cancelling_keeps_running_elapsed_but_clears_eta():
    progress = build_progress(
        job("cancelling"), "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it]", now=NOW
    )
    assert progress["label"] == "Stopping extraction"
    assert progress["etaSeconds"] is None
    assert progress["elapsedSeconds"] > 180


def test_log_tail_is_bounded_and_ansi_cleared():
    old = "Extracting slide features using titan: 50%| | 1/2 [00:10<00:10]\n" + "x" * 70000
    logs = (
        old
        + "\n\x1b[32mSegmenting tissue: 50%| | 1/2 [00:10<00:10, 10s/it, Segmenting <name=latest>]\x1b[0m\x1b[A"
    )
    progress = build_progress(job(), logs, now=datetime(2026, 9, 10, 6, 23, tzinfo=UTC))
    assert progress["stage"] == "segmentation"
    assert progress["currentSlide"] == "latest"
    json.dumps(progress, allow_nan=False)


@pytest.mark.parametrize("counter", ["3/2", "0/0", "?/100", "9" * 5000 + "/2"])
def test_malformed_counters_cannot_produce_invalid_percent_or_crash(counter):
    progress = build_progress(
        job(), "Segmenting tissue: 50%| | " + counter + " [00:10<00:10, ?it/s]", now=NOW
    )
    assert progress["percent"] is None
    json.dumps(progress, allow_nan=False)


def test_malformed_giant_batch_index_and_overflowing_inverse_rate_are_safe():
    logs = (
        "[CONSUMER] Processing batch " + "9" * 5000 + ": /cache\n"
        "Segmenting tissue: 50%| | 1/2 [00:10<00:10, 0." + "0" * 320 + "1s/it]"
    )
    progress = build_progress(job(), logs, now=NOW)
    assert progress["ratePerSecond"] is None
    assert progress["completed"] == 1
    json.dumps(progress, allow_nan=False)
