"""Prespecified case sampling and durable registration for the real-run helper."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_interpretation_gallery import gallery, study

__all__ = ["gallery", "study"]

spec = importlib.util.spec_from_file_location(
    "blca_final_analysis", Path(__file__).resolve().parents[1] / "scripts/blca_final_analysis.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def case(identity, label, probability):
    predicted = int(probability >= 0.5)
    outcome = ("correct" if predicted == label else
               "false_positive" if predicted else "false_negative")
    return {"id": identity, "label": ["low", "high"][label], "labelIndex": label,
            "probabilities": [1 - probability, probability], "outcome": outcome,
            "slides": [{"slidePath": f"/slides/{identity}.tiff"}]}


def test_prespecified_review_is_deterministic_distinct_and_balances_borderline_classes():
    rows = [case("fp", 0, .99), case("fn", 1, .01), case("tp", 1, .98), case("tn", 0, .02),
            case("positive-a", 1, .51), case("positive-b", 1, .49),
            case("negative-a", 0, .52), case("negative-b", 0, .48),
            case("extra-pos", 1, .8), case("extra-neg", 0, .2)]
    original = copy.deepcopy(rows)
    selected = helper.select_attention(rows, positive_index=1)
    assert helper.select_attention(list(reversed(rows)), positive_index=1) == selected
    assert len(selected["items"]) == len({row["slideId"] for row in selected["items"]}) == 8
    assert [row["slideId"] for row in selected["items"][:4]] == ["fp", "fn", "tp", "tn"]
    assert selected["bucketCounts"]["near_threshold_positive"] == 2
    assert selected["bucketCounts"]["near_threshold_negative"] == 2
    assert {row["slideId"] for row in selected["items"][4:]} == {
        "positive-a", "positive-b", "negative-a", "negative-b",
    }
    assert rows == original


def test_empty_error_buckets_are_not_filled_with_fabricated_errors():
    rows = [case(f"n{i}", 0, .1 + i * .05) for i in range(4)]
    rows += [case(f"p{i}", 1, .7 + i * .05) for i in range(4)]
    selected = helper.select_attention(rows, positive_index=1)
    assert len(selected["items"]) == 6
    assert selected["bucketCounts"]["confident_false_positive"] == 0
    assert selected["bucketCounts"]["confident_false_negative"] == 0
    assert all(row["outcome"] == "correct" for row in selected["items"])


def test_positive_class_order_is_not_assumed():
    rows = [case("fp", 0, .99), case("fn", 1, .01), case("tp", 1, .98), case("tn", 0, .02)]
    original = helper.select_attention(rows, positive_index=1)
    for row in rows:
        row["labelIndex"] = 1 - row["labelIndex"]
        row["probabilities"].reverse()
    assert helper.select_attention(rows, positive_index=0) == original


@pytest.mark.parametrize("change", ["duplicate", "threshold", "missing_image"])
def test_invalid_review_evidence_is_rejected(change):
    rows = [case("fp", 0, .99), case("fn", 1, .01)]
    if change == "duplicate":
        rows.append(rows[0])
    if change == "missing_image":
        rows[0]["slides"][0]["slidePath"] = None
    with pytest.raises(ValueError):
        helper.select_attention(rows, positive_index=1, threshold=.3 if change == "threshold" else .5)


def test_policy_registration_is_idempotent_preserves_state_and_rejects_tampering(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"projectId": "real-project", "otherProgress": 17}))
    registered = helper.prepare(path)
    policy_path = tmp_path / "final-analysis-plan.json"
    original = policy_path.read_bytes()
    assert helper.prepare(path) == registered
    assert policy_path.read_bytes() == original
    assert json.loads(path.read_text())["otherProgress"] == 17
    assert registered["evaluationIdsAtRegistration"] == {}
    changed = copy.deepcopy(registered)
    changed["policy"]["decisionThreshold"] = .3
    policy_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="policy changed"):
        helper.prepare(path)


def test_final_stage_waits_without_creating_project_or_starting_jobs(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"projectId": "real-project"}))
    monkeypatch.setattr(helper, "_services", lambda _: pytest.fail("must not open project yet"))
    result = helper.finalize(path)
    assert result["finalAnalysisStatus"] == "waiting_for_evaluations"
    assert not result.get("interpretationIds")


def test_services_allow_recorded_trident_geometry_and_segmentation_configs(tmp_path):
    from histopilot.storage.filesystem import FilesystemError

    project = tmp_path / "project"
    project.mkdir()
    (project / "histopilot-state.sqlite").touch()
    slides = tmp_path / "slides"
    slides.mkdir()
    features = tmp_path / "features" / "20x" / "features_uni_v1"
    features.mkdir(parents=True)
    config = features.parent / "_config_coords.json"
    config.write_text('{"patch_size":256,"level0_magnification":40,"target_magnification":20}')
    segmentation = features.parent.parent / "_config_segmentation.json"
    segmentation.write_text('{"segmentation_model":"recorded-model"}')
    _, filesystem = helper._services({"projectPath": str(project), "projectId": "test-project",
                                      "slideRoot": str(slides), "featurePath": str(features)})
    assert filesystem.directory(str(config.parent)) == features.parent
    assert filesystem.directory(str(segmentation.parent)) == features.parent.parent
    assert filesystem.directory(str(features)) == features
    with pytest.raises(FilesystemError):
        filesystem.directory(str(tmp_path))


def test_attention_batch_freshly_binds_each_request_and_resumes_one_saved_job(gallery, tmp_path, monkeypatch):
    service, source, executor, _ = gallery
    original_preview, original_save = service.preview, service.save
    calls = []

    def preview(request, **kwargs):
        assert kwargs.get("gallery_context") is None, "Preview needs its own native inspection deadline"
        calls.append("preview")
        return original_preview(request, **kwargs)

    def save(request, **kwargs):
        assert kwargs.get("gallery_context") is None, "Save must not inherit the preview's elapsed deadline"
        calls.append("save")
        return original_save(request, **kwargs)

    monkeypatch.setattr(service, "preview", preview)
    monkeypatch.setattr(service, "save", save)
    evaluation = service.store.publish_configuration(
        manifest={"kind": "model-evaluation", "predictorId": source["predictorId"],
                  "datasetId": service.predictors.get(source["predictorId"])["manifest"]["datasetId"]},
        operation_id="helper-test-evaluation",
    )
    state = {"bundleId": source["featureBundleId"], "slideRoot": source["slideFolder"],
             "predictorIds": {"ensemble": source["predictorId"]},
             "evaluationIds": {"ensemble": evaluation["id"]}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    selection = {"items": [{"slideId": identity,
                            "slidePath": str(Path(source["slideFolder"]) / f"{identity}.png")}
                           for identity in ("001", "Tumour-02", "病例 03")]}
    saved, status = helper._advance_attention(service, state, path, selection, None, launch=False)
    assert status == "attention_prepared"
    assert len(saved["manifest"]["slides"]) == 3
    assert all(row["patchWidthLevel0"] == 100 for row in saved["manifest"]["slides"])
    assert all(row["alignment"] == "embedded_verified" for row in saved["manifest"]["slides"])
    restored = helper.read_state(path)
    assert set(restored["interpretationIds"].values()) == {saved["id"]}
    again, status = helper._advance_attention(service, restored, path, selection, None, launch=False)
    assert again["id"] == saved["id"] and status == "attention_prepared"
    assert len(service.store.list_configurations("model-interpretation")) == 1
    assert calls == ["preview", "save"]
    assert not executor.calls
