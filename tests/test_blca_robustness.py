"""The real-study helper isolates invalid mutations and checkpoints honest evidence."""

import json
from contextlib import ExitStack

import h5py
import numpy as np
import pytest

from histopilot.application.evaluations import EvaluationService
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.application.imports import ImportService
from histopilot.application.protocols import ProtocolService
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job
from scripts import blca_robustness as helper


@pytest.fixture
def study(tmp_path):
    project = tmp_path / "blca-e2e-20260917"
    project.mkdir()
    store = ScientificStore(project, "project-test-blca-robustness")
    filesystem = LocalFilesystem((tmp_path,))
    slides = tmp_path / "slides"
    slides.mkdir()
    source = tmp_path / "metadata.csv"
    lines = ["slide,label,text"]
    for index in range(20):
        (slides / f"test-{index}.tiff").write_bytes(b"Nonempty metadata-only import fixture")
        lines.append(f"test-{index},{index % 2},{'high' if index % 2 else 'low'}")
    source.write_text("\n".join(lines) + "\n")
    spec = {"source": {"path": str(source)}, "slideIdColumn": "slide",
            "slideRoot": str(slides), "patientIdFallback": "slide_id",
            "attributes": [{"key": "who2022_binary", "sourceColumn": "label", "type": "categorical"},
                           {"key": "who2022_text", "sourceColumn": "text", "type": "categorical"}]}
    imports = ImportService(store, filesystem)
    draft = store.create_draft("import", "Test metadata", {"type": "dataset-import", "spec": spec})
    preview = imports.preview(draft["id"], 1)
    dataset = imports.freeze(draft["id"], 1, preview["previewHash"], "test-dataset")
    protocol = {"datasetId": dataset["id"], "target": {
        "field": "who2022_binary", "task": "binary_classification", "unit": "slide",
        "classes": ["low", "high"], "labels": {"0": "low", "1": "high"}, "positiveClass": "high"},
        "split": {"version": 4, "mode": "kfold", "folds": 2, "seeds": [42], "validationFraction": .2,
                  "pools": {"trainSelection": "remaining"}}}
    protocols = ProtocolService(store, filesystem)
    protocol_draft = store.create_draft("experiment", "Test design", {"type": "analysis-protocol", "spec": protocol})
    preview = protocols.preview(protocol_draft["id"], 1)
    assert preview["canFreeze"]
    frozen_protocol = protocols.freeze(protocol_draft["id"], 1, preview["previewHash"], "test-protocol")
    features = tmp_path / "features"
    features.mkdir()
    with h5py.File(features / "test-0.h5", "w") as handle:
        feature = handle.create_dataset("features", data=np.arange(48, dtype="float32").reshape(12, 4))
        feature.attrs["encoder"] = "uni_v1"
        coords = handle.create_dataset("coords", data=np.arange(24, dtype="int64").reshape(12, 2))
        coords.attrs["patch_size_level0"] = 512
    service = FeatureService(store, filesystem)
    feature_spec = FeatureSpec(path=str(features), encoderId="uni_v1")
    feature = service.freeze(feature_spec, service.preview(feature_spec)["previewHash"], "test-features")
    state = {"projectPath": str(project), "projectId": store.project_id, "workspace": str(tmp_path),
             "datasetId": dataset["id"], "importDraftId": draft["id"], "importSpec": spec,
             "metadataPath": str(source), "slideRoot": str(slides), "featurePath": str(features),
             "protocolId": frozen_protocol["id"], "featureId": feature["id"]}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    return path, store, source, features


def test_real_import_and_target_previews_do_not_change_original_records(study):
    path, store, source, _ = study
    state_before, csv_before = path.read_bytes(), source.read_bytes()
    drafts_before = store.list_drafts()
    datasets_before = store.list_datasets()
    assert helper.run_checks(path, "dataset")["passed"] == 5
    assert helper.run_checks(path, "target")["passed"] == 2
    assert path.read_bytes() == state_before
    assert source.read_bytes() == csv_before
    assert store.list_drafts() == drafts_before
    assert store.list_datasets() == datasets_before
    assert not list((path.parent / "robustness").glob(".dataset-*"))


def test_copy_mutation_invalidates_actual_worker_receipt_without_changing_source(study):
    path, store, _, features = study
    before = (features / "test-0.h5").read_bytes()
    configurations_before = store.list_configurations()
    with ExitStack() as stack:
        context = helper.Checks(path, "features", stack)
        assert context.features_wrong_encoder()["blockingCodes"]
        evidence = context.features_mutated_receipt()
    assert evidence["nativeWorkerSucceededBeforeMutation"] and evidence["receiptInvalidated"]
    assert (features / "test-0.h5").read_bytes() == before
    assert store.list_configurations() == configurations_before
    assert not (store.folder / "packing").exists()


def test_attention_checks_reject_geometry_dtype_alignment_and_source_escape(study, monkeypatch):
    path, _, _, features = study
    selection = {"slideId": "test-0", "slidePath": str(path.parent / "slides/test-0.tiff"),
                 "featurePath": str(features / "test-0.h5"), "featureKey": "features",
                 "coordinatesKey": "coords", "confirmRowAlignment": True}
    geometry = {"width": 1024, "height": 1024, "levelDownsamples": [1.0]}
    contract = {"dimensions": 4, "dtype": "float32", "encoderId": "uni_v1"}
    with ExitStack() as stack:
        context = helper.Checks(path, "attention", stack)
        monkeypatch.setattr(context, "attention_inputs", lambda: helper.copy.deepcopy(
            (selection, geometry, contract, "configuration-" + "a" * 64)))
        for check in (context.attention_geometry, context.attention_confirmation,
                      context.attention_wrong_dtype, context.attention_coordinates,
                      lambda: context.attention_coordinates(True), context.attention_source_boundary):
            assert check()["errorType"]


def test_dtype_check_uses_two_authenticated_bundles_and_exact_held_out_membership(study):
    path, store, _, source = study
    state = json.loads(path.read_text())
    filesystem = LocalFilesystem((path.parent,))
    for index in range(1, 20):
        (source / f"test-{index}.h5").write_bytes((source / "test-0.h5").read_bytes())
    service = FeatureService(store, filesystem)
    spec = FeatureSpec(path=str(source), encoderId="uni_v1")
    feature = service.freeze(spec, service.preview(spec)["previewHash"], "complete-features")
    packs = FeaturePackService(store, filesystem, helper.InlineValidationExecutor())
    request = FeaturePackSpec(featureSetId=feature["id"], action="validate")
    job = packs.submit(request, packs.preview(request)["previewHash"], "complete-validation")
    assert run_job(packs.folder / job["id"] / "plan.json")["state"] == "succeeded"
    bundles = FeatureBundleService(store, filesystem)
    request = FeatureBundleSpec(featureSetId=feature["id"])
    bundle = bundles.freeze(request, bundles.preview(request)["previewHash"], "complete-bundle")
    protocol = store.get_configuration(state["protocolId"])["manifest"]["spec"]
    protocol["eligibility"] = [{"field": "slideId", "op": "in", "value": [f"test-{index}" for index in range(16)]}]
    draft = store.create_draft("experiment", "Subset development", {"type": "analysis-protocol", "spec": protocol})
    protocols = ProtocolService(store, filesystem)
    reviewed = protocols.preview(draft["id"], 1)
    development = protocols.freeze(draft["id"], 1, reviewed["previewHash"], "subset-development")
    test_spec = {"datasetId": state["datasetId"], "target": protocol["target"],
                 "featureBundleId": bundle["id"], "eligibility": [{"field": "slideId", "op": "in",
                    "value": [f"test-{index}" for index in range(16, 20)]}]}
    draft = store.create_draft("experiment", "Held out", {"type": "evaluation-cohort", "spec": test_spec})
    cohorts = EvaluationService(store, filesystem)
    reviewed = cohorts.preview(draft["id"], 1)
    cohort = cohorts.freeze(draft["id"], 1, reviewed["previewHash"], "held-out-cohort")
    state.update(featureId=feature["id"], bundleId=bundle["id"], protocolId=development["id"], cohortId=cohort["id"])
    path.write_text(json.dumps(state))
    original_configurations = store.list_configurations()
    with ExitStack() as stack:
        context = helper.Checks(path, "evaluation", stack)
        result = context.evaluation_distinct_dtype()
    assert result["blockingCodes"] == ["FEATURE_DTYPE_MISMATCH"]
    assert result["bothBundlesAuthenticated"] and result["heldOutSlidesChecked"] == 1
    assert store.list_configurations() == original_configurations


def test_cached_pass_is_not_repeated_but_missing_dependency_is_retried(study, monkeypatch):
    path, _, _, _ = study
    counts = {"pass": 0, "gap": 0}

    def passed():
        counts["pass"] += 1
        return {"sample": 3}

    def gap():
        counts["gap"] += 1
        raise helper.Unavailable("Awaiting completed predictor")

    monkeypatch.setattr(helper.Checks, "checks", lambda _: {"pass": passed, "gap": gap})
    assert helper.run_checks(path, "experiments")["skipped"] == 1
    assert helper.run_checks(path, "experiments")["skipped"] == 1
    assert counts == {"pass": 1, "gap": 2}
    state = json.loads(path.read_text())
    state["batchId"] = "new-binding"
    path.write_text(json.dumps(state))
    helper.run_checks(path, "experiments")
    assert counts == {"pass": 2, "gap": 3}


def test_failure_is_persisted_and_private_exception_content_is_not_exposed(study, monkeypatch):
    path, _, _, _ = study

    def failed():
        raise ValueError("private clinical row must not escape")

    monkeypatch.setattr(helper.Checks, "checks", lambda _: {"deliberate_failure": failed})
    with pytest.raises(RuntimeError, match="deliberate_failure"):
        helper.run_checks(path, "dataset")
    content = (path.parent / "robustness/dataset.json").read_text()
    assert "private clinical" not in content
    assert json.loads(content)["summary"] == {"passed": 0, "failed": 1, "skipped": 0}


def test_unknown_stage_and_existing_unrelated_project_are_refused(study):
    path, _, _, _ = study
    with pytest.raises(ValueError, match="Unknown robustness stage"):
        helper.run_checks(path, "unsupported")
    state = json.loads(path.read_text())
    state["projectPath"] = str(path.parent)
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="restricted to the new BLCA project"):
        helper.run_checks(path, "dataset")
