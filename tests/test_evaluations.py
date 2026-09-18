"""Later evaluation setup verifies exact cohort membership and feature representations."""

import copy
import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api import create_app
from histopilot.application.evaluations import EvaluationService
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.config import Settings
from histopilot.schemas.evaluations import EvaluationSpec, InferenceSettings
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job


class FakeExecutor:
    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, session, runner, plan):
        pass


TARGET = {
    "field": "label",
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["low", "high"],
    "labels": {"0": "low", "1": "high"},
    "positiveClass": "high",
}


def dataset(store, operation="dataset", rows=None, inventory=None):
    if rows is None:
        rows = [
            {
                "slideId": f"s{i}",
                "patientId": f"p{i}",
                "attributes": {
                    "label": str(i % 2),
                    "cohort": "development" if i < 2 else "test",
                },
            }
            for i in range(4)
        ]
    draft = store.create_draft("import", operation, {})
    document = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        operation_id=operation,
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                for key in ("label", "cohort")
            ],
        },
        artifacts={
            "records.json": json.dumps(rows).encode(),
            **({"inventory.json": json.dumps(inventory).encode()} if inventory is not None else {}),
        },
    )
    return document, rows


def bundle(
    store,
    root,
    data,
    ids,
    name="features",
    dimension=4,
    encoder="uni_v1",
    pack=False,
    dtype="float32",
    feature_kind="patch",
):
    source = root / name
    source.mkdir()
    for identity in ids:
        with h5py.File(source / f"{identity}.h5", "w") as handle:
            handle.create_dataset("features", data=np.ones((1 if feature_kind == "slide" else 3, dimension), dtype=dtype))
            if feature_kind == "patch":
                handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
    filesystem = LocalFilesystem((root,))
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(datasetId=data["id"], path=str(source), encoderId=encoder, featureKind=feature_kind)
    frozen = features.freeze(spec, features.preview(spec)["previewHash"], name)
    packs = FeaturePackService(store, filesystem, FakeExecutor())
    packing = FeaturePackSpec(featureSetId=frozen["id"], action="pack" if pack else "validate")
    job = packs.submit(packing, packs.preview(packing)["previewHash"], name + "-validation")
    result = run_job(packs.folder / job["id"] / "plan.json")
    assert result["state"] == "succeeded", result
    bundle_service = FeatureBundleService(store, filesystem)
    pack_id = result["artifact"]["id"] if pack else None
    spec = FeatureBundleSpec(
        featureSetId=frozen["id"], packArtifactIds=[pack_id] if pack_id else []
    )
    preview = bundle_service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    return bundle_service.freeze(spec, preview["previewHash"], name + "-bundle"), pack_id, source


def setup(store, root, pack=False):
    data, rows = dataset(store)
    features, pack_id, source = bundle(
        store, root, data, [row["slideId"] for row in rows], pack=pack
    )
    protocol = store.publish_configuration(
        manifest={
            "kind": "protocol",
            "datasetId": data["id"],
            "spec": {"target": TARGET, "split": {"version": 4}},
            "memberships": [
                {"slideId": row["slideId"], "patientId": row["patientId"], "partition": "train"}
                for row in rows[:2]
            ],
        },
        operation_id="protocol",
    )
    spec = {
        "protocolId": protocol["id"],
        "developmentFeatureBundleId": features["id"],
        "datasetId": data["id"],
        "featureBundleId": features["id"],
        "target": TARGET,
        "eligibility": [{"field": "cohort", "op": "eq", "value": "test"}],
    }
    if pack:
        spec["inference"] = {"loadingPolicy": "packed", "packArtifactId": pack_id}
    return EvaluationService(store, LocalFilesystem((root,))), spec, source


@pytest.fixture
def evaluation(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    return setup(ScientificStore(folder, "project-evaluation"), tmp_path)


def draft(service, spec, name="Later cohort"):
    return service.store.create_draft(
        "experiment", name, {"type": "evaluation-cohort", "spec": spec}
    )


def preview(service, spec):
    item = draft(service, spec)
    return service.preview(item["id"], 1)


def codes(result):
    return {item["code"] for item in result["findings"] if item["severity"] == "error"}


@pytest.mark.parametrize("changed", ["identity", "content"])
def test_development_bundle_must_match_protocol_binding(evaluation, monkeypatch, changed):
    service, spec, _source = evaluation
    get_configuration = service.store.get_configuration

    def protocol_with_binding(identity):
        document = get_configuration(identity)
        if identity == spec["protocolId"]:
            document = copy.deepcopy(document)
            document["manifest"]["spec"]["featureBundleId"] = (
                "configuration-" + "d" * 64
                if changed == "identity"
                else spec["developmentFeatureBundleId"]
            )
            if changed == "content":
                document["manifest"]["featureBundle"] = {"contentHash": "0" * 64}
        return document

    monkeypatch.setattr(service.store, "get_configuration", protocol_with_binding)
    result = preview(service, spec)
    assert not result["canFreeze"]
    assert ("PROTOCOL_BUNDLE_MISMATCH" if changed == "identity" else "PROTOCOL_BUNDLE_CHANGED") in codes(result)


def test_combined_file_filter_freeze_reopen_and_exact_idempotency(evaluation):
    service, spec, _source = evaluation
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    assert result["canFreeze"], result["findings"]
    assert result["coverage"]["selectedSlideIds"] == ["s2", "s3"]
    assert result["summary"]["includedPatients"] == 2
    assert result["summary"]["labeledSlides"] == 2
    assert not result["executionEnabled"]
    intent = (item["id"], 1, result["previewHash"], "evaluation-freeze")
    frozen = service.freeze(*intent, version_label={"tag": "External cohort"})
    assert frozen["manifest"]["kind"] == "evaluation-cohort"
    assert "split" not in frozen["manifest"]["spec"]
    assert service.freeze(*intent, version_label={"tag": "External cohort"}) == frozen
    assert service.get(frozen["id"])["current"]
    assert len(service.list()["items"]) == 1
    assert service.store.get_draft(item["id"])["status"] == "frozen"
    with pytest.raises(StorageError, match="different request"):
        service.freeze(*intent, version_label={"tag": "Different tag"})


def test_separate_file_uses_existing_combined_feature_bundle_by_exact_slide_id(evaluation):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "test-import",
        [
            {"slideId": "s2", "patientId": "p2", "attributes": {"label": "0", "cohort": "test"}},
            {"slideId": "s3", "patientId": "p3", "attributes": {"label": "1", "cohort": "test"}},
        ],
    )
    spec["datasetId"] = data["id"]
    result = preview(service, spec)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 2


def test_missing_exact_slide_ids_block_even_for_compatible_feature_bundle(evaluation):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "new-slides",
        [
            {
                "slideId": "s002",
                "patientId": "p002",
                "attributes": {"label": "0", "cohort": "test"},
            },
        ],
    )
    spec["datasetId"] = data["id"]
    result = preview(service, spec)
    assert "MISSING_TEST_FEATURES" in codes(result)
    assert result["coverage"]["missingFeatureSlideIds"] == ["s002"]


def test_separate_dataset_and_compatible_test_features_can_freeze(evaluation, tmp_path):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "separate-source",
        [
            {"slideId": "t1", "patientId": "q1", "attributes": {"label": "0", "cohort": "test"}},
            {"slideId": "t2", "patientId": "q2", "attributes": {"label": "1", "cohort": "test"}},
        ],
    )
    features, _pack_id, _source = bundle(
        service.store, tmp_path, data, ["t1", "t2"], name="external-features"
    )
    spec.update(datasetId=data["id"], featureBundleId=features["id"])
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    assert result["canFreeze"], result["findings"]
    frozen = service.freeze(item["id"], 1, result["previewHash"], "external-freeze")
    assert frozen["manifest"]["coverage"]["selectedSlideIds"] == ["t1", "t2"]
    assert result["compatibility"]["development"] == result["compatibility"]["evaluation"]


def test_mapped_missing_label_policy_is_explicit(evaluation):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "missing-test-label",
        [
            {"slideId": "s2", "patientId": "p2", "attributes": {"label": "0", "cohort": "test"}},
            {"slideId": "s3", "patientId": "p3", "attributes": {"label": None, "cohort": "test"}},
        ],
    )
    spec["datasetId"] = data["id"]
    result = preview(service, spec)
    assert "MISSING_TARGET_LABEL" in codes(result)
    spec["target"] = {**TARGET, "missing": "exclude"}
    result = preview(service, spec)
    assert result["canFreeze"], result["findings"]
    assert result["coverage"]["selectedSlideIds"] == ["s2"]
    assert result["summary"]["labeledSlides"] == 1
    assert result["summary"]["excludedSlides"] == 1


def test_shared_patient_overlap_and_independent_namespace(evaluation):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "patient-collision",
        [
            {"slideId": "s2", "patientId": "p0", "attributes": {"label": "0", "cohort": "test"}},
        ],
    )
    spec["datasetId"] = data["id"]
    result = preview(service, spec)
    assert "DEVELOPMENT_PATIENT_OVERLAP" in codes(result)
    spec["patientIdentifiers"] = "independent"
    result = preview(service, spec)
    assert result["canFreeze"]
    assert not result["overlap"]["patientsComparable"]
    assert any(item["code"] == "PATIENT_OVERLAP_UNVERIFIABLE" for item in result["findings"])


def test_slide_overlap_cannot_be_bypassed_with_patient_namespace(evaluation):
    service, spec, _source = evaluation
    spec["eligibility"] = []
    spec["patientIdentifiers"] = "independent"
    result = preview(service, spec)
    assert {"DEVELOPMENT_SLIDE_OVERLAP", "SHARED_PATIENT_NAMESPACE"} <= codes(result)


@pytest.mark.parametrize("alias", ["canonical_path", "hardlink", "independent", "unused"])
def test_cross_import_slide_source_aliases_cannot_bypass_development_overlap(evaluation, alias):
    service, spec, _source = evaluation
    development_rows = [
        {
            "slideId": "s0",
            "patientId": "p0",
            "slidePath": "/slides/development.svs",
            "attributes": {"label": "0"},
        },
        {
            "slideId": "s1",
            "patientId": "p1",
            "slidePath": "/slides/unused.svs",
            "attributes": {"label": "1"},
        },
    ]
    stamp = {"device": 1, "inode": 100, "sizeBytes": 4096, "mtimeNs": 1234}
    development_data, _ = dataset(
        service.store,
        "source-development",
        development_rows,
        [{"path": development_rows[0]["slidePath"], **stamp}],
    )
    protocol = service.store.publish_configuration(
        manifest={
            "kind": "protocol",
            "datasetId": development_data["id"],
            "spec": {"target": TARGET, "split": {"version": 4}},
            "memberships": [{"slideId": "s0", "patientId": "p0", "partition": "train"}],
        },
        operation_id="source-protocol",
    )
    evaluation_path = {
        "canonical_path": "/slides/development.svs",
        "unused": "/slides/unused.svs",
    }.get(alias, "/external/renamed.svs")
    test_rows = [
        {
            "slideId": "s2",
            "patientId": "p2",
            "slidePath": evaluation_path,
            "attributes": {"label": "0", "cohort": "test"},
        }
    ]
    test_stamp = stamp if alias == "hardlink" else {**stamp, "inode": 101}
    test_data, _ = dataset(
        service.store,
        "source-evaluation",
        test_rows,
        [{"path": evaluation_path, **test_stamp}],
    )
    spec.update(
        protocolId=protocol["id"], datasetId=test_data["id"], patientIdentifiers="independent"
    )
    result = preview(service, spec)
    if alias in {"canonical_path", "hardlink"}:
        assert not result["canFreeze"]
        assert "DEVELOPMENT_SLIDE_SOURCE_OVERLAP" in codes(result)
        assert result["overlap"]["sourceSlideIds"] == ["s2"]
    else:
        assert result["canFreeze"], result["findings"]
        assert "sourceSlideIds" not in result["overlap"]


@pytest.mark.parametrize(
    "change", [{"classes": ["high", "low"]}, {"positiveClass": "low"}, {"unit": "slide"}]
)
def test_target_semantics_are_inherited(evaluation, change):
    service, spec, _source = evaluation
    spec["target"] = {**TARGET, **change}
    assert "TARGET_CONTRACT_MISMATCH" in codes(preview(service, spec))


def test_same_dataset_cannot_invert_development_label_mapping(evaluation):
    service, spec, _source = evaluation
    spec["target"] = {**TARGET, "labels": {"0": "high", "1": "low"}}
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    assert "TARGET_LABEL_MAPPING_MISMATCH" in codes(result)
    assert not result["canFreeze"]
    with pytest.raises(StorageError) as error:
        service.freeze(item["id"], 1, result["previewHash"], "inverted-cohort")
    assert error.value.code == "EVALUATION_PREFLIGHT_BLOCKED"


@pytest.mark.parametrize("overlapping_raw_labels", [False, True])
def test_external_dataset_label_encoding_is_allowed_and_remapping_is_visible(
    evaluation, overlapping_raw_labels
):
    service, spec, _source = evaluation
    labels = {"0": "high", "1": "low"} if overlapping_raw_labels else {"H": "high", "L": "low"}
    data, _rows = dataset(
        service.store,
        "external-encoding",
        [
            {"slideId": slide, "patientId": patient, "attributes": {"label": raw, "cohort": "test"}}
            for slide, patient, raw in zip(("s2", "s3"), ("p2", "p3"), labels, strict=True)
        ],
    )
    spec.update(datasetId=data["id"], target={**TARGET, "labels": labels})
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    assert result["canFreeze"], result["findings"]
    warnings = {item["code"] for item in result["findings"] if item["severity"] == "warning"}
    assert ("TARGET_LABEL_MAPPING_REMAPPED" in warnings) == overlapping_raw_labels
    frozen = service.freeze(item["id"], 1, result["previewHash"], "external-labels")
    assert [row["label"] for row in frozen["manifest"]["memberships"]] == ["high", "low"]
    assert service.get(frozen["id"])["current"]


@pytest.mark.parametrize(
    "mapped_empty,unmapped", [(True, "block"), (False, "block"), (False, "exclude")]
)
def test_empty_label_uses_explicit_mapping_or_unmapped_policy(evaluation, mapped_empty, unmapped):
    service, spec, _source = evaluation
    data, _rows = dataset(
        service.store,
        "empty-test-label",
        [
            {"slideId": "s2", "patientId": "p2", "attributes": {"label": "", "cohort": "test"}},
            {"slideId": "s3", "patientId": "p3", "attributes": {"label": "1", "cohort": "test"}},
        ],
    )
    spec.update(
        datasetId=data["id"],
        target={
            **TARGET,
            "labels": {"": "low", "1": "high"} if mapped_empty else TARGET["labels"],
            "missing": "exclude",
            "unmapped": unmapped,
        },
    )
    result = preview(service, spec)
    assert "MISSING_TARGET_LABEL" not in codes(result)
    assert ("UNMAPPED_TARGET_LABEL" in codes(result)) == (not mapped_empty and unmapped == "block")
    if mapped_empty:
        assert result["canFreeze"]
        assert result["memberships"][0]["label"] == "low"
        assert result["summary"]["includedSlides"] == 2
    elif unmapped == "exclude":
        assert result["canFreeze"]
        assert result["coverage"]["selectedSlideIds"] == ["s3"]
        assert result["summary"]["excludedSlides"] == 1


def test_unlabeled_inference_needs_no_target_column(evaluation):
    service, spec, _source = evaluation
    spec["target"] = None
    result = preview(service, spec)
    assert result["canFreeze"]
    assert result["summary"]["labeledSlides"] == 0
    assert result["target"]["classes"] == ["low", "high"]


@pytest.mark.parametrize(
    "dimension,encoder,expected",
    [
        (8, "uni_v1", "FEATURE_DIMENSION_MISMATCH"),
        (4, "other", "ENCODER_MISMATCH"),
        (4, None, "ENCODER_IDENTITY_REQUIRED"),
    ],
)
def test_distinct_feature_sources_must_match_representation(
    evaluation, tmp_path, dimension, encoder, expected
):
    service, spec, _source = evaluation
    data = service.store.get_dataset(spec["datasetId"])
    features, _pack_id, _source = bundle(
        service.store,
        tmp_path,
        data,
        ["s2", "s3"],
        name="test-features",
        dimension=dimension,
        encoder=encoder,
    )
    spec["featureBundleId"] = features["id"]
    assert expected in codes(preview(service, spec))


def test_cohort_blocks_dtype_mismatch_before_freezing(evaluation, tmp_path):
    service, spec, _source = evaluation
    data = service.store.get_dataset(spec["datasetId"])
    features, _pack_id, _source = bundle(
        service.store, tmp_path, data, ["s2", "s3"], name="half-features", dtype="float16"
    )
    spec["featureBundleId"] = features["id"]
    result = preview(service, spec)
    assert not result["canFreeze"]
    assert "FEATURE_DTYPE_MISMATCH" in codes(result)


def test_cohort_blocks_unsupported_patient_aggregation_before_freezing(evaluation):
    service, spec, _source = evaluation
    spec["inference"] = {"patientAggregation": "max"}
    result = preview(service, spec)
    assert not result["canFreeze"]
    assert "PATIENT_AGGREGATION_UNSUPPORTED" in codes(result)


def test_stale_bundle_and_freeze_race_are_blocked(evaluation, monkeypatch):
    service, spec, source = evaluation
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    publish = service.store.publish_configuration

    def change_before_publication(*args, **kwargs):
        with h5py.File(source / "s2.h5", "r+") as handle:
            handle["features"][0, 0] = 99
        return publish(*args, **kwargs)

    monkeypatch.setattr(service.store, "publish_configuration", change_before_publication)
    with pytest.raises(StorageError) as error:
        service.freeze(item["id"], 1, result["previewHash"], "raced-freeze")
    assert error.value.code == "PREVIEW_STALE"
    assert service.store.configuration_publication("raced-freeze") is None
    assert "STALE_FEATURE_BUNDLE" in codes(service.preview(item["id"], 1))


def test_pack_index_is_verified_for_selected_cohort(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    service, spec, _source = setup(
        ScientificStore(folder, "project-packed-test"), tmp_path, pack=True
    )
    result = preview(service, spec)
    assert result["canFreeze"], result["findings"]
    assert result["coverage"]["packChecked"]
    import histopilot.application.evaluations as module

    real_layout = module._layout

    def incomplete_index(path):
        layout = real_layout(path)
        return {**layout, "slides": [item for item in layout["slides"] if item["slideId"] != "s3"]}

    monkeypatch.setattr(module, "_layout", incomplete_index)
    result = preview(service, spec)
    assert result["coverage"]["missingPackSlideIds"] == ["s3"]
    assert "MISSING_TEST_PACK_SLIDES" in codes(result)


def test_no_split_in_evaluation_contract(evaluation):
    _service, spec, _source = evaluation
    with pytest.raises(ValidationError):
        EvaluationSpec.model_validate({**spec, "split": {"folds": 2}})
    with pytest.raises(ValidationError):
        InferenceSettings(loadingPolicy="packed")
    with pytest.raises(ValidationError):
        InferenceSettings(numWorkers=True)


def test_evaluation_api_draft_preview_freeze_and_reopen(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(
            "/api/v1/projects",
            json={"name": "Evaluation project", "storagePath": str(tmp_path / "api-project")},
        )
        assert response.status_code == 201, response.text
        identity = response.json()["id"]
        service, spec, _source = setup(app.state.projects.scientific_store(identity), tmp_path)
        base = f"/api/v1/projects/{identity}"
        response = client.post(
            base + "/drafts",
            json={
                "kind": "experiment",
                "name": "Later test cohort",
                "payload": {"type": "evaluation-cohort", "spec": spec},
            },
        )
        assert response.status_code == 201, response.text
        item = response.json()
        url = base + f"/drafts/{item['id']}"
        response = client.post(url + "/evaluation-preview", json={"expectedRevision": 1})
        assert response.status_code == 200, response.text
        result = response.json()
        intent = {
            "expectedRevision": 1,
            "previewHash": result["previewHash"],
            "operationId": "api-freeze",
            "versionLabel": {"tag": "Held-out cohort"},
        }
        frozen = client.post(url + "/evaluation-freeze", json=intent)
        assert frozen.status_code == 201, frozen.text
        assert client.post(url + "/evaluation-freeze", json=intent).json() == frozen.json()
        response = client.get(base + "/evaluation-cohorts")
        assert response.status_code == 200, response.text
        assert len(response.json()["items"]) == 1
        assert response.json()["items"][0]["current"]
        stale = client.post(url + "/evaluation-preview", json={"expectedRevision": 1})
        assert stale.status_code == 409


@pytest.mark.parametrize("legacy", [False, True])
def test_preview_hash_ignores_finding_wording_and_accepts_legacy_hash(
    evaluation, monkeypatch, legacy
):
    from histopilot.application import evaluations as module

    service, spec, _source = evaluation
    # Exercise an actual absent-class warning, rather than rewording an empty list.
    spec["eligibility"].append({"field": "slideId", "op": "eq", "value": "s2"})
    item = draft(service, spec)
    with monkeypatch.context() as old_release:
        if legacy:
            old_release.setattr(module, "preview_hash", module.legacy_preview_hash)
        result = service.preview(item["id"], 1)
        assert result["findings"]
        frozen = service.freeze(item["id"], 1, result["previewHash"], "freeze-hash")
    assert service.get(frozen["id"])["current"]
    # A release that rewords a warning must not mark saved cohorts stale.
    original = module.EvaluationService._prepare

    def reworded(self, spec):
        preview, guards = original(self, spec)
        findings = [
            {**item, "message": item["message"] + " (reworded)"} for item in preview["findings"]
        ]
        return {**preview, "findings": findings}, guards

    monkeypatch.setattr(module.EvaluationService, "_prepare", reworded)
    assert service.get(frozen["id"])["current"]
    document = service.store.get_configuration(frozen["id"])
    assert service._resolve(document)["current"]
    document["manifest"]["previewHash"] = "0" * 64
    assert not service._resolve(document)["current"]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("change", ["membership", "binding", "blocked", "stored_tamper"])
def test_cohort_freshness_still_rejects_scientific_changes(evaluation, monkeypatch, legacy, change):
    from histopilot.application import evaluations as module

    service, spec, _source = evaluation
    item = draft(service, spec)
    with monkeypatch.context() as old_release:
        if legacy:
            old_release.setattr(module, "preview_hash", module.legacy_preview_hash)
        result = service.preview(item["id"], 1)
        frozen = service.freeze(item["id"], 1, result["previewHash"], "freeze-hash")
    document = service.store.get_configuration(frozen["id"])
    if change == "stored_tamper":
        document["manifest"]["memberships"][0]["label"] = "high"
    else:
        original = service._prepare

        def changed(spec):
            preview, guards = original(spec)
            preview = copy.deepcopy(preview)
            if change == "membership":
                preview["memberships"][0]["label"] = "high"
            elif change == "binding":
                preview["bindings"]["dataset"]["contentHash"] = "0" * 64
            else:
                preview["canFreeze"] = False
                preview["findings"] = [
                    {"code": "NEW_SAFETY_BLOCK", "message": "Blocked", "severity": "error"}
                ]
            preview["previewHash"] = module.preview_hash(preview)
            return preview, guards

        monkeypatch.setattr(service, "_prepare", changed)
    assert not service._resolve(document)["current"]


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("source_kind", ["same_path", "hardlink", "distinct"])
def test_combined_cohorts_reject_duplicate_wsi_sources_across_imports(
    evaluation, tmp_path, independent, source_kind
):
    service, spec, _source = evaluation
    imported = []
    for index, slide in enumerate(("s2", "s3")):
        source = str(tmp_path / ("same.svs" if source_kind == "same_path" else f"{slide}.svs"))
        data, _rows = dataset(
            service.store,
            f"source-import-{index}",
            [
                {
                    "slideId": slide,
                    "patientId": f"external-patient-{index}",
                    "slidePath": source,
                    "attributes": {"label": str(index), "cohort": "test"},
                }
            ],
            inventory=[
                {
                    "path": source,
                    "device": 1,
                    "inode": 100 + index if source_kind == "distinct" else 100,
                    "sizeBytes": 200,
                    "mtimeNs": 300,
                }
            ],
        )
        imported.append(data["id"])
    spec.update(datasetId=imported[0], datasetIds=imported)
    if independent:
        for key in ("protocolId", "developmentFeatureBundleId", "featureBundleId"):
            spec.pop(key)
    item = draft(service, spec)
    result = service.preview(item["id"], 1)
    if source_kind == "distinct":
        assert result["canFreeze"], result["findings"]
        assert "duplicateSourceSlideIds" not in result["overlap"]
    else:
        assert not result["canFreeze"]
        assert "DUPLICATE_TEST_SLIDE_SOURCE" in codes(result)
        assert result["overlap"]["duplicateSourceSlideIds"] == ["s2", "s3"]
        with pytest.raises(StorageError) as error:
            service.freeze(item["id"], 1, result["previewHash"], "duplicate-source-freeze")
        assert error.value.code == "EVALUATION_PREFLIGHT_BLOCKED"
        # Filtering out an alias should make the retained physical slide usable.
        spec["eligibility"].append({"field": "slideId", "op": "eq", "value": "s2"})
        filtered = preview(service, spec)
        assert filtered["canFreeze"], filtered["findings"]
        assert filtered["coverage"]["selectedSlideIds"] == ["s2"]


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("identity_dataset_index", [0, 1])
def test_combined_cohort_target_checks_each_imports_own_dictionary_and_identity_mapping(
    evaluation, independent, identity_dataset_index
):
    service, spec, _source = evaluation
    datasets = []
    for index in (0, 1):
        source = "subject_code" if index == identity_dataset_index else "outcome"
        item = service.store.create_draft("import", f"import-{index}", {})
        datasets.append(
            service.store.publish_dataset(
                item["id"],
                expected_revision=1,
                operation_id=f"target-import-{index}",
                manifest={
                    "kind": "dataset",
                    "dictionary": [
                        {"key": "label", "sourceColumn": source, "owner": "slide", "type": "text"}
                    ],
                    "provenance": {
                        "mapping": {
                            "patientIdColumn": "subject_code"
                            if index == identity_dataset_index
                            else "participant_code"
                        }
                    },
                },
                artifacts={
                    "records.json": json.dumps(
                        [
                            {
                                "slideId": f"s{index + 2}",
                                "patientId": str(index),
                                "attributes": {"label": str(index)},
                            }
                        ]
                    ).encode()
                },
            )["id"]
        )
    spec.update(datasetId=datasets[0], datasetIds=datasets, eligibility=[])
    if independent:
        for key in ("protocolId", "developmentFeatureBundleId", "featureBundleId"):
            spec.pop(key)
    result = preview(service, spec)
    assert not result["canFreeze"]
    assert "IDENTIFIER_TARGET" in codes(result)
