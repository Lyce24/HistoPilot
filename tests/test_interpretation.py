"""Attention input provenance, review, local slide geometry and artifact boundaries."""

import copy
import hashlib
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.interpretation import InterpretationService
from histopilot.application.predictors import checkpoint_snapshot
from histopilot.schemas.interpretation import InterpretationSelection, SaveInterpretation
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.viewer.slide_images import inspect_slide, render_slide
from histopilot.workers.compute_job import verify_plan_inputs
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

Image = pytest.importorskip("PIL.Image")


class Executor:
    def __init__(self):
        self.sessions, self.calls = set(), []

    def running(self, session):
        return session in self.sessions

    def launch(self, session, python, plan, log, *, package_root):
        self.sessions.add(session)
        self.calls.append((session, python, plan, log, package_root))


@pytest.fixture
def study(tmp_path):
    (tmp_path / "project").mkdir()
    store = ScientificStore(tmp_path / "project", "interpret-project")
    draft = store.create_draft("import", "Development", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "Development"},
        artifacts={"slides.json": b'[{"slideId":"development-only"}]'},
        operation_id="data",
    )
    batch_id = "configuration-" + "1" * 64
    checkpoint_folder = store.folder / "training" / batch_id / "runs" / "run-one"
    checkpoint_folder.mkdir(parents=True)
    checkpoint_path = checkpoint_folder / "best.ckpt"
    checkpoint_path.write_bytes(b"test evidence, never deserialized by control API")
    predictor = store.publish_configuration(
        manifest={
            "kind": "frozen-predictor",
            "datasetId": dataset["id"],
            "name": "Model",
            "batchId": batch_id,
            "experimentId": "experiment-one",
            "method": "ensemble",
            "target": {
                "task": "binary_classification",
                "unit": "slide",
                "classes": ["yes", "no"],
                "positiveClass": "yes",
            },
            "recipe": {"model": "abmil", "embedDim": 8, "attentionDim": 4},
            "inputs": {
                "features": {"encoderId": "test-encoder", "dimensions": 4, "dtype": "float32"}
            },
            "checkpoints": [
                {**checkpoint_snapshot(checkpoint_path, checkpoint_folder), "runId": "run-one"}
            ],
        },
        operation_id="predictor",
    )
    image_path = tmp_path / "independent-slide.png"
    Image.new("RGB", (300, 200), color="pink").save(image_path)
    features_path = tmp_path / "features.h5"
    with h5py.File(features_path, "w") as handle:
        handle.create_dataset("features", data=np.arange(12, dtype=np.float32).reshape(3, 4))
        coords = handle.create_dataset(
            "coords", data=np.array([[0, 0], [100, 0], [100, 100]], dtype=np.int64)
        )
        coords.attrs["patch_size_level0"] = 100
        handle["features"].attrs["encoder_id"] = "test-encoder"
    selection = InterpretationSelection(
        name="Attention",
        predictorId=predictor["id"],
        encoderId="test-encoder",
        slides=[
            {
                "slideId": "independent",
                "slidePath": str(image_path),
                "featurePath": str(features_path),
                "confirmRowAlignment": True,
            }
        ],
    )
    service = InterpretationService(store, LocalFilesystem((tmp_path,)))
    executor = Executor()
    service.jobs = ComputeJobService(
        store,
        executor=executor,
        runtime=lambda: {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
            # The executor is fake; reservation checks must use a fixed host too.
            "host": {"cpuCount": 8, "totalRamGb": 16},
        },
    )
    return service, selection, executor


def save(study):
    service, selection, _ = study
    preview = service.preview(selection)
    assert preview["canSave"], preview
    request = SaveInterpretation(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="save-attention"
    )
    return service.save(request), request


def test_arbitrary_slide_review_geometry_identity_idempotency_and_execution(study):
    service, selection, executor = study
    preview = service.preview(selection)
    assert preview["canSave"], preview
    row = preview["manifest"]["slides"][0]
    assert row["slideId"] == "independent"
    assert row["width"] == 300 and row["height"] == 200
    assert row["patchWidthLevel0"] == row["patchHeightLevel0"] == 100
    assert row["alignment"] == "embedded_verified" and row["patchCount"] == 3
    assert len(row["featureSha256"]) == len(row["coordinatesSha256"]) == 64
    document, request = save(study)
    assert service.save(request)["id"] == document["id"]
    assert service.list()["items"][0]["execution"]["status"] == "not_started"
    result = service.launch(document["id"], "launch")
    assert result["status"] == "queued" and "interpretation" in result["sessionName"]
    assert service.launch(document["id"], "launch")["status"] == "queued"
    assert len(executor.calls) == 1
    plan = read_json(service.jobs.folder(document["id"]) / "plan.json")
    verify_plan_inputs(plan)
    executor.sessions.clear()
    assert service.execution(document["id"])["status"] == "interrupted"
    assert service.launch(document["id"], "resume", resume=True)["attempt"] == 2
    assert service.cancel(document["id"], "cancel")["cancellationRequested"]


def test_accepted_attention_retry_never_reinspects_complete_feature_arrays(study, monkeypatch):
    service, _, executor = study
    document, _ = save(study)
    first = service.launch(document["id"], "launch")
    monkeypatch.setattr(service, "_execution_plan", lambda *args, **kwargs: pytest.fail(
        "Accepted attention retry must not rescan feature tensors"))
    assert service.launch(document["id"], "launch")["planHash"] == first["planHash"]
    assert len(executor.calls) == 1


def test_nnmil_attention_publication_preserves_window_method_and_feature_provenance(study):
    service, selection, _ = study
    source = service.predictors.get(selection.predictorId)
    predictor = service.store.publish_configuration(
        manifest={**source["manifest"], "recipe": {
            "model": "nnmil", "attentionDim": 2, "nnmilWindowStrideDivisor": 2,
            "nnmilWindowAggregation": "mean_logits", "nnmilWindowSeed": 42,
        }},
        operation_id="nnmil-predictor",
    )
    selected = selection.model_copy(update={"predictorId": predictor["id"]})
    preview = service.preview(selected)
    assert preview["canSave"], preview
    document = service.save(SaveInterpretation(
        **selected.model_dump(), previewHash=preview["previewHash"], operationId="nnmil-attention"
    ))
    assert document["manifest"]["predictorId"] == predictor["id"]
    assert document["manifest"]["slides"][0]["patchCount"] == 3
    assert "nnMIL maps average normalized attention" in preview["executionNote"]


def test_geometry_and_slide_viewport_are_exact_and_bounded(study):
    service, selection, _ = study
    assert service.inspect_slide(selection.slides[0].slidePath)["width"] == 300
    document, _ = save(study)
    with Image.open(
        io.BytesIO(service.image(document["id"], "independent", max_size=128))
    ) as image:
        assert image.size == (128, 85)
    with Image.open(
        io.BytesIO(
            service.image(document["id"], "independent", max_size=128, region=(100, 0, 100, 100))
        )
    ) as image:
        assert image.size == (100, 100)
    with pytest.raises(StorageError, match="viewport"):
        service.image(document["id"], "independent", region=(250, 0, 100, 100))
    with pytest.raises(StorageError):
        render_slide(selection.slides[0].slidePath, max_size=100000)


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ("wrong_encoder", "encoder"),
        ("wrong_dimension", "dimensions"),
        ("wrong_dtype", "dtype"),
        ("missing_coords", "coords"),
        ("count", "one XY"),
        ("negative", "inside"),
        ("outside", "inside"),
        ("float_coords", "integer"),
        ("nan", "NaN"),
        ("geometry", "geometry"),
        ("no_geometry", "patch width"),
        ("coord_space", "level-0"),
    ],
)
def test_invalid_scientific_inputs_fail_review(study, mutation, fragment):
    service, selection, _ = study
    with h5py.File(selection.slides[0].featurePath, "a") as handle:
        if mutation == "wrong_encoder":
            handle["features"].attrs["encoder_id"] = "another"
        elif mutation in {"wrong_dimension", "wrong_dtype"}:
            del handle["features"]
            handle.create_dataset(
                "features",
                data=np.ones(
                    (3, 5 if mutation == "wrong_dimension" else 4),
                    dtype="float64" if mutation == "wrong_dtype" else "float32",
                ),
            )
        elif mutation in {"count", "float_coords"}:
            del handle["coords"]
            handle.create_dataset(
                "coords",
                data=np.zeros(
                    (2 if mutation == "count" else 3, 2),
                    dtype="float32" if mutation == "float_coords" else "int64",
                ),
            )
        elif mutation == "missing_coords":
            del handle["coords"]
        elif mutation in {"negative", "outside"}:
            handle["coords"][0] = [-1, 0] if mutation == "negative" else [300, 0]
        elif mutation == "nan":
            handle["features"][0, 0] = np.nan
        elif mutation == "geometry":
            selection.slides[0].patchWidthLevel0 = 200
            selection.slides[0].patchHeightLevel0 = 200
        elif mutation == "no_geometry":
            del handle["coords"].attrs["patch_size_level0"]
        elif mutation == "coord_space":
            handle["coords"].attrs["coordinate_space"] = "patch-level"
    result = service.preview(selection)
    assert result["canSave"] is False
    assert fragment in result["findings"][0]["message"], result


def test_separate_coords_validate_embedded_row_order_or_explicit_attestation(study, tmp_path):
    service, selection, _ = study
    external = tmp_path / "coords.h5"
    with h5py.File(selection.slides[0].featurePath) as source, h5py.File(external, "w") as dest:
        source.copy("coords", dest)
    selection.slides[0].coordinatesPath = str(external)
    assert service.preview(selection)["manifest"]["slides"][0]["alignment"] == "embedded_verified"
    with h5py.File(external, "a") as handle:
        handle["coords"][:] = handle["coords"][:][::-1]
    assert not service.preview(selection)["canSave"]
    with h5py.File(selection.slides[0].featurePath, "a") as handle:
        del handle["coords"]
    result = service.preview(selection)
    assert result["canSave"] and result["findings"][0]["severity"] == "warning"
    assert result["manifest"]["slides"][0]["alignment"] == "user_confirmed"


def test_linked_hdf5_and_unapproved_paths_are_blocked(study, tmp_path):
    service, selection, _ = study
    source = Path(selection.slides[0].featurePath)
    linked = tmp_path / "linked.h5"
    with h5py.File(linked, "w") as handle:
        handle["features"] = h5py.ExternalLink(str(source), "features")
    selection.slides[0].featurePath = str(linked)
    assert "embedded" in service.preview(selection)["findings"][0]["message"]
    selection.slides[0].slidePath = "/etc/passwd"
    assert service.preview(selection)["findings"][0]["code"] == "INTERPRETATION_PATH_INVALID"
    symlink = tmp_path / "slide-link.png"
    symlink.symlink_to(tmp_path / "independent-slide.png")
    selection.slides[0].slidePath = str(symlink)
    assert not service.preview(selection)["canSave"]


def test_stale_feature_preview_and_modified_slide_block_launch_and_view(study):
    service, selection, _ = study
    preview = service.preview(selection)
    with h5py.File(selection.slides[0].featurePath, "a") as handle:
        handle["features"][0, 0] += 1
    with pytest.raises(StorageError, match="Review again"):
        service.save(
            SaveInterpretation(
                **selection.model_dump(), previewHash=preview["previewHash"], operationId="stale"
            )
        )
    document, _ = save(study)
    Image.new("RGB", (300, 200), color="blue").save(selection.slides[0].slidePath)
    with pytest.raises(StorageError, match="changed"):
        service.launch(document["id"], "changed")
    with pytest.raises(StorageError, match="changed"):
        service.image(document["id"], "independent")


def test_prediction_and_clinical_lineage_reject_other_predictors(study):
    service, selection, _ = study
    model = service.predictors.get(selection.predictorId)

    def record(kind, **fields):
        return service.store.publish_configuration(
            manifest={"kind": kind, "datasetId": model["manifest"]["datasetId"], **fields},
            operation_id=kind + str(len(fields)),
        )

    evaluation = record("model-evaluation", predictorId=selection.predictorId)
    clinical = record(
        "clinical-analysis", evaluationId=evaluation["id"], predictorId=selection.predictorId
    )
    selection.clinicalAnalysisId = clinical["id"]
    assert service.preview(selection)["manifest"]["evaluationId"] == evaluation["id"]
    wrong = record("model-evaluation", predictorId="configuration-" + "f" * 64, name="wrong")
    selection.evaluationId = wrong["id"]
    assert service.preview(selection)["findings"][0]["code"] == "INTERPRETATION_LINEAGE_MISMATCH"


def test_attention_pagination_member_viewports_and_tamper_detection(study):
    service, _, _ = study
    document, _ = save(study)
    identity = document["id"]
    service.launch(identity, "launch")
    folder = service.jobs.folder(identity)
    value = {
        "slideId": "independent",
        "patchCount": 3,
        "classOrder": ["yes", "no"],
        "probabilities": [0.6, 0.4],
        "patches": [
            {"index": index, "x": xy[0], "y": xy[1], "weight": 1 / 3, "percentile": 0.5}
            for index, xy in enumerate([[0, 0], [100, 0], [100, 100]])
        ],
    }
    write_json(folder / "slide-0.json", value)
    content = (folder / "slide-0.json").read_bytes()
    result = {
        "runId": identity,
        "state": "succeeded",
        "artifacts": {
            "slide-0.json": {
                "path": str(folder / "slide-0.json"),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        },
    }
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)
    response = service.attention(identity, "independent", limit=1, offset=1)
    assert response["total"] == 3 and response["patches"][0]["index"] == 1
    response = service.attention(identity, "independent", region=(100, 100, 100, 100))
    assert response["total"] == 1 and response["patches"][0]["index"] == 2
    with pytest.raises(StorageError):
        service.attention(identity, "independent", member="1")
    with pytest.raises(StorageError):
        service.artifact(identity, "../../state.json")
    (folder / "slide-0.json").write_bytes(content + b" ")
    with pytest.raises(StorageError, match="changed"):
        service.attention(identity, "independent")


def test_schema_requires_alignment_and_distinct_slides(study):
    _, selection, _ = study
    data = selection.model_dump()
    data["slides"][0]["confirmRowAlignment"] = False
    with pytest.raises(ValidationError):
        InterpretationSelection.model_validate(data)
    data = selection.model_dump()
    data["slides"].append(copy.deepcopy(data["slides"][0]))
    with pytest.raises(ValidationError):
        InterpretationSelection.model_validate(data)


def test_corrupt_and_oversized_rasters_fail_clearly(tmp_path, monkeypatch):
    bad = tmp_path / "bad.svs"
    bad.write_bytes(b"not a slide")
    with pytest.raises(StorageError, match="opened safely"):
        inspect_slide(bad)
    image = tmp_path / "large.png"
    Image.new("RGB", (11, 10)).save(image)
    monkeypatch.setattr("histopilot.viewer.slide_images.MAX_RASTER_PIXELS", 100)
    with pytest.raises(StorageError, match="pyramidal"):
        inspect_slide(image)


def test_empty_explicit_resources_and_missing_geometry_infer_stable_preview(study):
    service, selection, _ = study
    selection = InterpretationSelection.model_validate(
        {**selection.model_dump(), "resources": {"gpuIds": [], "dataLoaderWorkers": 0}}
    )
    document, _ = save((service, selection, None))
    assert document["manifest"]["resources"]["ramGbPerRun"] == 8.0


def test_pending_extraction_lock_blocks_attention_review(study):
    service, selection, _ = study
    Path(selection.slides[0].featurePath + ".lock").touch()
    assert "lock remains" in service.preview(selection)["findings"][0]["message"]


def test_registered_attention_routes_authentication_roots_and_validation(tmp_path):
    from fastapi.testclient import TestClient

    from histopilot.api.app import create_app
    from histopilot.config import Settings

    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        token = client.get("/api/v1/session").json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        created = client.post(
            "/api/v1/projects",
            json={"name": "Attention", "storagePath": str(tmp_path / "attention-project")},
        )
        assert created.status_code == 201, created.text
        base = f"/api/v1/projects/{created.json()['id']}/interpretations"
        response = client.get(base)
        assert response.status_code == 200 and response.json()["items"] == []
        assert response.json()["executionEnabled"] is True
        slide = tmp_path / "slide.png"
        Image.new("RGB", (20, 10)).save(slide)
        response = client.post(base + "/slide-inspection", json={"path": str(slide)})
        assert response.status_code == 200 and response.json()["width"] == 20
        assert (
            client.post(base + "/slide-inspection", json={"path": "/etc/passwd"}).status_code == 403
        )
        assert client.post(base + "/preview", json={}).status_code == 422
        assert client.get(base + "/missing").status_code == 404
        client.headers.pop("X-HistoPilot-Token")
        for suffix in (
            "",
            "/preview",
            "/slide-inspection",
            "/missing/launch",
            "/missing/resume",
            "/missing/cancel",
        ):
            assert client.post(base + suffix, json={}).status_code == 401


def test_pyramid_render_uses_exact_fractional_field_of_view(tmp_path, monkeypatch):
    from types import SimpleNamespace

    recorded = {}

    class FakeSlide:
        dimensions = (10000, 6500)
        level_downsamples = [1.0, 64.0]

        @staticmethod
        def detect_format(path):
            return "test-pyramid"

        def __init__(self, path):
            pass

        def get_best_level_for_downsample(self, downsample):
            assert downsample > 64
            return 1

        def read_region(self, origin, level, size):
            recorded.update(origin=origin, level=level, size=size)
            return Image.new("RGBA", size)

        def close(self):
            recorded["closed"] = True

    class SlideError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "openslide", SimpleNamespace(OpenSlide=FakeSlide, OpenSlideError=SlideError)
    )
    resize = Image.Image.resize

    def record_resize(self, size, resample=None, box=None, **kwargs):
        recorded["box"] = box
        return resize(self, size, resample=resample, box=box, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", record_resize)
    path = tmp_path / "pyramid.svs"
    path.touch()
    render_slide(path, max_size=128)
    assert recorded["size"] == (157, 102)
    assert recorded["box"] == (0, 0, 156.25, 101.5625)
    assert recorded["closed"] is True


def test_mmap_attention_pages_verify_receipts_cache_stamps_and_reject_objects(
    tmp_path, monkeypatch
):
    from histopilot.viewer import attention_arrays

    path = tmp_path / "slide-0.npy"
    values = np.array([[0, 0, 0.2, 0.1], [100, 0, 0.3, 0.5], [100, 100, 0.5, 0.9]], dtype="<f8")
    np.save(path, values)

    def receipt():
        content = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    expected = receipt()
    slide = {
        "patchCount": 3,
        "patchWidthLevel0": 100,
        "patchHeightLevel0": 100,
        "width": 300,
        "height": 200,
    }
    page = attention_arrays.attention_page(path, expected, slide, offset=1, limit=1)
    assert page["total"] == 3 and page["patches"][0]["weight"] == 0.3
    original_hash = attention_arrays.hashlib.sha256
    monkeypatch.setattr(
        attention_arrays.hashlib,
        "sha256",
        lambda: pytest.fail("Unchanged files must reuse their verified signature"),
    )
    page = attention_arrays.attention_page(
        path, expected, slide, offset=0, limit=10, region=(100, 100, 100, 100)
    )
    assert page["total"] == 1 and page["patches"][0]["index"] == 2
    monkeypatch.setattr(attention_arrays.hashlib, "sha256", original_hash)
    values[0, 2] = 0.25
    np.save(path, values)
    with pytest.raises(StorageError, match="checksum changed"):
        attention_arrays.attention_page(path, expected, slide, offset=0, limit=1)
    np.save(path, np.array([{"dangerous": "object"}], dtype=object), allow_pickle=True)
    with pytest.raises(StorageError, match="invalid"):
        attention_arrays.attention_page(path, receipt(), slide, offset=0, limit=1)
    np.save(path, np.ones((3, 3), dtype=np.float64))
    with pytest.raises(StorageError, match="dimensions"):
        attention_arrays.attention_page(path, receipt(), slide, offset=0, limit=1)


def test_service_serves_indexed_mean_and_member_without_parsing_json(study, monkeypatch):
    service, _, _ = study
    document, _ = save(study)
    identity = document["id"]
    service.launch(identity, "launch")
    folder = service.jobs.folder(identity)
    artifacts = {}
    for suffix in ("", "-member-0"):
        path = folder / f"slide-0{suffix}.npy"
        np.save(
            path,
            np.array([[0, 0, 0.2, 0.1], [100, 0, 0.3, 0.5], [100, 100, 0.5, 0.9]], dtype="<f8"),
        )
        content = path.read_bytes()
        artifacts[path.name] = {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    result = {
        "runId": identity,
        "state": "succeeded",
        "artifacts": artifacts,
        "slides": [
            {
                "slideId": "independent",
                "patchCount": 3,
                "probabilities": [0.6, 0.4],
                "attentionArray": "slide-0.npy",
                "members": [
                    {
                        "index": 0,
                        "probabilities": [0.6, 0.4],
                        "attentionArray": "slide-0-member-0.npy",
                    }
                ],
            }
        ],
    }
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)
    monkeypatch.setattr(
        service,
        "artifact",
        lambda *args: pytest.fail("Indexed viewport must not read a full JSON artifact"),
    )
    for member in ("mean", "0"):
        response = service.attention(identity, "independent", member=member, offset=1, limit=1)
        assert response["probabilities"] == [0.6, 0.4]
        assert response["classOrder"] == ["yes", "no"]
        assert response["patchWidthLevel0"] == 100
        assert response["patches"][0]["index"] == 1
        assert response["member"] == member


def test_display_copy_updates_do_not_make_saved_attention_evidence_stale(study, monkeypatch):
    service, _, _ = study
    document, _ = save(study)
    monkeypatch.setattr(
        "histopilot.application.interpretation.EXECUTION_NOTE", "Updated help text."
    )
    assert service.launch(document["id"], "launch")["status"] == "queued"
