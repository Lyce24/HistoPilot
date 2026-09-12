"""Authenticated top-attention and original-patch API boundaries without a live server."""

import hashlib
import io

import pytest
from test_interpretation import save
from test_interpretation import study as study

from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def top_api(study, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api.app import create_app
    from histopilot.config import Settings

    service, _, _ = study
    record, _ = save(study)
    service.launch(record["id"], "top-api-launch")
    folder = service.jobs.folder(record["id"])
    artifacts = {}
    for member, filename, weights in [
        ("mean", "slide-0.json", [0.1, 0.6, 0.3]),
        ("0", "slide-0-member-0.json", [0.7, 0.2, 0.1]),
    ]:
        value = {
            "slideId": "independent",
            "patchCount": 3,
            "classOrder": ["yes", "no"],
            "probabilities": [0.6, 0.4],
            "member": member,
            "patches": [
                {"index": index, "x": xy[0], "y": xy[1], "weight": weight, "percentile": rank}
                for index, (xy, weight, rank) in enumerate(
                    zip([[0, 0], [100, 0], [100, 100]], weights, [1 / 6, 5 / 6, 0.5], strict=True)
                )
            ],
        }
        write_json(folder / filename, value)
        content = (folder / filename).read_bytes()
        artifacts[filename] = {
            "path": str(folder / filename),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    result = {"runId": record["id"], "state": "succeeded", "artifacts": artifacts}
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)
    monkeypatch.setattr(
        "histopilot.api.interpretation.InterpretationService", lambda *args: service
    )
    app = create_app(Settings(workspace=tmp_path / "top-api-registry", data_roots=(tmp_path,)))
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda identity: service.store)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        base = f"/api/v1/projects/test/interpretations/{record['id']}/slides/independent"
        token = client.get("/api/v1/session").json()["token"]
        yield client, base, token, service, record, folder


def test_top_attention_and_original_patch_require_session_token(top_api):
    client, base, token, _, _, _ = top_api
    for endpoint in ("/attention/top", "/patches/1/image"):
        assert client.get(base + endpoint).status_code == 401
        assert (
            client.get(base + endpoint, headers={"X-HistoPilot-Token": "wrong"}).status_code == 401
        )
    client.headers["X-HistoPilot-Token"] = token
    response = client.get(base + "/attention/top", params={"limit": 2})
    assert response.status_code == 200, response.text
    assert [patch["index"] for patch in response.json()["patches"]] == [1, 2]
    crop = client.get(base + "/patches/1/image", params={"max_size": 128})
    assert crop.status_code == 200, crop.text
    assert crop.headers["content-type"] == "image/png"
    assert "no-store" in crop.headers["cache-control"]
    with Image.open(io.BytesIO(crop.content)) as image:
        assert image.size == (100, 100)
        assert image.getpixel((0, 0)) == (255, 192, 203)
    blocked = client.get(base + "/patches/1/image", headers={"Origin": "https://untrusted.invalid"})
    assert blocked.status_code == 403


def test_top_attention_member_selection_and_query_bounds(top_api):
    client, base, token, _, _, _ = top_api
    client.headers["X-HistoPilot-Token"] = token
    member = client.get(base + "/attention/top", params={"member": "0", "limit": 1})
    assert member.status_code == 200, member.text
    assert member.json()["patches"][0]["index"] == 0
    for limit in (0, 21, -1, "many", 1.5):
        response = client.get(base + "/attention/top", params={"limit": limit})
        assert response.status_code == 422, response.text
    for invalid_member in ("1", "-1", "other"):
        for endpoint in ("/attention/top", "/patches/1/image"):
            response = client.get(base + endpoint, params={"member": invalid_member})
            assert response.status_code == 422, response.text
    for size in (63, 1025, "huge"):
        response = client.get(base + "/patches/1/image", params={"max_size": size})
        assert response.status_code == 422, response.text
    assert client.get(base + "/patches/-1/image").status_code == 422
    assert client.get(base + "/patches/1.5/image").status_code == 422
    assert client.get(base + "/patches/999/image").status_code in {404, 422}


def test_top_attention_unknown_study_slide_and_incomplete_execution(top_api):
    client, base, token, service, record, folder = top_api
    client.headers["X-HistoPilot-Token"] = token
    unknown_study = base.replace(record["id"], "configuration-" + "f" * 64)
    for endpoint in ("/attention/top", "/patches/1/image"):
        assert client.get(unknown_study + endpoint).status_code == 404
        assert (
            client.get(
                base.replace("/slides/independent", "/slides/missing") + endpoint
            ).status_code
            == 404
        )
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "running", "result": None})
    for endpoint in ("/attention/top", "/patches/1/image"):
        response = client.get(base + endpoint)
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "INTERPRETATION_NOT_COMPLETED"
    assert service.get(record["id"])["execution"]["status"] != "completed"


def test_original_patch_crop_refuses_changed_attention_evidence(top_api):
    client, base, token, _, _, folder = top_api
    client.headers["X-HistoPilot-Token"] = token
    content = (folder / "slide-0.json").read_bytes()
    (folder / "slide-0.json").write_bytes(content + b" ")
    for endpoint in ("/attention/top", "/patches/1/image"):
        response = client.get(base + endpoint)
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "INTERPRETATION_RESULT_CHANGED"
