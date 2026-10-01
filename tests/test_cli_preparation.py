"""Data preparation from spec files: inspect and import a table, relabel, attach features."""

import h5py
import numpy as np
import pytest
import yaml
from support.cli import Service


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        project = current.create_project()
        current.cli("use", project)
        table = current.data / "clinical.csv"
        table.write_text(
            "slide_id,patient_id,label\n"
            + "".join(f"s{index},p{index},{index % 2}\n" for index in range(12))
        )
        yield current


def spec_file(tmp_path, name, kind, body):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"kind": kind, "specVersion": 1, **body}, sort_keys=False))
    return path


def imported(service, tmp_path, tag="clinical v1"):
    body = service.cli("dataset", "template", "--json").envelope["data"]
    body.update(
        source={"path": str(service.data / "clinical.csv")},
        slideIdColumn="slide_id",
        patientIdColumn="patient_id",
        attributes=[{"key": "label", "sourceColumn": "label", "owner": "slide", "type": "text"}],
    )
    path = spec_file(tmp_path, "import.yaml", "import", body)
    return service.cli("dataset", "import", "--from", str(path), "--tag", tag, "--yes", "--json")


def test_a_table_is_inspected_and_imported_as_a_tagged_dataset(service, tmp_path):
    inspected = service.cli("dataset", "inspect", str(service.data / "clinical.csv"), "--json")
    assert inspected.code == 0, inspected.stdout
    assert {"slide_id", "patient_id", "label"} <= set(map(str, inspected.stdout.split('"')))
    outcome = imported(service, tmp_path)
    assert outcome.code == 0, outcome.stdout
    dataset = service.cli("dataset", "show", "@clinical v1", "--json").envelope["data"]
    assert dataset["id"] == outcome.envelope["data"]["result"]["id"]
    records = service.cli("dataset", "records", "@clinical v1", "--json").envelope["data"]
    assert len(records) == 12


def test_a_version_is_relabelled_with_both_tag_and_note(service, tmp_path):
    assert imported(service, tmp_path).code == 0
    pending = service.cli(
        "dataset", "label", "@clinical v1", "--tag", "clinical v2", "--note", "renamed", "--json"
    )
    assert pending.code == 7
    assert pending.envelope["data"]["preview"]["to"] == {"tag": "clinical v2", "note": "renamed"}
    relabelled = service.cli(
        "dataset",
        "label",
        "@clinical v1",
        "--tag",
        "clinical v2",
        "--note",
        "renamed",
        "--yes",
        "--json",
    )
    assert relabelled.code == 0, relabelled.stdout
    label = service.cli("dataset", "show", "@clinical v2", "--json").envelope["data"][
        "versionLabel"
    ]
    assert (label["tag"], label["note"]) == ("clinical v2", "renamed")
    # The service clears an omitted note, so the command never sends one without it.
    assert service.cli("dataset", "label", "@clinical v2", "--tag", "x", "--json").code == 2


def test_features_are_attached_and_validation_is_queued(service, tmp_path):
    assert imported(service, tmp_path).code == 0
    features = service.data / "features"
    features.mkdir()
    for index in range(12):
        with h5py.File(features / f"s{index}.h5", "w") as handle:
            handle.create_dataset("features", data=np.ones((3, 4), dtype="float32"))
            handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
    body = service.cli("features", "template", "--json").envelope["data"]
    body.update(datasetId="@clinical v1", path=str(features), encoderId="uni_v1")
    path = spec_file(tmp_path, "features.yaml", "features", body)
    attached = service.cli(
        "features", "attach", "--from", str(path), "--tag", "uni", "--yes", "--json"
    )
    assert attached.code == 0, attached.stdout
    assert service.cli("features", "show", "@uni", "--json").code == 0

    pack = service.cli("pack", "template", "--json").envelope["data"]
    pack["featureSetId"] = "@uni"
    path = spec_file(tmp_path, "pack.yaml", "feature-pack", pack)
    queued = service.cli("pack", "create", "--from", str(path), "--yes", "--json")
    assert queued.code == 0, queued.stdout
    jobs = service.cli("pack", "list", "--json").envelope["data"]
    assert [job["runState"] for job in jobs] == ["queued"]

    bundle = service.cli("bundle", "template", "--json").envelope["data"]
    bundle["featureSetId"] = "@uni"
    path = spec_file(tmp_path, "bundle.yaml", "feature-bundle", bundle)
    # Until validation has run, the bundle's review blocks it.
    blocked = service.cli(
        "bundle", "create", "--from", str(path), "--tag", "uni bundle", "--yes", "--json"
    )
    assert blocked.code == 3, blocked.stdout
    assert blocked.envelope["error"]["code"] == "PREVIEW_BLOCKED"
