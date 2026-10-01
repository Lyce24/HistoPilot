"""Experiments from a design file through the CLI: template, create, preview, freeze, start."""

import pytest
import yaml
from support.cli import Service
from support.projects import TARGET, bundle, dataset

from histopilot.application.target_splits import TargetSplitService
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.client import default_client


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A project with a dataset, a frozen Targets & splits version and a feature bundle."""
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        store = ScientificStore(service.settings.workspace / "projects" / "Study", project)
        rows = [
            {
                "slideId": f"slide-{index}",
                "patientId": f"patient-{index}",
                "attributes": {"label": str(index % 2), "cohort": "source"},
            }
            for index in range(40)
        ]
        data, _ = dataset(store, rows=rows)
        targets = TargetSplitService(store, LocalFilesystem((service.data,)))
        draft = store.create_draft(
            "experiment",
            "Train and test",
            {
                "type": "target-split",
                "spec": {
                    "datasetId": data["id"],
                    "splitUnit": "slide",
                    "target": {**TARGET, "unit": "slide"},
                    "split": {"method": "random", "testFraction": 0.2, "seed": 42},
                },
            },
        )
        reviewed = targets.preview(draft["id"], 1)
        frozen = targets.freeze(draft["id"], 1, reviewed["previewHash"], "freeze-targets")
        labelled = service.http.put(
            f"/api/v1/projects/{project}/configurations/{frozen['id']}/label",
            json={"tag": "targets v1", "note": "", "expectedRevision": 0},
        )
        assert labelled.status_code == 200, labelled.text
        training = [
            row["slideId"]
            for row in frozen["manifest"]["memberships"]
            if row["partition"] == "train"
        ]
        features, _, _ = bundle(store, service.data, data, training)
        service.cli("use", project)
        yield service, {"dataset": data["id"], "targets": frozen["id"], "bundle": features["id"]}


def design_file(service, ids, path):
    outcome = service.cli(
        "experiment", "template", "--preset", "quick", "--name", "CLI study", "--json"
    )
    design = outcome.envelope["data"]
    design["inputs"].update(
        datasetId=ids["dataset"], targetSplitId=ids["targets"], featureBundleId=ids["bundle"]
    )
    design["inputs"]["trainingSplit"]["folds"] = 2
    design["batches"][0]["predictorPolicy"] = {"method": "skip", "refitPercentile": None}
    path.write_text(
        yaml.safe_dump({"kind": "experiment", "specVersion": 1, **design}, sort_keys=False)
    )
    return design


def test_an_experiment_goes_from_a_design_file_to_queued_training(study, tmp_path, monkeypatch):
    from support.training import runtime

    # Starting checks the training runtime; nothing trains here, so the training extra
    # need not be installed.
    monkeypatch.setattr("histopilot.application.training.training_runtime", runtime)
    service, ids = study
    design_file(service, ids, tmp_path / "design.yaml")

    created = service.cli("experiment", "create", "--from", str(tmp_path / "design.yaml"), "--json")
    assert created.code == 0, created.stdout
    experiment = created.envelope["data"]["experimentId"]
    review = created.envelope["data"]["preview"]
    assert [plan["planId"] for plan in review["plans"]] == ["quick"]
    assert review["canFreeze"] is True, review["findings"]

    early = service.cli("experiment", "start", experiment, "--yes", "--json")
    assert early.code == 3 and early.envelope["error"]["code"] == "EXPERIMENT_SETUP_REQUIRED"
    pending = service.cli("experiment", "freeze", experiment, "--json")
    assert pending.code == 7 and pending.envelope["data"]["preview"]["canFreeze"] is True
    frozen = service.cli("experiment", "freeze", experiment, "--yes", "--json")
    assert frozen.code == 0, frozen.stdout
    assert frozen.envelope["data"]["result"]["frozenSetupId"]
    again = service.cli("experiment", "freeze", experiment, "--yes", "--json")
    assert again.code == 3 and again.envelope["error"]["code"] == "EXPERIMENT_SETUP_FROZEN"

    started = service.cli("experiment", "start", experiment, "--yes", "--wait", "--json")
    # Tests never run the Task Center runner, so the wait ends at once instead of hanging.
    assert started.code == 9, started.stdout
    assert started.envelope["error"]["code"] == "WORK_NEEDS_ATTENTION"
    queued = [
        task for task in default_client().store.list(limit=None) if task["kind"] == "mil-fold"
    ]
    assert len(queued) == 2


def test_a_design_exports_and_creates_again_as_the_same_science(study, tmp_path):
    service, ids = study
    design_file(service, ids, tmp_path / "design.yaml")
    first = service.cli("experiment", "create", "--from", str(tmp_path / "design.yaml"), "--json")
    experiment = first.envelope["data"]["experimentId"]
    exported = service.cli(
        "experiment", "export-design", experiment, "-o", str(tmp_path / "again.yaml"), "--json"
    )
    assert exported.code == 0, exported.stdout
    again = yaml.safe_load((tmp_path / "again.yaml").read_text())
    again["name"] = "CLI study, copy"
    (tmp_path / "again.yaml").write_text(yaml.safe_dump(again, sort_keys=False))
    second = service.cli("experiment", "create", "--from", str(tmp_path / "again.yaml"), "--json")
    assert second.code == 0, second.stdout
    copy = second.envelope["data"]["experimentId"]
    assert copy != experiment
    hashes = {
        service.cli("experiment", "export-design", identity, "--json").envelope["data"][
            "designHash"
        ]
        for identity in (experiment, copy)
    }
    assert len(hashes) == 1


def test_designs_that_leave_out_science_or_state_filled_fields_are_refused(study, tmp_path):
    service, ids = study
    path = tmp_path / "design.yaml"
    design = design_file(service, ids, path)
    del design["batches"][0]["recipe"]["bagSize"]
    path.write_text(yaml.safe_dump({"kind": "experiment", "specVersion": 1, **design}))
    refused = service.cli("experiment", "create", "--from", str(path), "--json")
    assert refused.code == 2 and refused.envelope["error"]["code"] == "SPEC_FIELD_REQUIRED"
    assert "recipe.bagSize" in refused.envelope["error"]["message"]
    design = design_file(service, ids, path)
    design["batches"][0]["experimentId"] = "draft-x"
    path.write_text(yaml.safe_dump({"kind": "experiment", "specVersion": 1, **design}))
    assert service.cli("experiment", "create", "--from", str(path), "--json").code == 2


def test_designs_name_versions_by_tag(study, tmp_path):
    service, ids = study
    path = tmp_path / "design.yaml"
    design = design_file(service, ids, path)
    design["inputs"]["targetSplitId"] = "@targets v1"
    path.write_text(yaml.safe_dump({"kind": "experiment", "specVersion": 1, **design}))
    created = service.cli("experiment", "create", "--from", str(path), "--json")
    assert created.code == 0, created.stdout
    resolved = [item for item in created.envelope["warnings"] if item["code"] == "TAG_RESOLVED"]
    assert [item["message"] for item in resolved] == [
        f"targetSplitId: @targets v1 is {ids['targets']}."
    ]
    exported = service.cli(
        "experiment", "export-design", created.envelope["data"]["experimentId"], "--json"
    )
    assert exported.envelope["data"]["spec"]["inputs"]["targetSplitId"] == ids["targets"]


def test_batch_plans_are_added_from_a_design_file_and_removed(study, tmp_path):
    service, ids = study
    design_file(service, ids, tmp_path / "design.yaml")
    created = service.cli("experiment", "create", "--from", str(tmp_path / "design.yaml"), "--json")
    experiment = created.envelope["data"]["experimentId"]

    # Another preset's batch, from a template whose inputs are still placeholders.
    extra = tmp_path / "nnmil.yaml"
    service.cli("experiment", "template", "--preset", "nnmil", "-o", str(extra))
    clash = service.cli(
        "experiment",
        "batches",
        "add",
        experiment,
        "--from",
        str(tmp_path / "design.yaml"),
        "--json",
    )
    assert clash.code == 2 and clash.envelope["error"]["code"] == "BATCH_PLAN_EXISTS"
    assert (
        service.cli(
            "experiment",
            "batches",
            "add",
            experiment,
            "--from",
            str(extra),
            "--batch",
            "other",
            "--json",
        ).code
        == 2
    )
    added = service.cli("experiment", "batches", "add", experiment, "--from", str(extra), "--json")
    assert added.code == 0, added.stdout
    assert added.envelope["data"]["added"] == ["nnmil"]
    assert [plan["planId"] for plan in added.envelope["data"]["preview"]["plans"]] == [
        "quick",
        "nnmil",
    ]
    listed = service.cli("experiment", "batches", "list", experiment, "--json").envelope["data"]
    assert [(row["id"], row["model"]) for row in listed] == [("quick", "abmil"), ("nnmil", "nnmil")]

    removed = service.cli("experiment", "batches", "remove", experiment, "quick", "--json")
    assert removed.code == 0 and removed.envelope["data"]["remaining"] == ["nnmil"]
    missing = service.cli("experiment", "batches", "remove", experiment, "quick", "--json")
    assert missing.code == 5 and missing.envelope["error"]["code"] == "BATCH_PLAN_NOT_FOUND"


def test_an_edited_design_updates_its_draft_instead_of_making_another(study, tmp_path):
    service, ids = study
    path = tmp_path / "design.yaml"
    design = design_file(service, ids, path)
    created = service.cli("experiment", "create", "--from", str(path), "--json")
    experiment = created.envelope["data"]["experimentId"]
    design["notes"] = "Second thoughts"
    design["batches"][0]["batchName"] = "Quick check, edited"
    path.write_text(
        yaml.safe_dump({"kind": "experiment", "specVersion": 1, **design}, sort_keys=False)
    )
    updated = service.cli("experiment", "update", experiment, "--from", str(path), "--json")
    assert updated.code == 0, updated.stdout
    assert updated.envelope["data"]["experimentId"] == experiment
    assert [row["id"] for row in service.cli("experiment", "list", "--json").envelope["data"]] == [
        experiment
    ]
    saved = service.cli("experiment", "export-design", experiment, "--json").envelope["data"]
    assert saved["spec"]["notes"] == "Second thoughts"
    assert saved["spec"]["batches"][0]["batchName"] == "Quick check, edited"
    assert service.cli("experiment", "freeze", experiment, "--yes", "--json").code == 0
    refused = service.cli("experiment", "update", experiment, "--from", str(path), "--json")
    assert refused.code == 3 and refused.envelope["error"]["code"] == "EXPERIMENT_SETUP_FROZEN"


def test_an_agent_revises_the_draft_it_previewed(study, tmp_path):
    from histopilot.agent import tools
    from histopilot.client import Client

    service, ids = study
    design = design_file(service, ids, tmp_path / "design.yaml")
    project = service.cli("use", "--json").envelope["data"]["project"]
    access = {"projectId": project, "scopes": ["read", "preview"], "exposure": "full"}
    agent = tools.AgentSession(Client(transport=service.transport), access)
    first = tools.preview_design(agent, design)
    design["notes"] = "Revised by the agent"
    second = tools.preview_design(agent, design, experiment=first["experimentId"])
    assert second["experimentId"] == first["experimentId"]
    assert second["revision"] > first["revision"] and second["canFreeze"] is True
    assert len(tools.list_records(agent, "experiment")["items"]) == 1


def test_a_person_freezes_exactly_the_preview_an_agent_handed_over(study, tmp_path):
    from histopilot.agent import tools
    from histopilot.client import Client

    service, ids = study
    design = design_file(service, ids, tmp_path / "design.yaml")
    project = service.cli("use", "--json").envelope["data"]["project"]
    exposed = service.http.put(
        f"/api/v1/projects/{project}/ai-exposure",
        json={"level": "metadata", "expectedLevel": "none"},
    )
    assert exposed.status_code == 200, exposed.text
    token = service.http.post(
        "/api/v1/tokens", json={"projectId": project, "scopes": ["read", "preview"]}
    ).json()["token"]
    client = Client(transport=service.transport, token=token)
    agent = tools.AgentSession(client, client.get("/access"))
    first = tools.preview_design(agent, design)
    experiment = first["experimentId"]
    # The person's own preview, unredacted, confirms the same hash.
    review = service.cli("experiment", "preview", experiment, "--json")
    assert review.envelope["data"]["previewHash"] == first["previewHash"]

    design["notes"] = "Revised by the agent"
    second = tools.preview_design(agent, design, experiment=experiment)
    assert second["previewHash"] != first["previewHash"]
    stale = service.cli(
        "experiment",
        "freeze",
        experiment,
        "--preview-hash",
        first["previewHash"],
        "--yes",
        "--json",
    )
    assert stale.envelope["error"]["code"] == "PREVIEW_CHANGED", stale.stdout
    frozen = service.cli(
        "experiment",
        "freeze",
        experiment,
        "--preview-hash",
        second["previewHash"],
        "--yes",
        "--json",
    )
    assert frozen.code == 0, frozen.stdout
    assert frozen.envelope["data"]["result"]["frozenSetupId"]


def test_comparisons_take_catalog_names_and_errors_list_them(study):
    service, _ = study
    compared = service.cli("experiment", "template", "--compare-model", "mean_pool", "--json")
    assert compared.code == 0, compared.stdout
    configurations = compared.envelope["data"]["batches"][0]["configurations"]
    assert [arm["model"] for arm in configurations] == ["abmil", "mean_pool"]
    for option, value, named in (
        ("--compare-model", "Mean pooling", "mean_pool"),
        ("--preset", "compare_model", "learning-rate"),
    ):
        refused = service.cli("experiment", "template", option, value, "--json")
        error = refused.envelope["error"]
        assert refused.code == 2 and error["code"] == "SPEC_INVALID" and named in error["message"]
