"""Objectives organize experiments without touching them: filing, lineage, routes and CLI."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError
from support import projects as fixtures
from support.cli import Service

from histopilot.application.lifecycle import CleanupService
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.application.objectives import OBJECTIVES_FILE, ObjectiveService
from histopilot.schemas.lifecycle import ApplyCleanup, CleanupSelection
from histopilot.schemas.model_experiments import CreateModelExperiment
from histopilot.schemas.objectives import (
    CreateObjective,
    ExperimentFiling,
    FileExperiment,
    RemoveObjective,
    UpdateObjective,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

NONE = {"objectiveId": None, "role": None, "series": None, "seriesMember": None}


class Training:
    def execution(self, identity, **_kwargs):
        return None


@pytest.fixture
def experiments(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-objectives")
    return ModelExperimentService(store, LocalFilesystem((tmp_path,)), training=Training())


@pytest.fixture
def objectives(experiments):
    return ObjectiveService(experiments.store)


def objective(objectives, name, rank="secondary"):
    return objectives.create(CreateObjective(name=name, rank=rank, operationId=uuid4().hex))


def experiment(experiments, name, **values):
    return experiments.create(
        CreateModelExperiment(name=name, operationId=uuid4().hex, setupVersion=1, **values)
    )


def filed(record):
    return {key: record[key] for key in NONE}


def cleanup(service, action, *keys):
    lifecycle = CleanupService(service.store, service.filesystem, training=service.training)
    selection = CleanupSelection(action=action, keys=list(keys))
    preview = lifecycle.preview(selection)
    assert preview["canApply"], preview
    lifecycle.apply(
        ApplyCleanup(
            **selection.model_dump(), previewHash=preview["previewHash"], operationId=uuid4().hex
        )
    )
    return preview


def test_objectives_list_by_rank_then_name_and_creation_replays(objectives):
    later = objective(objectives, "Aim 10 · Morphology", "exploratory")
    command = CreateObjective(name="Aim 1 · Validity", rank="primary", operationId="create-aim-1")
    first = objectives.create(command)
    objective(objectives, "Aim 2 · Transport")
    objective(objectives, "Aim 9 · Extra", "exploratory")
    listed = objectives.list()
    assert [item["name"] for item in listed["items"]] == [
        "Aim 1 · Validity",
        "Aim 2 · Transport",
        "Aim 9 · Extra",
        "Aim 10 · Morphology",
    ]
    assert listed["ranks"] == ["primary", "secondary", "exploratory"]
    assert "creationOperationId" not in listed["items"][0]
    assert listed["filings"] == {}
    # A lost answer replays; the same operation asking for something else is refused.
    assert objectives.create(command) == first
    with pytest.raises(StorageError) as reused:
        objectives.create(command.model_copy(update={"rank": "secondary"}))
    assert reused.value.code == "OPERATION_CONFLICT"
    with pytest.raises(StorageError) as taken:
        objective(objectives, "  aim 10   ·  MORPHOLOGY ")
    assert taken.value.code == "OBJECTIVE_EXISTS"
    assert objectives.get(later["id"])["rank"] == "exploratory"


def test_updates_rename_and_rerank_under_a_revision(objectives):
    aim = objective(objectives, "Aim 1", "primary")
    other = objective(objectives, "Aim 2")
    renamed = objectives.update(
        aim["id"],
        UpdateObjective(expectedRevision=1, name="Aim 1 · Validity", description=" Is it? "),
    )
    assert (renamed["name"], renamed["description"], renamed["revision"]) == (
        "Aim 1 · Validity",
        "Is it?",
        2,
    )
    # Sending the saved values again, as a retry does, changes nothing.
    again = objectives.update(
        aim["id"], UpdateObjective(expectedRevision=1, name="Aim 1 · Validity")
    )
    assert again["revision"] == 2
    with pytest.raises(StorageError) as stale:
        objectives.update(aim["id"], UpdateObjective(expectedRevision=1, rank="exploratory"))
    assert stale.value.code == "REVISION_CONFLICT"
    with pytest.raises(StorageError) as taken:
        objectives.update(other["id"], UpdateObjective(expectedRevision=1, name="AIM 1 · validity"))
    assert taken.value.code == "OBJECTIVE_EXISTS"
    with pytest.raises(StorageError) as missing:
        objectives.update(
            "objective-" + "0" * 32, UpdateObjective(expectedRevision=1, rank="primary")
        )
    assert missing.value.code == "OBJECTIVE_NOT_FOUND"


def test_filing_leaves_the_experiment_record_as_it_was(experiments, objectives):
    aim = objective(objectives, "Aim 2 · Transport")
    record = experiment(experiments, "LOCO cptac")
    assert filed(record) == NONE and record["objective"] is None
    saved = experiments.file(
        record["id"],
        FileExperiment(
            objectiveId=aim["id"],
            role="generalization",
            series="  LOCO ",
            seriesMember="cptac",
            expected=NONE,
        ),
    )
    assert filed(saved) == {
        "objectiveId": aim["id"],
        "role": "generalization",
        "series": "LOCO",
        "seriesMember": "cptac",
    }
    assert saved["objective"] == {"id": aim["id"], "name": aim["name"], "rank": "secondary"}
    # Downstream records group by this map: experiment ID to filing.
    assert objectives.list()["filings"] == {record["id"]: filed(saved)}
    # The record is untouched: same revision, same update time, nothing in its payload.
    assert (saved["revision"], saved["updatedAt"]) == (record["revision"], record["updatedAt"])
    payload = experiments.store.get_draft(record["id"])["payload"]
    assert not set(NONE) & set(payload)
    # Filing it the same way again is a no-op, whatever the caller read; another filing
    # made from a stale read is refused.
    experiments.file(record["id"], FileExperiment(**filed(saved), expected=NONE))
    with pytest.raises(StorageError) as stale:
        experiments.file(record["id"], FileExperiment(role="control", expected=NONE))
    assert stale.value.code == "EXPERIMENT_FILING_CONFLICT"
    with pytest.raises(StorageError) as unknown:
        experiments.file(
            record["id"],
            FileExperiment(objectiveId="objective-" + "1" * 32, expected=filed(saved)),
        )
    assert unknown.value.code == "OBJECTIVE_NOT_FOUND"
    with pytest.raises(StorageError) as missing:
        experiments.file("draft-missing", FileExperiment(role="primary", expected=NONE))
    assert missing.value.code == "EXPERIMENT_NOT_FOUND"
    unfiled = experiments.file(record["id"], FileExperiment(expected=filed(saved)))
    assert filed(unfiled) == NONE
    assert json.loads((experiments.store.folder / OBJECTIVES_FILE).read_text())["experiments"] == {}


@pytest.mark.parametrize(
    "values",
    [
        {"seriesMember": "cptac"},
        {"role": "baseline"},
        {"objectiveId": "aim-1"},
        {"series": "x" * 81},
    ],
)
def test_filings_name_known_roles_and_members_of_a_series(values):
    with pytest.raises(ValidationError):
        ExperimentFiling.model_validate(values)


def test_new_experiments_are_filed_at_creation_and_copies_name_their_source(
    experiments, objectives
):
    aim = objective(objectives, "Aim 1", "primary")
    command = CreateModelExperiment(
        name="Main",
        operationId="create-main",
        setupVersion=1,
        objectiveId=aim["id"],
        role="primary",
    )
    main = experiments.create(command)
    assert (main["objectiveId"], main["role"]) == (aim["id"], "primary")
    assert experiments.create(command)["id"] == main["id"]
    copy = experiment(experiments, "Main, smaller", sourceExperimentId=main["id"], role="tuning")
    assert copy["copiedFrom"] == {"id": main["id"], "name": "Main"}
    assert (copy["objectiveId"], copy["role"]) == (None, "tuning")
    before = len(experiments.list()["items"])
    with pytest.raises(StorageError) as unknown:
        experiment(experiments, "Nowhere", objectiveId="objective-" + "2" * 32)
    assert unknown.value.code == "OBJECTIVE_NOT_FOUND"
    assert len(experiments.list()["items"]) == before
    # Without a filing, the creation request hashes as it always did.
    plain = experiment(experiments, "Plain")
    payload = experiments.store.get_draft(plain["id"])["payload"]
    assert not set(NONE) & set(payload)


def test_lineage_is_not_a_cleanup_dependency_and_archived_work_can_be_filed(
    experiments, objectives
):
    aim = objective(objectives, "Aim 1")
    source = experiment(experiments, "v1")
    copy = experiment(experiments, "v2", sourceExperimentId=source["id"])
    review = cleanup(experiments, "trash", source["key"])
    assert review["requiredKeys"] == []
    assert experiments.get(copy["id"])["copiedFrom"]["id"] == source["id"]
    cleanup(experiments, "archive", copy["key"])
    saved = experiments.file(
        copy["id"], FileExperiment(objectiveId=aim["id"], role="tuning", expected=NONE)
    )
    assert (saved["state"], saved["role"]) == ("archived", "tuning")


def test_removing_an_objective_unfiles_its_experiments_and_replays(experiments, objectives):
    aim = objective(objectives, "Aim 3", "exploratory")
    kept = objective(objectives, "Aim 1", "primary")
    fine = experiment(experiments, "fine", objectiveId=aim["id"], role="primary", series="Fine")
    other = experiment(experiments, "main", objectiveId=kept["id"], role="primary")
    with pytest.raises(StorageError) as stale:
        objectives.remove(aim["id"], RemoveObjective(expectedRevision=2, operationId="stale"))
    assert stale.value.code == "REVISION_CONFLICT"
    command = RemoveObjective(expectedRevision=1, operationId="remove-aim-3")
    removed = objectives.remove(aim["id"], command)
    assert removed == {"objectiveId": aim["id"], "name": "Aim 3", "unfiled": 1}
    assert objectives.remove(aim["id"], command) == removed
    with pytest.raises(StorageError) as gone:
        objectives.remove(aim["id"], RemoveObjective(expectedRevision=1, operationId="again"))
    assert gone.value.code == "OBJECTIVE_NOT_FOUND"
    # The experiment keeps its role and series for whichever objective it joins next.
    assert filed(experiments.get(fine["id"])) == {
        "objectiveId": None,
        "role": "primary",
        "series": "Fine",
        "seriesMember": None,
    }
    assert experiments.get(other["id"])["objectiveId"] == kept["id"]


def test_an_unreadable_objectives_file_lists_experiments_unfiled(experiments, objectives):
    aim = objective(objectives, "Aim 1")
    record = experiment(experiments, "main", objectiveId=aim["id"])
    (experiments.store.folder / OBJECTIVES_FILE).write_text('{"version": 2}')
    listed = experiments.list()
    assert "objectives file is invalid" in listed["objectivesError"]
    assert filed(listed["items"][0]) == NONE and listed["items"][0]["id"] == record["id"]
    with pytest.raises(StorageError) as corrupt:
        objective(objectives, "Aim 2")
    assert corrupt.value.code == "OBJECTIVES_CORRUPT"
    # A filed creation is refused before anything is saved; a copy is made without its note.
    before = len(experiments.list()["items"])
    with pytest.raises(StorageError) as refused:
        experiment(experiments, "Filed", role="primary")
    assert refused.value.code == "OBJECTIVES_CORRUPT"
    assert len(experiments.list()["items"]) == before
    copy = experiment(experiments, "Copied", sourceExperimentId=record["id"])
    assert copy["copiedFrom"] is None and len(experiments.list()["items"]) == before + 1


def test_objective_routes_and_filing_route(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        base = f"/api/v1/projects/{project}"
        created = service.http.post(
            f"{base}/objectives",
            json={"name": "Grade", "rank": "primary", "operationId": "o1"},
        )
        assert created.status_code == 201, created.text
        aim = created.json()
        assert (
            service.http.post(
                f"{base}/objectives", json={"name": "x", "rank": "main", "operationId": "o2"}
            ).status_code
            == 422
        )
        record = service.http.post(
            f"{base}/model-experiments",
            json={"name": "gej_consented", "operationId": "e1", "setupVersion": 1},
        ).json()
        put = service.http.put(
            f"{base}/model-experiments/{record['id']}/objective",
            json={"objectiveId": aim["id"], "role": "primary", "expected": NONE},
        )
        assert put.status_code == 200, put.text
        assert put.json()["objective"]["name"] == "Grade"
        listed = service.http.get(f"{base}/model-experiments?summary=true").json()["items"]
        assert listed[0]["role"] == "primary"
        patched = service.http.patch(
            f"{base}/objectives/{aim['id']}",
            json={"expectedRevision": 1, "name": "Dysplasia grade"},
        )
        assert patched.status_code == 200 and patched.json()["revision"] == 2
        assert (
            service.http.get(f"{base}/objectives").json()["items"][0]["name"] == "Dysplasia grade"
        )
        missing = service.http.get(f"{base}/objectives/objective-{'0' * 32}")
        assert missing.status_code == 404 and missing.json()["code"] == "OBJECTIVE_NOT_FOUND"
        removed = service.http.post(
            f"{base}/objectives/{aim['id']}/remove",
            json={"expectedRevision": 2, "operationId": "r1"},
        )
        assert removed.status_code == 200 and removed.json()["unfiled"] == 1


def test_the_cli_files_experiments_under_objectives(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        service.cli("use", project)
        base = f"/api/v1/projects/{project}/model-experiments"
        for name in ("A2 LOCO cptac", "A2 LOCO rih"):
            response = service.http.post(
                base, json={"name": name, "operationId": name, "setupVersion": 1}
            )
            assert response.status_code == 201, response.text

        unconfirmed = service.cli("objective", "create", "--name", "Aim 2 · Transport", "--json")
        assert unconfirmed.code == 7
        made = service.cli("objective", "create", "--name", "Aim 2 · Transport", "--yes", "--json")
        assert made.code == 0, made.stdout
        aim = made.envelope["data"]["result"]
        assert aim["rank"] == "secondary"
        bad = service.cli("objective", "create", "--name", "x", "--rank", "main", "--json")
        assert bad.code == 2

        filed_one = service.cli(
            "experiment",
            "assign",
            "A2 LOCO cptac",
            "--objective",
            "aim 2 · transport",
            "--role",
            "generalization",
            "--series",
            "LOCO",
            "--member",
            "cptac",
            "--json",
        )
        assert filed_one.code == 0, filed_one.stdout
        assert filed_one.envelope["data"]["seriesMember"] == "cptac"
        text = service.cli("experiment", "assign", "A2 LOCO cptac", "--role", "generalization")
        assert text.code == 0 and text.stdout.startswith("Unchanged: A2 LOCO cptac")
        assert service.cli("experiment", "assign", "A2 LOCO rih", "--member", "rih").code == 2
        assert service.cli("experiment", "assign", "A2 LOCO rih", "--clear", "tag").code == 2
        assert service.cli("experiment", "assign", "A2 LOCO rih", "--role", "x", "--json").code == 2

        listed = service.cli("experiment", "list", "--objective", "Aim 2 · Transport", "--json")
        assert [item["name"] for item in listed.envelope["data"]] == ["A2 LOCO cptac"]
        unfiled = service.cli("experiment", "list", "--unfiled", "--json")
        assert [item["name"] for item in unfiled.envelope["data"]] == ["A2 LOCO rih"]
        table = service.cli("experiment", "list")
        assert "OBJECTIVE" in table.stdout and "generalization" in table.stdout
        shown = service.cli("experiment", "show", "A2 LOCO cptac").stdout.splitlines()
        shown = [" ".join(line.split()) for line in shown]
        assert "Objective Aim 2 · Transport (secondary)" in shown
        assert "Role generalization" in shown and "Series LOCO › cptac" in shown

        counted = service.cli("objective", "list", "--json")
        assert counted.envelope["data"][0]["experiments"] == 1
        detail = service.cli("objective", "show", "Aim 2 · Transport")
        assert "A2 LOCO cptac" in detail.stdout and "generalization" in detail.stdout
        renamed = service.cli(
            "objective", "update", "Aim 2 · Transport", "--rank", "primary", "--yes", "--json"
        )
        assert renamed.code == 0, renamed.stdout
        cleared = service.cli(
            "experiment", "assign", "A2 LOCO cptac", "--clear", "series", "--json"
        )
        assert (cleared.envelope["data"]["series"], cleared.envelope["data"]["seriesMember"]) == (
            None,
            None,
        )
        removed = service.cli("objective", "remove", aim["id"], "--yes", "--json")
        assert removed.code == 0, removed.stdout
        assert removed.envelope["data"]["result"]["unfiled"] == 1
        assert service.cli("objective", "show", "Aim 2 · Transport", "--json").code == 5


def test_downstream_lists_follow_the_objective_of_their_experiment(tmp_path, monkeypatch):
    from histopilot.client.records import experiments_of

    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        service.cli("use", project)
        base = f"/api/v1/projects/{project}/model-experiments"
        made = {}
        for name in ("Primary grade", "Encoder comparison", "Loose pilot"):
            response = service.http.post(
                base, json={"name": name, "operationId": name, "setupVersion": 1}
            )
            assert response.status_code == 201, response.text
            made[name] = response.json()["id"]
        aim = service.cli("objective", "create", "--name", "Aim 1", "--yes", "--json")
        assert aim.code == 0, aim.stdout
        for name, role in (("Primary grade", "primary"), ("Encoder comparison", "comparison")):
            filed_one = service.cli(
                "experiment", "assign", name, "--objective", "Aim 1", "--role", role, "--json"
            )
            assert filed_one.code == 0, filed_one.stdout
        store = ScientificStore(service.settings.workspace / "projects" / "Study", project)
        data, _rows = fixtures.dataset(store)
        predictors = {}
        for name, identity in made.items():
            published = store.publish_configuration(
                manifest={
                    "kind": "frozen-predictor",
                    "datasetId": data["id"],
                    "name": f"{name} model",
                    "experimentId": identity,
                    "method": "ensemble",
                    "recipe": {"model": "abmil"},
                    "checkpoints": [],
                },
                operation_id=f"predictor-{name}",
            )
            predictors[name] = published["id"]

        def listed(*options):
            outcome = service.cli("predictor", "list", *options, "--json")
            assert outcome.code == 0, outcome.stdout
            return sorted(row["id"] for row in outcome.envelope["data"])

        assert listed("--objective", "aim 1") == sorted(
            predictors[name] for name in ("Primary grade", "Encoder comparison")
        )
        assert listed("--objective", "Aim 1", "--role", "comparison") == [
            predictors["Encoder comparison"]
        ]
        assert listed("--unfiled") == [predictors["Loose pilot"]]
        assert listed("--role", "primary") == [predictors["Primary grade"]]
        assert service.cli("predictor", "list", "--objective", "x", "--unfiled").code == 2
        assert service.cli("predictor", "list", "--role", "main").code == 2
        assert service.cli("predictor", "list", "--objective", "Aim 9").code == 5
        # Runs, Apply models batches and studies take the same filters; none exist yet.
        for noun in ("run", "apply", "interpretation", "batch"):
            outcome = service.cli(noun, "list", "--objective", "Aim 1", "--json")
            assert outcome.code == 0 and outcome.envelope["data"] == [], noun
        shown = service.cli("predictor", "show", predictors["Primary grade"]).stdout.splitlines()
        shown = [" ".join(line.split()) for line in shown]
        assert "Objective Aim 1" in shown and "Role primary" in shown
        described = service.cli("predictor", "show", predictors["Loose pilot"], "--json")
        assert described.envelope["data"]["model"]["objectiveId"] is None

    # An Apply models batch comes from every experiment its items name.
    batch = {"items": [{"experimentId": "b"}, {"experimentId": "a"}, {"experimentId": "b"}, {}]}
    assert experiments_of("apply", batch) == ["a", "b"]
    assert experiments_of("run", {"manifest": {"experimentId": "a"}}) == ["a"]
    assert experiments_of("run", {"manifest": {}}) == []
