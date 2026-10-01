"""Trainable development designs beyond generated k-fold.

Predefined folds (fold assignments imported with the dataset), leave one site out and a
held-out assessment derive their plans from the frozen training set as k-fold does. Each
trains real models, collects out-of-fold predictions over exactly the units it assessed,
and builds predictors. Monte Carlo and nested designs stay planning-only.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from support import projects  # noqa: E402
from support.training import runtime  # noqa: E402

from histopilot.application.development import DevelopmentService  # noqa: E402
from histopilot.application.development_splits import training_split_issue  # noqa: E402
from histopilot.application.model_experiments import ModelExperimentService  # noqa: E402
from histopilot.application.predictors import PredictorService  # noqa: E402
from histopilot.application.protocols import ProtocolService  # noqa: E402
from histopilot.application.target_splits import TargetSplitService  # noqa: E402
from histopilot.application.training import TrainingService, plan_label  # noqa: E402
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe  # noqa: E402
from histopilot.schemas.model_experiments import (  # noqa: E402
    ConfigureModelExperimentSetup,
    CreateModelExperiment,
)
from histopilot.schemas.predictors import FreezePredictor, PredictorSelection  # noqa: E402
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.io import content_hash, read_json_bounded, write_json_atomic  # noqa: E402
from histopilot.storage.scientific import ScientificStore  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.workers.train_batch import collect_results, execute_plan  # noqa: E402

# Imported fold numbers, deliberately out of text order ("10" sorts before "2" as text).
FOLDS = ("1", "2", "10")
ROWS = [
    {
        "slideId": f"s{i:02}",
        "patientId": f"p{i:02}",
        "patientIdSource": "crosswalk",
        "attributes": {"label": str(i % 2), "site": "ABC"[i // 12], "fold": FOLDS[i % 3]},
    }
    for i in range(36)
]
SITE = {row["slideId"]: row["attributes"]["site"] for row in ROWS}
FOLD = {row["slideId"]: row["attributes"]["fold"] for row in ROWS}
POOLS = {"trainSelection": "remaining"}


@pytest.fixture
def study(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "training-designs")
    filesystem = LocalFilesystem((tmp_path,))
    draft = store.create_draft("import", "Sites and folds", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        operation_id="dataset",
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                for key in ("label", "site", "fold")
            ],
        },
        artifacts={"records.json": json.dumps(ROWS).encode()},
    )
    bundle = projects.slide_bundle(store, filesystem, tmp_path, dataset, ROWS, "slide-features")
    return store, filesystem, dataset, bundle


def review(study, split, target=None):
    store, filesystem, dataset, _bundle = study
    protocols = ProtocolService(store, filesystem)
    draft = store.create_draft(
        "experiment",
        "Design",
        {
            "type": "analysis-protocol",
            "spec": {
                "datasetId": dataset["id"],
                "target": target or projects.TARGET,
                "predictors": [],
                "eligibility": [],
                "split": {"version": 4, "pools": POOLS, **split},
            },
        },
    )
    return protocols, draft, protocols.preview(draft["id"], 1)


def codes(preview, severity="error"):
    return {item["code"] for item in preview["findings"] if item["severity"] == severity}


def train(study, protocol):
    """Freeze one small batch on ``protocol``, train every run and collect its OOF results."""
    store, filesystem, _dataset, bundle = study
    recipe = TrainingRecipe(
        model="slide_linear",
        maxEpochs=1,
        earlyStopping=False,
        checkpointMetric="validation_loss",
        batchSize=8,
        analysis=None,
    )
    spec = DevelopmentBatchSpec(
        experimentName="Designs",
        batchName="Linear probe",
        inputs={
            "protocolId": protocol["id"],
            "featureBundleId": bundle["id"],
            "loadingPolicy": "native",
        },
        recipe=recipe,
        trainingSeeds=[7],
        selectionMetric="validation_loss",
    )
    development = DevelopmentService(store, filesystem)
    preview = development.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    batch = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Design"})
    plan, _guard = TrainingService(store, filesystem, runtime=runtime)._prepare(batch)
    folder = store.folder / "training" / batch["id"]
    states = []
    for run in plan["runs"]:
        selected = execute_plan(plan, run, None)
        run_folder = folder / "runs" / run["id"]
        result = train_fold(selected, run_folder)
        write_json_atomic(run_folder / "plan.json", selected)
        states.append({**run, "status": "completed", "result": result})
    state = {
        "batchId": batch["id"],
        "status": "completed",
        "planHash": content_hash(plan),
        "runs": states,
    }
    write_json_atomic(folder / "plan.json", plan)
    write_json_atomic(folder / "state.json", state)
    collect_results(plan, state, folder)
    return batch, plan, read_json_bounded(folder / "results.json")


@pytest.mark.parametrize(
    "split,plans,assessed",
    [
        # Every slide is assessed once: by the model of the fold or site it is not in.
        ({"mode": "predefined_folds", "foldField": "fold"}, 3, 36),
        ({"mode": "leave_one_domain_out", "domainField": "site"}, 3, 36),
        # Only the held-out sites, or the held-out share, are assessed.
        (
            {
                "mode": "leave_one_domain_out",
                "domainField": "site",
                "domainPolicy": "selected",
                "heldOutDomains": ["C"],
            },
            1,
            12,
        ),
        # A quarter of each class's 18 slides, rounding halves up: 5 + 5.
        ({"mode": "held_out", "testFraction": 0.25}, 1, 10),
    ],
    ids=["predefined-folds", "every-site", "selected-site", "held-out"],
)
def test_each_trainable_design_trains_collects_its_assessed_units_and_builds_predictors(
    study, split, plans, assessed
):
    protocols, draft, preview = review(study, {"seeds": [42], **split})
    assert preview["canFreeze"], preview["findings"]
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "design")
    batch, plan, results = train(study, protocol)
    assert len(plan["splitPlans"]) == plans
    oof = read_json_bounded(results["oof"][0]["path"])
    tested = {row["slideId"] for row in oof["records"]}
    assert len(oof["records"]) == len(tested) == assessed
    assert results["candidates"][0]["complete"]
    assert results["candidates"][0]["metrics"]["count"] == assessed
    memberships = protocol["manifest"]["memberships"]
    for split_plan in plan["splitPlans"]:
        test = {
            row["slideId"]
            for row in memberships
            if row["planId"] == split_plan["planId"] and row["partition"] == "test"
        }
        fitting = {
            row["slideId"]
            for row in memberships
            if row["planId"] == split_plan["planId"] and row["partition"] in {"train", "val"}
        }
        assert test and fitting and not test & fitting
        if split["mode"] == "leave_one_domain_out":
            assert {SITE[slide] for slide in test} == {split_plan["domain"]}
            assert split_plan["domain"] not in {SITE[slide] for slide in fitting}
            assert plan_label(split_plan) == f"Held-out {split_plan['domain']}"
        elif split["mode"] == "predefined_folds":
            assert {FOLD[slide] for slide in test} == {FOLDS[split_plan["fold"]]}
            assert split_plan["planId"] == f"seed:42/fold:{split_plan['fold']}"
        else:
            assert split_plan["fold"] is None
            assert plan_label(split_plan) == "Held-out assessment"
    predictors = PredictorService(*study[:2])
    selection = PredictorSelection(
        experimentId=f"legacy-{batch['id']}",
        batchId=batch["id"],
        candidateId=batch["manifest"]["configurations"][0]["id"],
        trainingSeed=7,
        splitSeed=42,
        name="Design ensemble",
    )
    reviewed = predictors.preview(selection)
    assert reviewed["canFreeze"], reviewed["findings"]
    predictor = predictors.freeze(
        FreezePredictor(
            **selection.model_dump(), previewHash=reviewed["previewHash"], operationId="ensemble"
        )
    )
    assert len(predictor["manifest"]["checkpoints"]) == plans
    # A refit trains on every development slide, whichever units the design assessed.
    refit = predictors.preview(selection.model_copy(update={"method": "refit"}))
    assert refit["canFreeze"], refit["findings"]


def test_predefined_folds_take_their_values_in_numeric_order_and_record_them(study):
    _protocols, _draft, preview = review(
        study, {"mode": "predefined_folds", "foldField": "fold", "seeds": [42, 43]}
    )
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["predefinedFolds"] == [
        {"fold": 0, "value": "1"},
        {"fold": 1, "value": "2"},
        {"fold": 2, "value": "10"},
    ]
    assert preview["summary"]["oofCoverage"]["complete"]
    # The folds repeat for each split seed; only early-stop validation changes.
    assert "PREDEFINED_FOLDS_REUSED" in codes(preview, "warning")
    tests = {}
    for row in preview["memberships"]:
        assert row["foldValue"] == FOLDS[row["fold"]]
        if row["partition"] == "test":
            tests.setdefault(row["seed"], {})[row["slideId"]] = row["fold"]
    assert tests[42] == tests[43]


@pytest.mark.parametrize(
    "split,code",
    [
        # Every group needs exactly one fold value.
        ({"foldField": "site", "mode": "predefined_folds"}, None),
        ({"foldField": "label", "mode": "predefined_folds"}, "FOLD_TARGET_LEAKAGE"),
        ({"foldField": "slideId", "mode": "predefined_folds"}, "INVALID_FOLD_FIELD"),
        ({"foldField": "missing", "mode": "predefined_folds"}, "UNKNOWN_FIELD"),
    ],
)
def test_a_fold_column_is_any_assignment_column_but_never_the_target_or_an_identifier(
    study, split, code
):
    _protocols, _draft, preview = review(study, {"seeds": [42], **split})
    if code is None:
        # Sites can serve as folds too: a column name is not what makes a fold.
        assert preview["canFreeze"], preview["findings"]
    else:
        assert code in codes(preview)
        assert not preview["canFreeze"]


def test_predefined_folds_need_one_value_per_group_and_at_least_two_folds(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "folds")
    rows = [
        {
            "slideId": f"s{i}",
            # Slides s0 and s1 belong to one patient in different folds; s2 has no fold.
            "patientId": "shared" if i < 2 else f"p{i}",
            "patientIdSource": "crosswalk",
            "attributes": {
                "label": "0" if i < 2 else str(i % 2),
                "fold": "" if i == 2 else str(i % 2),
                "single": "only",
            },
        }
        for i in range(12)
    ]
    draft = store.create_draft("import", "Folds", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        operation_id="dataset",
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                for key in ("label", "fold", "single")
            ],
        },
        artifacts={"records.json": json.dumps(rows).encode()},
    )
    study = (store, LocalFilesystem((tmp_path,)), dataset, None)
    target = {**projects.TARGET, "missing": "block", "unmapped": "block"}
    _p, _d, preview = review(study, {"mode": "predefined_folds", "foldField": "fold"}, target)
    assert {"MISSING_FOLD", "INCONSISTENT_GROUP_FOLD"} <= codes(preview)
    _p, _d, single = review(study, {"mode": "predefined_folds", "foldField": "single"}, target)
    assert "INSUFFICIENT_FOLDS" in codes(single)


@pytest.mark.parametrize(
    "split,issue",
    [
        ({"mode": "kfold", "seeds": [1, 2]}, None),
        ({"mode": "predefined_folds", "seeds": [1, 2]}, None),
        ({"mode": "leave_one_domain_out", "seeds": [1, 2]}, None),
        ({"mode": "held_out", "seeds": [1]}, None),
        # A held-out set per seed would assess different units in each seed group.
        ({"mode": "held_out", "seeds": [1, 2]}, "TRAINING_SPLIT_UNSUPPORTED"),
        ({"mode": "monte_carlo", "seeds": [1]}, "TRAINING_SPLIT_UNSUPPORTED"),
        ({"mode": "nested_kfold", "seeds": [1]}, "NESTED_SELECTION_REQUIRED"),
    ],
)
def test_only_designs_that_assess_each_unit_once_per_seed_train(split, issue):
    found = training_split_issue({"version": 4, **split})
    assert (found[0] if found else None) == issue


@pytest.mark.parametrize(
    "split",
    [
        {"mode": "predefined_folds", "foldField": "fold"},
        {"mode": "leave_one_domain_out", "domainField": "site"},
    ],
    ids=["predefined-folds", "every-site"],
)
def test_an_experiment_design_assesses_every_training_slide_and_never_a_testing_one(study, split):
    store, filesystem, dataset, bundle = study
    targets = TargetSplitService(store, filesystem)
    draft = store.create_draft(
        "experiment",
        "Train and test",
        {
            "type": "target-split",
            "spec": {
                "datasetId": dataset["id"],
                "target": projects.TARGET,
                "split": {"method": "random", "testFraction": 0.25, "seed": 42},
            },
        },
    )
    reviewed = targets.preview(draft["id"], 1)
    assert reviewed["canFreeze"], reviewed["findings"]
    target_split = targets.freeze(draft["id"], 1, reviewed["previewHash"], "target-split")
    service = ModelExperimentService(store, filesystem)
    record = service.create(
        CreateModelExperiment(
            name="Design",
            operationId="create",
            setupVersion=1,
            predictorPolicy={"method": "skip", "refitPercentile": None},
        )
    )
    saved = service.setup_inputs(
        record["id"],
        ConfigureModelExperimentSetup(
            expectedRevision=record["revision"],
            datasetId=dataset["id"],
            targetSplitId=target_split["id"],
            featureBundleId=bundle["id"],
            trainingSplit={"version": 4, "seeds": [42], "pools": POOLS, **split},
        ),
    )
    assert saved["setupDesign"]["trainingSplit"]["mode"] == split["mode"]
    protocol = store.get_configuration(saved["inputs"]["protocolId"])["manifest"]
    training = {
        row["slideId"]
        for row in target_split["manifest"]["memberships"]
        if row["partition"] == "train"
    }
    tested = [row["slideId"] for row in protocol["memberships"] if row["partition"] == "test"]
    # Each frozen training slide is assessed once; the testing set stays reserved.
    assert sorted(tested) == sorted(training)


def test_results_describe_sites_and_held_out_sets_as_what_they_assess():
    from histopilot.application.experiment_results import _design, _fold

    def design(plans):
        return _design([{"plan": {"splitPlans": plans}}])

    sites = [
        {"id": f"site-{name}", "seed": 42, "fold": index, "domain": name}
        for index, name in enumerate("AB")
    ]
    held_out = [{"id": "held", "seed": 42, "fold": None, "planId": "seed:42/held_out"}]
    folds = [{"id": f"fold-{index}", "seed": 42, "fold": index} for index in range(3)]
    assert design(sites)["strategy"] == "leave_one_domain_out"
    assert design(held_out) | {"strategy": "held_out", "folds": 1} == design(held_out)
    assert design(folds)["strategy"] == "folds"
    target = {"classes": ["low", "high"]}
    assert _fold(sites[1], {}, target)["domain"] == "B"
    assert "domain" not in _fold(folds[0], {}, target)
