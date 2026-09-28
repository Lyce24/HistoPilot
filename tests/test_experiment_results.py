"""Experiment results: fold, seed, seed-average and paired evidence from batch files."""

import json
import statistics

import numpy as np
import pytest

from histopilot import cv_summary as cv
from histopilot.application import experiment_results as results

CLASSES = ["HG", "LG", "ND"]


def target(unit="slide"):
    return {
        "task": "multiclass_classification",
        "unit": unit,
        "classes": CLASSES,
        "positiveClass": None,
        "field": "grade",
    }


def members(fold, folds, slides):
    """Blocks of three patients (two slides each) per fold, so every fold holds every class."""
    return [index for index in range(slides) if (index // 6) % folds == fold]


def batch_files(
    batch_id,
    *,
    seeds=(42, 43, 44),
    folds=3,
    slides=36,
    skill=1.5,
    weak=None,
    complete=True,
    unit="slide",
    split_unit="slide",
    configurations=1,
    selection=True,
    slide_prefix="s",
    noise=0,
    skills=None,
):
    """A finished (or partial) batch as the workers leave it: plan, run receipts, OOF files."""
    tgt = target(unit)
    splits = [
        {
            "id": f"split-{fold}",
            "planId": f"seed:42/fold:{fold}",
            "seed": 42,
            "fold": fold,
            "phase": "evaluation",
            "slideCount": slides,
            "partitions": {"test": len(members(fold, folds, slides)), "train": 0, "val": 0},
        }
        for fold in range(folds)
    ]
    configs = [
        {
            "id": f"cand-{number}",
            "number": number,
            "recipe": {
                "model": "abmil" if number == 1 else "nnmil",
                "decisionThreshold": 0.5,
                "patientAggregation": "mean_probabilities",
                "analysis": {
                    "bootstrapResamples": 300,
                    "bootstrapSeed": 5,
                    "confidenceLevel": 0.95,
                    "oneSlideSeed": 42,
                    "version": 1,
                },
            },
        }
        for number in range(1, configurations + 1)
    ]
    plan = {
        "target": tgt,
        "splitPlans": splits,
        "configurations": configs,
        "runs": [],
        **({"splitUnit": split_unit} if split_unit else {}),
    }
    state_runs, candidates, oof = [], [], {}
    for config in configs:
        for seed in seeds:
            rng = np.random.default_rng([seed, config["number"], noise])
            everything = []
            for fold in range(folds):
                run_id = f"run-{config['id']}-{seed}-{fold}"
                plan["runs"].append(
                    {
                        "id": run_id,
                        "candidateId": config["id"],
                        "trainingSeed": seed,
                        "splitPlanId": f"split-{fold}",
                    }
                )
                rows = []
                for index in members(fold, folds, slides):
                    label = (index // 2) % len(CLASSES)
                    logits = rng.normal(size=len(CLASSES))
                    strength = skills[config["number"] - 1] if skills else skill
                    logits[label] += 0 if CLASSES[label] == weak else strength
                    logs = logits - np.logaddexp.reduce(logits)
                    rows.append(
                        {
                            "slideId": f"{slide_prefix}{index:03d}",
                            "patientId": f"p{index // 2:03d}",
                            "patientIdSource": "source",
                            "labelIndex": label,
                            "label": CLASSES[label],
                            "probabilities": np.exp(logs).tolist(),
                            "logProbabilities": logs.tolist(),
                        }
                    )
                done = complete or fold < folds - 1 or seed != seeds[-1]
                scored = rows if unit == "slide" else cv.patient_rows(rows)
                state_runs.append(
                    {
                        "id": run_id,
                        "candidateId": config["id"],
                        "trainingSeed": seed,
                        "splitPlanId": f"split-{fold}",
                        "status": "completed" if done else "running",
                        **(
                            {
                                "metrics": {
                                    "assessment": {
                                        "unit": unit,
                                        "selected": cv.point_metrics(scored, tgt),
                                    }
                                },
                                "result": {
                                    "bestEpoch": 1 + fold,
                                    "epochsCompleted": 8 + fold,
                                    "bestValidationScore": 0.9,
                                    "checkpointMetric": "validation_auroc",
                                },
                            }
                            if done
                            else {}
                        ),
                    }
                )
                everything.extend(rows)
            group_complete = complete or seed != seeds[-1]
            candidate = {
                "candidateId": config["id"],
                "trainingSeed": seed,
                "splitSeed": 42,
                "complete": group_complete,
                "completedRuns": folds if group_complete else folds - 1,
                "totalRuns": folds,
                "metrics": None,
                "oofPath": None,
            }
            if group_complete:
                scored = everything if unit == "slide" else cv.patient_rows(everything)
                candidate.update(
                    metrics=cv.point_metrics(scored, tgt), oofPath=f"oof-{config['id']}-{seed}.json"
                )
                oof[(config["id"], seed, 42)] = everything
            candidates.append(candidate)
    counts = {
        "total": len(state_runs),
        "completed": sum(run["status"] == "completed" for run in state_runs),
    }
    state = {
        "status": "completed" if complete else "running",
        "runCounts": counts,
        "runs": state_runs,
    }
    chosen = configs[-1]["id"]
    selection_block = {
        "metric": "validation_auroc",
        "ready": True,
        "selectedCandidateId": chosen,
        "candidates": [
            {"candidateId": row["id"], "score": 0.8 + row["number"] / 100} for row in configs
        ],
    }
    result_file = {
        "status": state["status"],
        "candidates": candidates,
        **({"selection": selection_block} if selection else {}),
    }
    batch = {
        "id": batch_id,
        "name": f"Batch {batch_id}",
        "state": "active",
        "status": state["status"],
        "manifest": {
            "spec": {"batchName": f"Batch {batch_id}"},
            "summary": {"runCount": len(state_runs)},
        },
        "execution": state,
    }
    return {"batch": batch, "plan": plan, "state": state, "results": result_file, "oof": oof}


def summarize(*loaded):
    return results.summarize_experiment({"id": "experiment-1"}, list(loaded))


def only_config(summary, index=0):
    return summary["batches"][index]["configurations"][0]


def test_seed_average_is_the_mean_of_per_seed_oof_and_folds_keep_their_receipts():
    item = batch_files("b1")
    summary = summarize(item)
    config = only_config(summary)
    seeds = config["splitSeeds"][0]["seeds"]
    assert [row["trainingSeed"] for row in seeds] == [42, 43, 44]
    per_seed = [
        cv.point_metrics(item["oof"][("cand-1", seed, 42)], target())["auroc"]
        for seed in (42, 43, 44)
    ]
    assert [row["oof"]["auroc"] for row in seeds] == pytest.approx(per_seed)
    assert config["seedAverage"]["auroc"]["mean"] == pytest.approx(statistics.fmean(per_seed))
    assert config["seedAverage"]["auroc"]["sd"] == pytest.approx(statistics.stdev(per_seed))
    assert config["seedCount"] == config["plannedSeedCount"] == 3 and config["complete"]
    fold_values = [fold["metrics"]["auroc"] for row in seeds for fold in row["folds"]]
    assert config["foldCount"] == config["plannedFoldCount"] == 9
    assert config["foldAverage"]["auroc"]["mean"] == pytest.approx(statistics.fmean(fold_values))
    assert seeds[0]["foldStats"]["auroc"]["n"] == 3
    fold = seeds[0]["folds"][1]
    assert fold["fold"] == 1 and fold["bestEpoch"] == 2 and fold["epochsCompleted"] == 9
    assert fold["testCount"] == 12 and fold["metrics"]["perClass"][0]["label"] == "HG"
    assert summary["design"] == {
        "splitUnit": "slide",
        "groupByPatient": False,
        "folds": 3,
        "splitSeeds": [42],
        "slideCount": 36,
        "resamplingUnit": "slide",
    }
    assert summary["policy"]["resamples"] == 300 and summary["policy"]["seed"] == 5


def test_seed_ensemble_per_class_confusion_and_interval():
    item = batch_files("b1", slides=60)
    config = only_config(summarize(item))
    groups = [item["oof"][("cand-1", seed, 42)] for seed in (42, 43, 44)]
    expected = cv.point_metrics(cv.seed_ensemble(groups), target())
    assert config["ensemble"]["auroc"] == pytest.approx(expected["auroc"])
    assert [row["label"] for row in config["perClass"]] == CLASSES
    assert config["perClass"][0]["support"] == 20
    assert config["perClass"][0]["auroc"]["n"] == 3
    assert config["confusion"]["seeds"] == 3
    assert sum(config["confusion"]["rowRates"][0]) == pytest.approx(1)
    interval = config["intervals"]
    assert interval["available"] and interval["unit"] == "slide" and interval["units"] == 60
    auroc = interval["seedAverage"]["intervals"]["auroc"]
    assert auroc["lower"] <= config["seedAverage"]["auroc"]["mean"] <= auroc["upper"]
    assert interval["ensemble"]["available"]
    # Identical files give identical summaries: the policy seed fixes the draws.
    assert summarize(batch_files("b1", slides=60))["batches"] == summarize(item)["batches"]


def test_paired_comparison_shares_draws_and_folds():
    strong, weak = batch_files("strong", skill=2.5), batch_files("weak", skill=0.8, noise=1)
    summary = summarize(strong, weak)
    (comparison,) = summary["comparisons"]
    assert comparison["available"] and comparison["difference"] == "left_minus_right"
    assert comparison["leftBatchId"] == "strong" and comparison["rightBatchId"] == "weak"
    left, right = only_config(summary, 0), only_config(summary, 1)
    assert comparison["oof"]["auroc"]["difference"] == pytest.approx(
        left["seedAverage"]["auroc"]["mean"] - right["seedAverage"]["auroc"]["mean"]
    )
    interval = comparison["oofInterval"]
    assert interval["available"] and interval["unit"] == "slide"
    assert 0 < interval["intervals"]["auroc"]["lower"] < interval["intervals"]["auroc"]["upper"]
    folds = comparison["folds"]["auroc"]
    assert folds["n"] == 3 and folds["better"] == 3
    assert comparison["folds"]["loss"]["better"] == 3  # lower loss is better


def test_comparison_refuses_to_pair_different_slides():
    summary = summarize(batch_files("a"), batch_files("b", slide_prefix="t"))
    (comparison,) = summary["comparisons"]
    assert comparison["available"]
    assert not comparison["oofInterval"]["available"]
    assert "different assessment units" in comparison["oofInterval"]["reason"]


def test_partial_results_cover_finished_seeds_and_say_so():
    summary = summarize(batch_files("b1", complete=False))
    config = only_config(summary)
    assert config["seedCount"] == 2 and config["plannedSeedCount"] == 3 and not config["complete"]
    assert config["seedAverage"]["auroc"]["n"] == 2
    last = config["splitSeeds"][0]["seeds"][-1]
    assert (
        last["oof"] is None and last["completedRuns"] == 2 and last["folds"][-1]["metrics"] is None
    )
    assert config["foldCount"] == 8 and config["plannedFoldCount"] == 9
    assert "PARTIAL_RESULTS" in {finding["code"] for finding in summary["findings"]}


def test_findings_name_weak_classes_and_single_seeds():
    summary = summarize(batch_files("b1", weak="LG", slides=60), batch_files("b2", seeds=(42,)))
    codes = [(finding["batchId"], finding["code"]) for finding in summary["findings"]]
    assert ("b1", "CLASS_RECALL_LOW") in codes
    weak = next(finding for finding in summary["findings"] if finding["code"] == "CLASS_RECALL_LOW")
    assert weak["message"].startswith("Batch b1: LG recall is") and weak["severity"] == "warning"
    assert ("b2", "SINGLE_TRAINING_SEED") in codes
    assert ("b1", "EARLY_CHECKPOINTS") in codes  # best epochs 1-3 in every fold


def test_patient_level_scoring_resamples_patients():
    summary = summarize(batch_files("b1", unit="patient", split_unit=None, slides=40))
    config = only_config(summary)
    assert summary["design"]["resamplingUnit"] == "patient"
    assert config["intervals"]["unit"] == "patient" and config["intervals"]["units"] == 20
    assert config["splitSeeds"][0]["seeds"][0]["oof"]["count"] == 20


def test_selection_follows_validation_and_never_oof():
    summary = summarize(batch_files("b1", configurations=2, seeds=(42,)))
    batch = summary["batches"][0]
    assert batch["selection"]["source"] == "validation" and batch["selectedCandidateId"] == "cand-2"
    assert [row["selected"] for row in batch["configurations"]] == [False, True]
    assert batch["configurations"][1]["validationScore"] == pytest.approx(0.82)
    # Only the reported configuration gets intervals, the ensemble and per-class ranking.
    other, reported = batch["configurations"]
    assert reported["intervals"]["available"] and not other["intervals"]["available"]
    assert "reported configuration" in other["intervals"]["reason"]
    assert other["seedAverage"]["auroc"]["n"] == 1 and other["ensemble"] is None
    assert other["perClass"][0]["auroc"] is None and reported["perClass"][0]["auroc"] is not None
    fallback = summarize(batch_files("b1", configurations=2, seeds=(42,), selection=False))[
        "batches"
    ][0]
    assert (
        fallback["selection"]["source"] == "first" and fallback["selectedCandidateId"] == "cand-1"
    )
    assert "SELECTION_UNAVAILABLE" in {finding["code"] for finding in fallback["findings"]}


def test_loader_reads_predictions_of_the_reported_configuration_only(tmp_path):
    item = batch_files("grid", configurations=2, seeds=(42, 43))
    folder = tmp_path / "grid"
    folder.mkdir()
    (folder / "plan.json").write_text(json.dumps(item["plan"]))
    for candidate in item["results"]["candidates"]:
        records = item["oof"][(candidate["candidateId"], candidate["trainingSeed"], 42)]
        path = folder / candidate["oofPath"]
        path.write_text(json.dumps({"records": records}))
        candidate["oofPath"] = str(path)
    (folder / "results.json").write_text(json.dumps(item["results"]))
    loaded = results._load(folder, item["batch"])
    assert set(loaded["oof"]) == {("cand-2", 42, 42), ("cand-2", 43, 42)}
    assert summarize(loaded)["design"]["folds"] == 3


def test_loader_reads_every_arm_of_a_declared_comparison(tmp_path):
    item = batch_files("arms", configurations=2, seeds=(42, 43))
    item["batch"]["manifest"]["spec"]["comparison"] = {"reference": 1}
    folder = tmp_path / "arms"
    folder.mkdir()
    (folder / "plan.json").write_text(json.dumps(item["plan"]))
    for candidate in item["results"]["candidates"]:
        records = item["oof"][(candidate["candidateId"], candidate["trainingSeed"], 42)]
        path = folder / candidate["oofPath"]
        path.write_text(json.dumps({"records": records}))
        candidate["oofPath"] = str(path)
    (folder / "results.json").write_text(json.dumps(item["results"]))
    loaded = results._load(folder, item["batch"])
    assert {key[0] for key in loaded["oof"]} == {"cand-1", "cand-2"}


def test_declared_comparison_contrasts_every_arm_with_the_reference():
    item = batch_files("ablation", configurations=3, skills=(2.5, 0.4, 2.5))
    item["batch"]["manifest"]["spec"]["comparison"] = {"reference": 1, "primaryMetric": "auroc"}
    batch = summarize(item)["batches"][0]
    # The declared reference is reported, whatever validation would have chosen.
    assert batch["selectedCandidateId"] == "cand-1"
    assert batch["selection"]["source"] == "reference"
    assert all(row["intervals"]["available"] for row in batch["configurations"])
    comparison = batch["comparison"]
    assert (comparison["referenceNumber"], comparison["primaryMetric"]) == (1, "auroc")
    weaker, matched = comparison["contrasts"]
    assert (weaker["armNumber"], matched["armNumber"]) == (2, 3)
    assert weaker["difference"] == "reference_minus_arm"
    assert weaker["oof"]["auroc"]["difference"] > 0
    assert weaker["oofInterval"]["intervals"]["auroc"]["lower"] > 0
    assert weaker["pValue"] < 0.01
    for row in comparison["contrasts"]:
        assert row["pValueHolm"] >= row["pValue"]


def test_bootstrap_p_values_and_holm_adjustment():
    valid = np.ones(99, dtype=bool)
    assert cv.bootstrap_p_value(np.full(99, 0.1), valid) == pytest.approx(0.02)
    assert cv.bootstrap_p_value(np.linspace(-1, 1, 99), valid) == 1.0
    assert cv.holm([0.01, 0.04, 0.03, None]) == [0.03, 0.06, 0.06, None]


def test_batch_without_a_plan_is_listed_as_not_started():
    item = {
        "batch": {
            "id": "new",
            "name": "New",
            "state": "active",
            "status": "planned",
            "manifest": {"spec": {"batchName": "New"}, "summary": {"runCount": 15}},
        },
        "plan": None,
        "state": None,
        "results": None,
        "oof": {},
    }
    summary = summarize(item, batch_files("b1"))
    assert (
        summary["batches"][0]["configurations"] == []
        and summary["batches"][0]["progress"]["totalRuns"] == 15
    )
    assert summary["comparisons"][0]["available"] is False


def test_loading_reads_only_the_batch_folder_and_caches_by_file_identity(tmp_path, monkeypatch):
    item = batch_files("b1")
    folder = tmp_path / "training" / "b1"
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(item["plan"]))
    outside = tmp_path / "oof-elsewhere.json"
    outside.write_text(json.dumps({"records": item["oof"][("cand-1", 44, 42)]}))
    for candidate in item["results"]["candidates"]:
        records = item["oof"][(candidate["candidateId"], candidate["trainingSeed"], 42)]
        if candidate["trainingSeed"] == 44:
            candidate["oofPath"] = str(outside)  # results never name files outside their folder
        else:
            path = folder / candidate["oofPath"]
            path.write_text(json.dumps({"records": records}))
            candidate["oofPath"] = str(path)
    (folder / "results.json").write_text(json.dumps(item["results"]))
    loaded = results._load(folder, item["batch"])
    assert set(loaded["oof"]) == {("cand-1", 42, 42), ("cand-1", 43, 42)}

    calls = []

    class Store:
        pass

    store = Store()
    store.folder = tmp_path
    monkeypatch.setattr(results.ModelExperimentService, "__init__", lambda self, *_: None)
    monkeypatch.setattr(
        results.ModelExperimentService,
        "get",
        lambda self, identity: (
            calls.append(identity) or {"id": identity, "batches": [item["batch"]]}
        ),
    )
    monkeypatch.setattr(results, "_CACHE", type(results._CACHE)())
    computed = []
    real = results.summarize_experiment
    monkeypatch.setattr(
        results, "summarize_experiment", lambda *args: computed.append(1) or real(*args)
    )
    first = results.experiment_results(store, None, "experiment-1")
    second = results.experiment_results(store, None, "experiment-1")
    assert first is second and len(computed) == 1 and calls == ["experiment-1"] * 2
    # A seed without its OOF file keeps its recorded metrics but no per-class ranking rows.
    seed44 = first["batches"][0]["configurations"][0]["splitSeeds"][0]["seeds"][2]
    assert seed44["oof"]["auroc"] == pytest.approx(
        item["results"]["candidates"][2]["metrics"]["auroc"]
    )
    assert seed44["oof"]["perClass"][0].get("auroc") is None
    (folder / "results.json").write_text(
        json.dumps({**item["results"], "status": "completed"}) + " "
    )
    results.experiment_results(store, None, "experiment-1")
    assert len(computed) == 2  # a changed file is a new identity


def test_results_route_reads_the_experiment_and_unknown_ids_are_404(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects", json={"name": "Results", "storagePath": str(tmp_path / "project")}
        ).json()
        prefix = f"/api/v1/projects/{project['id']}/model-experiments"
        missing = client.get(f"{prefix}/draft-missing/results")
        assert missing.status_code == 404 and missing.json()["code"] == "EXPERIMENT_NOT_FOUND"
        seen = []
        monkeypatch.setattr(
            "histopilot.api.model_experiments.experiment_results",
            lambda store, _filesystem, identity: (
                seen.append((store.folder, identity)) or {"experimentId": identity, "batches": []}
            ),
        )
        response = client.get(f"{prefix}/draft-1/results")
        assert response.status_code == 200 and response.json() == {
            "experimentId": "draft-1",
            "batches": [],
        }
        assert seen == [(tmp_path / "project", "draft-1")]


def test_headlines_use_the_validation_choice_and_only_complete_seeds(tmp_path, monkeypatch):
    item = batch_files("b1", configurations=2, complete=False)
    folder = tmp_path / "training" / "b1"
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps(item["results"]))
    headline = results._headline(folder, item["batch"])
    assert headline["batchId"] == "b1" and headline["configurations"] == 2
    assert headline["seeds"] == 2 and headline["plannedSeeds"] == 3  # the running seed waits
    chosen = [
        row
        for row in item["results"]["candidates"]
        if row["candidateId"] == "cand-2" and row["complete"]
    ]
    assert headline["metrics"]["auroc"]["mean"] == pytest.approx(
        statistics.fmean(row["metrics"]["auroc"] for row in chosen)
    )
    assert headline["task"] == "multiclass_classification"
    assert results._headline(tmp_path / "missing", item["batch"]) is None

    class Store:
        folder = tmp_path

    listed = {
        "items": [
            {
                "id": "e1",
                "batches": [item["batch"], {**item["batch"], "id": "gone", "state": "trashed"}],
            },
            {"id": "e2", "batches": []},
        ]
    }
    monkeypatch.setattr(results.ModelExperimentService, "__init__", lambda self, *_: None)
    monkeypatch.setattr(results.ModelExperimentService, "list", lambda self, **_: listed)
    assert [
        row["experimentId"] for row in results.experiment_headlines(Store(), None)["items"]
    ] == ["e1"]


def test_headlines_route_is_not_taken_for_an_experiment_id(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects", json={"name": "Headlines", "storagePath": str(tmp_path / "p")}
        ).json()
        response = client.get(f"/api/v1/projects/{project['id']}/model-experiments/headlines")
        assert response.status_code == 200 and response.json() == {"items": []}


def test_older_plans_without_run_lists_fold_numbers_or_run_metrics_still_summarize():
    item = batch_files("old")
    item["plan"].pop("runs")
    for split in item["plan"]["splitPlans"]:
        split.pop("fold")
    for run in item["state"]["runs"]:
        # Adopted receipts keep metrics only inside the fold result.
        run["result"]["metrics"] = run.pop("metrics")
    config = only_config(summarize(item))
    assert config["foldCount"] == 9 and config["seedCount"] == 3
    assert [fold["fold"] for fold in config["splitSeeds"][0]["seeds"][0]["folds"]] == [0, 1, 2]
    assert config["splitSeeds"][0]["seeds"][0]["folds"][0]["metrics"]["auroc"] is not None
