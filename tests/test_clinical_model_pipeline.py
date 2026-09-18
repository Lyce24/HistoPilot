"""Immutable import → matched folds → saved predictors → external clinical inference."""

import json
import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.development import DevelopmentService  # noqa: E402
from histopilot.application.evaluation_runs import EvaluationRunService  # noqa: E402
from histopilot.application.evaluations import EvaluationService  # noqa: E402
from histopilot.application.feature_bundles import _hash  # noqa: E402
from histopilot.application.predictors import PredictorService  # noqa: E402
from histopilot.application.protocols import ProtocolService  # noqa: E402
from histopilot.application.refits import RefitService  # noqa: E402
from histopilot.application.training import TrainingService  # noqa: E402
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe  # noqa: E402
from histopilot.schemas.evaluations import InferenceSettings  # noqa: E402
from histopilot.schemas.predictors import (  # noqa: E402
    EvaluationRunSelection,
    FreezePredictor,
    LaunchRefit,
    PredictorSelection,
    SaveEvaluationRun,
)
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.scientific import ScientificStore  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402
from histopilot.workers.train_batch import _run_plan  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))
refit_support = runpy.run_path(str(Path(__file__).with_name("test_refit_predictors.py")))


def test_matched_clinical_models_keep_frozen_covariates_through_publication(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "clinical-pipeline")
    filesystem = LocalFilesystem((tmp_path,))
    rows = [
        {
            "slideId": f"s{i:03}",
            "patientId": f"p{i:03}",
            "patientIdSource": "crosswalk",
            "attributes": {
                "label": str(i % 2),
                "age": str(30 + i % 13),
                "site": "A" if i % 3 else "B",
                "cohort": "development" if i < 36 else "external",
            },
        }
        for i in range(48)
    ]
    draft = store.create_draft("import", "Clinical cohort", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        operation_id="clinical-data",
        manifest={
            "kind": "dataset",
            "dictionary": [
                {
                    "key": name,
                    "sourceColumn": name,
                    "owner": "patient",
                    "type": "number" if name == "age" else "text",
                }
                for name in ("label", "age", "site", "cohort")
            ],
        },
        artifacts={"records.json": json.dumps(rows).encode()},
    )
    bundle, _, _ = support["bundle"](store, tmp_path, dataset, [row["slideId"] for row in rows])
    protocols = ProtocolService(store, filesystem)
    draft = store.create_draft(
        "experiment",
        "Clinical comparison",
        {
            "type": "analysis-protocol",
            "spec": {
                "datasetId": dataset["id"],
                "target": support["TARGET"],
                "predictors": ["age", "site"],
                "eligibility": [{"field": "cohort", "op": "eq", "value": "development"}],
                "split": {
                    "version": 4,
                    "mode": "kfold",
                    "folds": 3,
                    "seeds": [42],
                    "pools": {"trainSelection": "remaining"},
                },
            },
        },
    )
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "clinical-protocol")
    fields = [{"field": "age", "kind": "numeric"}, {"field": "site", "kind": "categorical"}]
    recipes = [
        TrainingRecipe(
            inputMode=mode,
            clinicalFields=[] if mode == "image" else fields,
            maxEpochs=1,
            earlyStopping=False,
            checkpointMetric="validation_loss",
            embedDim=4,
            attentionDim=2,
            dropout=0,
            batchSize=8,
            analysis=None,
        )
        for mode in ("image", "clinical", "multimodal")
    ]
    spec = DevelopmentBatchSpec(
        experimentName="Matched modalities",
        batchName="All three arms",
        inputs={
            "protocolId": protocol["id"],
            "featureBundleId": bundle["id"],
            "loadingPolicy": "native",
        },
        recipe=recipes[0],
        mode="explicit",
        configurations=recipes,
        trainingSeeds=[7],
        candidateSelection="all",
        selectionMetric="validation_loss",
        resources={"gpuIds": [], "dataLoaderWorkers": 0, "cpuThreadsPerRun": 1, "ramGbPerRun": 0.1},
    )
    development = DevelopmentService(store, filesystem)
    preview = development.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    batch = development.freeze(
        spec, preview["previewHash"], "clinical-batch", {"tag": "Matched inputs"}
    )

    def runtime():
        return {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
            "findings": [],
        }

    service = TrainingService(store, filesystem, runtime=runtime)
    plan, _guard = service._prepare(batch)
    assert len(plan["data"]["clinicalValues"]) == 36
    states = []
    folder = store.folder / "training" / batch["id"]
    for run in plan["runs"]:
        selected = _run_plan(plan, run, None)
        run_folder = folder / "runs" / run["id"]
        result = train_fold(selected, run_folder)
        write_json(run_folder / "plan.json", selected)
        states.append({**run, "status": "completed", "result": result})
    write_json(folder / "plan.json", plan)
    write_json(
        folder / "state.json",
        {"batchId": batch["id"], "status": "completed", "planHash": _hash(plan), "runs": states},
    )
    cohorts = EvaluationService(store, filesystem)
    draft = support["draft"](
        cohorts,
        {
            "datasetId": dataset["id"],
            "target": support["TARGET"],
            "eligibility": [{"field": "cohort", "op": "eq", "value": "external"}],
        },
    )
    preview = cohorts.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    cohort = cohorts.freeze(draft["id"], 1, preview["previewHash"], "clinical-test")
    predictors = PredictorService(store, filesystem)
    evaluations = EvaluationRunService(store, filesystem)
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", runtime)
    memberships = []
    jobs = refit_support["FakeJobs"](store)
    refits = RefitService(store, filesystem, jobs)
    for configuration in batch["manifest"]["configurations"]:
        mode = configuration["recipe"].get("inputMode", "image")
        selection = PredictorSelection(
            experimentId=f"legacy-{batch['id']}",
            batchId=batch["id"],
            candidateId=configuration["id"],
            trainingSeed=7,
            splitSeed=42,
            name=mode,
        )
        preview = predictors.preview(selection)
        assert preview["canFreeze"], preview
        predictor = predictors.freeze(
            FreezePredictor(
                **selection.model_dump(),
                previewHash=preview["previewHash"],
                operationId=f"predictor-{mode}",
            )
        )
        refit_preview = predictors.preview(selection.model_copy(update={"method": "refit"}))
        assert refit_preview["canFreeze"], refit_preview
        assert (
            refit_preview["manifest"]["planTemplate"]["data"]["clinicalValues"]
            == plan["data"]["clinicalValues"]
        )
        choice = EvaluationRunSelection(
            predictorId=predictor["id"],
            cohortId=cohort["id"],
            featureBundleId=bundle["id"],
            name=mode,
            inference=InferenceSettings(
                device="cpu", decisionThreshold="predictor", patientAggregation="predictor"
            ),
        )
        preview = evaluations.preview(choice)
        assert preview["canSave"], preview
        saved = evaluations.save(
            SaveEvaluationRun(
                **choice.model_dump(),
                previewHash=preview["previewHash"],
                operationId=f"evaluation-{mode}",
            )
        )
        execution = evaluations._execution_plan(saved["id"])
        if mode != "image":
            assert execution["data"]["clinicalFields"] == fields
            assert set(execution["data"]["clinicalValues"]) == {row["slideId"] for row in rows[36:]}
            assert saved["manifest"]["clinical"]["valuesSha256"] == _hash(
                execution["data"]["clinicalValues"]
            )
        output = tmp_path / f"evaluation-{mode}"
        assert evaluate(execution, output)["patientCount"] == 12
        memberships.append(
            [
                row["patientId"]
                for row in json.loads((output / "predictions.json").read_text())["patientRecords"]
            ]
        )
        refit_selection = selection.model_copy(update={"method": "refit"})
        refit = refits.create(
            FreezePredictor(
                **refit_selection.model_dump(),
                previewHash=refit_preview["previewHash"],
                operationId=f"refit-{mode}",
            )
        )
        refits.launch(refit["id"], LaunchRefit(operationId=f"launch-refit-{mode}"))
        refit_folder = jobs.folder(refit["id"])
        refit_plan = json.loads((refit_folder / "plan.json").read_text())
        refit_result = train_refit(refit_plan, refit_folder)
        jobs.states[refit["id"]].update(status="completed", result=refit_result)
        refit_predictor = refits.publish(refit["id"], f"publish-refit-{mode}")
        if mode != "image":
            assert (
                refit_predictor["manifest"]["checkpoints"][0]["clinicalPreprocessing"][
                    "trainingPatientCount"
                ]
                == 36
            )
        refit_choice = choice.model_copy(update={"predictorId": refit_predictor["id"]})
        refit_review = evaluations.preview(refit_choice)
        assert refit_review["canSave"], refit_review
        refit_evaluation = evaluations.save(
            SaveEvaluationRun(
                **refit_choice.model_dump(),
                previewHash=refit_review["previewHash"],
                operationId=f"evaluation-refit-{mode}",
            )
        )
        refit_output = tmp_path / f"evaluation-refit-{mode}"
        assert (
            evaluate(evaluations._execution_plan(refit_evaluation["id"]), refit_output)[
                "patientCount"
            ]
            == 12
        )
        assert [
            row["patientId"]
            for row in json.loads((refit_output / "predictions.json").read_text())["patientRecords"]
        ] == memberships[-1]
    assert memberships[0] == memberships[1] == memberships[2]
