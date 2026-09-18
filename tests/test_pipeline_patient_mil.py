"""Real frozen inputs through five-fold slide fitting and both reporting units."""

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from test_evaluations import TARGET, bundle, dataset
from test_training_execution import FakeExecutor, runtime

pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.development import DevelopmentService  # noqa: E402
from histopilot.application.protocols import ProtocolService  # noqa: E402
from histopilot.application.training import TrainingService  # noqa: E402
from histopilot.application.training_exports import training_oof_csv  # noqa: E402
from histopilot.schemas.development import DevelopmentBatchSpec  # noqa: E402
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.scientific import ScientificStore  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.module import aggregate_patients  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402
from histopilot.workers.train_batch import _run_plan, collect_results  # noqa: E402
from histopilot.workers.training_process import read_json  # noqa: E402


def test_independent_bundle_patient_kfold_slide_training_and_both_result_units(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-pipeline-audit")
    filesystem = LocalFilesystem((tmp_path,))
    rows = [
        {"slideId": f"p{i:02}-s{j}", "patientId": f"p{i:02}",
         "attributes": {"label": str(i % 2), "cohort": "TCGA" if (i // 2) % 2 else "SurGen"}}
        for i in range(30) for j in range(1 + i % 3)
    ]
    eligible_ids = {row["slideId"] for row in rows}
    # Publish/verify features before any dataset exists. One source slide has no
    # dataset row; two dataset rows are ineligible for different explicit reasons.
    features, _, _ = bundle(store, tmp_path, {"id": None},
                            sorted(eligible_ids | {"source-only", "rih-slide"}))
    rows.extend([
        {"slideId": "no-features", "patientId": "p-uncovered",
         "attributes": {"label": "unmapped", "cohort": "TCGA"}},
        {"slideId": "rih-slide", "patientId": "p-rih",
         "attributes": {"label": None, "cohort": "RIH"}},
    ])
    data, _ = dataset(store, rows=rows)
    spec = {
        "datasetId": data["id"], "featureBundleId": features["id"], "target": TARGET,
        "eligibility": [{"field": "cohort", "op": "in", "value": ["TCGA", "SurGen"]}],
        "split": {"version": 4, "mode": "kfold", "folds": 5, "seeds": [42],
                  "pools": {"trainSelection": "remaining"}},
    }
    draft = store.create_draft("experiment", "Patient folds", {"type": "analysis-protocol", "spec": spec})
    protocols = ProtocolService(store, filesystem)
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["includedPatients"] == 30
    assert preview["summary"]["includedSlides"] == 60
    assert preview["summary"]["featureExclusions"] == 1
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "patient-protocol")
    assessed = Counter()
    for fold in range(5):
        grouped = defaultdict(set)
        selected = [row for row in protocol["manifest"]["memberships"] if row["fold"] == fold]
        assert {row["slideId"] for row in selected} == eligible_ids
        for row in selected:
            grouped[row["patientId"]].add(row["partition"])
        assert len(grouped) == 30 and all(len(roles) == 1 for roles in grouped.values())
        for role in ("train", "val", "test"):
            assert {row["label"] for row in selected if row["partition"] == role} == set(TARGET["classes"])
        assessed.update(patient for patient, roles in grouped.items() if roles == {"test"})
    assert set(assessed.values()) == {1} and len(assessed) == 30

    common = {"maxEpochs": 1, "earlyStopping": False, "batchSize": 5,
              "accumulateGradBatches": 2, "bagSize": 2, "embedDim": 4,
              "attentionDim": 2, "dropout": 0, "analysis": {"bootstrapResamples": 200},
              "decisionThreshold": 0.7}
    batch_spec = DevelopmentBatchSpec(
        experimentName="Pipeline audit", batchName="Both MIL models",
        inputs={"protocolId": protocol["id"], "featureBundleId": features["id"]},
        mode="explicit", configurations=[{**common, "model": "abmil"},
                                         {**common, "model": "nnmil", "bagSizeMode": "training_median"}],
        resources={"gpuIds": [], "cpuThreadsPerRun": 1, "dataLoaderWorkers": 0},
    )
    development = DevelopmentService(store, filesystem)
    review = development.preview(batch_spec)
    assert review["canFreeze"], review["findings"]
    assert len(review["runs"]) == 10 and len(review["nnmilPlanning"]) == 5
    frozen = development.freeze(batch_spec, review["previewHash"], "both-models", {"tag": "Audit"})
    monkeypatch.setattr("histopilot.application.training.gpu_snapshot", lambda: {"gpus": []})
    training = TrainingService(store, filesystem, executor=FakeExecutor(), runtime=runtime)
    plan, _ = training._prepare(frozen)
    assert set(plan["data"]["featureFiles"]) == eligible_ids
    output = folder / "training" / frozen["id"]
    output.mkdir(parents=True)
    state = {"status": "completed", "runs": []}
    for run in plan["runs"]:
        worker = _run_plan(plan, run, None)
        run_folder = output / "runs" / run["id"]
        result = train_fold(worker, run_folder)
        assert result["state"] == "succeeded" and result["checkpointUnit"] == "patient"
        assert result["patientAggregation"] == "mean_probabilities"
        for role, partition in (("validation", "val"), ("assessment", "test")):
            selected = [row for row in worker["data"]["memberships"] if row["partition"] == partition]
            metrics = result["metrics"][role]
            assert metrics["slide"]["count"] == len(selected)
            assert metrics["patient"]["count"] == len({row["patientId"] for row in selected})
            assert metrics["selected"] == metrics["patient"]
        predictions = read_json(Path(result["predictions"]["assessment"]))["records"]
        for patient in aggregate_patients(predictions):
            slides = [row for row in predictions if row["patientId"] == patient["patientId"]]
            np.testing.assert_allclose(patient["probabilities"], np.mean([row["probabilities"] for row in slides], axis=0))
        write_json(run_folder / "plan.json", worker)
        state["runs"].append({**run, "status": "completed", "result": result})
    collect_results(plan, state, output)
    write_json(output / "plan.json", plan)
    write_json(output / "state.json", state)
    results = read_json(output / "results.json")
    assert len(results["candidates"]) == 2
    for candidate in results["candidates"]:
        assert candidate["complete"] and candidate["assessmentSlideCount"] == 60
        metrics = candidate["metricDetails"]
        assert metrics["patient"]["count"] == 30 and metrics["slide"]["count"] == 60
        assert metrics["patientAnalysis"]["uncertainty"]["patientCount"] == 30
        assert set(metrics["patient"]["confidenceIntervals"]) >= {"auroc", "auprc"}
        records = read_json(Path(candidate["oofPath"]))["records"]
        assert {row["slideId"] for row in records} == eligible_ids and len(records) == 60
        for unit, count in (("slide", 60), ("patient", 30)):
            exported = training_oof_csv(store, frozen["id"], candidate["candidateId"],
                                        candidate["trainingSeed"], candidate["splitSeed"], unit)
            parsed = list(csv.DictReader(io.StringIO(exported.decode())))
            assert len(parsed) == count
            assert {int(row["assessmentFold"]) for row in parsed} == set(range(5))
    assert not training.executor.launches
