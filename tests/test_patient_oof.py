"""OOF patient evidence is complete, reusable and guarded against cache corruption."""

import copy
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.training import module  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402
from histopilot.workers.train_batch import collect_results  # noqa: E402
from histopilot.workers.training_process import read_json  # noqa: E402


@pytest.mark.parametrize("corrupt_value", [0, float("nan")])
def test_completed_oof_patients_have_intervals_and_only_verified_analysis_is_reused(
    tmp_path, monkeypatch, corrupt_value
):
    target = {
        "unit": "patient",
        "task": "binary_classification",
        "classes": ["a", "b"],
        "positiveClass": "b",
    }
    rows = [
        {
            "slideId": f"s{i}",
            "patientId": f"p{i}",
            "label": target["classes"][i % 2],
            "labelIndex": i % 2,
            "probabilities": [0.8, 0.2] if i % 2 == 0 else [0.3, 0.7],
        }
        for i in range(8)
    ]
    runs, memberships = [], {}
    for fold in range(2):
        split = f"fold-{fold}"
        selected = rows[fold * 4 : fold * 4 + 4]
        memberships[split] = [{**row, "partition": "test"} for row in selected]
        path = tmp_path / f"assessment-{fold}.json"
        write_json(path, {"classOrder": target["classes"], "records": selected})
        runs.append(
            {
                "id": split,
                "candidateId": "candidate",
                "splitPlanId": split,
                "trainingSeed": 42,
                "status": "completed",
                "result": {"predictions": {"assessment": str(path)}},
            }
        )
    plan = {
        "batchId": "batch",
        "protocolId": "protocol",
        "target": target,
        "configurations": [
            {"id": "candidate", "number": 1, "recipe": {"analysis": {"bootstrapResamples": 200}}}
        ],
        "memberships": memberships,
        "splitPlans": [{"id": f"fold-{fold}", "seed": 42} for fold in range(2)],
    }
    state = {"status": "completed", "runs": runs}
    calls = []
    original = module.classification_metrics

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "classification_metrics", measured)
    collect_results(plan, state, tmp_path)
    result = read_json(tmp_path / "results.json")["candidates"][0]
    assert result["metricDetails"]["patientAnalysis"]["uncertainty"]["patientCount"] == 8
    assert result["metrics"]["confidenceIntervals"]["auroc"] == {"lower": 1, "upper": 1}
    collect_results(plan, state, tmp_path)
    assert len(calls) == 1
    assert read_json(tmp_path / "results.json")["candidates"][0] == result
    path = tmp_path / result["oofPath"].split("/")[-1]
    damaged = copy.deepcopy(read_json(path))
    damaged["summary"]["patientAnalysis"]["uncertainty"]["intervals"]["auroc"]["lower"] = corrupt_value
    path.write_text(json.dumps(damaged))
    collect_results(plan, state, tmp_path)
    assert len(calls) == 2
    assert read_json(tmp_path / "results.json")["candidates"][0] == result
