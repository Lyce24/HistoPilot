"""A worker authenticates the evaluation's chosen pack, independently of cohort setup."""

import copy
import runpy
from pathlib import Path

import pytest

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.predictors import feature_contract, reference
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.pack_import import _layout
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.compute_job import verify_plan_inputs
from histopilot.workers.pack_features import run_job

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))


@pytest.fixture
def packed_evaluation(tmp_path):
    (tmp_path / "project").mkdir()
    store = ScientificStore(tmp_path / "project", "pack-binding-test")
    data, rows = support["dataset"](store)
    initial, first_id, _ = support["bundle"](
        store, tmp_path, data, [row["slideId"] for row in rows], pack=True,
    )
    feature = store.get_configuration(initial["manifest"]["feature"]["id"])
    filesystem = LocalFilesystem((tmp_path,))
    packing = FeaturePackService(store, filesystem, support["FakeExecutor"]())
    spec = FeaturePackSpec(featureSetId=feature["id"], action="pack", dtype="float16",
                           outputPath=str(tmp_path / "second-pack"))
    job = packing.submit(spec, packing.preview(spec)["previewHash"], "second-pack")
    result = run_job(packing.folder / job["id"] / "plan.json")
    assert result["state"] == "succeeded"
    selected = result["artifact"]
    assert selected["id"] != first_id
    bundles = FeatureBundleService(store, filesystem)
    spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[first_id, selected["id"]])
    bundle = bundles.freeze(spec, bundles.preview(spec)["previewHash"], "both-packs")
    target = support["TARGET"]

    def publish(kind, **values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": data["id"], **values}, operation_id=kind,
        )

    predictor = publish("frozen-predictor", target=target, method="ensemble", checkpoints=[])
    memberships = [{"slideId": row["slideId"], "patientId": row["patientId"], "label": "low"}
                   for row in rows]

    def make_plan(*, historical=False):
        cohort = store.publish_configuration(manifest={
            "kind": "evaluation-cohort", "datasetId": data["id"], "memberships": memberships,
            "pack": packing.artifact(first_id) if historical else None,
        }, operation_id=f"cohort-{historical}")
        inference = {"loadingPolicy": "packed", "packArtifactId": selected["id"]}
        manifest = {"kind": "model-evaluation", "datasetId": data["id"],
                    "predictorId": predictor["id"], "cohortId": cohort["id"],
                    "features": feature_contract(feature, bundle), "target": target,
                    "inference": inference}
        evaluation = store.publish_configuration(manifest=manifest, operation_id=f"evaluation-{historical}")
        files = {row["slideId"]: row for row in feature["manifest"]["files"]}
        return {"kind": "evaluation", "recordId": evaluation["id"],
                "recordContentHash": evaluation["contentHash"], "projectFolder": str(store.folder),
                "projectId": store.project_id, "target": target, "method": "ensemble",
                "checkpoints": [], "inference": inference,
                "references": [reference(row) for row in (predictor, cohort, feature, bundle)],
                "data": {"memberships": memberships, "featureDim": 4, "featureFiles": files,
                         "sourceStamps": {row["path"]: row for row in files.values()},
                         "loadingPolicy": "mmap", "packPath": selected["outputPath"],
                         "packStamps": _layout(Path(selected["outputPath"]))["packStamps"]}}

    return make_plan, packing.artifact(first_id)


@pytest.mark.parametrize("historical", [False, True])
def test_worker_uses_saved_evaluation_pack_for_independent_and_overridden_cohorts(packed_evaluation, historical):
    make_plan, _ = packed_evaluation
    verify_plan_inputs(make_plan(historical=historical))


def test_other_valid_pack_cannot_replace_reviewed_evaluation_selection(packed_evaluation):
    make_plan, other = packed_evaluation
    plan = make_plan(historical=True)
    plan["data"].update(packPath=other["outputPath"],
                        packStamps=_layout(Path(other["outputPath"]))["packStamps"])
    with pytest.raises(ValueError, match="loading contract changed"):
        verify_plan_inputs(plan)


def test_changed_loading_policy_cannot_hide_pack_binding(packed_evaluation):
    make_plan, _ = packed_evaluation
    plan = copy.deepcopy(make_plan())
    plan["data"].update(loadingPolicy="native", packPath=None, packStamps=None)
    with pytest.raises(ValueError, match="loading contract changed"):
        verify_plan_inputs(plan)
