"""Run a real BLCA study through HistoPilot services and persistent native workers.

Use a new project only. Source slides/metadata/features are read-only. All study
choices, receipts and checks survive a coordinator restart; no HTTP server is run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histopilot.application.development import DevelopmentService  # noqa: E402
from histopilot.application.evaluation_runs import EvaluationRunService  # noqa: E402
from histopilot.application.evaluations import EvaluationService  # noqa: E402
from histopilot.application.feature_bundles import FeatureBundleService  # noqa: E402
from histopilot.application.feature_packs import FeaturePackService  # noqa: E402
from histopilot.application.features import FeatureService  # noqa: E402
from histopilot.application.imports import ImportService  # noqa: E402
from histopilot.application.local_workspace import LocalWorkspace  # noqa: E402
from histopilot.application.model_experiments import ModelExperimentService  # noqa: E402
from histopilot.application.predictors import PredictorService  # noqa: E402
from histopilot.application.project_workspace import ProjectWorkspace  # noqa: E402
from histopilot.application.protocols import ProtocolService  # noqa: E402
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe  # noqa: E402
from histopilot.schemas.feature_bundles import FeatureBundleSpec  # noqa: E402
from histopilot.schemas.feature_packs import FeaturePackSpec  # noqa: E402
from histopilot.schemas.features import FeatureSpec  # noqa: E402
from histopilot.schemas.model_experiments import (  # noqa: E402
    CreateModelExperiment,
    SubmitModelExperiment,
    UpdateModelExperiment,
)
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun  # noqa: E402
from histopilot.schemas.workspace import ProjectRequest  # noqa: E402
from histopilot.storage.database import Database  # noqa: E402
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.scientific import ScientificStore  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402


def stamp():
    return datetime.now(UTC).isoformat()


def log(message, **values):
    print(json.dumps({"time": stamp(), "message": message, **values}, allow_nan=False), flush=True)


class Study:
    def __init__(self, path, *, exercise_recovery=False):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "schemaVersion": 1,
            "createdAt": stamp(),
            "workspace": str(Path.home() / ".histopilot/workspace"),
            "projectPath": str(Path.home() / ".histopilot/workspace/blca-e2e-20260917"),
            "projectName": "BLCA end-to-end robustness · 2026-09-17",
            "metadataPath": "/mnt/d/YC.Liu/manifests/BLCA/blca.csv",
            "slideRoot": "/mnt/d/YC.Liu/slides/BLCA",
            "featurePath": "/mnt/d/YC.Liu/features/blca/20x_256px_0px_overlap/features_uni_v1",
            "checks": [],
        }
        self.filesystem = LocalFilesystem((
            Path(self.state["workspace"]), Path("/mnt/d/YC.Liu"),
            Path("/mnt/wsl/oceanpath-hot"), self.path.parent,
        ))
        self.exercise_recovery = exercise_recovery or self.state.get("trainingRecoveryRequested", False)
        if self.exercise_recovery:
            self.state["trainingRecoveryRequested"] = True
        self.save()

    def save(self, **values):
        self.state.update(values, updatedAt=stamp())
        write_json(self.path, self.state)

    def evidence(self, name, value):
        folder = self.path.parent / "evidence"
        folder.mkdir(exist_ok=True)
        write_json(folder / f"{name}.json", value)

    def check(self, stage, name, condition, **details):
        item = {"stage": stage, "name": name, "passed": bool(condition), **details}
        self.state["checks"] = [entry for entry in self.state["checks"]
                                if (entry["stage"], entry["name"]) != (stage, name)] + [item]
        self.save()
        log("check", **item)
        if not condition:
            raise AssertionError(f"{stage}: {name}")

    def services(self):
        self.store = ScientificStore(Path(self.state["projectPath"]), self.state["projectId"])
        self.imports = ImportService(self.store, self.filesystem)
        self.features = FeatureService(self.store, self.filesystem)
        self.packs = FeaturePackService(self.store, self.filesystem)
        self.bundles = FeatureBundleService(self.store, self.filesystem)
        self.protocols = ProtocolService(self.store, self.filesystem)
        self.cohorts = EvaluationService(self.store, self.filesystem)
        self.experiments = ModelExperimentService(self.store, self.filesystem)

    def project(self):
        database = Database(Path(self.state["workspace"]))
        database.initialize()
        try:
            legacy = LocalWorkspace(database, self.filesystem)
            projects = ProjectWorkspace(database, legacy, self.filesystem)
            destination = Path(self.state["projectPath"])
            if (destination / "histopilot-project.json").exists():
                project = projects.open(str(destination))
                if self.state.get("projectId") and project["id"] != self.state["projectId"]:
                    raise RuntimeError("The saved project identity changed.")
                if project["name"] != self.state["projectName"]:
                    raise RuntimeError("Refusing to use a different existing project.")
            else:
                project = projects.create(ProjectRequest(
                    name=self.state["projectName"], storagePath=str(destination),
                    description="Real BLCA workflow robustness study; slide/case identities only. "
                    "Grade-2 evaluation is exploratory due to prior historical model selection.",
                    dataPath=str(Path(self.state["metadataPath"]).parent),
                    slidePath=self.state["slideRoot"], featurePath=self.state["featurePath"],
                ))
            self.save(projectId=project["id"])
            reopened = projects.open(str(destination))
            self.check("project", "Folder reopen retains registered identity", reopened["id"] == project["id"])
        finally:
            database.close()
        self.services()

    def dataset(self):
        self.save(stage="dataset")
        spec = {
            "source": {"path": self.state["metadataPath"]},
            "slideIdColumn": "De ID", "patientIdColumn": None,
            "patientIdFallback": "slide_id", "slideRoot": self.state["slideRoot"],
            "recursive": False, "includeMissingSlides": False,
            "attributes": [
                {"key": "who2022_binary", "sourceColumn": "Binary WHO 2022", "type": "categorical", "categories": ["0", "1"]},
                {"key": "who2022_text", "sourceColumn": "WHO 2022", "type": "categorical"},
                {"key": "who1973", "sourceColumn": "WHO 1973", "type": "integer"},
                {"key": "historical_fold", "sourceColumn": "k_fold", "type": "integer"},
            ],
        }
        self.save(importSpec=spec)
        if not self.state.get("importDraftId"):
            draft = self.store.create_draft("import", "BLCA prepared manifest and original slides", {"type": "dataset-import", "spec": spec})
            self.save(importDraftId=draft["id"])
        draft_id = self.state["importDraftId"]
        if not self.state.get("datasetId"):
            preview = self.imports.preview(draft_id, 1)
            self.evidence("dataset-preview", preview)
            self.check("dataset", "Mapped data passes import preflight", preview["canFreeze"], findings=preview["findings"])
            self.check("dataset", "Repeated preview has identical content identity", self.imports.preview(draft_id, 1)["previewHash"] == preview["previewHash"])
            label = {"tag": "BLCA 138 original slides", "note": "De ID is slide/case; patient independence unverified."}
            document = self.imports.freeze(draft_id, 1, preview["previewHash"], "blca-dataset-v1", version_label=label)
            self.save(datasetId=document["id"])
            self.check("dataset", "Publication retry returns same dataset", self.imports.freeze(draft_id, 1, preview["previewHash"], "blca-dataset-v1", version_label=label)["id"] == document["id"])
        rows = self.imports.records(self.state["datasetId"])
        if isinstance(rows, dict):
            rows = rows.get("records", rows.get("items"))
        self.rows = rows
        self.check("dataset", "138 exact slides with explicit unverified identity", len(rows) == 138 and all(row["patientIdSource"] == "slide_fallback" for row in rows), slides=len(rows), verifiedPatients=0)
        self.save(targetSpec={
            "field": "who2022_binary", "task": "binary_classification", "unit": "slide",
            "classes": ["low_grade", "high_grade"], "labels": {"0": "low_grade", "1": "high_grade"},
            "positiveClass": "high_grade", "missing": "block", "unmapped": "block",
        })

    def feature_bundle(self):
        self.save(stage="features")
        spec = FeatureSpec(datasetId=self.state["datasetId"], path=self.state["featurePath"], encoderId="uni_v1", recursive=False)
        if not self.state.get("featureId"):
            preview = self.features.preview(spec)
            self.evidence("features-preview", preview)
            self.check("features", "Exact native features pass attachment preflight", preview["canFreeze"], findings=preview["findings"])
            feature = self.features.freeze(spec, preview["previewHash"], "blca-features-v1", version_label={"tag": "UNI v1 native float32", "note": "Original D: inventory; independent of the different hot-cache feature version."})
            self.save(featureId=feature["id"])
        feature = self.store.get_configuration(self.state["featureId"])
        self.check("features", "One feature file per frozen slide", {row["slideId"] for row in feature["manifest"]["files"]} == {row["slideId"] for row in self.rows})
        pack_path = str(Path(self.state["projectPath"]) / "feature-materializations" / "uni-v1-float32")
        self.save(packPath=pack_path)
        pack_spec = FeaturePackSpec(featureSetId=feature["id"], action="pack", outputPath=pack_path, dtype="preserve")
        if not self.state.get("packJobId"):
            preview = self.packs.preview(pack_spec)
            self.evidence("packing-preview", preview)
            self.check("features", "Full validation and lossless pack preflight", preview["canRun"], findings=preview["findings"])
            job = self.packs.submit(pack_spec, preview["previewHash"], "blca-pack-v1")
            self.save(packJobId=job["id"])
            retry = self.packs.submit(pack_spec, preview["previewHash"], "blca-pack-v1")
            self.check("features", "Packing launch retry does not duplicate work", retry["id"] == job["id"])
        last = None
        while True:
            job = next(row for row in self.packs.list()["jobs"] if row["id"] == self.state["packJobId"])
            if job["state"] != last:
                log("packing", status=job["state"], job=job["id"])
                last = job["state"]
            if job["state"] == "succeeded":
                result = self.packs._result(self.packs._job_record(job["id"]))
                self.evidence("packing-result", result)
                self.save(packId=result["artifact"]["id"])
                break
            if job["state"] in {"failed", "cancelled", "interrupted"}:
                self.evidence("packing-failure", job)
                raise RuntimeError(f"Packing stopped: {job['state']}")
            time.sleep(3)
        if not self.state.get("bundleId"):
            request = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[self.state["packId"]])
            preview = self.bundles.preview(request)
            self.evidence("bundle-preview", preview)
            self.check("features", "Full validation permits frozen feature bundle", preview["canFreeze"], findings=preview["findings"])
            bundle = self.bundles.freeze(request, preview["previewHash"], "blca-bundle-v1", version_label={"tag": "Verified UNI v1 + float32 pack"})
            self.save(bundleId=bundle["id"], featureBundleId=bundle["id"])
        self.check("features", "Frozen bundle remains current", self.bundles.get(self.state["bundleId"])["current"])

    def design(self):
        self.save(stage="target-and-protocol")
        spec = {
            "datasetId": self.state["datasetId"], "target": self.state["targetSpec"],
            "predictors": [], "eligibility": [{"field": "who1973", "op": "in", "value": [1, 3]}],
            "featureBundleId": self.state["bundleId"], "featureCoverage": "require",
            "split": {"version": 4, "mode": "kfold", "folds": 5, "seeds": [42], "validationFraction": .2, "pools": {"trainSelection": "remaining"}},
        }
        if not self.state.get("protocolDraftId"):
            draft = self.store.create_draft("experiment", "WHO 2022 grading · grades 1/3 development", {"type": "analysis-protocol", "spec": spec})
            self.save(protocolDraftId=draft["id"])
        if not self.state.get("protocolId"):
            preview = self.protocols.preview(self.state["protocolDraftId"], 1)
            self.evidence("protocol-preview", preview)
            self.check("target", "Explicit slide target and five-fold design pass", preview["canFreeze"], findings=preview["findings"])
            self.check("target", "Split preview is reproducible", preview["previewHash"] == self.protocols.preview(self.state["protocolDraftId"], 1)["previewHash"])
            protocol = self.protocols.freeze(self.state["protocolDraftId"], 1, preview["previewHash"], "blca-protocol-v1", version_label={"tag": "Grades 1+3 · five development folds"})
            self.save(protocolId=protocol["id"])
        if not self.state.get("cohortDraftId"):
            cohort_spec = {
                "datasetId": self.state["datasetId"], "target": self.state["targetSpec"],
                "eligibility": [{"field": "who1973", "op": "eq", "value": 2}],
                "featureBundleId": self.state["bundleId"],
                "inference": {"device": "cuda", "loadingPolicy": "packed", "packArtifactId": self.state["packId"], "batchSize": 1, "patientAggregation": "mean", "decisionThreshold": .5},
            }
            draft = self.store.create_draft("experiment", "Grade 2 · prespecified exploratory evaluation", {"type": "evaluation-cohort", "spec": cohort_spec})
            self.save(cohortDraftId=draft["id"])
        if not self.state.get("cohortId"):
            preview = self.cohorts.preview(self.state["cohortDraftId"], 1)
            self.evidence("cohort-preview", preview)
            self.check("target", "Grade-2 cohort frozen before fitting", preview["canFreeze"], findings=preview["findings"])
            cohort = self.cohorts.freeze(self.state["cohortDraftId"], 1, preview["previewHash"], "blca-cohort-v1", version_label={"tag": "76 grade-2 slides · exploratory"})
            self.save(cohortId=cohort["id"])
        protocol = self.store.get_configuration(self.state["protocolId"])["manifest"]
        cohort = self.store.get_configuration(self.state["cohortId"])["manifest"]
        development = {row["slideId"] for row in protocol["memberships"]}
        external = {row["slideId"] for row in cohort["memberships"]}
        self.check("target", "62 development and 76 evaluation slides are disjoint", len(development) == 62 and len(external) == 76 and not development.intersection(external), developmentSlides=len(development), evaluationSlides=len(external))
        self.save(studyPlan={
            "unit": "slide", "verifiedPatients": 0, "developmentSlides": 62,
            "evaluationSlides": 76, "splitSeed": 42, "trainingSeed": 42,
            "primaryPredictor": "five-fold probability ensemble", "secondaryPredictor": "median-best-epoch full-development refit",
            "modelSelection": "One fixed default ABMIL recipe; validation loss checkpointing; no search or test-set tuning.",
            "evaluationStatus": "Exploratory; grade-2 data has historical model-selection exposure.",
            "attentionSelection": "Up to eight distinct evaluated slides: confident FP/FN, correct high/low, then class-balanced closest to threshold 0.5. No retraining from selected examples.",
        })
        self.evidence("prespecified-study-plan", self.state["studyPlan"])

    def train(self):
        self.save(stage="experiments")
        from blca_final_analysis import prepare
        prepare(self.path)
        self.state = json.loads(self.path.read_text())
        inputs = {"protocolId": self.state["protocolId"], "featureBundleId": self.state["bundleId"], "loadingPolicy": "mmap", "packArtifactId": self.state["packId"]}
        policy = {"method": "both", "refitPercentile": 50.0}
        if not self.state.get("experimentId"):
            experiment = self.experiments.create(CreateModelExperiment(
                name="BLCA UNI v1 · fixed ABMIL · complete workflow", inputs=inputs,
                predictorPolicy=policy, operationId="blca-experiment-v1",
                notes="Exploratory slide-level analysis. Fixed recipe, no grade-2 tuning. Ensemble primary; refit secondary. Patient identities unverified.",
                tags=["BLCA", "end-to-end", "robustness", "slide-level"],
            ))
            self.save(experimentId=experiment["id"])
        experiment = self.experiments.get(self.state["experimentId"])
        if not self.state.get("batchPlanSaved"):
            recipe = TrainingRecipe(model="abmil", maxEpochs=40, earlyStopping=True, patience=8, checkpointMetric="validation_loss", decisionThreshold=.5)
            spec = DevelopmentBatchSpec(
                experimentId=experiment["id"], experimentRevision=experiment["revision"], experimentName=experiment["name"], batchName="Prespecified ABMIL baseline", inputs=inputs,
                recipe=recipe, trainingSeeds=[42], candidateSelection="all", selectionMetric="validation_loss", predictorPolicy=policy,
                resources={"gpuIds": [0], "maxConcurrentRuns": 1, "runsPerGpu": 1, "dataLoaderWorkers": 0, "cpuThreadsPerRun": 4, "ramGbPerRun": 8.0},
            )
            preview = DevelopmentService(self.store, self.filesystem).preview(spec)
            self.evidence("batch-preview", preview)
            self.check("experiments", "Fixed recipe passes training preflight", preview["canFreeze"], findings=preview["findings"])
            experiment = self.experiments.update(experiment["id"], UpdateModelExperiment(
                name=experiment["name"], expectedRevision=experiment["revision"], inputs=inputs, predictorPolicy=policy, notes=experiment["notes"], tags=experiment["tags"],
                batchPlans=[{"id": "fixed-abmil", "spec": spec.model_dump()}],
            ))
            self.save(batchPlanSaved=True, batchSpec=spec.model_dump())
        if not self.state.get("submitted"):
            request = SubmitModelExperiment(expectedRevision=experiment["revision"], operationId="blca-submit-v1")
            experiment = self.experiments.submit(experiment["id"], request)
            self.evidence("experiment-submission", experiment)
            if experiment.get("submission", {}).get("status") != "submitted":
                raise RuntimeError("Experiment submission needs recovery; see experiment-submission evidence.")
            retry = self.experiments.submit(experiment["id"], request)
            self.check("experiments", "Submission retry retains experiment identity", retry["id"] == experiment["id"])
            self.save(submitted=True)
        last = None
        while True:
            batches = [row for row in self.store.list_configurations("mil-batch") if row["manifest"]["spec"].get("experimentId") == self.state["experimentId"]]
            if len(batches) != 1:
                raise RuntimeError(f"Expected exactly one submitted batch, found {len(batches)}")
            batch = batches[0]
            self.save(batchId=batch["id"])
            training = self.experiments.training.execution(batch["id"])
            predictor_work = self.experiments._predictors().status(self.state["experimentId"])
            if training is None or predictor_work is None:
                self.evidence("experiment-missing-execution", self.experiments.get(self.state["experimentId"]))
                raise RuntimeError("A submitted worker has no execution receipt; review saved experiment evidence.")
            summary = (training["status"], predictor_work["status"], sum(row["status"] == "completed" for row in training.get("runs", [])))
            if summary != last:
                log("training-and-predictors", training=summary[0], predictors=summary[1], completeFolds=summary[2])
                self.evidence("training-status", training)
                self.evidence("predictor-status", predictor_work)
                last = summary
            if self.exercise_recovery:
                from blca_training_recovery import advance
                if advance(self, training, predictor_work):
                    time.sleep(5)
                    continue
            if training["status"] == "completed" and predictor_work["status"] == "completed":
                break
            if training["status"] in {"failed", "cancelled", "interrupted"} or predictor_work["status"] in {"failed", "cancelled", "interrupted", "attention"}:
                raise RuntimeError(f"Study worker requires recovery: {summary}")
            time.sleep(5)
        predictors = [row for row in PredictorService(self.store, self.filesystem).list()["items"] if row["manifest"]["experimentId"] == self.state["experimentId"]]
        ids = {row["manifest"]["method"]: row["id"] for row in predictors}
        self.check("experiments", "Five folds produce ensemble and refit", len(predictors) == 2 and set(ids) == {"ensemble", "refit"})
        self.save(predictorIds=ids, predictorId=ids["ensemble"])

    def evaluate(self):
        self.save(stage="evaluation")
        service = EvaluationRunService(self.store, self.filesystem)
        for method, predictor_id in self.state["predictorIds"].items():
            if method not in self.state.get("evaluationIds", {}):
                selection = EvaluationRunSelection(predictorId=predictor_id, cohortId=self.state["cohortId"], featureBundleId=self.state["bundleId"], name=f"Grade 2 · {method} · exploratory")
                preview = service.preview(selection)
                self.evidence(f"evaluation-{method}-preview", preview)
                self.check("evaluation", f"{method} independent-of-current-fit inputs pass", preview["canSave"], findings=preview["findings"])
                request = SaveEvaluationRun(**selection.model_dump(), previewHash=preview["previewHash"], operationId=f"blca-evaluate-{method}")
                document = service.save(request)
                self.check("evaluation", f"{method} evaluation save is idempotent", service.save(request)["id"] == document["id"])
                self.save(evaluationIds={**self.state.get("evaluationIds", {}), method: document["id"]})
            identity = self.state["evaluationIds"][method]
            manifest = service.get(identity)["manifest"]
            inference = manifest["inference"]
            self.check("evaluation", f"{method} preserves packed CUDA inference and cutoff", inference["device"] == "cuda" and inference["loadingPolicy"] == "packed" and inference["packArtifactId"] == self.state["packId"] and inference["decisionThreshold"] == .5)
            status = service.execution(identity)
            if status["status"] == "not_started":
                status = service.launch(identity, f"blca-inference-{method}")
                retry = service.launch(identity, f"blca-inference-{method}")
                self.check("evaluation", f"{method} launch retry reuses plan", retry["planHash"] == status["planHash"])
            while status["status"] != "completed":
                if status["status"] in {"failed", "cancelled", "interrupted"}:
                    self.evidence(f"evaluation-{method}-failure", status)
                    raise RuntimeError(f"Evaluation {method} requires recovery: {status['status']}")
                time.sleep(5)
                status = service.execution(identity)
            self.evidence(f"evaluation-{method}-status", status)
            predictions = json.loads(service.artifact(identity, "predictions.json"))
            self.check("evaluation", f"{method} predicts every grade-2 slide once", len(predictions["records"]) == 76 and len({row["slideId"] for row in predictions["records"]}) == 76)
        self.save(evaluationId=self.state["evaluationIds"]["ensemble"])

    def finalize(self):
        from blca_final_analysis import finalize
        self.save(stage="final-analysis-and-attention")
        last = None
        while True:
            try:
                result = finalize(self.path)
            finally:
                self.state = json.loads(self.path.read_text())
            self.evidence("finalize-return", result)
            status = self.state.get("finalAnalysisStatus")
            if status != last:
                log("final-analysis-and-attention", status=status)
                last = status
            if status == "complete":
                history = list(self.state.get("errorHistory", []))
                if self.state.get("lastError") and self.state["lastError"] not in history:
                    history.append(self.state["lastError"])
                self.save(stage="complete", lastError=None, errorHistory=history)
                return
            if status in {"failed", "cancelled", "interrupted", "attention_needs_review", "waiting_for_evaluations"}:
                raise RuntimeError(f"Attention requires recovery: {status}")
            time.sleep(15)

    def robustness(self):
        from blca_robustness import run_checks
        for stage in ("dataset", "target", "features", "experiments", "evaluation", "attention"):
            result = run_checks(self.path, stage)
            log("robustness-stage", stage=stage, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / ".local/blca-e2e-20260917/state.json")
    parser.add_argument("--phase", choices=("prepare", "run", "complete", "finalize", "robustness"), default="prepare")
    parser.add_argument("--exercise-recovery", action="store_true", help="Interrupt and resume one owned training worker after a saved checkpoint.")
    args = parser.parse_args()
    study = Study(args.state, exercise_recovery=args.exercise_recovery)
    try:
        if args.phase == "prepare":
            study.project()
            study.dataset()
            study.feature_bundle()
            study.design()
            study.save(stage="prepared")
        else:
            study.services()
            if args.phase in {"run", "complete"}:
                study.train()
                study.evaluate()
                study.save(stage="evaluated")
                if args.phase == "complete":
                    study.finalize()
            elif args.phase == "finalize":
                study.finalize()
            else:
                study.robustness()
        log("phase-completed", phase=args.phase, state=str(study.path))
    except Exception as error:
        # Helpers checkpoint publication and launch receipts before later work.
        # Preserve those acknowledgements when recording a recoverable failure.
        study.state = json.loads(study.path.read_text())
        study.save(lastError={"type": type(error).__name__, "message": str(error), "at": stamp()})
        log("phase-failed", phase=args.phase, errorType=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
