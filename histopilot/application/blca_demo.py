"""Reproducible, source-free BLCA walkthrough.

Only aggregate study structure is encoded here. Every identifier, prediction,
curve, resource sample and artifact reference is newly invented. This module
never opens a clinical table, slide, checkpoint, feature tensor or local project.

Regenerate the packaged fixture explicitly with:
    python -m histopilot.application.blca_demo --write
Without --write, the command prints the generated workspace to standard output.
"""

import argparse
import json
import math
import random
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path

DEMO_ID = "blca-demo-v1"
SEED = 20260912
STAMP = "2026-09-12T12:00:00Z"
RESOURCE = "blca_demo_workspace.json"
SYNTHETIC_NOTICE = (
    "Synthetic illustration, not measured performance. All identifiers, predictions, "
    "loss curves, resource samples and artifact references are invented. No training runs."
)
GROUPING_NOTICE = (
    "Groups are synthetic slide fallbacks. No verified patient identifiers are provided; "
    "slide-disjoint folds do not establish patient independence."
)
EXPORT_NOTICE = {
    "mode": "synthetic-demo",
    "synthetic": True,
    "executable": False,
    "notice": SYNTHETIC_NOTICE,
}


def demo_summary() -> dict:
    return {
        "id": DEMO_ID,
        "name": "BLCA demo",
        "description": "A synthetic walkthrough based on the Bladder project's aggregate design.",
        "storagePath": "",
        "mode": "synthetic-demo",
        "createdAt": STAMP,
        "updatedAt": STAMP,
        "config": {
            "task": "binary_classification",
            "targetColumn": "grade_binary",
            "positiveLabel": "high",
            "seed": 42,
            "folds": 5,
            "encoderId": "uni",
            "milId": "abmil",
        },
        "sources": [],
        "available": True,
    }


def load_demo() -> dict:
    """Read the shipped fixture afresh; older SQLite seeds cannot shadow it."""
    return json.loads(files("histopilot.resources").joinpath(RESOURCE).read_text())


def synthetic_cases() -> list[dict]:
    cases = []
    # Counts describe the workflow, never retain a correspondence to source rows.
    for partition, grade, label, count in (
        ("development", "1", "low", 33),
        ("development", "3", "high", 29),
        ("test", "2", "low", 54),
        ("test", "2", "high", 22),
    ):
        for _ in range(count):
            number = len(cases) + 1
            cases.append(
                {
                    "id": f"BLCA-DEMO-{number:03d}",
                    "caseId": f"SYNTHETIC-CASE-{number:03d}",
                    "grade": grade,
                    "label": label,
                    "partition": partition,
                    "groupId": f"slide:BLCA-DEMO-{number:03d}",
                }
            )
    return cases


def predictions(
    cases: list[dict], *, seed: int, shift: float = 0, signal: float = 0.24
) -> list[dict]:
    """Invent overlapping score distributions without looking up any real row."""
    rng = random.Random(seed)
    rows = []
    for case in cases:
        label = int(case["label"] == "high")
        score = min(0.99, max(0.01, 0.42 + shift + signal * label + rng.uniform(-0.40, 0.40)))
        rows.append({"id": case["id"], "label": label, "probability": round(score, 6)})
    return rows


def metrics(rows: list[dict], threshold: float = 0.5) -> dict:
    """Exact binary metrics from the released synthetic prediction rows."""
    positive = [row["probability"] for row in rows if row["label"] == 1]
    negative = [row["probability"] for row in rows if row["label"] == 0]
    tp = sum(row["label"] == 1 and row["probability"] >= threshold for row in rows)
    fp = sum(row["label"] == 0 and row["probability"] >= threshold for row in rows)
    tn, fn = len(negative) - fp, len(positive) - tp
    auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    # Average precision uses tied-score groups, matching a stepwise PR curve.
    seen = hits = 0
    average_precision = 0.0
    for score in sorted({row["probability"] for row in rows}, reverse=True):
        group = [row for row in rows if row["probability"] == score]
        new_hits = sum(row["label"] for row in group)
        seen += len(group)
        hits += new_hits
        average_precision += new_hits / len(positive) * hits / seen
    # Python's built-in float summation changed in 3.12. Keep the packaged
    # synthetic fixture identical across supported interpreter versions.
    loss = -math.fsum(
        math.log(row["probability"] if row["label"] else 1 - row["probability"]) for row in rows
    ) / len(rows)
    f1_high = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    f1_low = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    return {
        "available": True,
        "count": len(rows),
        "loss": loss,
        "accuracy": (tp + tn) / len(rows),
        "balancedAccuracy": (tp / len(positive) + tn / len(negative)) / 2,
        "macroF1": (f1_high + f1_low) / 2,
        "auroc": auc / (len(positive) * len(negative)),
        "auprc": average_precision,
        "classCounts": {"low": len(negative), "high": len(positive)},
        "missingClasses": [],
        "confusionMatrix": [[tn, fp], [fn, tp]],
    }


def metric_details(rows: list[dict]) -> dict:
    return {
        "unit": "slide",
        "classOrder": ["low", "high"],
        "positiveClass": "high",
        "patientAggregation": "not applicable: no verified patient identifiers",
        "slide": metrics(rows),
        "patient": {"available": False, "reason": "Verified patient identifiers unavailable"},
        "selected": metrics(rows),
    }


def roc_points(rows: list[dict]) -> list[dict]:
    points = [{"x": 0.0, "y": 0.0}]
    for threshold in sorted({row["probability"] for row in rows}, reverse=True):
        result = metrics(rows, threshold)
        (tn, fp), (fn, tp) = result["confusionMatrix"]
        points.append({"x": fp / (fp + tn), "y": tp / (tp + fn)})
    return points


def facts(**values: object) -> list[dict]:
    return [
        {"label": label.replace("_", " "), "value": str(value)} for label, value in values.items()
    ]


def table(columns: list[str], rows: list, title: str | None = None) -> dict:
    return {"columns": columns, "rows": rows, **({"title": title} if title else {})}


def score_rows(rows: list[dict]) -> list[list]:
    return [
        [
            row["id"],
            "high" if row["label"] else "low",
            row["probability"],
            "high" if row["probability"] >= 0.5 else "low",
        ]
        for row in rows
    ]


def record(identity: str, module: str, name: str, description: str, steps: list, tags=None) -> dict:
    return {
        "id": identity,
        "module": module,
        "name": name,
        "description": description,
        "tags": tags or ["Synthetic", "Read only"],
        "steps": steps,
    }


def _recipe() -> dict:
    return {
        "model": "abmil",
        "learningRate": 0.0003,
        "weightDecay": 0.0001,
        "maxEpochs": 40,
        "optimizer": "adamw",
        "batchSize": 1,
        "bagSize": 4096,
        "earlyStopping": True,
        "patience": 8,
        "checkpointMetric": "validation_loss",
        "embedDim": 512,
        "attentionDim": 384,
        "numFcLayers": 1,
        "gatedAttention": True,
        "dropout": 0.25,
        "inputDropout": 0,
        "precision": "32-true",
        "gradientCheckpointing": False,
        "gradientClipNorm": 0,
        "accumulateGradBatches": 1,
        "lrScheduler": "none",
        "warmupEpochs": 0,
        "minEpochs": 1,
    }


def assessment_folds(cases: list[dict]) -> dict[str, int]:
    membership = {}
    for label in ("low", "high"):
        group = [case for case in cases if case["label"] == label]
        random.Random(42).shuffle(group)
        membership.update({case["id"]: index % 5 for index, case in enumerate(group)})
    return membership


def _training(version: int, cases: list[dict]) -> tuple[dict, list[dict]]:
    batch_id = f"blca-demo-batch-v{version}"
    candidate = f"blca-demo-config-v{version}"
    recipe = _recipe()
    resources = {
        "maxConcurrentRuns": 1,
        "gpuIds": [0],
        "runsPerGpu": 1,
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 2,
        "ramGbPerRun": 8,
    }
    inputs = {
        "protocolId": "blca-protocol",
        "featureBundleId": "blca-features",
        "loadingPolicy": "native",
        "packArtifactId": None,
    }
    policy = {
        "method": "both" if version == 2 else "refit",
        "refitPercentile": 75 if version == 2 else 50,
    }
    predictions_all = predictions(cases, seed=SEED + version, shift=-0.22, signal=0.65)
    by_id = {row["id"]: row for row in predictions_all}
    membership = assessment_folds(cases)
    split_plans, planned, runs, histories = [], [], [], {}
    for fold in range(5):
        run_id = f"blca-demo-v{version}-fold-{fold + 1}"
        split_id = f"blca-demo-split-{fold + 1}"
        assessment = [case for case in cases if membership[case["id"]] == fold]
        fitting = [case for case in cases if membership[case["id"]] != fold]
        validation = []
        for label in ("low", "high"):
            validation.extend([case for case in fitting if case["label"] == label][:5])
        valid_rows = predictions(
            validation, seed=SEED + 100 + version * 5 + fold, shift=-0.22, signal=0.65
        )
        final_validation = metrics(valid_rows)
        epochs = [18, 22, 26, 30, 34][fold] if version == 2 else [16, 20, 24, 28, 32][fold]
        best_epoch = epochs - recipe["patience"]
        history_rows = []
        for epoch in range(1, epochs + 1):
            loss = round(0.71 * math.exp(-epoch / 5) + 0.014 + fold * 0.002, 6)
            validation_metric = {
                **final_validation,
                "loss": final_validation["loss"]
                + (
                    0.18 * (best_epoch - epoch) / best_epoch
                    if epoch <= best_epoch
                    else 0.004 * (epoch - best_epoch)
                ),
            }
            history_rows.append(
                {
                    "epoch": epoch,
                    "trainingLoss": loss,
                    "validation": validation_metric,
                    "learningRate": 0.0003,
                    "checkpointUnit": "slide",
                }
            )
        histories[run_id] = {
            **EXPORT_NOTICE,
            "runId": run_id,
            "rows": history_rows,
            "totalRows": len(history_rows),
            "truncated": False,
        }
        split_plans.append(
            {
                "id": split_id,
                "planId": "blca-protocol",
                "seed": 42,
                "fold": fold,
                "phase": "development",
                "slideCount": len(cases),
                "partitions": {
                    "train": len(fitting) - len(validation),
                    "validation": len(validation),
                    "assessment": len(assessment),
                },
            }
        )
        planned.append(
            {
                "id": run_id,
                "candidateId": candidate,
                "trainingSeed": 42,
                "splitPlanId": split_id,
                "status": "planned",
            }
        )
        runs.append(
            {
                **planned[-1],
                "status": "completed",
                "metrics": {
                    "validation": metric_details(valid_rows),
                    "assessment": metric_details([by_id[case["id"]] for case in assessment]),
                },
                "checkpointPath": f"demo://blca/v{version}/fold-{fold + 1}/illustrative.ckpt",
                "outputPath": f"demo://blca/v{version}/fold-{fold + 1}",
                "progress": {
                    "epoch": epochs,
                    "maxEpochs": 40,
                    "globalStep": epochs * (len(fitting) - len(validation)),
                    "trainingLoss": history_rows[-1]["trainingLoss"],
                    "validation": history_rows[-1]["validation"],
                    "learningRate": 0.0003,
                    "cudaPeakAllocatedBytes": 1073741824,
                    "cudaPeakReservedBytes": 2147483648,
                },
            }
        )
    batch = {
        "id": batch_id,
        "createdAt": STAMP,
        "manifest": {
            "kind": "mil-batch",
            "version": 1,
            "datasetId": "blca-dataset",
            "spec": {
                "version": 1,
                "experimentName": f"BLCA baseline v{version}",
                "batchName": "ABMIL baseline · synthetic",
                "inputs": inputs,
                "recipe": recipe,
                "mode": "single",
                "grid": {"learningRates": [], "weightDecays": [], "maxEpochs": []},
                "configurations": [recipe],
                "trainingSeeds": [42],
                "resources": resources,
                "notes": SYNTHETIC_NOTICE,
                "predictorPolicy": policy,
            },
            "configurations": [{"id": candidate, "number": 1, "recipe": recipe}],
            "splitPlans": split_plans,
            "runs": planned,
            "summary": {
                "configurationCount": 1,
                "trainingSeedCount": 1,
                "splitPlanCount": 5,
                "runCount": 5,
            },
            "executionImplemented": False,
            "previewHash": "synthetic:not-a-scientific-preview",
            "resolvedInputs": {
                "canPlan": False,
                "findings": [],
                "resolvedLoadingPolicy": "native",
                "packArtifactId": None,
                "featureSetId": "blca-features",
                "bundleId": "blca-features",
                "executionImplemented": False,
            },
        },
    }
    samples = []
    start = datetime(2026, 9, 12, 12, version, tzinfo=UTC)
    for index in range(20):
        used = round(2.8 + 0.3 * math.sin(index / 2), 3)
        samples.append(
            {
                "at": (start + timedelta(seconds=15 * index)).isoformat(),
                "host": {
                    "cpuCount": 16,
                    "totalRamGb": 64,
                    "availableRamGb": 52 - index / 10,
                    "cpuUtilizationPercent": round(20 + 9 * math.sin(index / 3), 2),
                    "kernel": "Synthetic Linux runtime",
                    "bootId": "demo-boot",
                },
                "gpus": [
                    {
                        "index": 0,
                        "uuid": "DEMO-GPU-0",
                        "name": "Synthetic GPU · 24 GiB",
                        "driverVersion": "illustrative",
                        "totalMemoryGb": 24,
                        "usedMemoryGb": used,
                        "freeMemoryGb": 24 - used,
                        "utilizationPercent": round(57 + 16 * math.sin(index / 4), 2),
                    }
                ],
                "runs": [
                    {
                        "runId": runs[index // 4]["id"],
                        "pid": 10000 + index // 4,
                        "rssGb": round(1.5 + index / 40, 3),
                    }
                ],
            }
        )
    resource_history = {
        **EXPORT_NOTICE,
        "batchId": batch_id,
        "rows": samples,
        "totalRows": len(samples),
        "truncated": False,
    }
    execution = {
        "batchId": batch_id,
        "status": "completed",
        "sessionName": "demo-only-no-worker",
        "logPath": f"demo://blca/v{version}/synthetic.log",
        "outputPath": f"demo://blca/v{version}",
        "findings": [],
        "runs": runs,
        "createdAt": STAMP,
        "updatedAt": samples[-1]["at"],
        "runCounts": {
            "total": 5,
            "queued": 0,
            "running": 0,
            "completed": 5,
            "failed": 0,
            "cancelled": 0,
            "interrupted": 0,
        },
        "computeVersion": "Synthetic illustration · no runtime executed",
        "resourcePlan": {
            "requestedConcurrency": 1,
            "effectiveConcurrency": 1,
            "cpuSlotsPerRun": 6,
            "cpuLimit": 2,
            "ramLimit": 6,
            "gpuSlotLimit": 1,
            "note": "Illustrative scheduling, not an observed worker.",
        },
        "telemetry": {
            "path": f"demo://blca/v{version}/synthetic-resources.jsonl",
            "intervalSeconds": 15,
            "latest": samples[-1],
            "peak": {
                "hostUsedRamGb": max(64 - sample["host"]["availableRamGb"] for sample in samples),
                "gpuUsedMemoryGb": {
                    "0": max(sample["gpus"][0]["usedMemoryGb"] for sample in samples)
                },
                "runRssGb": {
                    run["id"]: max(
                        sample["runs"][0]["rssGb"]
                        for sample in samples
                        if sample["runs"][0]["runId"] == run["id"]
                    )
                    for run in runs
                },
            },
        },
    }
    return {
        "batch": batch,
        "execution": execution,
        "histories": histories,
        "resources": resource_history,
    }, predictions_all


def _experiment(version: int, cases: list[dict]) -> dict:
    training, oof = _training(version, cases)
    result = metrics(oof)
    epochs = [
        min(history["rows"], key=lambda row: row["validation"]["loss"])["epoch"]
        for history in training["histories"].values()
    ]
    budget = sorted(epochs)[3 if version == 2 else 2]
    predictors = (
        [["Fold ensemble", "Illustrative ready", "5 fold checkpoints", "No artifact files"]]
        if version == 2
        else []
    )
    predictors.append(
        [
            "Full-development refit",
            "Illustrative ready",
            f"P{75 if version == 2 else 50} · {budget} epochs",
            "No artifact files",
        ]
    )
    return record(
        f"blca-baseline-v{version}",
        "experiments",
        f"BLCA baseline v{version}",
        "Inspect a five-fold ABMIL plan, synthetic run curves and predictor creation.",
        [
            {
                "id": "inputs",
                "title": "Inputs",
                "description": "Protocol and slide feature bundle stay linked to the batch.",
                "facts": facts(
                    Protocol="Grade 1/3 development",
                    Features="UNI v1 · 1024 dimensions",
                    Development="62 synthetic slides",
                    Target="low / high · high is positive",
                ),
                "notice": GROUPING_NOTICE,
            },
            {
                "id": "batches",
                "title": "Batches",
                "description": "The plan mirrors the Bladder baseline recipe; no compute is launched.",
                "facts": facts(
                    Model="Gated ABMIL",
                    Optimizer="AdamW",
                    Learning_rate="0.0003",
                    Maximum_epochs=40,
                    Bag_size=4096,
                    Dropout=0.25,
                    Folds=5,
                    Training_seed=42,
                    Split_seed=42,
                ),
                "table": table(
                    ["Fold", "Train", "Validation", "Assessment"],
                    [
                        [
                            index + 1,
                            *[
                                plan["partitions"][part]
                                for part in ("train", "validation", "assessment")
                            ],
                        ]
                        for index, plan in enumerate(training["batch"]["manifest"]["splitPlans"])
                    ],
                ),
                "notice": SYNTHETIC_NOTICE,
            },
            {
                "id": "runs",
                "title": "Runs",
                "description": "Runs first, resource usage below, then predictor creation.",
                "notice": SYNTHETIC_NOTICE,
                "runs": training,
                "table": table(
                    ["Method", "Status", "Source / budget", "Artifacts"],
                    predictors,
                    "Predictor creation",
                ),
            },
            {
                "id": "results",
                "title": "Results",
                "description": "Out-of-fold assessment summarizes synthetic held-out slides, separate from checkpoint validation.",
                "facts": facts(
                    Assessment_unit="Slide",
                    OOF_slides=result["count"],
                    AUROC=f"{result['auroc']:.3f}",
                    AUPRC=f"{result['auprc']:.3f}",
                    Accuracy=f"{result['accuracy']:.3f}",
                ),
                "table": table(
                    ["Synthetic slide", "Label", "P(high)", "Predicted at 0.5"], score_rows(oof)
                ),
                "notice": GROUPING_NOTICE,
            },
        ],
        ["Synthetic", "ABMIL", "5 folds", "Ensemble + P75 refit" if version == 2 else "P50 refit"],
    )


def generate_demo() -> dict:
    cases = synthetic_cases()
    development = [case for case in cases if case["partition"] == "development"]
    test = [case for case in cases if case["partition"] == "test"]
    test_scores = predictions(test, seed=SEED + 300, shift=-0.22, signal=0.50)
    evaluation = metrics(test_scores)
    confusion = evaluation["confusionMatrix"]
    records = [
        record(
            "blca-dataset",
            "dataset",
            "BLCA synthetic slides",
            "138 invented slide records reflect the source study's aggregate grade structure.",
            [
                {
                    "id": "source",
                    "title": "Source",
                    "description": "Example identifiers are created from scratch; no source files are bundled.",
                    "facts": facts(
                        Synthetic_slides=138,
                        Verified_patients=0,
                        Development_slides=62,
                        Test_slides=76,
                    ),
                    "notice": GROUPING_NOTICE,
                    "table": table(
                        ["Grade", "low", "high", "Use"],
                        [
                            [1, 33, 0, "Development"],
                            [3, 0, 29, "Development"],
                            [2, 54, 22, "Test cohort"],
                        ],
                    ),
                },
                {
                    "id": "mapping",
                    "title": "Mapping",
                    "description": "Map synthetic case, slide, grade and binary label fields.",
                    "table": table(
                        ["Synthetic slide", "Synthetic case", "Grade", "Binary label", "Use"],
                        [
                            [
                                case["id"],
                                case["caseId"],
                                case["grade"],
                                case["label"],
                                case["partition"],
                            ]
                            for case in cases
                        ],
                    ),
                    "notice": "A synthetic case label is not a verified patient identifier.",
                },
                {
                    "id": "review",
                    "title": "Review",
                    "description": "A dataset record preserves mappings and grouping limitations for downstream stages.",
                    "facts": facts(
                        Unique_slide_IDs=138,
                        Missing_labels=0,
                        Source_files="None included",
                        Pixels="None included",
                        Verified_patient_independence="Unavailable",
                    ),
                    "notice": "This illustrative dataset cannot train a model: slide pixels and feature tensors are intentionally absent.",
                },
            ],
        )
    ]
    records.append(
        record(
            "blca-protocol",
            "cohort",
            "Grade 1/3 development protocol",
            "Freeze the binary target, eligible development slides and five assessment folds.",
            [
                {
                    "id": "target",
                    "title": "Target",
                    "description": "Predict low versus high grade; high is the positive class.",
                    "facts": facts(
                        Target_column="grade_binary",
                        Class_order="low, high",
                        Positive_class="high",
                        Unit="Slide",
                        Development="Grade 1 and Grade 3",
                    ),
                    "notice": "Grade 2 slides are reserved for the later test cohort. Their binary labels are supplied separately in this synthetic example.",
                },
                {
                    "id": "splits",
                    "title": "Splits",
                    "description": "Five slide-disjoint assessment folds with separate checkpoint-validation subsets.",
                    "facts": facts(
                        Eligible_slides=62,
                        Low=33,
                        High=29,
                        Folds=5,
                        Split_seed=42,
                        Grouping="slide_id fallback",
                    ),
                    "notice": GROUPING_NOTICE,
                    "table": table(
                        ["Synthetic slide", "Label", "Assessment fold"],
                        [
                            [
                                case["id"],
                                case["label"],
                                assessment_folds(development)[case["id"]] + 1,
                            ]
                            for case in development
                        ],
                    ),
                },
                {
                    "id": "frozen",
                    "title": "Frozen protocol",
                    "description": "The example protocol fixes memberships before comparing baseline configurations.",
                    "table": table(
                        ["Input", "Rule"],
                        [
                            ["Development", "Grade 1/3 only"],
                            ["Assessment", "Each eligible slide appears in exactly one fold"],
                            [
                                "Validation",
                                "Checkpoint selection inside the remaining development slides",
                            ],
                            ["Test", "Grade 2 held aside"],
                            ["Patient independence", "Not established"],
                        ],
                    ),
                },
            ],
        )
    )
    records.append(
        record(
            "blca-features",
            "features",
            "UNI v1 feature bundle",
            "Explain feature attachment and coverage without publishing embeddings.",
            [
                {
                    "id": "source",
                    "title": "Feature source",
                    "description": "The project uses existing UNI v1 embeddings with 1024 dimensions.",
                    "facts": facts(
                        Encoder="UNI v1",
                        Dimensions=1024,
                        Coverage="138 synthetic slide references",
                        Mode="Metadata only",
                    ),
                    "notice": "No feature tensors, patch coordinates, encoder weights or real artifact paths are bundled.",
                },
                {
                    "id": "validation",
                    "title": "Validation",
                    "description": "A real bundle checks dimensions, finite values, slide IDs and optional coordinates before training.",
                    "table": table(
                        ["Check", "Demo representation"],
                        [
                            ["Dimensions", "1024, illustrative metadata"],
                            ["ID coverage", "138 unique synthetic references"],
                            ["Tensor integrity", "Not executed; no tensors included"],
                            ["Patch coordinates", "Not included"],
                            ["Slide matching", "Synthetic identifiers only"],
                        ],
                    ),
                },
                {
                    "id": "bundle",
                    "title": "Bundle",
                    "description": "A shared feature contract links the development and test stages.",
                    "facts": facts(
                        Development=62, Test=76, Loading="Native, illustrative", Pack="Not required"
                    ),
                    "notice": "This sample bundle is documentation metadata and is not an executable or verified scientific artifact.",
                },
            ],
        )
    )
    records.extend([_experiment(2, development), _experiment(3, development)])
    records.append(
        record(
            "blca-test-cohort",
            "test-data",
            "Grade 2 test cohort",
            "A separate set of 76 synthetic slides illustrates later evaluation.",
            [
                {
                    "id": "selection",
                    "title": "Selection",
                    "description": "Keep Grade 2 slides outside baseline development and checkpoint selection.",
                    "facts": facts(Test_slides=76, Low=54, High=22, Development_overlap=0),
                    "notice": GROUPING_NOTICE,
                },
                {
                    "id": "labels",
                    "title": "Labels",
                    "description": "Map the binary target and freeze slide membership independently of feature preparation.",
                    "table": table(
                        ["Synthetic slide", "Binary label", "Grade"],
                        [[case["id"], case["label"], 2] for case in test],
                    ),
                    "notice": "A test cohort can be prepared before features or predictors are ready. Evaluation checks feature compatibility later.",
                },
                {
                    "id": "frozen",
                    "title": "Frozen test cohort",
                    "description": "A prepared test cohort is reusable across predictors without changing its membership.",
                    "facts": facts(
                        Class_order="low, high",
                        Positive_class="high",
                        Unit="Slide",
                        Status="Illustrative prepared cohort",
                    ),
                },
            ],
        )
    )
    records.append(
        record(
            "blca-evaluation",
            "evaluation",
            "Grade 2 predictor evaluation",
            "Inspect synthetic predictions and metrics derived from exactly those predictions.",
            [
                {
                    "id": "plan",
                    "title": "Plan",
                    "description": "Pair the baseline v2 P75 refit example with the Grade 2 test cohort.",
                    "facts": facts(
                        Predictor="Baseline v2 · P75 refit",
                        Cohort="Grade 2 · 76 slides",
                        Positive_class="high",
                        Scoring_unit="Slide",
                        Feature_contract="UNI v1 · 1024 dimensions · metadata only",
                        Threshold=0.5,
                    ),
                    "notice": SYNTHETIC_NOTICE,
                },
                {
                    "id": "metrics",
                    "title": "Metrics",
                    "description": "All metrics below are recomputed from the released synthetic P(high) values.",
                    "facts": facts(
                        AUROC=f"{evaluation['auroc']:.3f}",
                        AUPRC=f"{evaluation['auprc']:.3f}",
                        Accuracy=f"{evaluation['accuracy']:.3f}",
                        Balanced_accuracy=f"{evaluation['balancedAccuracy']:.3f}",
                    ),
                    "chart": {
                        "title": "Synthetic test ROC",
                        "xLabel": "False-positive rate",
                        "yLabel": "True-positive rate",
                        "series": [
                            {
                                "label": "P75 refit · illustrative",
                                "points": roc_points(test_scores),
                            },
                            {"label": "Chance", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
                        ],
                        "yRange": [0, 1],
                    },
                    "table": table(
                        ["Actual / predicted", "low", "high"],
                        [["low", *confusion[0]], ["high", *confusion[1]]],
                        "Confusion matrix · threshold 0.5",
                    ),
                    "notice": "Good checkpoint validation does not guarantee test discrimination or useful thresholds. These invented scores illustrate that distinction, not clinical evidence.",
                },
                {
                    "id": "predictions",
                    "title": "Predictions",
                    "description": "Every row is newly generated; there is no correspondence to a real subject or slide.",
                    "table": table(
                        ["Synthetic slide", "Label", "P(high)", "Predicted at 0.5"],
                        score_rows(test_scores),
                    ),
                    "notice": GROUPING_NOTICE,
                },
            ],
        )
    )
    prevalence = sum(row["label"] for row in test_scores) / len(test_scores)
    thresholds = [index / 20 for index in range(1, 20)]
    utility, treat_all, utility_rows = [], [], []
    for threshold in thresholds:
        result = metrics(test_scores, threshold)
        (tn, fp), (fn, tp) = result["confusionMatrix"]
        net_benefit = tp / len(test_scores) - fp / len(test_scores) * threshold / (1 - threshold)
        all_benefit = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        utility.append({"x": threshold, "y": net_benefit})
        treat_all.append({"x": threshold, "y": all_benefit})
        utility_rows.append(
            [threshold, tn, fp, fn, tp, round(net_benefit, 6), round(all_benefit, 6)]
        )
    calibration_rows = []
    for lower in [index / 5 for index in range(5)]:
        selected = [row for row in test_scores if lower <= row["probability"] < lower + 0.2]
        if selected:
            calibration_rows.append(
                [
                    f"{lower:.1f}–{lower + 0.2:.1f}",
                    len(selected),
                    math.fsum(row["probability"] for row in selected) / len(selected),
                    sum(row["label"] for row in selected) / len(selected),
                ]
            )
    brier = math.fsum((row["probability"] - row["label"]) ** 2 for row in test_scores) / len(test_scores)
    records.append(
        record(
            "blca-clinical-utility",
            "clinical-utility",
            "Synthetic threshold review",
            "Explain calibration and decision curves using the same synthetic test predictions.",
            [
                {
                    "id": "calibration",
                    "title": "Calibration",
                    "description": "Compare predicted probabilities with observed synthetic class fractions.",
                    "facts": facts(
                        Brier_score=f"{brier:.3f}",
                        Positive_fraction=f"{prevalence:.3f}",
                        Analysis_unit="Slide, not verified patient",
                    ),
                    "table": table(
                        ["P(high) bin", "Slides", "Mean predicted", "Observed high fraction"],
                        calibration_rows,
                    ),
                    "notice": "Calibration bins are descriptive synthetic examples. No clinical recommendation or uncertainty claim is supported.",
                },
                {
                    "id": "thresholds",
                    "title": "Thresholds",
                    "description": "Changing the threshold changes false positives, false negatives and net benefit.",
                    "chart": {
                        "title": "Synthetic decision curve",
                        "xLabel": "Decision threshold",
                        "yLabel": "Net benefit",
                        "series": [
                            {"label": "Example predictor", "points": utility},
                            {"label": "Treat all", "points": treat_all},
                            {
                                "label": "Treat none",
                                "points": [{"x": t, "y": 0} for t in thresholds],
                            },
                        ],
                    },
                    "table": table(
                        ["Threshold", "TN", "FP", "FN", "TP", "Net benefit", "Treat all"],
                        utility_rows,
                    ),
                    "notice": "Net benefit = TP/N − FP/N × threshold/(1−threshold). Values come from the synthetic predictions and do not select a deployable threshold.",
                },
                {
                    "id": "report",
                    "title": "Report",
                    "description": "A useful report keeps discrimination, calibration, thresholds and data limitations together.",
                    "facts": facts(
                        Use="Pipeline education",
                        Clinical_validity="Not established",
                        Source_data="Excluded",
                        Real_predictions="Excluded",
                    ),
                    "notice": GROUPING_NOTICE,
                },
            ],
        )
    )
    records.append(
        record(
            "blca-interpretation",
            "interpretation",
            "Attention interpretation guide",
            "Explain how a local predictor, coordinates and pixels support a slide overlay.",
            [
                {
                    "id": "inputs",
                    "title": "Inputs",
                    "description": "A real attention overlay needs a compatible ABMIL checkpoint, features, coordinates and slide pixels.",
                    "table": table(
                        ["Required input", "Demo status"],
                        [
                            ["ABMIL checkpoint", "Placeholder only"],
                            ["Patch embeddings", "Not included"],
                            ["Level-0 coordinates", "Not included"],
                            ["Slide pixels", "Not included"],
                        ],
                    ),
                    "notice": "No attention has been computed for this demo. The following page is a conceptual schematic, not tissue or a measured heatmap.",
                },
                {
                    "id": "schematic",
                    "title": "Schematic",
                    "description": "Slide → patches → UNI embeddings → ABMIL attention → ranked regions.",
                    "table": table(
                        ["Step", "Purpose"],
                        [
                            ["Slide", "Select a local slide"],
                            ["Patches", "Align embeddings to level-0 coordinates"],
                            ["Attention", "Obtain weights from the selected ABMIL predictor"],
                            ["Overlay", "Place the weights over the corresponding slide pixels"],
                            ["Review", "Inspect ranked regions and retain source provenance"],
                        ],
                    ),
                    "notice": "Attention ranks model contributions; it is not a diagnosis or a validated localization of disease.",
                },
            ],
        )
    )
    return {
        "project": demo_summary(),
        "mode": "synthetic-demo",
        "executionEnabled": False,
        "dataset": {
            "id": "blca-dataset",
            "name": "BLCA synthetic slides",
            "patientCount": 0,
            "specimenCount": 0,
            "slideCount": 138,
            "fallbackSlideCount": 138,
            "groupCount": 138,
        },
        "patients": [],
        "slides": [
            {
                "id": case["id"],
                "patientId": "",
                "specimenId": "",
                "site": "Synthetic",
                "status": "Synthetic reference",
                "filename": f"{case['id']}.demo",
            }
            for case in cases
        ],
        "encoders": [
            {
                "id": "uni",
                "name": "UNI v1",
                "description": "Illustrative feature contract",
                "dimensions": 1024,
                "adapter": "Metadata only",
            }
        ],
        "milModels": [
            {
                "id": "abmil",
                "name": "Gated ABMIL",
                "description": "Illustrative baseline",
                "adapter": "No execution in demo",
            }
        ],
        "featureSets": [],
        "split": {"id": "blca-protocol", "seed": 42, "groupBy": "slide_id_fallback"},
        "results": [],
        "cohortSnapshots": [],
        "drafts": [],
        "sources": [],
        "exampleManifests": [],
        "demoPipeline": {
            "version": 1,
            "synthetic": True,
            "readOnly": True,
            "seed": SEED,
            "sourceBasis": [
                "The Bladder project supplied aggregate workflow structure only: 138 slides; Grade 1/3 development has 62 slides (33 low, 29 high); Grade 2 test has 76 slides (54 low, 22 high).",
                "UNI v1 1024-dimensional features and five-fold gated ABMIL baselines with training/split seed 42, AdamW, learning rate 0.0003, bag size 4096 and a 40-epoch maximum informed the sample configuration.",
                "Baseline v2 illustrates a fold ensemble and P75 refit; baseline v3 illustrates a P50 refit. All histories, checkpoint references, probabilities, derived metrics and resource usage are newly synthesized.",
                GROUPING_NOTICE,
                "No patient identifiers, original filenames, paths, dates, row-level labels or predictions, slide pixels, patch coordinates, embeddings, weights or checkpoints were copied. The generator reads no local project or real source data.",
            ],
            "records": records,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Regenerate the packaged JSON fixture")
    mode.add_argument(
        "--check", action="store_true", help="Check the packaged fixture is reproducible"
    )
    args = parser.parse_args()
    content = json.dumps(generate_demo(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.check:
        if load_demo() != generate_demo():
            parser.exit(1, "The packaged BLCA demo differs from its deterministic generator.\n")
        print("BLCA demo fixture is reproducible.")
    elif args.write:
        path = Path(__file__).resolve().parents[1] / "resources" / RESOURCE
        path.write_text(content)
        print(path)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
