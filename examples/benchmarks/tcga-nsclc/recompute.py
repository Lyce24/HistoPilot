"""Recompute the TCGA-NSCLC benchmark from a project folder, without HistoPilot.

Reads only the files HistoPilot wrote (out-of-fold predictions, the external runs'
patient predictions, the dataset shift report) and the label table, recomputes every
AUROC with scikit-learn, checks the point estimates against the ones HistoPilot stored,
and prints the tables of docs/benchmarks.md with 95% patient bootstrap intervals.

    python recompute.py PROJECT_FOLDER nsclc_dataset.csv

Needs numpy and scikit-learn.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

POSITIVE = "LUSC"
RESAMPLES = 2000
NAMES = {
    "abmil": "ABMIL",
    "clam_sb": "CLAM-SB",
    "clam_mb": "CLAM-MB",
    "transmil": "TransMIL",
    "dsmil": "DSMIL",
    "dtfd": "DTFD-MIL",
    "acmil": "ACMIL",
    "nnmil": "nnMIL",
    "mean_pool": "Mean pooling",
    "max_pool": "Max pooling",
    "oceanpath_mamba2": "Mamba-2",
}


def interval(labels, scores, seed=42):
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(RESAMPLES):
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() != labels[index].max():
            values.append(roc_auc_score(labels[index], scores[index]))
    return np.percentile(values, [2.5, 97.5])


def cell(labels, scores):
    value = roc_auc_score(labels, scores)
    lower, upper = interval(labels, scores)
    return value, f"{value:.3f} ({lower:.3f}–{upper:.3f})"


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    with args.labels.open() as stream:
        table = {row["Slide_ID"]: row for row in csv.DictReader(stream)}
    mismatches = 0

    # Development: out-of-fold predictions of each configuration, one row per patient.
    print("## TCGA-NSCLC, 5-fold cross-validation (out of fold)\n")
    print("| Model | Training seed | Patients | AUROC (95% CI) | Stored |")
    print("| --- | --- | ---: | --- | --- |")
    run_model = {}
    for batch in sorted((args.project / "training").glob("*/")):
        if not (batch / "results.json").exists():
            continue
        plan, results = read_json(batch / "plan.json"), read_json(batch / "results.json")
        number_model = {row["number"]: row["recipe"]["model"] for row in plan["configurations"]}
        candidate_model = {
            row["candidateId"]: number_model[row["number"]]
            for row in results["selection"]["candidates"]
        }
        for run in plan["runs"]:
            run_model[run["id"]] = candidate_model[run["candidateId"]]
        for path in sorted(batch.glob("oof-*.json")):
            document = read_json(path)
            positive = document["classOrder"].index(POSITIVE)
            patients, label = defaultdict(list), {}
            for record in document["records"]:
                patients[record["patientId"]].append(record["probabilities"][positive])
                label[record["patientId"]] = int(record["label"] == POSITIVE)
            keys = sorted(patients)
            value, text = cell([label[k] for k in keys], [np.mean(patients[k]) for k in keys])
            stored = document["summary"]["patient"]["auroc"]
            mismatches += abs(stored - value) > 1e-9
            model = NAMES.get(candidate_model[document["candidateId"]], "?")
            print(f"| {model} | {document['trainingSeed']} | {len(keys)} | {text} | {stored:.4f} |")

    # External: each fold ensemble's patient predictions on CPTAC, joined to the labels here.
    print("\n## CPTAC-NSCLC, external (fold ensembles)\n")
    print("| Model | Patients | AUROC (95% CI) | Stored |")
    print("| --- | ---: | --- | --- |")
    external = {}
    for job in sorted((args.project / "compute-jobs").glob("*/")):
        plan_path, table_path = job / "plan.json", job / "patient-predictions.csv"
        if not plan_path.exists() or not table_path.exists():
            continue
        plan = read_json(plan_path)
        if plan.get("kind") != "evaluation":
            continue
        models = {run_model.get(row["runId"], "?") for row in plan["checkpoints"]}
        rows = []
        with table_path.open() as stream:
            for row in csv.DictReader(stream):
                slides = json.loads(row["slideIds"])
                subtypes = {table[slide]["Subtype"] for slide in slides if slide in table}
                if len(subtypes) == 1:
                    rows.append(
                        (
                            slides,
                            int(subtypes.pop() == POSITIVE),
                            float(row[f"probability:{POSITIVE}"]),
                        )
                    )
        value, text = cell([r[1] for r in rows], [r[2] for r in rows])
        metrics_path = job / "service-metrics.json"
        stored = (
            read_json(metrics_path)["metrics"]["patient"]["auroc"]
            if metrics_path.exists()
            else None
        )
        if stored is not None:
            mismatches += abs(stored - value) > 1e-9
        model = NAMES.get("/".join(sorted(models)), "/".join(sorted(models)))
        external[model] = rows
        print(
            f"| {model} | {len(rows)} | {text} | {stored:.4f} |"
            if stored is not None
            else f"| {model} | {len(rows)} | {text} | — |"
        )

    # Dataset shift: atypicality of each CPTAC slide against the development slides.
    reports = [
        read_json(job / "shift.json")
        for job in sorted((args.project / "compute-jobs").glob("*/"))
        if (job / "shift.json").exists() and read_json(job / "plan.json").get("kind") == "shift"
    ]
    if reports:
        report = reports[-1]
        summary = report["summary"]
        domain = summary["domainAuroc"]
        print("\n## Dataset shift of CPTAC against the TCGA development slides\n")
        print(f"- Domain AUROC {domain['value']:.3f} ({domain['lower']:.3f}–{domain['upper']:.3f})")
        print(
            f"- Beyond the development range: {summary['beyond']['count']} of "
            f"{summary['cohortSlides']} slides ({summary['beyond']['fraction']:.1%})"
        )
        slides = {row["slideId"]: row for row in report["slides"] if row["set"] == "cohort"}
        print(
            "\n| Model | Least atypical third | Middle third | Most atypical third | Within range | Beyond range |"
        )
        print("| --- | --- | --- | --- | --- | --- |")
        for model, rows in sorted(external.items()):
            scored = sorted(
                (
                    float(np.mean([slides[s]["percentile"] for s in slide_ids])),
                    any(slides[s]["beyond"] for s in slide_ids),
                    label,
                    score,
                    slide_ids[0],
                )
                for slide_ids, label, score in rows
                if all(s in slides for s in slide_ids)
            )
            scored.sort(key=lambda item: (item[0], item[4]))
            n = len(scored)
            edges = [0, n // 3 + (n % 3 > 0), 2 * n // 3 + (n % 3 > 1), n]
            groups = [scored[a:b] for a, b in pairwise(edges)]
            groups += [[r for r in scored if not r[1]], [r for r in scored if r[1]]]
            cells = [
                cell([r[2] for r in g], [r[3] for r in g])[1] + f", n={len(g)}" for g in groups
            ]
            print(f"| {model} | " + " | ".join(cells) + " |")

    print(f"\n{mismatches} recomputed AUROC(s) differ from the stored values.", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
