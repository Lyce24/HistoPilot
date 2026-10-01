"""The labels that score one model-evaluation run, joined to its verified predictions.

Runs saved before label-blind prediction were scored by their worker, which received the
cohort labels and stored them beside every prediction. Labeled runs saved since then carry
``labelSource: "cohort"``: their job never receives a label, and the labels come from the
frozen cohort membership here. Inference runs have no labels at all.

Metrics, case review, clinical utility, comparisons and downloads read labels through this
module, so the two storage generations answer every question the same way. The same
predictions can also be scored against a reference standard attached to the cohort later
(``join_reference``), which replaces the cohort's labels rather than mixing with them.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict

from histopilot.inference_summary import describe, patient_member_probabilities
from histopilot.scoring import patient_predictions

LABELS_FROM_COHORT = "cohort"


def labels_withheld(manifest):
    """True for a run predicted without labels and scored from its frozen cohort."""
    return manifest.get("labelSource") == LABELS_FROM_COHORT


def strip_labels(memberships):
    """Membership rows as a label-blind job receives them."""
    return [{key: value for key, value in row.items() if key != "label"} for row in memberships]


def join_labels(manifest, cohort_manifest, records):
    """Prediction records carrying the labels that score them.

    Raises ValueError when the predictions and the frozen membership disagree. A
    label-blind run must have stored no labels; an earlier run must have stored exactly
    the cohort's labels.
    """
    memberships = {row["slideId"]: row for row in cohort_manifest["memberships"]}
    if len(memberships) != len(cohort_manifest["memberships"]):
        raise ValueError("The cohort contains duplicate slide membership.")
    if not labels_withheld(manifest):
        observed = {row["slideId"]: (row.get("patientId"), row.get("label")) for row in records}
        expected = {
            key: (row.get("patientId"), row.get("label")) for key, row in memberships.items()
        }
        if observed != expected:
            raise ValueError("Predictions differ from frozen cohort membership or labels.")
        return records
    if any(row.get("label") is not None or row.get("labelIndex") is not None for row in records):
        raise ValueError("A label-blind run stored labels with its predictions.")
    observed = {row["slideId"]: row.get("patientId") for row in records}
    if observed != {key: row.get("patientId") for key, row in memberships.items()}:
        raise ValueError("Predictions differ from frozen cohort membership.")
    classes = manifest["target"]["classes"]
    joined = []
    for row in records:
        label = memberships[row["slideId"]].get("label")
        if label is not None and label not in classes:
            raise ValueError("Cohort labels must preserve the frozen class order.")
        joined.append(
            {**row, "label": label, "labelIndex": None if label is None else classes.index(label)}
        )
    return joined


def join_reference(manifest, reference_manifest, records):
    """Prediction records carrying a reference standard's labels, in the run's class order.

    Returns the records and the number of patients left unlabeled. Labels a run stored
    itself (earlier labeled runs) are replaced, never mixed in. A reference is taken from a
    dataset column after the cohort was frozen, so under a patient target the slides of one
    patient can disagree; such a patient is left unlabeled instead of guessed.
    """
    classes = manifest["target"]["classes"]
    if sorted(reference_manifest["classes"]) != sorted(classes):
        raise ValueError("The reference standard's classes differ from this run's classes.")
    labels = {row["slideId"]: row.get("label") for row in reference_manifest["memberships"]}
    if len(labels) != len(reference_manifest["memberships"]):
        raise ValueError("The reference standard contains duplicate slides.")
    if {row["slideId"] for row in records} != set(labels):
        raise ValueError("The reference standard covers different slides than this run.")
    joined = []
    for row in records:
        label = labels[row["slideId"]]
        if label is not None and label not in classes:
            raise ValueError("Reference labels must be classes of this run.")
        joined.append(
            {**row, "label": label, "labelIndex": None if label is None else classes.index(label)}
        )
    if manifest["target"]["unit"] != "patient":
        return joined, 0
    seen = defaultdict(set)
    for row in joined:
        if row.get("patientId") and row["labelIndex"] is not None:
            seen[row["patientId"]].add(row["labelIndex"])
    conflicting = {patient for patient, values in seen.items() if len(values) > 1}
    if conflicting:
        joined = [
            {**row, "label": None, "labelIndex": None}
            if row.get("patientId") in conflicting
            else row
            for row in joined
        ]
    return joined, len(conflicting)


def patient_evidence(manifest, slides, saved, *, rebuild=None):
    """Patient records whose labels agree with ``slides``.

    Earlier runs stored labeled patient records; they are returned for the caller's own
    validation. A label-blind run stored unlabeled ones, so they are rebuilt from the
    labeled slides with the rule its worker used; so are the patients of any run scored
    against a reference standard (``rebuild=True``), whose labels no worker saw. Where that
    rule refuses (slides of one patient disagree, or slide-ID fallback groups), the worker
    of a labeled evaluation stored no patient records, and neither does this.
    """
    if not (labels_withheld(manifest) if rebuild is None else rebuild):
        return saved
    aggregation = manifest["inference"]["patientAggregation"]
    try:
        return patient_predictions(
            slides, "mean_logits" if aggregation == "mean_logits" else "mean"
        )
    except ValueError:
        return []


def development_patients(manifest):
    """Patients shared with development, or None when overlap cannot be compared."""
    if manifest.get("splitUnit") == "slide":
        return None
    overlap = manifest.get("overlap") or {}
    if not overlap.get("patientsComparable"):
        return None
    return set(overlap.get("patientIds") or [])


def development_flag(row, shared):
    """True/False for verified patient IDs; None when overlap cannot be established.

    A slide-ID fallback group can never match a development patient, so it is
    unknown rather than evidence of a new patient.
    """
    if shared is None or not row.get("patientId") or row.get("patientIdSource") == "slide_fallback":
        return None
    return row["patientId"] in shared


def csv_text(value):
    """A CSV cell that spreadsheet applications cannot read as a formula."""
    value = str(value if value is not None else "")
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value


def prediction_table(manifest, slides, saved_patients, *, patient, rebuild=None):
    """A prediction table with the labels that score it joined: a label-blind run's cohort
    labels, or a reference standard's (``rebuild=True``, as for ``patient_evidence``).

    Earlier labeled runs serve their worker's CSV unchanged. This table keeps that file's
    identity, label, decision and probability columns, adds the label-free descriptions
    inference exports carry, and says whether each row counts toward the metrics.
    """
    target = manifest["target"]
    classes = target["classes"]
    threshold = manifest["inference"]["decisionThreshold"]
    shared = development_patients(manifest)
    rows = slides
    if patient:
        aggregation = manifest["inference"]["patientAggregation"]
        groups = defaultdict(list)
        for row in slides:
            groups[row.get("patientId")].append(row)
        rows = []
        for row in patient_evidence(manifest, slides, saved_patients, rebuild=rebuild):
            members = patient_member_probabilities(groups[row["patientId"]], aggregation)
            rows.append({**row, **({"memberProbabilities": members} if members else {})})
    identity = ["patientId", "slideIds"] if patient else ["slideId", "patientId"]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            *identity,
            "label",
            "predictedLabel",
            "confidence",
            "margin",
            "membersAgreeing",
            "memberCount",
            "developmentPatient",
            "scored",
            *(f"probability:{label}" for label in classes),
        ]
    )
    for row in rows:
        described = describe(row, target, threshold)
        agreement = described.get("memberAgreement")
        flag = development_flag(row, shared)
        cells = (
            [row["patientId"], json.dumps(row["slideIds"], ensure_ascii=False)]
            if patient
            else [row["slideId"], row.get("patientId")]
        )
        writer.writerow(
            [
                csv_text(value)
                for value in [
                    *cells,
                    row.get("label"),
                    described["predictedLabel"],
                    described["confidence"],
                    described["margin"],
                    agreement["agree"] if agreement else None,
                    agreement["total"] if agreement else None,
                    "" if flag is None else "yes" if flag else "no",
                    "yes" if row.get("labelIndex") is not None and flag is not True else "no",
                    *row["probabilities"],
                ]
            ]
        )
    return output.getvalue().encode("utf-8")
