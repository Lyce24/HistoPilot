"""Data-dependent recipe checks shared by planning and training workers."""

from collections import defaultdict


def validate_selection_metric(metric, target, rows):
    """Configuration ranking must be defined independently of checkpoint stopping."""
    if metric == "validation_auroc" and {
        row["label"] for row in rows if row["partition"] == "val"
    } != set(target["classes"]):
        raise ValueError(
            "Validation AUROC cannot rank configurations when a frozen class is absent. "
            "Choose validation loss for configuration selection or revise the validation split. "
            "A fixed checkpoint epoch budget does not make AUROC defined."
        )


def validate_split_unit(recipe, target, split_unit=None):
    """Explicit slide experiments must never invoke patient-based computation."""
    if split_unit not in {None, "slide", "patient"}:
        raise ValueError("Choose slide-level or patient-level splitting.")
    if split_unit is not None and target["unit"] != split_unit:
        raise ValueError("The target unit must match the frozen split unit.")
    if split_unit == "slide":
        if recipe.get("samplingStrategy", "slide_uniform") != "slide_uniform":
            raise ValueError("Slide-level experiments require slide-uniform sampling; patient sampling is unavailable.")
        if recipe.get("inputMode", "image") != "image" or recipe.get("clinicalFields"):
            raise ValueError("Patient-based clinical preprocessing is unavailable for slide-level experiments.")


def resolve_stopping(recipe, target, rows, *, split_unit=None):
    """Resolve explicit epoch budgets and small-validation fallback policies."""
    threshold = recipe.get("minValidationPositives")
    budget = recipe.get("fixedEpochBudget")
    if threshold is None and budget is None:
        return recipe, None
    if threshold is not None and target["task"] != "binary_classification":
        raise ValueError("The minimum validation positives fallback requires a binary target.")
    positive = target.get("positiveClass")
    units = {
        row["slideId" if split_unit == "slide" else "patientId"]
        for row in rows if row["partition"] == "val" and row["label"] == positive
    }
    if threshold is not None and len(units) >= threshold:
        return recipe, None
    if not budget:
        raise ValueError("Too few validation positives: provide a fixed epoch budget.")
    decision = {
        "reason": "explicit_fixed_epoch_budget"
        if threshold is None
        else f"insufficient_validation_positive_{'slides' if split_unit == 'slide' else 'patients'}",
        **(
            ({"positiveSlides": len(units), "minimumPositiveSlides": threshold}
             if split_unit == "slide" else
             {"positivePatients": len(units), "minimumPositivePatients": threshold})
            if threshold is not None
            else {}
        ),
        "epochs": budget,
        "checkpointSelection": "final_epoch",
        "requestedCheckpointMetric": recipe["checkpointMetric"],
    }
    return {
        **recipe,
        "earlyStopping": False,
        "maxEpochs": budget,
        "minEpochs": budget,
        # Loss remains defined with one validation class. It is logged but does
        # not choose the deployed checkpoint under the fixed-budget fallback.
        "checkpointMetric": "validation_loss",
    }, decision


def sampling_memberships(rows, recipe, cohort_values):
    """Bind the candidate's cohort column from metadata pinned in the plan."""
    if recipe.get("samplingStrategy", "slide_uniform") not in {
        "cohort_balanced",
        "cohort_label_balanced",
    }:
        return rows
    column = recipe.get("cohortColumn", "cohort")
    values = cohort_values.get(column, {})
    return [{**row, "cohort": values.get(row["slideId"])} for row in rows]


def validate_training_controls(recipe, target, rows, *, split_unit=None):
    """Fail before launching workers when a choice cannot use these folds."""
    validate_split_unit(recipe, target, split_unit)
    classes = target["classes"]
    if recipe.get("lossType") == "bce" and target["task"] != "binary_classification":
        raise ValueError("Single-logit BCE requires a binary classification target.")
    if recipe.get("classWeights") is not None and len(recipe["classWeights"]) != len(classes):
        raise ValueError("Provide one class weight per target class, in the frozen class order.")
    training = [row for row in rows if row["partition"] == "train"]
    if recipe.get("nnmilBatchSampler", "patient_weighted") != "patient_weighted":
        if {row["label"] for row in training} != set(classes):
            raise ValueError("nnMIL batch sampling requires every target class in fitting data.")
        if recipe["nnmilBatchSampler"] == "class_balanced" and recipe["batchSize"] < len(classes):
            raise ValueError("Class-balanced nnMIL batches need at least one slide per class.")
    if (
        recipe.get("classWeighting") == "inverse_prevalence" or recipe.get("classWeightedSampling")
    ) and {row["label"] for row in training} != set(classes):
        raise ValueError("Automatic class weights require every target class in training.")
    strategy = recipe.get("samplingStrategy", "slide_uniform")
    if strategy != "slide_uniform":
        patients = defaultdict(set)
        for row in training:
            if not row.get("patientId"):
                raise ValueError("Patient sampling requires a patient ID for each training slide.")
            patients[row["patientId"]].add(row["label"])
        if any(len(labels) != 1 for labels in patients.values()):
            raise ValueError("Patient sampling requires consistent training labels per patient.")
    if strategy in {"cohort_balanced", "cohort_label_balanced"}:
        patients, cohort_labels = defaultdict(set), defaultdict(set)
        for row in training:
            cohort = row.get("cohort")
            if not isinstance(cohort, str) or not cohort.strip():
                raise ValueError(
                    "Cohort sampling requires a nonempty cohort value for every training slide."
                )
            patients[row["patientId"]].add(cohort)
            cohort_labels[cohort].add(row["label"])
        if any(len(values) != 1 for values in patients.values()):
            raise ValueError("Cohort sampling requires one cohort per training patient.")
        if strategy == "cohort_label_balanced" and (
            target["task"] != "binary_classification"
            or any(labels != set(classes) for labels in cohort_labels.values())
        ):
            raise ValueError(
                "Cohort-label sampling requires both binary classes in every training cohort."
            )
    resolved, decision = resolve_stopping(recipe, target, rows, split_unit=split_unit)
    if resolved["checkpointMetric"] == "validation_auroc" and {
        row["label"] for row in rows if row["partition"] == "val"
    } != set(classes):
        raise ValueError(
            "Validation AUROC cannot select checkpoints when a frozen class is absent. Choose validation loss or configure a fixed-budget fallback."
        )
    return decision
