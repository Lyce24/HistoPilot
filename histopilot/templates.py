"""Server-owned starting points for the specs people and agents write.

The browser's starters and presets read `web/src/lib/templates.json`, which
`scripts/export_templates.py` writes from this module and `tests/test_templates.py`
checks against it. The CLI and the agent tools read the same values from the service. A
starter writes out every field whose service default differs from the browser's choice,
so an omitted field can never change the science. Pure data: no Pydantic, no Torch.
"""

from copy import deepcopy

VERSION = 1

ANALYSIS = {
    "version": 1,
    "confidenceLevel": 0.95,
    "bootstrapResamples": 2000,
    "bootstrapSeed": 42,
    "oneSlideSeed": 42,
}

DEFAULT_RECIPE = {
    "model": "abmil",
    "learningRate": 0.0003,
    "weightDecay": 0.0001,
    "maxEpochs": 40,
    "optimizer": "adamw",
    "batchSize": 1,
    "bagSize": 4096,
    "earlyStopping": True,
    "patience": 8,
    "checkpointMetric": "validation_auroc",
    "analysis": ANALYSIS,
    "decisionThreshold": 0.5,
    "embedDim": 512,
    "attentionDim": 384,
    "numFcLayers": 1,
    "gatedAttention": True,
    "dropout": 0.25,
    "inputDropout": 0,
    "gradientCheckpointing": False,
    "precision": "32-true",
    "gradientClipNorm": 0,
    "accumulateGradBatches": 1,
    "lrScheduler": "none",
    "warmupEpochs": 0,
    "finalLrFraction": 0.01,
    "earlyStoppingMinDelta": 0,
    "minEpochs": 1,
}

# Settings every recipe has, shown by the advanced editor and the nnMIL template.
EXPERIMENTAL_DEFAULTS = {
    "lossType": "ce",
    "classWeighting": "none",
    "classWeights": None,
    "focalGamma": 2,
    "labelSmoothing": 0,
    "patientAggregation": "mean_probabilities",
    "ensembleAggregation": "mean_probability",
    "adamBetas": [0.9, 0.999],
    "adamEps": 1e-8,
    "aggregatorLearningRate": None,
    "headLearningRate": None,
    "lrStepSize": 10,
    "lrGamma": 0.5,
    "lrPlateauPatience": 5,
    "samplingStrategy": "slide_uniform",
    "classWeightedSampling": False,
    "samplingPositivePrevalence": 0.4,
    "cohortColumn": "cohort",
    "instanceDropout": 0,
    "featureNoiseStd": 0,
    "bagCurriculum": False,
    "bagCurriculumStart": 512,
    "bagCurriculumEnd": 8000,
    "bagCurriculumWarmupEpochs": 5,
    "evalBagSize": None,
    "evalBatchSize": None,
    "minValidationPositives": None,
    "fixedEpochBudget": None,
    "bagSizeMode": "fixed",
    "bagSizeFraction": 0.5,
    "nnmilFeatureSampling": True,
    "nnmilWindowStrideDivisor": 4,
    "nnmilWindowShuffle": True,
    "nnmilWindowSeed": 42,
    "nnmilWindowSeedFromTraining": False,
    "nnmilWindowAggregation": "mean_logits",
    "nnmilBatchSampler": "patient_weighted",
    "nnmilCheckpointSelection": "best_validation",
    "weightDecayPolicy": "all",
    "lrScheduleInterval": "epoch",
}

# A preset is its base recipes merged in order, then its own values.
RECIPE_PRESETS = {
    "nnmil": {
        "base": ["default", "experimental"],
        "values": {
            "model": "nnmil",
            "attentionDim": 256,
            "nnmilWindowSeedFromTraining": True,
            "dropout": 0.25,
            "batchSize": 32,
            "bagSize": None,
            "bagSizeMode": "training_median",
            "bagSizeFraction": 0.5,
            "optimizer": "adamw",
            "lrScheduler": "cosine",
            "warmupEpochs": 5,
            "maxEpochs": 100,
            "patience": 10,
            "minEpochs": 1,
            "evalBagSize": None,
            "evalBatchSize": 1,
            "checkpointMetric": "validation_auroc",
        },
    },
    "oceanpath": {
        "base": ["default"],
        "values": {
            "maxEpochs": 20,
            "minEpochs": 10,
            "patience": 5,
            "lrScheduler": "cosine",
            "finalLrFraction": 0.01,
            "gradientClipNorm": 1,
            "checkpointMetric": "validation_auroc",
            "patientAggregation": "mean_probabilities",
            "ensembleAggregation": "mean_probability",
            "bagSize": None,
        },
    },
    "oceanpath-kras": {
        "base": ["default"],
        "values": {
            "weightDecay": 0.01,
            "maxEpochs": 40,
            "minEpochs": 0,
            "patience": 8,
            "lrScheduler": "cosine",
            "finalLrFraction": 0.001,
            "gradientClipNorm": 1,
            "checkpointMetric": "validation_auroc",
            "patientAggregation": "mean_probabilities",
            "ensembleAggregation": "mean_probability",
            "bagSize": 4096,
            "lossType": "bce",
            "classWeighting": "none",
        },
    },
    "quick": {"base": ["default"], "values": {"maxEpochs": 5, "bagSize": 1024, "patience": 3}},
}

BATCH_PRESETS = [
    {
        "id": "blank",
        "name": "Start blank",
        "description": "Start with default settings and give this batch a name.",
        "recipe": "default",
    },
    {
        "id": "baseline",
        "name": "ABMIL baseline",
        "description": "One configuration with standard training settings.",
        "recipe": "default",
    },
    {
        "id": "nnmil",
        "name": "nnMIL",
        "description": "Editable feature-sampling attention, automatic fitting-fold patch limits, "
        "and feature-window testing with the patient protocol.",
        "recipe": "nnmil",
    },
    {
        "id": "oceanpath",
        "name": "OceanPath standard",
        "description": "20 epochs, whole training bags, cosine decay, and patient AUROC "
        "checkpoints with mean probabilities.",
        "recipe": "oceanpath",
    },
    {
        "id": "oceanpath-kras",
        "name": "OceanPath binary (BCE)",
        "description": "Binary cross entropy, equal patient weight, 4096-patch bags, weight decay "
        "0.01, and mean probabilities.",
        "recipe": "oceanpath-kras",
    },
    {
        "id": "quick",
        "name": "Quick check",
        "description": "Five epochs and smaller sampled bags to check the training setup.",
        "recipe": "quick",
    },
    {
        "id": "learning-rate",
        "name": "Learning-rate comparison",
        "description": "Compare three learning rates with the same folds and training seed.",
        "recipe": "default",
        "mode": "grid",
        "learningRates": [0.0001, 0.0003, 0.001],
    },
]
# The preset names, in the order the browser offers them.
PRESETS = tuple(item["id"] for item in BATCH_PRESETS)
DEFAULT_PRESET = "baseline"

PREDICTOR_POLICY = {"method": "ensemble", "refitPercentile": None}

BATCH_DEFAULTS = {
    "version": 1,
    "mode": "single",
    "configurations": [],
    "trainingSeeds": [42],
    "notes": "",
    "predictorPolicy": PREDICTOR_POLICY,
    "selectionMetric": "validation_auroc",
    "candidateSelection": "best_validation",
}

# Settings that change together when the model or input changes.
_NNMIL_ON = {"attentionDim": 256, "gatedAttention": True, "nnmilWindowSeedFromTraining": True}
_NNMIL_OFF = {
    "nnmilBatchSampler": "patient_weighted",
    "nnmilCheckpointSelection": "best_validation",
    "nnmilWindowSeedFromTraining": False,
}
COUPLING = {
    # Choosing a model in the recipe editor.
    "modelSelect": {
        "nnmil": _NNMIL_ON,
        "other": {"bagSizeMode": "fixed", **_NNMIL_OFF},
        "slide": {
            "bagSize": 1,
            "bagCurriculum": False,
            "instanceDropout": 0,
            "evalBagSize": None,
            "gradientCheckpointing": False,
        },
    },
    # Arms of a controlled comparison keep the base recipe's optimisation and bags, so the
    # comparison measures the model alone.
    "comparisonArm": {"nnmil": _NNMIL_ON, "nnmilOff": _NNMIL_OFF},
    # A clinical-only arm fits a logistic model; image and nnMIL options do not apply.
    "clinicalOnly": {
        "inputMode": "clinical",
        "model": "abmil",
        "bagSizeMode": "fixed",
        "bagCurriculum": False,
        **_NNMIL_OFF,
    },
    "nonCosineSchedule": {"warmupEpochs": 0, "lrScheduleInterval": "epoch"},
}

TEST_TARGET = {
    "field": "",
    "task": "",
    "unit": "slide",
    "classes": [],
    "labels": {},
    "missing": "block",
    "unmapped": "block",
}

COHORT_INFERENCE = {
    "loadingPolicy": "per_slide",
    "packArtifactId": None,
    "batchSize": 1,
    "numWorkers": 0,
    "device": "auto",
    "precision": "float32",
    "patientAggregation": "mean",
    "decisionThreshold": 0.5,
}

STARTERS = {
    "datasetImport": {
        "source": {"path": ""},
        "slideIdColumn": "",
        "recursive": True,
        "includeMissingSlides": True,
        "missingValues": [""],
        "attributes": [],
        "patientIdFallback": "unresolved",
    },
    "testTarget": TEST_TARGET,
    "targetSplit": {
        "datasetId": "",
        "splitUnit": "slide",
        "target": TEST_TARGET,
        "predictors": [],
        "eligibility": [],
        "split": {
            "method": "random",
            "testFraction": 0.2,
            "seed": 42,
            "stratify": False,
            "trainRules": [],
            "testRules": [],
            "trainValues": [],
            "testValues": [],
        },
    },
    "cohort": {
        "protocolId": "",
        "developmentFeatureBundleId": "",
        "datasetId": "",
        "featureBundleId": "",
        "target": TEST_TARGET,
        "eligibility": [],
        "patientIdentifiers": "shared",
        "inference": COHORT_INFERENCE,
    },
    # A development-only training design: version 4 with its pools.
    "trainingSplit": {
        "version": 4,
        "pools": {
            "source": "rules",
            "trainSelection": "remaining",
            "validationSource": "training_fraction",
            "rules": {"train": [], "val": [], "test": []},
        },
        "mode": "kfold",
        "folds": 5,
        "seeds": [42],
        "stratify": True,
        "validationFraction": 0.15,
        "testFraction": 0.2,
        "repeats": 5,
        "outerFolds": 5,
        "innerFolds": 3,
        "domainPolicy": "all",
        "heldOutDomains": [],
        "heldOutSource": "fractions",
        "ratios": {"train": 0.8, "val": 0.2, "test": 0},
        "rules": {"train": [], "val": [], "test": []},
    },
}

APPLY = {
    "scope": "selected",
    "namePrefix": {"labeled": "Evaluation", "unlabeled": "Inference"},
    # A predictor applies its own frozen aggregation and threshold.
    "inference": {
        **COHORT_INFERENCE,
        "patientAggregation": "predictor",
        "decisionThreshold": "predictor",
    },
    "patientIdentifiers": "shared",
}

RESOURCES = {
    "training": {
        "maxConcurrentRuns": 1,
        "gpuIds": [0],
        "runsPerGpu": 1,
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 2,
        "ramGbPerRun": 8,
    },
    "interpretation": {"device": "cpu", "gpu": 0, "cpuThreadsPerRun": 4, "ramGbPerRun": 8},
}

# Starters for records the CLI and agent tools author from spec files. Placeholders begin
# with @ (a version tag) or end in … (an ID to fill in).
SPEC_STARTERS = {
    "targets": {**STARTERS["targetSplit"], "datasetId": "@dataset-tag"},
    "cohort": {
        **STARTERS["cohort"],
        "protocolId": "configuration-…",
        "developmentFeatureBundleId": "configuration-…",
        "datasetId": "@dataset-tag",
        "featureBundleId": "configuration-…",
    },
    "unlabeled-cohort": {
        **STARTERS["cohort"],
        "protocolId": "configuration-…",
        "developmentFeatureBundleId": "configuration-…",
        "datasetId": "@dataset-tag",
        "featureBundleId": "configuration-…",
        "purpose": "inference",
        "target": None,
    },
    "import": {**STARTERS["datasetImport"], "source": {"path": "/absolute/path/to/table.csv"}},
    "reference": {
        "cohortId": "@cohort-tag",
        "name": "",
        "datasetIds": ["@dataset-tag"],
        "field": "",
        "classes": [],
        "labels": {},
    },
    "features": {
        "datasetId": "@dataset-tag",
        "path": "/absolute/path/to/features",
        "encoderId": None,
        "fileSuffix": ".h5",
        "idSuffix": "",
        "recursive": True,
        "layout": "auto",
        "featureKind": "patch",
    },
    "feature-bundle": {"featureSetId": "@features-tag", "packArtifactIds": []},
    "feature-pack": {
        "featureSetId": "@features-tag",
        "action": "validate",
        "outputPath": None,
        "existingPath": None,
        "dtype": "preserve",
    },
    "extraction": {
        "datasetId": "@dataset-tag",
        "slideRoot": "/absolute/path/to/slides",
        "recursive": True,
        "outputPath": "/absolute/path/to/output",
        "options": {},
    },
    "interpretation": {
        "name": "Attention study",
        "predictorId": "configuration-…",
        "evaluationId": None,
        "clinicalAnalysisId": None,
        "featureBundleId": "configuration-…",
        "packArtifactId": None,
        "slideFolder": None,
        "encoderId": "",
        "slides": [
            {"slideId": "", "slidePath": "/absolute/path/to/slide.svs", "sourceFormat": "h5"}
        ],
        # The training policy on the CPU, with the interpretation page's threads and memory.
        "resources": {
            **RESOURCES["training"],
            "gpuIds": [],
            "cpuThreadsPerRun": RESOURCES["interpretation"]["cpuThreadsPerRun"],
            "dataLoaderWorkers": 0,
            "ramGbPerRun": RESOURCES["interpretation"]["ramGbPerRun"],
        },
    },
    "clinical-analysis": {
        "evaluationId": "configuration-…",
        "name": "Clinical utility analysis",
        "unit": "selected",
        "positiveClass": None,
        "threshold": None,
        "bins": 10,
        "thresholdMin": 0.01,
        "thresholdMax": 0.99,
        "thresholdSteps": 99,
        "referenceId": None,
    },
    "apply": {
        "cohortId": "@cohort-tag",
        "scope": APPLY["scope"],
        "predictorIds": ["configuration-…"],
        "namePrefix": APPLY["namePrefix"]["labeled"],
        "inference": APPLY["inference"],
        "patientIdentifiers": APPLY["patientIdentifiers"],
    },
}


def spec_starter(kind: str) -> dict:
    """A copy of one spec starter, every science field written out."""
    if kind not in SPEC_STARTERS:
        raise ValueError(f"Unknown spec kind {kind!r}; choose from: {', '.join(SPEC_STARTERS)}.")
    return deepcopy(SPEC_STARTERS[kind])


def recipe(preset: str = "default") -> dict:
    """A training recipe: the default, or a preset over its bases."""
    if preset == "default":
        return deepcopy(DEFAULT_RECIPE)
    if preset not in RECIPE_PRESETS:
        raise ValueError(
            f"Unknown recipe {preset!r}; choose from: default, {', '.join(RECIPE_PRESETS)}."
        )
    chosen = RECIPE_PRESETS[preset]
    merged: dict = {}
    for base in chosen["base"]:
        merged.update(DEFAULT_RECIPE if base == "default" else EXPERIMENTAL_DEFAULTS)
    return deepcopy({**merged, **chosen["values"]})


def batch(preset: str, *, inputs: dict, experiment_name: str) -> dict:
    """A development batch spec from a preset, as the browser's batch templates build it."""
    info = next((item for item in BATCH_PRESETS if item["id"] == preset), None)
    if info is None:
        raise ValueError(f"Unknown preset {preset!r}; choose from: {', '.join(PRESETS)}.")
    chosen = recipe(info["recipe"])
    grid = {
        "learningRates": info.get("learningRates") or [chosen["learningRate"]],
        "weightDecays": [chosen["weightDecay"]],
        "maxEpochs": [chosen["maxEpochs"]],
    }
    return deepcopy(
        {
            **BATCH_DEFAULTS,
            "experimentName": experiment_name,
            "batchName": "" if preset == "blank" else info["name"],
            "inputs": inputs,
            "recipe": chosen,
            "mode": info.get("mode", BATCH_DEFAULTS["mode"]),
            "grid": grid,
        }
    )


def describe() -> dict:
    """Everything above as one JSON document, for the browser and for clients."""
    return deepcopy(
        {
            "version": VERSION,
            "analysis": ANALYSIS,
            "recipes": {
                "default": DEFAULT_RECIPE,
                "experimentalDefaults": EXPERIMENTAL_DEFAULTS,
                "presets": RECIPE_PRESETS,
            },
            "batchPresets": BATCH_PRESETS,
            "batchDefaults": BATCH_DEFAULTS,
            "predictorPolicy": PREDICTOR_POLICY,
            "coupling": COUPLING,
            "starters": STARTERS,
            "specStarters": SPEC_STARTERS,
            "apply": APPLY,
            "resources": RESOURCES,
        }
    )
