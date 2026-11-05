"""Dataset shift between a predictor's development slides and a cohort, in feature space.

Every slide becomes one embedding, the mean of its patch features: the input every MIL model
here starts from. The cohort is then compared with the slides the predictor was developed on
in three ways.

- **Atypicality.** Each slide's mean cosine distance to its ``k`` nearest development slides;
  a development slide leaves itself out. A cohort slide lies *beyond the development range*
  when that distance exceeds the 99th percentile of the development slides' own distances,
  so about 1% of development slides do by construction.
- **Separability.** How well a ridge-penalised logistic regression on leading principal
  components, cross-validated in stratified folds, tells cohort slides from development
  slides (domain AUROC). 0.5 means the sets are indistinguishable in feature space; values
  near 1 mean a linear model can name each slide's source, as scanners, stains and sites do.
- **A two-dimensional view.** Both sets projected on the development slides' first two
  principal components.

The analysis is label-free: its job never reads a label. Performance by atypicality joins the
per-slide distances with a scored run's predictions and labels in the control service
(``application/shift.py``). Everything here is descriptive: shift explains degraded
performance, it does not prove it.

numpy only, so the control service and the worker share it without Torch.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 1
NEIGHBOURS = 10
BEYOND_QUANTILE = 0.99
DOMAIN_FOLDS = 5
DOMAIN_COMPONENTS = 32
DOMAIN_RIDGE = 1.0
BOOTSTRAP_RESAMPLES = 1000
SEED = 42
# Rows read at once from one slide's features, to bound the worker's memory.
CHUNK_ROWS = 65536
METHOD = {
    "version": VERSION,
    "embedding": "mean of each slide's patch features",
    "distance": "mean cosine distance to the k nearest development slides",
    "neighbours": NEIGHBOURS,
    "beyondQuantile": BEYOND_QUANTILE,
    "domainClassifier": (
        "logistic regression with a ridge penalty on the leading principal components, "
        "standardised and fitted within each training fold; out-of-fold AUROC"
    ),
    "domainFolds": DOMAIN_FOLDS,
    "domainComponents": DOMAIN_COMPONENTS,
    "domainRidge": DOMAIN_RIDGE,
    "bootstrapResamples": BOOTSTRAP_RESAMPLES,
    "seed": SEED,
}
# Plain-language summaries of the two measures. Descriptive guidance, not a test.
VERDICTS = {
    "little": "The cohort looks like the development data in feature space.",
    "moderate": "Part of the cohort differs from the development data in feature space.",
    "strong": "The cohort differs clearly from the development data in feature space.",
}


def _unit(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms == 0):
        raise ValueError("Slide embeddings must be finite and nonzero.")
    return values / norms


def neighbour_distances(reference, query=None, k=NEIGHBOURS, block=1024):
    """Mean cosine distance from each query row to its ``k`` nearest reference rows.

    Without ``query`` the reference is its own query, and each row leaves itself out.
    """
    ref = _unit(reference)
    leave_out = query is None
    rows = ref if leave_out else _unit(query)
    k = min(k, ref.shape[0] - (1 if leave_out else 0))
    if k < 1:
        raise ValueError("At least two development slides are needed.")
    distances = np.empty(rows.shape[0])
    for start in range(0, rows.shape[0], block):
        similarity = rows[start : start + block] @ ref.T
        if leave_out:
            index = np.arange(start, start + similarity.shape[0])
            similarity[index - start, index] = -np.inf
        nearest = np.partition(similarity, -k, axis=1)[:, -k:]
        distances[start : start + similarity.shape[0]] = 1.0 - nearest.mean(axis=1)
    return distances


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    new = np.r_[True, ordered[1:] != ordered[:-1]]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(values)] - 1
    ranks = np.empty(len(values))
    ranks[order] = ((starts + ends) / 2 + 1)[np.cumsum(new) - 1]
    return ranks


def auroc(labels, scores):
    """Area under the ROC curve with ties counted half; None without both classes."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _sigmoid(values):
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def _logistic(features, labels, ridge, iterations=100):
    """Newton's method for ridge-penalised logistic regression; the intercept is free."""
    design = np.c_[np.ones(len(features)), features]
    weights = np.zeros(design.shape[1])
    penalty = np.full(design.shape[1], float(ridge))
    penalty[0] = 0.0
    for _ in range(iterations):
        probability = _sigmoid(design @ weights)
        gradient = design.T @ (probability - labels) + penalty * weights
        curvature = probability * (1.0 - probability)
        hessian = (design * curvature[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hessian + 1e-9 * np.eye(len(weights)), gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def _folds(labels, folds, rng):
    assignment = np.empty(len(labels), dtype=int)
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        rng.shuffle(members)
        assignment[members] = np.arange(len(members)) % folds
    return assignment


def domain_scores(
    development,
    cohort,
    *,
    folds=DOMAIN_FOLDS,
    components=DOMAIN_COMPONENTS,
    ridge=DOMAIN_RIDGE,
    seed=SEED,
):
    """Out-of-fold probability that each slide is a cohort slide, development first.

    Standardisation and principal components are fitted on each training fold alone, so a
    held-out slide never shapes the space it is scored in.
    """
    values = np.vstack([development, cohort]).astype(np.float64)
    labels = np.r_[np.zeros(len(development)), np.ones(len(cohort))]
    folds = min(folds, len(development), len(cohort))
    if folds < 2:
        raise ValueError("Each set needs at least two slides to cross-validate.")
    assignment = _folds(labels, folds, np.random.default_rng(seed))
    scores = np.empty(len(labels))
    for fold in range(folds):
        train, test = assignment != fold, assignment == fold
        mean = values[train].mean(axis=0)
        spread = values[train].std(axis=0) + 1e-8
        fitted = (values[train] - mean) / spread
        _, singular, basis = np.linalg.svd(fitted, full_matrices=False)
        # Components with no variance in the training fold carry only rounding noise, and
        # whitening would magnify it without bound.
        keep = int(min(components, np.sum(singular > singular[0] * 1e-6)))
        scale = singular[:keep] / np.sqrt(max(1, train.sum() - 1))
        project = basis[:keep].T / scale
        weights = _logistic(fitted @ project, labels[train], ridge)
        held_out = ((values[test] - mean) / spread) @ project
        scores[test] = _sigmoid(np.c_[np.ones(test.sum()), held_out] @ weights)
    return scores, labels


def bootstrap_auroc(labels, scores, *, resamples=BOOTSTRAP_RESAMPLES, seed=SEED, level=0.95):
    """Percentile interval over slides, resampled within each set; the scores stay fixed."""
    labels = np.asarray(labels, dtype=bool)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        values.append(auroc(labels[index], scores[index]))
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def wilson(count, total, level_z=1.959963984540054):
    """Wilson score interval for a proportion."""
    if total == 0:
        return None, None
    share = count / total
    denominator = 1 + level_z**2 / total
    centre = (share + level_z**2 / (2 * total)) / denominator
    half = level_z * np.sqrt(share * (1 - share) / total + level_z**2 / (4 * total**2))
    half /= denominator
    lower = 0.0 if count == 0 else float(max(0.0, centre - half))
    upper = 1.0 if count == total else float(min(1.0, centre + half))
    return lower, upper


def plane(development, cohort):
    """Both sets on the development slides' first two principal components."""
    reference = np.asarray(development, dtype=np.float64)
    mean = reference.mean(axis=0)
    _, _, basis = np.linalg.svd(reference - mean, full_matrices=False)
    axes = basis[:2].T
    if axes.shape[1] < 2:
        axes = np.c_[axes, np.zeros((axes.shape[0], 2 - axes.shape[1]))]
    return (reference - mean) @ axes, (np.asarray(cohort, dtype=np.float64) - mean) @ axes


def verdict(domain, beyond_fraction):
    if domain is None:
        return None
    if domain >= 0.85 or beyond_fraction >= 0.25:
        return "strong"
    if domain >= 0.65 or beyond_fraction >= 0.05:
        return "moderate"
    return "little"


def _quantiles(values):
    return {
        name: float(np.quantile(values, q))
        for name, q in (("p05", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95))
    }


def analyze(development: dict, cohort: dict, *, seed=SEED, resamples=BOOTSTRAP_RESAMPLES) -> dict:
    """The shift report for slide embeddings keyed by slide ID (development, cohort)."""
    if len(development) < 2 or len(cohort) < 2:
        raise ValueError("Each set needs at least two slides.")
    shared = set(development) & set(cohort)
    if shared:
        raise ValueError(f"{len(shared)} slides are in both sets; a cohort must be distinct.")
    development_ids, cohort_ids = sorted(development), sorted(cohort)
    reference = np.stack(
        [np.asarray(development[key], dtype=np.float64) for key in development_ids]
    )
    query = np.stack([np.asarray(cohort[key], dtype=np.float64) for key in cohort_ids])
    if reference.shape[1] != query.shape[1]:
        raise ValueError("Development and cohort embeddings differ in dimension.")
    own = neighbour_distances(reference)
    distance = neighbour_distances(reference, query)
    threshold = float(np.quantile(own, BEYOND_QUANTILE))
    ordered = np.sort(own)

    def percentile(values):
        return np.searchsorted(ordered, values, side="right") / len(ordered)

    beyond = distance > threshold
    scores, labels = domain_scores(reference, query, seed=seed)
    domain = auroc(labels, scores)
    lower, upper = bootstrap_auroc(labels, scores, resamples=resamples, seed=seed)
    share = float(beyond.mean())
    share_lower, share_upper = wilson(int(beyond.sum()), len(beyond))
    xy_development, xy_cohort = plane(reference, query)
    slides = [
        {
            "slideId": key,
            "set": "development",
            "distance": float(value),
            "percentile": float(rank),
            "beyond": bool(value > threshold),
            "x": float(x),
            "y": float(y),
        }
        for key, value, rank, (x, y) in zip(
            development_ids, own, percentile(own), xy_development, strict=True
        )
    ] + [
        {
            "slideId": key,
            "set": "cohort",
            "distance": float(value),
            "percentile": float(rank),
            "beyond": bool(flag),
            "domainScore": float(score),
            "x": float(x),
            "y": float(y),
        }
        for key, value, rank, flag, score, (x, y) in zip(
            cohort_ids,
            distance,
            percentile(distance),
            beyond,
            scores[len(development_ids) :],
            xy_cohort,
            strict=True,
        )
    ]
    kind = verdict(domain, share)
    return {
        "version": VERSION,
        "method": METHOD,
        "summary": {
            "developmentSlides": len(development_ids),
            "cohortSlides": len(cohort_ids),
            "dimensions": int(reference.shape[1]),
            "domainAuroc": {"value": domain, "lower": lower, "upper": upper},
            "beyond": {
                "count": int(beyond.sum()),
                "fraction": share,
                "lower": share_lower,
                "upper": share_upper,
                "threshold": threshold,
            },
            "distance": {"development": _quantiles(own), "cohort": _quantiles(distance)},
            "verdict": kind,
            "verdictText": VERDICTS.get(kind),
        },
        "slides": slides,
    }


def slide_embedding(entry: dict, dimensions: int) -> np.ndarray:
    """The mean of one slide's patch features, read in chunks from its verified file."""
    from histopilot.storage.packed import embedded_dataset, open_source

    path = Path(entry["path"])
    total, count = np.zeros(dimensions), 0
    with open_source(path, entry) as (stream, _stamp):
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(stream, "r") as handle:
                source = embedded_dataset(handle, "features")
                shape = (1, source.shape[0]) if len(source.shape) == 1 else source.shape
                if len(shape) != 2 or shape[1] != dimensions:
                    raise ValueError(f"{path}: features are not [patches, {dimensions}].")
                if len(source.shape) == 1:
                    total, count = np.asarray(source[:], dtype=np.float64), 1
                else:
                    for start in range(0, shape[0], CHUNK_ROWS):
                        block = np.asarray(source[start : start + CHUNK_ROWS], dtype=np.float64)
                        total += block.sum(axis=0)
                        count += block.shape[0]
        elif suffix == ".npy":
            values = np.load(stream, allow_pickle=False)
            values = values.reshape(1, -1) if values.ndim == 1 else values
            if values.ndim != 2 or values.shape[1] != dimensions:
                raise ValueError(f"{path}: features are not [patches, {dimensions}].")
            total, count = values.astype(np.float64).sum(axis=0), values.shape[0]
        else:
            raise ValueError(f"{path}: dataset shift reads HDF5 or NumPy feature files.")
    if count == 0:
        raise ValueError(f"{path}: the slide has no patch features.")
    embedding = total / count
    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"{path}: features contain values that are not finite.")
    return embedding.astype(np.float32)


def _receipt(path: Path) -> dict:
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def analyze_plan(plan: dict, folder: Path) -> dict:
    """Run a frozen shift-analysis plan in its job folder and return the job's result."""
    from histopilot.storage.io import content_hash, write_json_atomic

    folder = Path(folder)
    data = plan["data"]
    dimensions = data["featureDim"]
    sets = {name: data[name]["featureFiles"] for name in ("development", "cohort")}
    total = sum(len(files) for files in sets.values())
    embeddings, done = {}, 0
    for name, files in sets.items():
        embeddings[name] = {}
        for slide_id in sorted(files):
            embeddings[name][slide_id] = slide_embedding(files[slide_id], dimensions)
            done += 1
            if done % 25 == 0 or done == total:
                write_json_atomic(
                    folder / "progress.json",
                    {"stage": "embedding", "completed": done, "total": total, "unit": "slides"},
                )
    report = analyze(embeddings["development"], embeddings["cohort"])
    report["inputHash"] = content_hash(data)
    output = folder / "shift.json"
    write_json_atomic(output, report)
    arrays = folder / "slide-embeddings.npz"
    with arrays.open("wb") as stream:
        np.savez(
            stream,
            **{f"{name}_ids": np.array(sorted(embeddings[name]), dtype=str) for name in embeddings},
            **{
                name: np.stack([embeddings[name][key] for key in sorted(embeddings[name])])
                for name in embeddings
            },
        )
    result = {
        "state": "succeeded",
        # The compute record this job ran for, as every compute result names it.
        "runId": plan.get("recordId"),
        "kind": "shift",
        "summary": report["summary"],
        "artifacts": {"shift.json": _receipt(output), "slide-embeddings.npz": _receipt(arrays)},
        "inputHash": report["inputHash"],
    }
    write_json_atomic(folder / "result.json", result)
    return result


def describe(summary: dict) -> str:
    """One sentence for a report or a terminal."""
    domain = summary["domainAuroc"]
    beyond = summary["beyond"]
    parts = [
        f"domain AUROC {domain['value']:.2f} ({domain['lower']:.2f}–{domain['upper']:.2f})",
        (
            f"{beyond['fraction']:.0%} of cohort slides beyond the development range "
            f"({beyond['count']} of {summary['cohortSlides']})"
        ),
    ]
    return f"{VERDICTS.get(summary.get('verdict'), '')} {'; '.join(parts)}.".strip()


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
