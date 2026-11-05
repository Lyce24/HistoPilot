"""Dataset shift in feature space: what it reports for data with and without a shift."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from histopilot import shift
from histopilot.storage.packed import stat_stamp

DIMENSIONS = 24


def sets(rng, offset=0.0, development=300, cohort=150):
    """Slide embeddings around a shared centre; the cohort moved by ``offset`` along one axis."""
    centre = rng.normal(size=DIMENSIONS) * 3
    direction = np.eye(DIMENSIONS)[0]
    reference = centre + rng.normal(size=(development, DIMENSIONS))
    query = centre + offset * direction + rng.normal(size=(cohort, DIMENSIONS))
    return (
        {f"dev-{i:04d}": row for i, row in enumerate(reference)},
        {f"cohort-{i:04d}": row for i, row in enumerate(query)},
    )


def test_a_cohort_drawn_like_the_development_data_shows_little_shift():
    development, cohort = sets(np.random.default_rng(0))
    report = shift.analyze(development, cohort, resamples=200)
    summary = report["summary"]
    assert 0.35 < summary["domainAuroc"]["value"] < 0.65
    assert summary["domainAuroc"]["lower"] < 0.5 < summary["domainAuroc"]["upper"]
    assert summary["beyond"]["fraction"] < 0.06
    assert summary["verdict"] == "little"
    # About 1% of development slides lie beyond their own 99th percentile, by construction.
    own = [row for row in report["slides"] if row["set"] == "development"]
    assert sum(row["beyond"] for row in own) <= 0.02 * len(own) + 1


def test_a_shifted_cohort_is_separable_and_beyond_the_development_range():
    development, cohort = sets(np.random.default_rng(1), offset=6.0)
    report = shift.analyze(development, cohort, resamples=200)
    summary = report["summary"]
    assert summary["domainAuroc"]["value"] > 0.95
    assert summary["beyond"]["fraction"] > 0.5
    assert summary["verdict"] == "strong"
    assert summary["cohortSlides"] == 150 and summary["developmentSlides"] == 300
    rows = {row["slideId"]: row for row in report["slides"]}
    assert len(rows) == 450
    assert all("domainScore" in row for row in rows.values() if row["set"] == "cohort")
    assert "verdict" in summary and summary["verdictText"]
    assert shift.describe(summary).startswith("The cohort differs clearly")


def test_the_report_is_deterministic():
    development, cohort = sets(np.random.default_rng(2), offset=1.0)
    first = shift.analyze(development, cohort, resamples=100)
    second = shift.analyze(dict(reversed(development.items())), cohort, resamples=100)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_neighbour_distances_leave_each_development_slide_out():
    points = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    own = shift.neighbour_distances(points, k=1)
    # The duplicate is each first slide's nearest neighbour, never the slide itself.
    assert own[0] == pytest.approx(0.0) and own[1] == pytest.approx(0.0)
    assert own[2] == pytest.approx(1.0)
    query = shift.neighbour_distances(points, np.array([[0.0, 1.0]]), k=1)
    assert query[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite and nonzero"):
        shift.neighbour_distances(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_auroc_counts_ties_half_and_needs_both_classes():
    assert shift.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert shift.auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert shift.auroc([1, 1], [0.2, 0.3]) is None


def test_wilson_interval_and_verdicts():
    lower, upper = shift.wilson(0, 100)
    assert lower == 0.0 and 0.03 < upper < 0.04
    assert shift.wilson(5, 0) == (None, None)
    assert shift.verdict(0.5, 0.01) == "little"
    assert shift.verdict(0.7, 0.01) == "moderate"
    assert shift.verdict(0.6, 0.30) == "strong"


def test_overlapping_or_tiny_sets_are_refused():
    development, cohort = sets(np.random.default_rng(3), development=5, cohort=5)
    with pytest.raises(ValueError, match="both sets"):
        shift.analyze(development, {**cohort, "dev-0000": development["dev-0000"]})
    with pytest.raises(ValueError, match="at least two"):
        shift.analyze(development, {"cohort-0000": cohort["cohort-0000"]})


def write_slide(path, values):
    with h5py.File(path, "w") as handle:
        handle["features"] = values
        handle["coords"] = np.zeros((len(values), 2), dtype=np.int32)
    return {"path": str(path), "slideId": path.stem, **stat_stamp(path.stat())}


def test_slide_embedding_is_the_mean_of_verified_patch_features(tmp_path, monkeypatch):
    values = np.random.default_rng(4).normal(size=(37, DIMENSIONS)).astype(np.float16)
    entry = write_slide(tmp_path / "slide.h5", values)
    monkeypatch.setattr(shift, "CHUNK_ROWS", 8)  # several chunks, as a large slide reads
    embedding = shift.slide_embedding(entry, DIMENSIONS)
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(embedding, values.astype(np.float64).mean(axis=0), rtol=1e-5)
    np.save(tmp_path / "slide.npy", values)
    numpy_entry = {
        "path": str(tmp_path / "slide.npy"),
        **stat_stamp((tmp_path / "slide.npy").stat()),
    }
    np.testing.assert_allclose(shift.slide_embedding(numpy_entry, DIMENSIONS), embedding, rtol=1e-6)
    with pytest.raises(ValueError, match="not \\[patches"):
        shift.slide_embedding(entry, DIMENSIONS + 1)
    (tmp_path / "slide.h5").write_bytes(b"changed")
    with pytest.raises(Exception, match=r"changed|differs|stamp|Cannot read"):
        shift.slide_embedding(entry, DIMENSIONS)


def test_a_plan_runs_end_to_end_and_receipts_its_outputs(tmp_path):
    rng = np.random.default_rng(5)
    files = {"development": {}, "cohort": {}}
    # Real patch features share a large common direction; slides vary around it.
    centre = rng.normal(size=DIMENSIONS) * 3
    for name, offset, count in (("development", 0.0, 12), ("cohort", 4.0, 8)):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(count):
            values = centre + rng.normal(size=(20, DIMENSIONS))
            values[:, 0] += offset
            values = values.astype(np.float32)
            entry = write_slide(folder / f"{name}-{index}.h5", values)
            files[name][entry["slideId"]] = entry
    plan = {
        "kind": "shift",
        "data": {
            "featureDim": DIMENSIONS,
            "development": {"featureFiles": files["development"]},
            "cohort": {"featureFiles": files["cohort"]},
        },
    }
    job = tmp_path / "job"
    job.mkdir()
    result = shift.analyze_plan(plan, job)
    assert result["state"] == "succeeded" and result["kind"] == "shift"
    for name, receipt in result["artifacts"].items():
        content = (job / name).read_bytes()
        assert receipt == {
            "path": str(job / name),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    report = json.loads((job / "shift.json").read_text())
    assert report["summary"]["cohortSlides"] == 8 and report["summary"]["verdict"] == "strong"
    assert json.loads((job / "progress.json").read_text())["completed"] == 20
    arrays = np.load(job / "slide-embeddings.npz")
    assert arrays["development"].shape == (12, DIMENSIONS)
    assert list(arrays["cohort_ids"]) == sorted(files["cohort"])
