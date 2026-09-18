"""Bounded, isolated adversarial checks for the real BLCA study coordinator.

This module never starts a server, trains a model, or cancels a study job. Native
previews read the real study; invalid drafts and altered inputs live in disposable
snapshots. Reports contain aggregate evidence, never clinical rows or slide IDs.
Run only after the main coordinator has written its state.json.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from histopilot.application.development import DevelopmentService  # noqa: E402
from histopilot.application.evaluation_runs import EvaluationRunService  # noqa: E402
from histopilot.application.evaluations import EvaluationService  # noqa: E402
from histopilot.application.feature_bundles import FeatureBundleService  # noqa: E402
from histopilot.application.feature_packs import FeaturePackService  # noqa: E402
from histopilot.application.features import FeatureService  # noqa: E402
from histopilot.application.imports import ImportService  # noqa: E402
from histopilot.application.interpretation import InterpretationService  # noqa: E402
from histopilot.application.model_experiments import ModelExperimentService  # noqa: E402
from histopilot.application.protocols import ProtocolService  # noqa: E402
from histopilot.schemas.development import DevelopmentBatchSpec  # noqa: E402
from histopilot.schemas.evaluations import EvaluationSpec, InferenceSettings  # noqa: E402
from histopilot.schemas.feature_bundles import FeatureBundleSpec  # noqa: E402
from histopilot.schemas.feature_packs import FeaturePackSpec  # noqa: E402
from histopilot.schemas.features import FeatureSpec  # noqa: E402
from histopilot.schemas.imports import ImportSpec  # noqa: E402
from histopilot.schemas.interpretation import (  # noqa: E402
    InterpretationSelection,
    InterpretationSlide,
)
from histopilot.schemas.predictors import EvaluationRunSelection  # noqa: E402
from histopilot.storage.attention_inputs import inspect_inputs  # noqa: E402
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.packed import PackedFeatureStore  # noqa: E402
from histopilot.storage.project_lock import StorageError, writer_lock  # noqa: E402
from histopilot.storage.scientific import DATABASE_FILE, ScientificStore  # noqa: E402
from histopilot.viewer.slide_images import allowed_file, inspect_slide  # noqa: E402
from histopilot.workers.pack_features import run_job  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402

STAGES = ("dataset", "target", "features", "experiments", "evaluation", "attention")
CHECK_VERSION = 1
MAX_COPY_BYTES = 64 * 1024**2


class Unavailable(Exception):
    """A missing prerequisite is an explicit gap, never a successful check."""


class InlineValidationExecutor:
    """Allocate a native receipt; run its bounded copied-input worker synchronously."""

    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, session, runner, plan):
        # The caller invokes run_job immediately. No background process is launched.
        pass


def _stamp():
    return datetime.now(UTC).isoformat()


def _codes(result):
    return sorted({item["code"] for item in result.get("findings", [])
                   if item.get("severity") == "error"})


def _blocked(result, code, flag="canFreeze"):
    codes = _codes(result)
    if result.get(flag) is not False or code not in codes:
        raise AssertionError(f"Expected blocked {code}; observed codes {codes}")
    return {"blockingCodes": codes}


def _reject(call, code=None, exception=StorageError, contains=None):
    try:
        call()
    except exception as error:
        if code is not None and getattr(error, "code", None) != code:
            raise AssertionError(f"Expected {code}; observed {getattr(error, 'code', None)}") from None
        if contains and contains not in str(error):
            raise AssertionError("Rejection occurred for an unexpected reason") from None
        return {"errorType": type(error).__name__, **({"code": code} if code else {})}
    raise AssertionError("Invalid input was accepted")


class Checks:
    def __init__(self, state_path, stage, stack):
        self.path = Path(state_path).resolve(strict=True)
        self.state = json.loads(self.path.read_text())
        self.stage, self.stack = stage, stack
        self.project = Path(self.state["projectPath"]).resolve(strict=True)
        if self.project.name != "blca-e2e-20260917":
            raise ValueError("Robustness checks are restricted to the new BLCA project")
        self.store = ScientificStore(self.project, self.state["projectId"])
        self.output = self.path.parent / "robustness"
        self.output.mkdir(exist_ok=True)
        self.scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(
            prefix=f".{stage}-", dir=self.output)))
        roots = [self.project, self.scratch]
        for key in ("workspace", "slideRoot", "featurePath"):
            if self.state.get(key):
                roots.append(Path(self.state[key]))
        if self.state.get("featurePath"):
            # Attached TRIDENT inventories also pin sibling coordinate files and
            # configuration/provenance under the encoder's enclosing job folder.
            roots.append(Path(self.state["featurePath"]).parent.parent)
        if self.state.get("metadataPath"):
            roots.append(Path(self.state["metadataPath"]).parent)
        self.filesystem = LocalFilesystem(tuple(roots))
        self._clone_store = None
        self._copied_feature = None
        self._attention = None

    def require(self, *keys):
        missing = [key for key in keys if not self.state.get(key)]
        if missing:
            raise Unavailable("Awaiting state fields: " + ", ".join(missing))

    def clone(self):
        """Consistent SQLite backup plus bounded immutable metadata, never tensors."""
        if self._clone_store is not None:
            return self._clone_store
        folder = self.scratch / "snapshot"
        folder.mkdir()
        database = self.project / DATABASE_FILE
        if database.stat().st_size > MAX_COPY_BYTES:
            raise Unavailable("Scientific database exceeds the bounded snapshot budget")
        deadline = time.monotonic() + 15

        def progress(*_):
            if time.monotonic() > deadline:
                raise Unavailable("Scientific snapshot exceeded its 15-second budget")

        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(folder / DATABASE_FILE) as target:
                source.backup(target, pages=128, progress=progress, sleep=0.05)
        if self.state.get("datasetId"):
            source = self.project / "datasets" / self.state["datasetId"]
            files = list(source.rglob("*"))
            if any(item.is_symlink() for item in files):
                raise Unavailable("Dataset snapshot refuses symbolic links")
            if sum(item.stat().st_size for item in files if item.is_file()) > MAX_COPY_BYTES:
                raise Unavailable("Dataset metadata exceeds the bounded snapshot budget")
            shutil.copytree(source, folder / "datasets" / source.name)
        self._clone_store = ScientificStore(folder, self.state["projectId"])
        self._clone_store.initialize()
        return self._clone_store

    def import_spec(self):
        self.require("datasetId")
        return copy.deepcopy(self.state.get("importSpec") or self.store.get_dataset(
            self.state["datasetId"])["manifest"]["provenance"]["mapping"])

    def import_preview(self, spec):
        service = ImportService(self.clone(), self.filesystem)
        draft = service.store.create_draft("import", "Disposable robustness preview",
                                           {"type": "dataset-import", "spec": spec})
        return service.preview(draft["id"], 1)

    def dataset_wrong_column(self):
        spec = self.import_spec()
        spec["slideIdColumn"] = "__nonexistent_slide_identity__"
        return _reject(lambda: self.import_preview(spec), "MAPPING_COLUMN_MISSING")

    def dataset_blank_identity(self):
        spec = self.import_spec()
        table = ImportService(self.store, self.filesystem)._table(ImportSpec.model_validate(spec).source)
        # Modify an in-memory upload only; original CSV and frozen records stay intact.
        text = table["bytes"].decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise Unavailable("The study source has no rows")
        rows[0][spec["slideIdColumn"]] = ""
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        spec["source"] = {"filename": "disposable.csv", "contentBase64": base64.b64encode(
            stream.getvalue().encode()).decode()}
        return _blocked(self.import_preview(spec), "SLIDE_ID_MISSING")

    def dataset_revision(self):
        self.require("importDraftId")
        service = ImportService(self.store, self.filesystem)
        revision = self.store.get_draft(self.state["importDraftId"])["revision"]
        return _reject(lambda: service.preview(self.state["importDraftId"], revision + 1),
                       "REVISION_CONFLICT")

    def dataset_stale_preview(self):
        service = ImportService(self.clone(), self.filesystem)
        draft = service.store.create_draft("import", "Disposable stale review",
                                           {"type": "dataset-import", "spec": self.import_spec()})
        return _reject(lambda: service.freeze(draft["id"], 1, "0" * 64,
                                             "disposable-stale-preview"), "PREVIEW_STALE")

    def dataset_label_agreement(self):
        self.require("datasetId")
        rows = json.loads(self.store.read_artifact(self.state["datasetId"], "records.json"))
        expected = {"low": "0", "high": "1"}
        mismatched = 0
        for row in rows:
            values = row.get("attributes", {})
            text = str(values.get("who2022_text", "")).strip().lower()
            binary = str(values.get("who2022_binary", "")).strip()
            if text not in expected or binary != expected[text]:
                mismatched += 1
        if mismatched:
            raise AssertionError(f"WHO 2022 text/binary mapping disagrees in {mismatched} rows")
        return {"rowsChecked": len(rows), "mismatches": mismatched,
                "mapping": {"low": 0, "high": 1}}

    def protocol_spec(self):
        self.require("protocolId")
        return copy.deepcopy(self.store.get_configuration(self.state["protocolId"])["manifest"]["spec"])

    def protocol_preview(self, spec):
        # Target checks do not need to rescan 138 feature headers per invalid input.
        # Keep the real frozen dataset, target and partitions; detach optional features.
        spec.pop("featureBundleId", None)
        spec.pop("featureSetId", None)
        spec["featureCoverage"] = "require"
        service = ProtocolService(self.clone(), self.filesystem)
        draft = service.store.create_draft("experiment", "Disposable target preflight",
                                           {"type": "analysis-protocol", "spec": spec})
        return service.preview(draft["id"], 1)

    def target_unknown_labels(self):
        spec = self.protocol_spec()
        spec["target"]["labels"] = {f"__unknown_{i}__": value
                                    for i, value in enumerate(spec["target"]["classes"])}
        return _blocked(self.protocol_preview(spec), "UNMAPPED_LABEL")

    def target_leakage(self):
        spec = self.protocol_spec()
        spec["predictors"] = [spec["target"]["field"]]
        return _blocked(self.protocol_preview(spec), "TARGET_PREDICTOR_LEAKAGE")

    def copied_feature(self):
        self.require("featureId")
        if self._copied_feature is None:
            document = self.store.get_configuration(self.state["featureId"])
            entry = min(document["manifest"]["files"], key=lambda row: row["patchCount"])
            source = Path(entry["path"])
            if source.stat().st_size > MAX_COPY_BYTES:
                raise Unavailable("Smallest feature file exceeds the 64 MiB copy budget")
            destination = self.scratch / "source-copy" / source.name
            destination.parent.mkdir()
            shutil.copyfile(source, destination)
            self._copied_feature = (destination, entry)
        return self._copied_feature

    def features_wrong_encoder(self):
        path, _ = self.copied_feature()
        spec = FeatureSpec(path=str(path.parent), encoderId="phikon")
        return _blocked(FeatureService(self.clone(), self.filesystem).preview(spec),
                        "INVALID_FEATURE_HEADER")

    def features_pack_parity(self):
        import h5py
        import numpy as np

        self.require("featureId", "packPath", "packId")
        files = sorted(self.store.get_configuration(self.state["featureId"])["manifest"]["files"],
                       key=lambda row: row["slideId"])
        selected = sorted({0, len(files) // 2, len(files) - 1})
        rows_checked = 0
        with PackedFeatureStore(Path(self.state["packPath"]), verify=False) as packed:
            if packed.manifest["outputDtype"] != "float32":
                raise AssertionError("Expected the preserved float32 pack")
            for index in selected:
                entry = files[index]
                with h5py.File(entry["path"], "r") as original:
                    rows = sorted({0, len(original["features"]) // 2, len(original["features"]) - 1})
                    source = original["features"][rows]
                    materialized = packed.read_features(entry["slideId"], rows)
                    if source.dtype != materialized.dtype or source.tobytes() != materialized.tobytes():
                        raise AssertionError("Native and packed feature bits differ")
                    if not np.array_equal(original["coords"][rows], packed.read_coords(entry["slideId"], rows)):
                        raise AssertionError("Native and packed coordinates differ")
                    rows_checked += len(rows)
        return {"slidesSampled": len(selected), "rowsSampled": rows_checked,
                "featureBitsIdentical": True, "coordinatesEqual": True, "dtype": "float32",
                "scope": "Bounded independent sample; authoritative worker verifies the full pack"}

    def features_slide_integrity(self):
        import h5py
        import numpy as np
        import openslide

        from histopilot.storage.attention_inputs import _geometry, _metadata

        self.require("datasetId", "featureId")
        entries = self.store.get_configuration(self.state["featureId"])["manifest"]["files"]
        rows = json.loads(self.store.read_artifact(self.state["datasetId"], "records.json"))
        indexed = {row["slideId"]: row for row in rows}
        decoded, patches, clipped = 0, 0, 0
        geometries = set()
        deadline = time.monotonic() + 12 * 60
        for entry in entries:
            if time.monotonic() > deadline:
                raise Unavailable(f"Bounded slide check reached its 12-minute budget after {decoded} slides")
            row = indexed[entry["slideId"]]
            path = row.get("slidePath") or str(Path(self.state["slideRoot"]) / (entry["slideId"] + ".tiff"))
            path = allowed_file(self.filesystem, path)
            with openslide.OpenSlide(str(path)) as slide:
                width, height = slide.dimensions
                if width <= 0 or height <= 0 or slide.level_count < 1:
                    raise AssertionError("An original slide has invalid pyramid dimensions")
                size = (min(64, width), min(64, height))
                region = slide.read_region(((width - size[0]) // 2, (height - size[1]) // 2), 0, size)
                region.load()
                if region.size != size:
                    raise AssertionError("An original slide crop could not be decoded")
                geometry = {"width": width, "height": height,
                            "levelDownsamples": list(slide.level_downsamples)}
            with h5py.File(entry["path"], "r") as source:
                coords = source["coords"]
                patch_width, patch_height = _geometry(_metadata(source, coords), {}, geometry)
                geometries.add((patch_width, patch_height))
                for offset in range(0, len(coords), 8192):
                    xy = coords[offset:offset + 8192]
                    if np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height):
                        raise AssertionError("An embedded patch origin lies outside its original slide")
                    clipped += int(np.count_nonzero((xy[:, 0] + patch_width > width) |
                                                   (xy[:, 1] + patch_height > height)))
                    patches += len(xy)
            decoded += 1
        if decoded != len(rows):
            raise AssertionError("Original-slide verification did not cover every frozen dataset row")
        if geometries != {(512.0, 512.0)}:
            raise AssertionError("Original UNI v1 coordinate footprint differs from 512 level-0 pixels")
        return {"slidesDecoded": decoded, "datasetSlides": len(rows), "patchOriginsChecked": patches,
                "outOfBoundsOrigins": 0, "edgeClippedFootprints": clipped,
                "patchWidthLevel0": 512, "patchHeightLevel0": 512, "decodedCropMaximumPixels": 4096,
                "scope": "All headers, one crop per slide and every coordinate; not whole-slide pixel decoding"}

    def features_mutated_receipt(self):
        import h5py
        import numpy as np

        path, _ = self.copied_feature()
        store = self.clone()
        service = FeatureService(store, self.filesystem)
        spec = FeatureSpec(path=str(path.parent), encoderId="uni_v1")
        preview = service.preview(spec)
        if not preview["canFreeze"]:
            raise AssertionError("The untouched disposable source must attach before mutation")
        feature = service.freeze(spec, preview["previewHash"], "disposable-source")
        packs = FeaturePackService(store, self.filesystem, InlineValidationExecutor())
        request = FeaturePackSpec(featureSetId=feature["id"], action="validate")
        preview = packs.preview(request)
        if not preview["canRun"]:
            raise AssertionError("Copied-input validation could not be prepared")
        job = packs.submit(request, preview["previewHash"], "disposable-validation")
        result = run_job(packs.folder / job["id"] / "plan.json")
        if result["state"] != "succeeded" or not packs.validation_for(feature["id"])["current"]:
            raise AssertionError("The native copied-input worker did not authenticate its receipt")
        bundles = FeatureBundleService(store, self.filesystem)
        request = FeatureBundleSpec(featureSetId=feature["id"])
        if not bundles.preview(request)["canFreeze"]:
            raise AssertionError("The unmodified validated copy must permit bundle publication")
        with h5py.File(path, "r+") as handle:
            before = handle["features"][0, 0]
            handle["features"][0, 0] = np.nextafter(before, np.float32(np.inf))
        receipt = packs.validation_for(feature["id"])
        if receipt["current"]:
            raise AssertionError("Mutation failed to invalidate the full validation receipt")
        evidence = _blocked(bundles.preview(request), "FULL_FEATURE_VALIDATION_REQUIRED")
        return {**evidence, "copiedFiles": 1, "nativeWorkerSucceededBeforeMutation": True,
                "receiptInvalidated": True, "originalInputsMutated": False}

    def batch_spec(self):
        self.require("batchId")
        document = self.store.get_configuration(self.state["batchId"])
        spec = copy.deepcopy(document["manifest"]["spec"])
        spec.pop("experimentId", None)
        spec.pop("experimentRevision", None)
        return spec

    def experiment_model(self, model, code):
        spec = self.batch_spec()
        spec["recipe"]["model"] = model
        spec["mode"] = "single"
        spec["grid"] = {}
        spec["configurations"] = []
        return _blocked(DevelopmentService(self.store, self.filesystem).preview(
            DevelopmentBatchSpec.model_validate(spec)), code)

    def experiment_locked(self):
        self.require("experimentId", "batchId")
        return _reject(lambda: ModelExperimentService(self.store, self.filesystem).require_editable(
            self.state["experimentId"]), "EXPERIMENT_CONFIGURATION_LOCKED")

    def experiment_retry_evidence(self):
        # The coordinator performs the actual launch retry. Independently require
        # its persisted success evidence; do not issue a new launch from this helper.
        checks = [item for item in self.state.get("checks", [])
                  if item.get("stage") == "experiments" and "retry" in item.get("name", "").lower()]
        if not checks:
            raise Unavailable("Coordinator has not recorded an experiment launch-retry check")
        if not all(item.get("passed") is True for item in checks):
            raise AssertionError("The coordinator's launch-retry check failed")
        return {"coordinatorRetryChecks": len(checks), "scope": "Read-only corroboration of coordinator receipts"}

    def evaluation_overlap(self):
        self.require("datasetId", "protocolId", "bundleId", "cohortId")
        manifest = self.store.get_configuration(self.state["cohortId"])["manifest"]
        spec = EvaluationSpec.model_validate({**manifest["spec"], "eligibility": [],
            "protocolId": self.state["protocolId"],
            "developmentFeatureBundleId": self.state["bundleId"],
            "featureBundleId": self.state["bundleId"], "inference": {}})
        review, _ = EvaluationService(self.store, self.filesystem)._prepare_bound(spec)
        if "DEVELOPMENT_SLIDE_OVERLAP" not in _codes(review):
            raise AssertionError("Development slides were accepted into the evaluation cohort")
        return {"blockingCodes": _codes(review), "scope": "All frozen slides proposed as evaluation; preview only"}

    def evaluation_selection(self):
        self.require("predictorIds", "cohortId", "bundleId")
        return {"name": "Disposable evaluation preflight", "cohortId": self.state["cohortId"],
                "featureBundleId": self.state["bundleId"],
                "predictorId": next(iter(self.state["predictorIds"].values()))}

    def evaluation_wrong_bundle(self):
        self.require("featureId")
        selection = {**self.evaluation_selection(), "featureBundleId": self.state["featureId"]}
        return _blocked(EvaluationRunService(self.store, self.filesystem).preview(
            EvaluationRunSelection.model_validate(selection)), "FEATURE_BUNDLE_NOT_FOUND", "canSave")

    def evaluation_threshold(self):
        selection = self.evaluation_selection()
        predictor = self.store.get_configuration(selection["predictorId"])
        threshold = predictor["manifest"]["recipe"].get("decisionThreshold")
        if threshold is None:
            raise Unavailable("Selected predictor has no frozen decision threshold")
        selection["inference"] = InferenceSettings(
            decisionThreshold=0.25 if threshold != 0.25 else 0.75).model_dump()
        return _blocked(EvaluationRunService(self.store, self.filesystem).preview(
            EvaluationRunSelection.model_validate(selection)), "EVALUATION_THRESHOLD_MISMATCH", "canSave")

    def attention_inputs(self):
        self.require("featureId", "datasetId", "predictorIds")
        if self._attention is None:
            feature = self.store.get_configuration(self.state["featureId"])
            entry = min(feature["manifest"]["files"], key=lambda row: row["patchCount"])
            rows = json.loads(self.store.read_artifact(self.state["datasetId"], "records.json"))
            row = next(row for row in rows if row["slideId"] == entry["slideId"])
            source = row.get("slidePath")
            if not source:
                source = str(Path(self.state["slideRoot"]) / (entry["slideId"] + ".tiff"))
            predictor = self.store.get_configuration(next(iter(self.state["predictorIds"].values())))
            contract = predictor["manifest"]["inputs"]["features"]
            selection = InterpretationSlide(slideId=entry["slideId"], slidePath=source,
                featurePath=entry["path"], confirmRowAlignment=True).model_dump()
            self._attention = (selection, inspect_slide(source), contract, predictor["id"])
        return copy.deepcopy(self._attention)

    def attention_geometry(self):
        selection, slide, contract, _ = self.attention_inputs()
        selection.update(patchWidthLevel0=513.0, patchHeightLevel0=513.0)
        return _reject(lambda: inspect_inputs(selection, slide, contract), exception=ValueError,
                       contains="geometry differs")

    def attention_confirmation(self):
        selection, _, _, _ = self.attention_inputs()
        selection["confirmRowAlignment"] = False
        return _reject(lambda: InterpretationSlide.model_validate(selection), exception=ValidationError)

    def attention_wrong_dtype(self):
        selection, slide, contract, _ = self.attention_inputs()
        contract["dtype"] = "float64" if contract["dtype"] != "float64" else "float32"
        return _reject(lambda: inspect_inputs(selection, slide, contract), exception=ValueError,
                       contains="predictor's exact dimensions and dtype")

    def attention_coordinates(self, outside=False):
        import h5py
        import numpy as np

        selection, slide, contract, _ = self.attention_inputs()
        destination = self.scratch / ("outside.h5" if outside else "reordered.h5")
        with h5py.File(selection["featurePath"], "r") as source:
            coords = source["coords"][:]
            if len(coords) < 2:
                raise Unavailable("At least two coordinate rows are needed")
            if outside:
                coords[0, 0] = slide["width"]
            else:
                different = np.flatnonzero(np.any(coords != coords[0], axis=1))
                if not len(different):
                    raise Unavailable("Distinct coordinate rows are needed for alignment check")
                index = int(different[0])
                coords[[0, index]] = coords[[index, 0]]
            with h5py.File(destination, "w") as target:
                value = target.create_dataset("coords", data=coords)
                for key, item in source.attrs.items():
                    target.attrs[key] = item
                for key, item in source["coords"].attrs.items():
                    value.attrs[key] = item
        selection["coordinatesPath"] = str(destination)
        return _reject(lambda: inspect_inputs(selection, slide, contract), exception=ValueError,
                       contains="inside the selected slide" if outside else "row order differ")

    def attention_encoder(self):
        selection, _, _, predictor = self.attention_inputs()
        request = InterpretationSelection(name="Disposable encoder mismatch", predictorId=predictor,
                                          encoderId="__wrong_encoder__", slides=[selection])
        return _blocked(InterpretationService(self.store, self.filesystem).preview(request),
                        "INTERPRETATION_ENCODER_MISMATCH", "canSave")

    def attention_source_boundary(self):
        # Use an existing unrelated local file; failure must come from root scope.
        return _reject(lambda: allowed_file(self.filesystem, "/etc/passwd"),
                       "INTERPRETATION_PATH_INVALID")

    def evaluation_distinct_dtype(self):
        """Authenticate one altered held-out input in a disposable project snapshot."""
        import h5py
        import numpy as np

        self.require("featureId", "bundleId", "cohortId", "protocolId")
        cohort = self.store.get_configuration(self.state["cohortId"])["manifest"]
        eligible = {row["slideId"] for row in cohort["memberships"]}
        source = self.store.get_configuration(self.state["featureId"])
        entry = min((row for row in source["manifest"]["files"] if row["slideId"] in eligible),
                    key=lambda row: row["patchCount"])
        if Path(entry["path"]).stat().st_size > MAX_COPY_BYTES:
            raise Unavailable("Smallest held-out feature file exceeds the bounded copy budget")
        store = self.clone()
        # Existing immutable validation receipts preserve the original development
        # bundle in the snapshot. Pack arrays stay at their original read-only path.
        receipts = []
        for folder in (self.project / "packing").iterdir():
            if folder.is_dir() and not folder.is_symlink():
                for name in ("job.json", "result.json", "plan.json"):
                    path = folder / name
                    if path.is_file() and not path.is_symlink():
                        receipts.append(path)
        if sum(path.stat().st_size for path in receipts) > MAX_COPY_BYTES:
            raise Unavailable("Packing receipts exceed the bounded snapshot budget")
        for path in receipts:
            destination = store.folder / "packing" / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        copied = self.scratch / "dtype-copy" / Path(entry["path"]).name
        copied.parent.mkdir()
        shutil.copyfile(entry["path"], copied)
        # Float64 preserves each original float32 value exactly but violates the
        # predictor's declared storage dtype. No alternate full inventory or pack.
        with h5py.File(copied, "r+") as handle:
            values = handle["features"][:].astype(np.float64)
            attributes = dict(handle["features"].attrs)
            del handle["features"]
            features = handle.create_dataset("features", data=values)
            for key, value in attributes.items():
                features.attrs[key] = value
        features = FeatureService(store, self.filesystem)
        spec = FeatureSpec(path=str(copied.parent), encoderId="uni_v1")
        reviewed = features.preview(spec)
        if not reviewed["canFreeze"]:
            raise AssertionError("The disposable dtype variant must attach before comparison")
        feature = features.freeze(spec, reviewed["previewHash"], "disposable-dtype-source")
        packs = FeaturePackService(store, self.filesystem, InlineValidationExecutor())
        request = FeaturePackSpec(featureSetId=feature["id"], action="validate")
        reviewed = packs.preview(request)
        if not reviewed["canRun"]:
            raise AssertionError("The disposable dtype variant must permit full validation")
        job = packs.submit(request, reviewed["previewHash"], "disposable-dtype-validation")
        if run_job(packs.folder / job["id"] / "plan.json")["state"] != "succeeded":
            raise AssertionError("The disposable dtype worker did not validate successfully")
        bundles = FeatureBundleService(store, self.filesystem)
        request = FeatureBundleSpec(featureSetId=feature["id"])
        reviewed = bundles.preview(request)
        bundle = bundles.freeze(request, reviewed["previewHash"], "disposable-dtype-bundle")
        spec = EvaluationSpec.model_validate({**cohort["spec"],
            "eligibility": [*cohort["spec"].get("eligibility", []),
                            {"field": "slideId", "op": "eq", "value": entry["slideId"]}],
            "protocolId": self.state["protocolId"],
            "developmentFeatureBundleId": self.state["bundleId"],
            "featureBundleId": bundle["id"], "inference": {}})
        review, _ = EvaluationService(store, self.filesystem)._prepare_bound(spec)
        codes = _codes(review)
        if codes != ["FEATURE_DTYPE_MISMATCH"]:
            raise AssertionError(f"Expected only FEATURE_DTYPE_MISMATCH; observed codes {codes}")
        return {"blockingCodes": codes, "heldOutSlidesChecked": 1,
                "originalDtype": entry["dtype"], "disposableDtype": "float64",
                "bothBundlesAuthenticated": True, "originalInputsMutated": False}

    def experiment_recovery_evidence(self):
        from scripts.blca_training_recovery import _hash as file_hash
        from scripts.blca_training_recovery import _verify_completed

        self.require("batchId", "experimentId")
        path = Path(self.state.get("interruptionResumeEvidencePath") or
                    self.path.parent / "evidence/interruption-resume.json")
        if not path.exists():
            raise Unavailable("The owning coordinator has not completed its opt-in interruption/recovery exercise")
        evidence = json.loads(path.read_text())
        if any(evidence.get(key) != self.state[key] for key in ("projectId", "batchId", "experimentId")):
            raise AssertionError("Recovery evidence belongs to different study bindings")
        if not evidence.get("complete"):
            raise Unavailable("The owning coordinator's checkpoint recovery exercise is still in progress")
        required = ("interruptedAfterCheckpoint", "resumedFromCheckpoint", "planHashUnchanged",
                    "completedReceiptsUnchanged", "allFoldsCompleted")
        if not all(evidence.get(key) is True for key in required):
            raise AssertionError("Completed recovery evidence lacks required verification")
        service = ModelExperimentService(self.store, self.filesystem).training
        training = service.execution(self.state["batchId"])
        folder = self.project / "training" / self.state["batchId"]
        if training["status"] != "completed" or not all(row["status"] == "completed" for row in training["runs"]):
            raise AssertionError("Current fold execution does not confirm completed recovery")
        if (training["planHash"] != evidence["planHashBefore"]
                or file_hash(folder / "plan.json") != evidence["planFileSha256Before"]):
            raise AssertionError("Current execution plan differs from recovery evidence")
        _verify_completed(evidence["completedBefore"], training["runs"], folder)
        row = next(row for row in training["runs"] if row["id"] == evidence["runId"])
        result = json.loads((folder / "runs" / row["id"] / "result.json").read_text())
        if (result.get("resumedFrom") != str(folder / "runs" / row["id"] / "last.ckpt")
                or row.get("attempt") != evidence["attemptBefore"] + 1):
            raise AssertionError("The resumed fold receipt does not confirm the native checkpoint attempt")
        return {**{key: True for key in required}, "completedFoldsPreserved": len(evidence["completedBefore"]),
                "totalFolds": len(training["runs"]), "scope": "Independent read-only verification of owned coordinator recovery"}

    def checks(self):
        return {
            "dataset": {"missing_slide_id_column": self.dataset_wrong_column,
                        "blank_slide_identity": self.dataset_blank_identity,
                        "stale_draft_revision": self.dataset_revision,
                        "stale_import_preview": self.dataset_stale_preview,
                        "text_binary_label_agreement": self.dataset_label_agreement},
            "target": {"unmapped_target_labels": self.target_unknown_labels,
                       "target_predictor_leakage": self.target_leakage},
            "features": {"wrong_encoder": self.features_wrong_encoder,
                         "all_original_slide_geometry_and_decode": self.features_slide_integrity,
                         "float32_pack_sample_parity": self.features_pack_parity,
                         "modified_copy_invalidates_worker_receipt": self.features_mutated_receipt},
            "experiments": {
                "patch_features_reject_slide_model": lambda: self.experiment_model(
                    "slide_linear", "TRAINING_FEATURE_KIND_MISMATCH"),
                "unsupported_model": lambda: self.experiment_model(
                    "unsupported_robustness_model", "TRAINING_MODEL_UNSUPPORTED"),
                "submitted_configuration_locked": self.experiment_locked,
                "coordinator_retry_receipt": self.experiment_retry_evidence,
                "training_interruption_resume": self.experiment_recovery_evidence},
            "evaluation": {"development_overlap": self.evaluation_overlap,
                           "wrong_feature_document": self.evaluation_wrong_bundle,
                           "frozen_threshold_change": self.evaluation_threshold,
                           "distinct_dtype_bundle": self.evaluation_distinct_dtype},
            "attention": {"wrong_level0_geometry": self.attention_geometry,
                          "alignment_confirmation_required": self.attention_confirmation,
                          "wrong_feature_dtype": self.attention_wrong_dtype,
                          "reordered_coordinates": self.attention_coordinates,
                          "out_of_slide_coordinates": lambda: self.attention_coordinates(True),
                          "wrong_encoder": self.attention_encoder,
                          "outside_source_roots": self.attention_source_boundary},
        }[self.stage]


def run_checks(state_path, stage):
    """Run available checks, checkpoint each result, and fail loudly after recording failures."""
    if stage not in STAGES:
        raise ValueError("Unknown robustness stage")
    with ExitStack() as stack:
        context = Checks(state_path, stage, stack)
        report_path = context.output / f"{stage}.json"
        stack.enter_context(writer_lock(context.output, timeout=5))
        previous = json.loads(report_path.read_text()) if report_path.exists() else {}
        # Cached passes belong to this exact project and frozen stage bindings.
        dependencies = {
            "dataset": ("datasetId", "importDraftId"),
            "target": ("datasetId", "protocolId"),
            "features": ("featureId", "bundleId", "packId", "packPath"),
            "experiments": ("protocolId", "bundleId", "experimentId", "batchId"),
            "evaluation": ("protocolId", "cohortId", "featureId", "bundleId", "predictorIds"),
            "attention": ("datasetId", "featureId", "predictorIds"),
        }
        binding = {key: context.state.get(key) for key in ("projectId", *dependencies[stage])}
        binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        cached = previous.get("checks", {}) if (
            previous.get("checkVersion") == CHECK_VERSION and
            previous.get("bindingHash") == binding_hash) else {}
        # A newly completed pack must not cause an eight-minute all-slide image
        # audit to repeat. Before packing completed, only bundle/pack IDs were
        # absent: prove that exact prior binding hash before retaining checks
        # whose inputs are the unchanged native inventory. Pack parity reruns.
        if (not cached and stage == "features"
                and previous.get("checkVersion") == CHECK_VERSION):
            native_binding = {**binding, "bundleId": None, "packId": None}
            native_hash = hashlib.sha256(json.dumps(native_binding, sort_keys=True).encode()).hexdigest()
            if previous.get("bindingHash") == native_hash:
                cached = {name: value for name, value in previous.get("checks", {}).items()
                          if name != "float32_pack_sample_parity"}
        report = {"schemaVersion": 1, "checkVersion": CHECK_VERSION, "stage": stage,
                  "projectId": context.state["projectId"], "bindingHash": binding_hash, "bindings": binding,
                  "startedAt": _stamp(), "checks": {},
                  "policy": "Native previews and isolated copies; no original source or valid study record mutations"}
        for name, check in context.checks().items():
            if cached.get(name, {}).get("status") == "passed":
                report["checks"][name] = cached[name]
                continue
            started = time.monotonic()
            try:
                evidence = check()
                result = {"status": "passed", "evidence": evidence}
            except Unavailable as error:
                result = {"status": "skipped", "reason": str(error)}
            except Exception as error:
                # Do not put exception text, source row identifiers or clinical data in reports.
                result = {"status": "failed", "errorType": type(error).__name__}
                if isinstance(error, StorageError):
                    result["errorCode"] = error.code
                if isinstance(error, AssertionError):
                    result["reason"] = str(error)
            result.update(checkedAt=_stamp(), seconds=round(time.monotonic() - started, 3))
            report["checks"][name] = result
            write_json(report_path, report)
        summary = {status: sum(item["status"] == status for item in report["checks"].values())
                   for status in ("passed", "failed", "skipped")}
        report.update(summary=summary, finishedAt=_stamp(), complete=not summary["failed"] and not summary["skipped"])
        write_json(report_path, report)
        if summary["failed"]:
            names = [name for name, value in report["checks"].items() if value["status"] == "failed"]
            raise RuntimeError(f"{stage} robustness failed: {', '.join(names)}; see {report_path}")
        return {"stage": stage, **summary, "reportPath": str(report_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    arguments = parser.parse_args()
    print(json.dumps(run_checks(arguments.state, arguments.stage)))
