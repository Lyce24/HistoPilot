"""Immutable, independently selected test cohorts with exact feature coverage checks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from histopilot.application.feature_bundles import FeatureBundleService, _hash
from histopilot.application.protocols import (
    CANONICAL,
    FilterEvaluator,
    FilterFailure,
    ProtocolService,
    _forbidden_name,
    _key,
)
from histopilot.schemas.evaluations import EvaluationSpec
from histopilot.schemas.protocols import TargetSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.pack_import import _layout
from histopilot.storage.packed import PackedStoreError
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


def _reference(document):
    return {key: document[key] for key in ("id", "contentHash")}


def _representation(feature):
    manifest = feature["manifest"]
    dimensions = {item["dimensions"] for item in manifest["files"]}
    return {
        "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "encoderId": manifest.get("layout", {}).get("encoderId")
        or manifest.get("spec", {}).get("encoderId"),
    }


def _slide_sources(store, dataset, rows, selected):
    """Use immutable import evidence to recognize renamed/symlinked/hardlinked WSIs.

    No live WSI reads are needed. Feature-only datasets can lack this evidence;
    their exact slide and patient identifiers are still checked separately.
    """
    paths = defaultdict(set)
    for row in rows:
        if (
            row["slideId"] in selected
            and isinstance(row.get("slidePath"), str)
            and row["slidePath"]
        ):
            paths[row["slidePath"]].add(row["slideId"])
    sources = defaultdict(set)
    for path, slides in paths.items():
        for slide in slides:
            sources[slide].add(("path", path))
    if paths and "inventory.json" in dataset.get("artifacts", {}):
        try:
            inventory = json.loads(store.read_artifact(dataset["id"], "inventory.json"))
            if not isinstance(inventory, list) or any(
                not isinstance(row, dict) for row in inventory
            ):
                raise ValueError
            for row in inventory:
                if row.get("path") not in paths:
                    continue
                stamp = tuple(row.get(key) for key in ("device", "inode", "sizeBytes", "mtimeNs"))
                if all(type(value) is int for value in stamp) and stamp[1] > 0 and stamp[2] > 0:
                    for slide in paths[row["path"]]:
                        sources[slide].add(("file", *stamp))
        except (ValueError, TypeError) as error:
            raise StorageError(
                "The frozen slide-source inventory is malformed.", "STORAGE_CORRUPT", 409
            ) from error
    return sources


# Finding messages, the freeze verdict and capability flags are presentation, not
# scientific content: the underlying facts (coverage, overlap, bindings, memberships)
# are already hashed. Hashing prose froze every saved cohort whenever wording changed.
VOLATILE_PREVIEW_KEYS = frozenset({"findings", "canFreeze", "executionEnabled", "previewHash"})


def preview_hash(preview):
    return _hash({key: value for key, value in preview.items() if key not in VOLATILE_PREVIEW_KEYS})


def legacy_preview_hash(preview):
    """Pre-v2 cohorts hashed the whole preview, findings included."""
    return _hash({key: value for key, value in preview.items() if key != "previewHash"})


def preview_current(preview, manifest):
    # These three fields wrap the preview at publication; all remaining fields
    # belong to the reviewed evidence, including any future scientific fields.
    saved = {
        key: value
        for key, value in manifest.items()
        if key not in {"kind", "schemaVersion", "datasetId"}
    }
    stored_hash = saved.get("previewHash")
    if stored_hash not in {preview_hash(saved), legacy_preview_hash(saved)}:
        return False
    # Compare scientific content only after verifying the saved preview in its
    # own format. A legacy warning must not be reconstructed from today's prose.
    return preview["canFreeze"] and preview_hash(preview) == preview_hash(saved)


class EvaluationService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem):
        self.store = store
        self.protocols = ProtocolService(store, filesystem)
        self.bundles = FeatureBundleService(store, filesystem)

    def _draft_spec(self, draft_id, expected_revision):
        draft = self.store.get_draft(draft_id)
        if draft["revision"] != expected_revision:
            raise StorageError("The test-cohort draft changed. Reload it.", "REVISION_CONFLICT")
        if draft["status"] != "editable":
            raise StorageError(
                "This test cohort is frozen. Clone it to make changes.", "DRAFT_FROZEN"
            )
        payload = draft["payload"]
        if (
            draft["kind"] != "experiment"
            or not isinstance(payload, dict)
            or set(payload) != {"type", "spec"}
            or payload["type"] != "evaluation-cohort"
        ):
            raise StorageError("Select a test-cohort draft.", "INVALID_EVALUATION_DRAFT", 422)
        try:
            return EvaluationSpec.model_validate(payload["spec"])
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                for item in error.errors(include_input=False, include_url=False)[:8]
            )
            raise StorageError(
                f"Invalid test-cohort settings: {details}", "INVALID_EVALUATION_SPEC", 422
            ) from error

    def _prepare(self, spec):
        findings = []

        def finding(code, message, severity="error"):
            item = {"code": code, "message": message, "severity": severity}
            if item not in findings:
                findings.append(item)

        protocol = self.store.get_configuration(spec.protocolId)
        protocol_manifest = protocol["manifest"]
        if protocol_manifest.get("kind") != "protocol":
            raise StorageError("Select a frozen development protocol.", "INVALID_PROTOCOL", 422)
        if protocol_manifest.get("spec", {}).get("split", {}).get("version", 1) != 4:
            finding(
                "LEGACY_DEVELOPMENT_PROTOCOL",
                "Create a development-only protocol revision in Stage 2 before preparing test cohorts. Legacy protocols may include externally reserved data and remain unchanged.",
            )
        try:
            target = TargetSpec.model_validate(protocol_manifest["spec"]["target"])
        except (KeyError, ValidationError) as error:
            raise StorageError(
                "The development target is invalid.", "INVALID_PROTOCOL", 422
            ) from error
        dataset, fields, rows = self.protocols._load_dataset(spec.datasetId)
        same_dataset = spec.datasetId == protocol_manifest["datasetId"]
        if same_dataset and spec.patientIdentifiers == "independent":
            finding(
                "SHARED_PATIENT_NAMESPACE", "The same dataset must use shared patient identifiers."
            )
        elif spec.patientIdentifiers == "independent":
            finding(
                "PATIENT_OVERLAP_UNVERIFIABLE",
                "Separate patient identifier namespaces were declared. Patient overlap cannot be verified across these datasets; exact slide IDs are still checked.",
                "warning",
            )
        if spec.inference.patientAggregation != "mean":
            finding(
                "PATIENT_AGGREGATION_UNSUPPORTED",
                "Frozen predictors use mean class probabilities across each patient's slides. Select mean probabilities before freezing this cohort.",
            )
        for condition in spec.eligibility:
            if condition.field not in fields and condition.field not in CANONICAL:
                finding(
                    "UNKNOWN_FIELD", f"Filter field '{condition.field}' is not in this dataset."
                )
        if spec.target:
            for key in ("task", "unit", "classes", "positiveClass"):
                if getattr(spec.target, key) != getattr(target, key):
                    finding(
                        "TARGET_CONTRACT_MISMATCH",
                        "Test labels must preserve development task, prediction unit, class order, and positive class.",
                    )
            remapped = {
                raw
                for raw in spec.target.labels.keys() & target.labels.keys()
                if spec.target.labels[raw] != target.labels[raw]
            }
            if remapped:
                if same_dataset and spec.target.field == target.field:
                    finding(
                        "TARGET_LABEL_MAPPING_MISMATCH",
                        "The same label field in the same dataset must preserve development label mappings. Restore the development class for each conflicting raw label before freezing this test cohort.",
                    )
                else:
                    finding(
                        "TARGET_LABEL_MAPPING_REMAPPED",
                        f"{len(remapped)} raw label values map to different classes than in development. Verify this test dataset's label meanings and confirm that its mappings preserve the same class meanings before freezing.",
                        "warning",
                    )
            if spec.target.field not in fields:
                finding("UNKNOWN_TARGET_FIELD", "Select an available target label field.")
            source = fields.get(spec.target.field, {}).get("sourceColumn", spec.target.field)
            mapping = dataset["manifest"].get("provenance", {}).get("mapping", {})
            identity_columns = {
                _key(mapping[key])
                for key in (
                    "slideIdColumn",
                    "patientIdColumn",
                    "patientSourceSlideIdColumn",
                    "patientSourcePatientIdColumn",
                )
                if isinstance(mapping.get(key), str)
            }
            if (
                spec.target.field in CANONICAL
                or _forbidden_name(source, target=True)
                or _key(source) in identity_columns
            ):
                finding(
                    "IDENTIFIER_TARGET",
                    "Identifiers and partition fields cannot serve as target labels.",
                )
        evaluator = FilterEvaluator()
        included, excluded = [], 0
        try:
            evaluator.prepare(spec.eligibility)
            for row in rows:
                if not evaluator.conjunction(row, spec.eligibility):
                    excluded += 1
                    continue
                label = None
                if spec.target:
                    raw = evaluator.field(row, spec.target.field)
                    # Match development protocols: an empty string can be an
                    # explicitly mapped label; only null is intrinsically missing.
                    missing = raw is None
                    label = spec.target.labels.get(raw) if not missing else None
                    if label is None:
                        policy = spec.target.missing if missing else spec.target.unmapped
                        if policy == "exclude":
                            excluded += 1
                            continue
                        finding(
                            "MISSING_TARGET_LABEL" if missing else "UNMAPPED_TARGET_LABEL",
                            "Selected slides have missing or unmapped target labels. Map them, explicitly exclude them, or choose unlabeled inference.",
                        )
                included.append({**row, "label": label})
        except FilterFailure as error:
            finding(error.code, str(error))
        if not included:
            finding("EMPTY_TEST_COHORT", "The test-cohort conditions select no slides.")
        slide_ids = [row["slideId"] for row in included]
        selected_ids = set(slide_ids)
        if len(slide_ids) != len(selected_ids):
            finding("DUPLICATE_SLIDE_ID", "Selected test slides have duplicate slide identifiers.")
        groups = defaultdict(list)
        for row in included:
            groups[row.get("patientId")].append(row)
        self.protocols._identity_findings(included, groups, finding)
        if target.unit == "patient" and spec.target:
            if any(
                len({row["label"] for row in group if row["label"] is not None}) > 1
                for group in groups.values()
            ):
                finding(
                    "CONFLICTING_PATIENT_LABELS",
                    "A test patient has conflicting labels across slides.",
                )
        development_memberships = protocol_manifest.get("memberships", [])
        if not development_memberships:
            finding(
                "DEVELOPMENT_MEMBERSHIPS_MISSING",
                "The development protocol has no frozen membership evidence.",
            )
        development_slides = {item["slideId"] for item in development_memberships}
        development_patients = {
            item["patientId"] for item in development_memberships if item.get("patientId")
        }
        slide_overlap = sorted(selected_ids & development_slides)
        patient_overlap = (
            sorted((set(groups) - {None}) & development_patients)
            if same_dataset or spec.patientIdentifiers == "shared"
            else []
        )
        if slide_overlap:
            finding(
                "DEVELOPMENT_SLIDE_OVERLAP",
                f"{len(slide_overlap)} selected slide IDs occur in model development.",
            )
        if patient_overlap:
            finding(
                "DEVELOPMENT_PATIENT_OVERLAP",
                f"{len(patient_overlap)} selected patient IDs occur in model development.",
            )
        development_dataset, development_rows = dataset, rows
        if not same_dataset:
            development_dataset, _fields, development_rows = self.protocols._load_dataset(
                protocol_manifest["datasetId"]
            )
        development_sources = _slide_sources(
            self.store, development_dataset, development_rows, development_slides
        )
        source_identities = (
            set().union(*development_sources.values()) if development_sources else set()
        )
        selected_sources = _slide_sources(self.store, dataset, included, selected_ids)
        source_overlap = sorted(
            slide
            for slide, sources in selected_sources.items()
            if sources & source_identities and slide not in development_slides
        )
        if source_overlap:
            finding(
                "DEVELOPMENT_SLIDE_SOURCE_OVERLAP",
                f"{len(source_overlap)} selected slides refer to source files used in model development under different slide IDs. Reconcile their identities and select independent slides.",
            )
        guards, bindings, representations, feature_dtypes = [], {}, {}, {}
        feature_ids = set()
        selected_bundle = None
        for name, identity in (
            ("development", spec.developmentFeatureBundleId),
            ("evaluation", spec.featureBundleId),
        ):
            bundle = self.bundles.get(identity)
            if not bundle["current"]:
                finding(
                    "STALE_FEATURE_BUNDLE",
                    f"The {name} feature bundle is stale or no longer verified.",
                )
            for item in bundle["findings"]:
                finding(
                    item["code"],
                    f"{name.capitalize()} features: {item['message']}",
                    item["severity"],
                )
            feature = self.store.get_configuration(bundle["manifest"]["feature"]["id"])
            available = [item["slideId"] for item in feature["manifest"]["files"]]
            if len(available) != len(set(available)):
                finding(
                    "DUPLICATE_FEATURE_ID",
                    f"The {name} feature inventory contains duplicate slide IDs.",
                )
            bindings[name] = {"bundle": _reference(bundle), "feature": _reference(feature)}
            representations[name] = _representation(feature)
            feature_dtypes[name] = {item["dtype"] for item in feature["manifest"]["files"]}
            if name == "development":
                missing_development = development_slides - set(available)
                if missing_development:
                    finding(
                        "DEVELOPMENT_FEATURE_COVERAGE",
                        f"The selected development bundle is missing {len(missing_development)} development slide IDs.",
                    )
                expected_feature = protocol_manifest["spec"].get("featureSetId")
                if expected_feature and expected_feature != feature["id"]:
                    finding(
                        "DEVELOPMENT_FEATURE_BINDING_MISMATCH",
                        "The development bundle differs from the feature version pinned by this protocol.",
                    )
            else:
                feature_ids = set(available)
                selected_bundle = bundle
            if bundle["current"]:
                guards.append(self.bundles._freshness_guard(feature, bundle["manifest"]))
        missing_features = sorted(selected_ids - feature_ids)
        if missing_features:
            finding(
                "MISSING_TEST_FEATURES",
                f"{len(missing_features)} selected test slide IDs have no features in the selected bundle.",
            )
        left, right = representations["development"], representations["evaluation"]
        if left["dimensions"] is None or left["dimensions"] != right["dimensions"]:
            finding(
                "FEATURE_DIMENSION_MISMATCH",
                "Test feature dimensions must match the development features.",
            )
        if (
            len(feature_dtypes["development"]) != 1
            or feature_dtypes["development"] != feature_dtypes["evaluation"]
        ):
            finding(
                "FEATURE_DTYPE_MISMATCH",
                "Test feature dtype must match the single dtype used by the development features. Select or rebuild a compatible feature bundle before freezing.",
            )
        same_feature = bindings["development"]["feature"] == bindings["evaluation"]["feature"]
        if not same_feature and (not left["encoderId"] or not right["encoderId"]):
            finding(
                "ENCODER_IDENTITY_REQUIRED",
                "Distinct development and test feature sources require an explicit encoder identity on both inventories.",
            )
        elif left["encoderId"] != right["encoderId"]:
            finding(
                "ENCODER_MISMATCH", "The test feature encoder differs from the development encoder."
            )
        missing_pack, pack_binding, pack_checked = [], None, False
        if spec.inference.packArtifactId:
            pack_id = spec.inference.packArtifactId
            expected = next(
                (item for item in selected_bundle["manifest"]["packs"] if item["id"] == pack_id),
                None,
            )
            if expected is None:
                finding(
                    "PACK_NOT_IN_BUNDLE",
                    "Select a verified pack included in the test feature bundle.",
                )
            else:
                try:
                    resolved = self.bundles.packing.resolve_artifact(
                        bindings["evaluation"]["feature"]["id"], pack_id
                    )
                    if not resolved["current"]:
                        finding("STALE_TEST_PACK", "The selected test pack is no longer current.")
                    for item in resolved["findings"]:
                        finding(item["code"], item["message"], item["severity"])
                    layout = _layout(Path(resolved["artifact"]["outputPath"]))
                    packed_ids = {item["slideId"] for item in layout["slides"]}
                    missing_pack = sorted(selected_ids - packed_ids)
                    pack_checked = True
                    pack_binding = expected
                    if missing_pack:
                        finding(
                            "MISSING_TEST_PACK_SLIDES",
                            f"{len(missing_pack)} selected test slide IDs are absent from the selected pack index.",
                        )
                except (StorageError, PackedStoreError, OSError) as error:
                    finding(
                        "TEST_PACK_UNAVAILABLE", f"The selected pack cannot be verified: {error}"
                    )
        counts = Counter(row["label"] for row in included if row["label"] is not None)
        if spec.target and set(counts) != set(target.classes):
            finding(
                "TEST_CLASSES_ABSENT",
                "Some development classes are absent in the selected test cohort; some metrics will be unavailable.",
                "warning",
            )
        preview = {
            "spec": spec.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "summary": {
                "includedSlides": len(included),
                "includedPatients": len(set(groups) - {None}),
                "excludedSlides": excluded,
                "labeledSlides": sum(counts.values()),
                "classCounts": {label: counts[label] for label in target.classes},
                "developmentSlideOverlap": len(slide_overlap),
                "developmentPatientOverlap": len(patient_overlap),
            },
            "coverage": {
                "selectedSlideIds": sorted(selected_ids),
                "featureSlideCount": len(feature_ids),
                "missingFeatureSlideIds": missing_features,
                "missingPackSlideIds": missing_pack,
                "packChecked": pack_checked,
            },
            "overlap": {
                "slideIds": slide_overlap,
                "patientIds": patient_overlap,
                "patientsComparable": same_dataset or spec.patientIdentifiers == "shared",
                **({"sourceSlideIds": source_overlap} if source_overlap else {}),
            },
            "bindings": {
                "protocol": _reference(protocol),
                "dataset": _reference(dataset),
                **bindings,
            },
            "compatibility": representations,
            "pack": pack_binding,
            "memberships": [
                {key: row.get(key) for key in ("slideId", "patientId", "patientIdSource", "label")}
                for row in included
            ],
            "findings": findings,
            "canFreeze": not any(item["severity"] == "error" for item in findings),
            "executionEnabled": False,
        }
        return {**preview, "previewHash": preview_hash(preview)}, guards

    def preview(self, draft_id, expected_revision):
        return self._prepare(self._draft_spec(draft_id, expected_revision))[0]

    def freeze(
        self, draft_id, expected_revision, preview_hash, operation_id, *, version_label=None
    ):
        prior = self.store.configuration_publication(operation_id)
        if prior:
            manifest = prior["manifest"]
            if (
                manifest.get("kind") != "evaluation-cohort"
                or manifest.get("previewHash") != preview_hash
            ):
                raise StorageError(
                    "This operation belongs to another test-cohort freeze.", "OPERATION_CONFLICT"
                )
            return self.store.publish_configuration(
                draft_id,
                expected_revision=expected_revision,
                manifest=manifest,
                operation_id=operation_id,
                version_label=version_label,
            )
        preview, guards = self._prepare(self._draft_spec(draft_id, expected_revision))
        if preview["previewHash"] != preview_hash:
            raise StorageError(
                "The cohort or input verification changed. Preview again.", "PREVIEW_STALE"
            )
        if not preview["canFreeze"]:
            raise StorageError(
                "Resolve the test-cohort findings before freezing.",
                "EVALUATION_PREFLIGHT_BLOCKED",
                422,
            )

        def check():
            for guard in guards:
                guard()

        return self.store.publish_configuration(
            draft_id,
            expected_revision=expected_revision,
            manifest={
                "kind": "evaluation-cohort",
                "schemaVersion": 1,
                "datasetId": preview["spec"]["datasetId"],
                **preview,
            },
            operation_id=operation_id,
            version_label=version_label,
            before_publish=check,
        )

    def _resolve(self, document):
        if document["manifest"].get("kind") != "evaluation-cohort":
            raise StorageError("Test cohort not found.", "EVALUATION_NOT_FOUND", 404)
        try:
            preview, _guards = self._prepare(
                EvaluationSpec.model_validate(document["manifest"]["spec"])
            )
            current = preview_current(preview, document["manifest"])
            findings = preview["findings"]
        except (StorageError, ValidationError) as error:
            current = False
            findings = [
                {"severity": "error", "code": "EVALUATION_INPUT_UNAVAILABLE", "message": str(error)}
            ]
        return {**document, "current": current, "findings": findings, "executionEnabled": False}

    def get(self, identity):
        return self._resolve(self.store.get_configuration(identity))

    def list(self):
        return {
            "items": [
                self._resolve(item) for item in self.store.list_configurations("evaluation-cohort")
            ]
        }
