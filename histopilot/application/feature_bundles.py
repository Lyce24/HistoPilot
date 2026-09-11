"""Freeze verified feature inventories and optional packs without choosing a loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from histopilot.application.feature_packs import FeaturePackService
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _finding(code: str, message: str) -> dict:
    return {"severity": "error", "code": code, "message": message}


def _validation_snapshot(report: dict | None) -> dict | None:
    if report is None:
        return None
    return {
        key: report[key]
        for key in (
            "jobId",
            "sourceContentHash",
            "tensorValidationComplete",
            "provenanceComplete",
            "validatedAt",
        )
        if key in report
    }


def bundle_pack_snapshot(artifact: dict) -> dict:
    """A specific verified representation, without a mutable loading preference."""
    return {
        **{
            key: artifact[key]
            for key in (
                "id",
                "materializationId",
                "featureSetId",
                "outputPath",
                "outputDtype",
                "sourceContentHash",
                "verification",
                "jobId",
            )
            if key in artifact
        },
        "validation": _validation_snapshot({**artifact["validation"], "jobId": artifact["jobId"]}),
    }


class FeatureBundleService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem):
        self.store = store
        self.packing = FeaturePackService(store, filesystem)

    def _pack(self, feature_id: str, artifact_id: str, source_hash: str | None) -> tuple:
        resolved = self.packing.resolve_artifact(feature_id, artifact_id)
        artifact = resolved["artifact"]
        findings = list(resolved["findings"])
        report = artifact.get("validation", {})
        if (
            resolved.get("current") is not True
            or artifact.get("verification") not in {"exact-source-values", "validated-source-copy"}
            or report.get("valid") is not True
            or report.get("tensorValidationComplete") is not True
            or not source_hash
            or artifact.get("sourceContentHash") != source_hash
            or report.get("sourceContentHash") != source_hash
        ):
            findings.append(
                _finding(
                    "PACK_VERIFICATION_REQUIRED",
                    "Every included pack must have current full verification against these exact features.",
                )
            )
        return bundle_pack_snapshot(artifact), findings

    def _prepare(self, spec: FeatureBundleSpec) -> tuple[dict, dict]:
        configuration = self.store.get_configuration(spec.featureSetId)
        manifest = configuration["manifest"]
        if manifest.get("kind") != "feature":
            raise StorageError("Select a feature inventory.", "INVALID_FEATURE_SET", 422)
        validation = self.packing.validation_for(spec.featureSetId)
        findings = self.packing._source_findings(configuration)
        if (
            validation is None
            or validation.get("current") is not True
            or validation.get("valid") is not True
            or validation.get("tensorValidationComplete") is not True
        ):
            findings.append(
                _finding(
                    "FULL_FEATURE_VALIDATION_REQUIRED",
                    "Validate every feature value and coordinate before freezing this bundle. "
                    "Creating or verifying a pack also performs this validation.",
                )
            )
        source_hash = validation.get("sourceContentHash") if validation else None
        feature = {
            "id": configuration["id"],
            "contentHash": configuration["contentHash"],
            "datasetId": manifest["datasetId"],
            "sourceContentHash": source_hash,
            "validation": _validation_snapshot(validation),
        }
        packs = []
        for identity in spec.packArtifactIds:
            try:
                snapshot, pack_findings = self._pack(spec.featureSetId, identity, source_hash)
                packs.append(snapshot)
                findings.extend(pack_findings)
            except StorageError as error:
                findings.append(_finding(error.code, f"Pack {identity}: {error}"))
        files = manifest["files"]
        dimensions = {item["dimensions"] for item in files}
        dtypes = {item["dtype"] for item in files}
        summary = {
            "slideCount": len(files),
            "patchCount": sum(item["patchCount"] for item in files),
            "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else None,
            "dtype": next(iter(dtypes)) if len(dtypes) == 1 else None,
            "packCount": len(spec.packArtifactIds),
        }
        preview = {
            "spec": spec.model_dump(mode="json"),
            "summary": summary,
            "feature": feature,
            "packs": packs,
            "findings": findings,
            "canFreeze": not findings,
        }
        return {**preview, "previewHash": _hash(preview)}, configuration

    def preview(self, spec: FeatureBundleSpec) -> dict:
        return self._prepare(spec)[0]

    def _freshness_guard(self, configuration: dict, preview: dict):
        """Pin external files for the final check; the callback never re-enters the store."""
        from histopilot.storage.pack_import import pack_file_stamps
        from histopilot.storage.packed import PackedStoreError, _check_sources

        manifest = configuration["manifest"]
        sources = {item["path"]: item for item in manifest["files"]}
        sources.update(
            {
                item["coordinatePath"]: item["coordinateFile"]
                for item in manifest["files"]
                if item.get("coordinatePath") and item.get("coordinateFile")
            }
        )
        sources.update({item["path"]: item for item in manifest.get("provenance", [])})
        evidence = {}
        job_ids = {preview["feature"]["validation"]["jobId"]}
        job_ids.update(item["jobId"] for item in preview["packs"])
        for identity in job_ids:
            path = self.packing.folder / identity / "result.json"
            evidence[path] = hashlib.sha256(
                ScientificStore._read_file(path, 64 * 1024 * 1024)
            ).hexdigest()
        packs = []
        for snapshot in preview["packs"]:
            artifact = self.packing.artifact(snapshot["id"])
            packs.append((Path(artifact["outputPath"]), artifact["packStamps"]))

        def check():
            try:
                _check_sources({"sourceStamps": sources})
                # Extraction receipts are separate from feature/container stamps. This
                # helper only reads files, so it does not re-enter the project lock.
                if manifest.get("sourceExtraction") and self.packing._source_findings(
                    configuration
                ):
                    raise PackedStoreError(
                        "Recorded extraction provenance changed before publication."
                    )
                for path, stamps in packs:
                    if pack_file_stamps(path) != stamps:
                        raise PackedStoreError("A pack changed before bundle publication.")
                for path, digest in evidence.items():
                    current = ScientificStore._read_file(path, 64 * 1024 * 1024)
                    if hashlib.sha256(current).hexdigest() != digest:
                        raise PackedStoreError("Verification evidence changed before publication.")
            except (PackedStoreError, OSError) as error:
                raise StorageError(
                    f"Feature bundle inputs changed. Preview again: {error}", "PREVIEW_STALE"
                ) from error

        return check

    def freeze(
        self,
        spec: FeatureBundleSpec,
        preview_hash: str,
        operation_id: str,
        *,
        version_label: dict | None = None,
    ) -> dict:
        prior = self.store.configuration_publication(operation_id)
        if prior:
            manifest = prior["manifest"]
            if (
                manifest.get("kind") != "feature-bundle"
                or manifest.get("spec") != spec.model_dump(mode="json")
                or manifest.get("previewHash") != preview_hash
            ):
                raise StorageError(
                    "This operation ID belongs to another bundle request.", "OPERATION_CONFLICT"
                )
            # The storage layer verifies the original tag/note intent as well.
            document = self.store.publish_configuration(
                manifest=manifest, operation_id=operation_id, version_label=version_label
            )
            return self._resolve(document)
        preview, configuration = self._prepare(spec)
        if preview["previewHash"] != preview_hash:
            raise StorageError(
                "Feature bundle inputs or verification changed. Preview again.", "PREVIEW_STALE"
            )
        if not preview["canFreeze"]:
            raise StorageError(
                "Resolve all verification findings before freezing this bundle.",
                "FEATURE_BUNDLE_INVALID",
                422,
            )
        document = self.store.publish_configuration(
            manifest={
                "kind": "feature-bundle",
                "schemaVersion": 1,
                "datasetId": configuration["manifest"]["datasetId"],
                **{
                    key: preview[key]
                    for key in ("spec", "summary", "feature", "packs", "previewHash")
                },
            },
            operation_id=operation_id,
            version_label=version_label,
            before_publish=self._freshness_guard(configuration, preview),
        )
        return self._resolve(document)

    def _resolve(self, document: dict) -> dict:
        manifest = document["manifest"]
        if manifest.get("kind") != "feature-bundle":
            raise StorageError("Feature bundle not found.", "FEATURE_BUNDLE_NOT_FOUND", 404)
        findings = []
        feature = manifest["feature"]
        try:
            configuration = self.store.get_configuration(feature["id"])
            if configuration["contentHash"] != feature["contentHash"]:
                findings.append(
                    _finding("FEATURE_SOURCE_CHANGED", "The frozen feature inventory changed.")
                )
            findings.extend(self.packing._source_findings(configuration))
            proof = feature["validation"]
            job = self.packing.get(proof["jobId"])
            report = job.get("result", {}).get("validation")
            if (
                job["state"] != "succeeded"
                or report is None
                or not report.get("valid")
                or _validation_snapshot(
                    {**report, "jobId": job["id"], "validatedAt": job["updatedAt"]}
                )
                != proof
            ):
                findings.append(
                    _finding(
                        "FEATURE_VERIFICATION_CHANGED",
                        "Frozen feature verification is unavailable or changed.",
                    )
                )
        except StorageError as error:
            findings.append(_finding(error.code, str(error)))
        for expected in manifest["packs"]:
            try:
                actual, pack_findings = self._pack(
                    feature["id"], expected["id"], feature["sourceContentHash"]
                )
                findings.extend(pack_findings)
                if actual != expected:
                    findings.append(
                        _finding(
                            "PACK_BINDING_CHANGED",
                            "An included pack no longer matches its frozen identity.",
                        )
                    )
            except StorageError as error:
                findings.append(_finding(error.code, str(error)))
        return {**document, "current": not findings, "findings": findings}

    def get(self, identity: str) -> dict:
        return self._resolve(self.store.get_configuration(identity))

    def list(self) -> dict:
        return {
            "items": [
                self._resolve(item) for item in self.store.list_configurations("feature-bundle")
            ]
        }
