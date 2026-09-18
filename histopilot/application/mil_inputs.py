"""Resolve execution input intent without mutating scientific bundles or running models."""

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.protocols import pack_binding_snapshot, protocol_bundle_findings
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


class MILInputService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem):
        self.store = store
        self.filesystem = filesystem

    def preview(self, spec: MILInputSpec) -> dict:
        findings = []

        def error(code, message):
            findings.append({"code": code, "message": message, "severity": "error"})

        result = {
            "canPlan": False,
            "findings": findings,
            "resolvedLoadingPolicy": None,
            "packArtifactId": None,
            "featureSetId": None,
            "featureKind": "patch",
            "bundleId": spec.featureBundleId,
            "executionImplemented": False,
        }
        try:
            bundle = FeatureBundleService(self.store, self.filesystem).get(spec.featureBundleId)
            protocol = self.store.get_configuration(spec.protocolId)["manifest"]
        except StorageError as failure:
            error(failure.code, str(failure))
            return result
        if protocol.get("kind") != "protocol":
            error("INVALID_PROTOCOL", "Choose a frozen target and split protocol.")
            return result
        manifest = bundle["manifest"]
        feature_id = manifest["spec"]["featureSetId"]
        result["featureSetId"] = feature_id
        if not bundle["current"]:
            error(
                "BUNDLE_STALE",
                "Bundle inputs changed. Prepare and freeze a current bundle in PFM & features.",
            )
            findings.extend(bundle["findings"])
        findings.extend(protocol_bundle_findings(protocol, bundle))
        protocol_spec = protocol["spec"]
        if protocol_spec.get("predictors"):
            findings.append({
                "severity": "info", "code": "CLINICAL_INPUTS_AVAILABLE",
                "message": "Clinical fields are declared. Choose image-only, clinical-only, or combined inputs in each training recipe; all arms share this feature-covered cohort.",
            })
        if protocol_spec.get("featureSetId") not in (None, "", feature_id):
            error(
                "BUNDLE_FEATURE_MISMATCH",
                "The protocol pins a different feature source. Choose a matching bundle or a new protocol revision.",
            )
        try:
            feature = self.store.get_configuration(feature_id)
        except StorageError as failure:
            error(failure.code, str(failure))
            return result
        # Which extraction output this bundle holds decides which architectures can
        # read it. Records frozen before slide encoders were supported hold patches.
        result["featureKind"] = feature["manifest"].get("spec", {}).get("featureKind", "patch")
        present = {row["slideId"] for row in feature["manifest"].get("files", [])}
        required = {row["slideId"] for row in protocol.get("memberships", [])}
        if required - present:
            error(
                "MISSING_FEATURES",
                f"The bundle lacks features for {len(required - present)} eligible protocol slides.",
            )

        packs = {item["id"]: item for item in manifest["packs"]}
        pack_id = spec.packArtifactId
        mode = spec.loadingPolicy
        if pack_id and pack_id not in packs:
            error("PACK_NOT_IN_BUNDLE", "The selected pack is not part of this frozen bundle.")
        elif mode == "auto":
            if pack_id:
                mode = "mmap"
            elif not packs:
                mode = "native"
            elif len(packs) == 1:
                candidate = next(iter(packs.values()))
                source_dtype = manifest["summary"].get("dtype")
                source_types = source_dtype if isinstance(source_dtype, list) else [source_dtype]
                if source_types and all(
                    value == candidate["outputDtype"] for value in source_types
                ):
                    mode, pack_id = "mmap", candidate["id"]
                else:
                    error(
                        "PACK_PRECISION_CHOICE_REQUIRED",
                        "The pack changes feature precision. Select that pack explicitly or choose original files.",
                    )
            else:
                error(
                    "PACK_CHOICE_REQUIRED",
                    "This bundle contains multiple packs. Choose which pack this experiment should read.",
                )
        elif mode == "mmap" and not pack_id:
            error(
                "PACK_CHOICE_REQUIRED",
                "Choose a verified pack included in this bundle for memory-mapped loading.",
            )

        legacy_pack = protocol_spec.get("featurePackId")
        if legacy_pack:
            if mode != "mmap" or pack_id != legacy_pack:
                error(
                    "PROTOCOL_PACK_CONFLICT",
                    "This older protocol pins a specific pack. Use that pack or create a protocol revision without a loading binding.",
                )
            else:
                try:
                    status = FeaturePackService(self.store, self.filesystem).resolve_artifact(
                        feature_id, legacy_pack
                    )
                    if not status["current"] or pack_binding_snapshot(
                        status["artifact"]
                    ) != protocol.get("featurePack"):
                        error(
                            "PROTOCOL_PACK_CHANGED",
                            "The pack no longer matches the representation pinned by this protocol.",
                        )
                except StorageError as failure:
                    error(failure.code, str(failure))
        if not any(item["severity"] == "error" for item in findings):
            result.update(canPlan=True, resolvedLoadingPolicy=mode, packArtifactId=pack_id)
        return result
