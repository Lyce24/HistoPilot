"""Feature artifacts retain encoder, preprocessing, and patch coordinate lineage."""

from dataclasses import dataclass


def representation_kind(manifest: dict) -> str:
    """Read the representation of a feature inventory or saved input contract.

    Frozen patch records predate the field. Their absence must keep meaning
    patch without rewriting their serialized identity.
    """
    return manifest.get("featureKind", manifest.get("spec", {}).get("featureKind", "patch"))


@dataclass(frozen=True, slots=True)
class FeatureSet:
    id: str
    dataset_version_id: str
    encoder: str
    encoder_checkpoint_uri: str
    encoder_checkpoint_hash: str
    slide_ids: tuple[str, ...]
    features_uri: str
    patch_coordinates_uri: str
    extraction_config_uri: str
    embedding_dimension: int
    patch_size_px: int
    target_mpp: float
    coordinate_space: str = "level0_pixels"
