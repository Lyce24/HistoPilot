"""Versioned patient → specimen → slide hierarchy and source references."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    id: str
    project_id: str
    name: str
    manifest_uri: str
    source_table_uris: tuple[str, ...]
    labels_uri: str
    content_hash: str
    parent_version_id: str | None = None


@dataclass(frozen=True, slots=True)
class Patient:
    id: str
    dataset_version_id: str
    site: str | None = None


@dataclass(frozen=True, slots=True)
class Specimen:
    id: str
    patient_id: str
    dataset_version_id: str
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class Block:
    """Optional intermediate entity for specimen/block/slide source resolution."""

    id: str
    specimen_id: str
    dataset_version_id: str


@dataclass(frozen=True, slots=True)
class Slide:
    id: str
    specimen_id: str
    dataset_version_id: str
    uri: str
    content_hash: str
    stain: str = "H&E"
    block_id: str | None = None
