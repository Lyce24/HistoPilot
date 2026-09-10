"""Local artifact identities; large files are stored separately from metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    kind: str
    uri: str
    content_hash: str
    created_by_job_id: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    id: str
    display_name: str
    kind: str
    adapter: str
    backend_identifier: str
    config_uri: str
    available: bool = False
