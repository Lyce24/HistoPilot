"""Canonical scientific geometry is expressed in level-0 WSI pixels."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchGeometry:
    patch_id: str
    slide_id: str
    x_level0: int
    y_level0: int
    width_level0: int
    height_level0: int
    source_level: int
    source_mpp: float
    requested_mpp: float
    patch_size: int


@dataclass(frozen=True, slots=True)
class AttentionRegion:
    run_id: str
    feature_set_id: str
    geometry: PatchGeometry
    score: float
