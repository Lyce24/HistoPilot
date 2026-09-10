"""WSI access boundary; core code does not import OpenSlide, cuCIM, or Pillow."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    width: int
    height: int
    level_downsamples: tuple[float, ...]
    mpp_x: float | None = None
    mpp_y: float | None = None


class WSIPort(Protocol):
    def metadata(self, slide_uri: str) -> SlideMetadata:
        """Read slide geometry without exposing a backend-specific object."""
        ...

    def read_region(
        self, slide_uri: str, *, x: int, y: int, level: int, width: int, height: int
    ) -> bytes:
        """Return PNG bytes; x/y use level-0 pixels, dimensions use the chosen level."""
        ...
