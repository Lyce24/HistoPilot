"""Bounded level-0 slide views; optional image libraries stay outside API startup."""

import io
import math
import warnings
from contextlib import contextmanager
from pathlib import Path

from histopilot.storage.project_lock import StorageError, _reject_symlink_components

MAX_RASTER_PIXELS = 32_000_000
MAX_RASTER_BYTES = 256 * 1024 * 1024
MAX_VIEW_SIDE = 2048


def allowed_file(filesystem, value):
    path = Path(value)
    try:
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError
        _reject_symlink_components(path)
        resolved = path.resolve(strict=True)
        if not filesystem._contains(resolved) or not resolved.is_file():
            raise ValueError
        return resolved
    except (OSError, ValueError, RuntimeError) as error:
        raise StorageError(
            "Select a regular local file inside a configured data root, without symbolic links.",
            "INTERPRETATION_PATH_INVALID",
            403,
        ) from error


def _pillow():
    try:
        from PIL import Image

        return Image
    except ImportError as error:
        raise StorageError(
            "Slide viewing needs the optional imaging dependencies (histopilot[imaging]).",
            "SLIDE_VIEWER_UNAVAILABLE",
            503,
        ) from error


@contextmanager
def _open(path):
    """Open pyramidal WSI with OpenSlide; never fall back to decoding a huge raster."""
    _reject_symlink_components(Path(path))
    Image = _pillow()
    slide = None
    try:
        import openslide

    except ImportError:
        openslide = None
    if openslide is not None:
        try:
            if openslide.OpenSlide.detect_format(str(path)):
                slide = openslide.OpenSlide(str(path))
        except (openslide.OpenSlideError, OSError) as error:
            raise StorageError(
                "The whole-slide image is corrupt or unreadable.", "SLIDE_FORMAT_UNSUPPORTED", 422
            ) from error
    if slide is not None:
        try:
            yield slide, "openslide"
        except (openslide.OpenSlideError, OSError) as error:
            raise StorageError(
                "The whole-slide region cannot be read.", "SLIDE_FORMAT_UNSUPPORTED", 422
            ) from error
        finally:
            slide.close()
        return
    if Path(path).stat().st_size > MAX_RASTER_BYTES:
        raise StorageError(
            "This slide requires OpenSlide support; oversized images cannot use raster fallback.",
            "SLIDE_FORMAT_UNSUPPORTED",
            422,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as raster:
                if raster.width * raster.height > MAX_RASTER_PIXELS:
                    raise StorageError(
                        "This image requires a pyramidal slide reader; raster decoding is bounded to 32 megapixels.",
                        "SLIDE_RASTER_TOO_LARGE",
                        422,
                    )
                yield raster, "pillow"
    except StorageError:
        raise
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise StorageError(
            "The slide cannot be opened safely. Install OpenSlide support for whole-slide formats.",
            "SLIDE_FORMAT_UNSUPPORTED",
            422,
        ) from error


def inspect_slide(path):
    with _open(path) as (slide, backend):
        width, height = slide.dimensions if backend == "openslide" else slide.size
        if not 0 < width <= 2**31 or not 0 < height <= 2**31:
            raise StorageError("The slide dimensions are invalid.", "SLIDE_GEOMETRY_INVALID", 422)
        return {
            "width": width,
            "height": height,
            "backend": backend,
            "levelDownsamples": list(slide.level_downsamples) if backend == "openslide" else [1.0],
            "coordinateSpace": "level0",
        }


def render_slide(path, *, max_size=1024, region=None):
    """Render at most 2048² pixels; a WSI read never decodes its entire base level."""
    Image = _pillow()
    if type(max_size) is not int or not 64 <= max_size <= MAX_VIEW_SIDE:
        raise StorageError("View size must be 64–2048 pixels.", "SLIDE_VIEW_INVALID", 422)
    with _open(path) as (slide, backend):
        width, height = slide.dimensions if backend == "openslide" else slide.size
        x, y, view_width, view_height = region or (0, 0, width, height)
        if (
            any(
                not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (x, y, view_width, view_height)
            )
            or x < 0
            or y < 0
            or view_width <= 0
            or view_height <= 0
            or x + view_width > width
            or y + view_height > height
        ):
            raise StorageError(
                "Choose a positive viewport inside the slide.", "SLIDE_VIEW_INVALID", 422
            )
        scale = min(1.0, max_size / max(view_width, view_height))
        output_size = max(1, round(view_width * scale)), max(1, round(view_height * scale))
        if backend == "openslide":
            level = slide.get_best_level_for_downsample(1 / scale)
            downsample = slide.level_downsamples[level]
            read_size = math.ceil(view_width / downsample), math.ceil(view_height / downsample)
            # Sparse pyramids must not silently decode a massive intermediate image.
            if read_size[0] * read_size[1] > MAX_RASTER_PIXELS:
                raise StorageError(
                    "This pyramid level is too large for a bounded view. Zoom into a smaller region.",
                    "SLIDE_VIEW_TOO_LARGE",
                    422,
                )
            tile = slide.read_region((int(x), int(y)), level, read_size)
            source_box = (0, 0, view_width / downsample, view_height / downsample)
        else:
            tile = slide.crop(
                (int(x), int(y), math.ceil(x + view_width), math.ceil(y + view_height))
            )
            source_box = (x - int(x), y - int(y), x - int(x) + view_width, y - int(y) + view_height)
        # A ceil-sized pyramid read can include part of an extra source pixel.
        # Resize only the exact level-0 field of view so the image and patch
        # coordinates share one affine transform, including nondivisible bounds.
        tile = tile.convert("RGB").resize(output_size, Image.Resampling.LANCZOS, box=source_box)
        output = io.BytesIO()
        tile.save(output, format="PNG")
        return output.getvalue()
