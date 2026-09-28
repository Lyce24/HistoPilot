"""Bounded level-0 slide views; optional image libraries stay outside API startup."""

import io
import math
import stat
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
        from histopilot.viewer.reader_cache import OPENSLIDE_READERS

        stamp = Path(path).stat()
        key = (
            str(Path(path).absolute()),
            stamp.st_dev,
            stamp.st_ino,
            stamp.st_size,
            stamp.st_mtime_ns,
            stamp.st_ctime_ns,
            id(openslide.OpenSlide),
        )

        def create():
            if not openslide.OpenSlide.detect_format(str(path)):
                return None
            value = openslide.OpenSlide(str(path))
            try:
                if hasattr(openslide, "OpenSlideCache") and hasattr(value, "set_cache"):
                    try:
                        value.set_cache(openslide.OpenSlideCache(32 * 1024**2))
                    except openslide.OpenSlideVersionError:
                        pass  # Older libraries retain their built-in private cache.
                return value
            except BaseException:
                value.close()
                raise

        try:
            with OPENSLIDE_READERS.lease(key, create) as slide:
                if slide is not None:
                    yield slide, "openslide"
                    return
        except (openslide.OpenSlideError, OSError) as error:
            raise StorageError(
                "The whole-slide region cannot be read.", "SLIDE_FORMAT_UNSUPPORTED", 422
            ) from error
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


def _inspect_open_slide(slide, backend):
    width, height = (
        slide.level_dimensions[0]
        if backend == "opensdpc"
        else slide.dimensions
        if backend == "openslide"
        else slide.size
    )
    if (
        type(width) is not int
        or type(height) is not int
        or not 0 < width <= 2**31
        or not 0 < height <= 2**31
    ):
        raise StorageError("The slide dimensions are invalid.", "SLIDE_GEOMETRY_INVALID", 422)
    downsamples = list(slide.level_downsamples) if backend != "pillow" else [1.0]
    if (
        not downsamples
        or len(downsamples) > 64
        or downsamples[0] != 1
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 1 for v in downsamples)
        or any(a >= b for a, b in zip(downsamples, downsamples[1:]))
    ):
        raise StorageError("The slide pyramid is invalid.", "SLIDE_GEOMETRY_INVALID", 422)
    return {
        "width": width,
        "height": height,
        "backend": backend,
        "levelDownsamples": downsamples,
        "coordinateSpace": "level0",
    }


def inspect_slide(path):
    _reject_symlink_components(Path(path))
    if Path(path).suffix.lower() == ".sdpc":
        from histopilot.viewer.sdpc import read_sdpc

        return read_sdpc(path)
    from histopilot.viewer.sdpc import read_openslide

    return read_openslide(path)


def _validate_view_size(max_size):
    if type(max_size) is not int or not 64 <= max_size <= MAX_VIEW_SIDE:
        raise StorageError("View size must be 64–2048 pixels.", "SLIDE_VIEW_INVALID", 422)


def _render_open_slide(slide, backend, *, max_size, region):
    """Shared affine mapping; callable in the standalone SDPC interpreter."""
    Image = _pillow()
    _validate_view_size(max_size)
    metadata = _inspect_open_slide(slide, backend)
    width, height = metadata["width"], metadata["height"]
    if region is not None and (not isinstance(region, (list, tuple)) or len(region) != 4):
        raise StorageError("Provide four viewport bounds.", "SLIDE_VIEW_INVALID", 422)
    x, y, view_width, view_height = region if region is not None else (0, 0, width, height)
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v)
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
    if backend != "pillow":
        if backend == "opensdpc":
            # OpenSDPC chooses the nearest level, which can undersample tissue.
            # Select the coarsest level that still contains the requested detail.
            level = max(i for i, value in enumerate(slide.level_downsamples) if value <= 1 / scale)
        else:
            level = slide.get_best_level_for_downsample(1 / scale)
        downsample = slide.level_downsamples[level]
        origin = (int(x), int(y))
        if backend == "opensdpc":
            # Its native decoder floors the origin in level pixels. Account for
            # this remainder or attention drifts by up to one pyramid pixel.
            level_x, level_y = (math.floor(value / downsample) for value in origin)
            offset_x, offset_y = x / downsample - level_x, y / downsample - level_y
        else:
            offset_x, offset_y = (x - origin[0]) / downsample, (y - origin[1]) / downsample
        source_box = (
            offset_x,
            offset_y,
            offset_x + view_width / downsample,
            offset_y + view_height / downsample,
        )
        read_size = math.ceil(source_box[2]), math.ceil(source_box[3])
        if read_size[0] * read_size[1] > MAX_RASTER_PIXELS:
            raise StorageError(
                "This pyramid level is too large for a bounded view. Zoom into a smaller region.",
                "SLIDE_VIEW_TOO_LARGE",
                422,
            )
        if backend == "opensdpc":
            # The native SDPC API does not promise OpenSlide's out-of-bounds
            # padding. Never pass an overhanging read to the native decoder.
            level_width, level_height = slide.level_dimensions[level]
            safe_size = (
                min(read_size[0], max(0, level_width - level_x)),
                min(read_size[1], max(0, level_height - level_y)),
            )
            tile = Image.new("RGB", read_size, "white")
            if min(safe_size) > 0:
                tile.paste(slide.read_region(origin, level, safe_size).convert("RGB"), (0, 0))
        else:
            tile = slide.read_region(origin, level, read_size)
    else:
        tile = slide.crop((int(x), int(y), math.ceil(x + view_width), math.ceil(y + view_height)))
        source_box = (x - int(x), y - int(y), x - int(x) + view_width, y - int(y) + view_height)
    # Resize the exact level-0 field of view, including nondivisible bounds.
    tile = tile.convert("RGB").resize(output_size, Image.Resampling.LANCZOS, box=source_box)
    output = io.BytesIO()
    tile.save(output, format="PNG", compress_level=1)
    return output.getvalue()


def _source_identity(path):
    _reject_symlink_components(path)
    try:
        value = path.stat()
        if not stat.S_ISREG(value.st_mode):
            raise OSError("Not a regular slide")
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    except OSError as error:
        raise StorageError(
            "The slide changed or is no longer available. Reopen it before viewing.",
            "SLIDE_SOURCE_CHANGED",
            409,
        ) from error


def _view_key(region):
    if region is None:
        return None
    if (
        not isinstance(region, (list, tuple))
        or len(region) != 4
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in region)
    ):
        raise StorageError(
            "Provide four finite numeric viewport bounds.", "SLIDE_VIEW_INVALID", 422
        )
    return tuple(region)


def render_slide(path, *, max_size=1024, region=None):
    """Render at most 2048² pixels; a WSI read never decodes its entire base level."""
    from histopilot.viewer.image_cache import SLIDE_IMAGES

    _validate_view_size(max_size)
    region = _view_key(region)
    path = Path(path).absolute()
    source = _source_identity(path)
    from histopilot.viewer.sdpc import (
        _openslide_python,
        _python,
        _runtime_key,
        read_openslide,
        read_sdpc,
    )

    backend = "opensdpc" if path.suffix.lower() == ".sdpc" else "openslide"
    runtime = _runtime_key(_python() if backend == "opensdpc" else _openslide_python())
    key = (str(path), source, backend, runtime, max_size, region)

    def verify_source():
        if _source_identity(path) != source:
            raise StorageError(
                "The slide changed while it was being read. Reopen it before viewing.",
                "SLIDE_SOURCE_CHANGED",
                409,
            )

    def render():
        reader = read_sdpc if backend == "opensdpc" else read_openslide
        content = reader(path, max_size=max_size, region=region)
        verify_source()  # Never publish a render made while its source was changing.
        return content

    content = SLIDE_IMAGES.get_or_create(key, render)
    try:
        verify_source()  # Cache hits still validate the file after retrieving the bytes.
        return content
    except BaseException:
        SLIDE_IMAGES.discard(key)
        raise
