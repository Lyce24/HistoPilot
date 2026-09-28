"""Bounded real-image viewer smoke benchmark; no HTTP server or full-slide conversion.

Run with an imaging-enabled Python and --slide arguments. SDPC uses the configured
HISTOPILOT_SDPC_PYTHON/HISTOPILOT_TRIDENT_PYTHON environment through production code.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from histopilot.application.morphology import _image_fingerprint  # noqa: E402
from histopilot.storage.packed import _stamp  # noqa: E402
from histopilot.viewer.image_cache import SLIDE_IMAGES  # noqa: E402
from histopilot.viewer.sdpc import close_readers  # noqa: E402
from histopilot.viewer.slide_images import (  # noqa: E402
    _source_identity,
    inspect_slide,
    render_slide,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def tissue_centers(content, width, height):
    """Choose colorful non-background pixels only; this does not assign pathology labels."""
    with Image.open(io.BytesIO(content)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = maximum - minimum
    mask = (saturation > 18) & (rgb.mean(axis=2) < 240) & (maximum > 40)
    h, w = mask.shape
    candidates = []
    for y in range(0, h, max(8, h // 24)):
        for x in range(0, w, max(8, w // 24)):
            cell = mask[y : y + max(8, h // 24), x : x + max(8, w // 24)]
            yy, xx = np.nonzero(cell)
            if len(xx) < 3:
                continue
            # Snap the center back onto an actual foreground pixel.
            near = np.argmin((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2)
            px, py = x + int(xx[near]), y + int(yy[near])
            score = float(cell.mean() * saturation[py, px])
            candidates.append((score, (px + 0.5) / w, (py + 0.5) / h))
    candidates.sort(reverse=True)
    selected = []
    for score, x, y in candidates:
        if all((x - row[1]) ** 2 + (y - row[2]) ** 2 >= 0.08**2 for row in selected):
            selected.append((score, x, y))
        if len(selected) == 3:
            break
    if not selected:
        selected = [(0, 0.5, 0.5)]
    return [
        {"x": round(x * width), "y": round(y * height), "foregroundScore": score}
        for score, x, y in selected
    ], float(mask.mean())


def region_at(center, width, height, downsample):
    span = 512 * downsample
    w, h = min(span, width), min(span, height)
    return [
        max(0, min(width - w, center["x"] - w // 2)),
        max(0, min(height - h, center["y"] - h // 2)),
        w,
        h,
    ]


def timed_render(path, region):
    start = time.perf_counter()
    content = render_slide(path, max_size=512, region=region)
    elapsed = time.perf_counter() - start
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        size = image.size
    return {
        "region": region,
        "seconds": elapsed,
        "bytes": len(content),
        "pixels": list(size),
    }, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.slide) <= 5:
        parser.error("Choose 1–5 image files for this bounded smoke benchmark.")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = {"slides": [], "failures": [], "complete": False}
    report = {
        "slides": [],
        "failures": catalog["failures"],
        "limitations": "Local image render/decode smoke test on selected existing images; no diagnostic labels, no full-slide decoding/conversion, and no HTTP/browser transport included. Cold means fresh reader process, not flushed operating-system caches.",
    }
    write_json(args.output / "catalog.json", catalog)
    try:
        for value in args.slide:
            path = Path(value).absolute()
            close_readers()
            SLIDE_IMAGES.clear()
            row = {"path": str(path), "label": path.name, "fileBytes": path.stat().st_size}
            try:
                initial = _stamp(path.stat())
                start = time.perf_counter()
                metadata = inspect_slide(path)
                row.update(metadata, metadataColdSeconds=time.perf_counter() - start)
                start = time.perf_counter()
                overview = render_slide(path, max_size=1024)
                row["overviewSeconds"] = time.perf_counter() - start
                if _stamp(path.stat()) != initial:
                    raise RuntimeError("Source changed during benchmark preparation")
                index = len(catalog["slides"])
                folder = args.output / f"slide-{index}"
                folder.mkdir(exist_ok=True)
                (folder / "overview.png").write_bytes(overview)
                row["slideIndex"] = index
                row["sourceFingerprint"] = _image_fingerprint(path, initial)
                row["sourceStamp"] = list(_source_identity(path))
                row["overviewPath"] = str(folder / "overview.png")
                row["tissueCenters"], row["overviewForegroundFraction"] = tissue_centers(
                    overview, row["width"], row["height"]
                )
                catalog["slides"].append(row)
                write_json(args.output / "catalog.json", catalog)
                print(json.dumps({"catalogReady": row}), flush=True)
                reads, exact = [], []
                for offset, downsample in enumerate((1, 4, 16)):
                    center = row["tissueCenters"][offset % len(row["tissueCenters"])]
                    region = region_at(center, row["width"], row["height"], downsample)
                    result, content = timed_render(path, region)
                    target = folder / f"tissue-{downsample}x.png"
                    target.write_bytes(content)
                    result["imagePath"] = str(target)
                    reads.append(result)
                    exact.append(content)
                repeat = []
                for read, expected in zip(reads, exact):
                    result, content = timed_render(path, read["region"])
                    if content != expected:
                        raise RuntimeError("Cached render pixels changed")
                    repeat.append(result["seconds"])
                SLIDE_IMAGES.clear()
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as workers:
                    futures = [
                        workers.submit(timed_render, path, read["region"]) for read in reads[:2]
                    ]
                    batch = [future.result(timeout=30) for future in futures]
                elapsed = time.perf_counter() - start
                for (_, content), expected in zip(batch, exact):
                    if content != expected:
                        raise RuntimeError("Concurrent render pixels changed")
                if _stamp(path.stat()) != initial:
                    raise RuntimeError("Source changed during tissue benchmark reads")
                row.update(
                    tissueReads=reads,
                    cachedSeconds=repeat,
                    cachedMedianSeconds=statistics.median(repeat),
                    concurrentSeconds=elapsed,
                    concurrentReads=[result for result, _ in batch],
                )
                report["slides"].append(row)
                write_json(args.output / "report.json", report)
                write_json(args.output / "catalog.json", catalog)
                print(
                    json.dumps(
                        {
                            "benchmarked": row["label"],
                            "tissueSeconds": [item["seconds"] for item in reads],
                            "cachedMedianSeconds": row["cachedMedianSeconds"],
                            "concurrentSeconds": elapsed,
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "path": str(path),
                    "error": str(error),
                    "code": getattr(error, "code", type(error).__name__),
                }
                catalog["failures"].append(failure)
                # Metadata-ready entries remain useful to the browser, but failures are explicit.
                write_json(args.output / "catalog.json", catalog)
                write_json(args.output / "report.json", report)
                print(json.dumps({"failed": failure}), flush=True)
        catalog["complete"] = True
        write_json(args.output / "catalog.json", catalog)
        write_json(args.output / "report.json", report)
    finally:
        close_readers()
        SLIDE_IMAGES.clear()
    return 1 if catalog["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
