"""Real slide pixels over local stdio for offline browser performance checks.

Uses production readers/cache; deliberately starts no HistoPilot or HTTP server.
With --api, includes actual authentication and dataset validation in-process.
Network latency is excluded in both modes.
"""

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from histopilot.viewer.image_cache import SLIDE_IMAGES  # noqa: E402
from histopilot.viewer.sdpc import close_readers  # noqa: E402
from histopilot.viewer.slide_images import _source_identity, render_slide  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--api",
        action="store_true",
        help="Include real authenticated FastAPI/dataset work in-process.",
    )
    args = parser.parse_args()
    slides = {row["slideIndex"]: row for row in json.loads(args.catalog.read_text())["slides"]}
    stamps = {
        index: tuple(row.get("sourceStamp") or _source_identity(Path(row["path"])))
        for index, row in slides.items()
    }
    output_lock = Lock()
    api = None
    if args.api:
        from real_slide_api_fixture import create_real_slide_api

        api = create_real_slide_api(args.catalog)

    def respond(value):
        with output_lock:
            print(json.dumps(value, allow_nan=False), flush=True)

    def handle(request):
        start = time.perf_counter()
        response = {"id": request["id"]}
        try:
            index = request["slideIndex"]
            row = slides[index]
            path = Path(row["path"])
            if _source_identity(path) != stamps[index]:
                raise RuntimeError("The benchmark slide changed after catalog inspection.")
            value = request.get("region")
            region = tuple(value[key] for key in ("x", "y", "width", "height")) if value else None
            max_size = int(request.get("maxSize", 512))
            if api is None:
                content = render_slide(path, max_size=max_size, region=region)
            else:
                client, project_id, datasets, _, _ = api
                params = {
                    "datasetId": datasets[index],
                    "slideId": f"real-slide-{index}",
                    "sourceFingerprint": row["sourceFingerprint"],
                    "max_size": max_size,
                }
                if value:
                    params.update(value)
                result = client.get(
                    f"/api/v1/projects/{project_id}/morphology/image", params=params
                )
                if result.status_code != 200:
                    from histopilot.storage.project_lock import StorageError

                    detail = result.json()
                    raise StorageError(
                        str(detail.get("detail", detail)),
                        detail.get("code", "API_IMAGE_FAILED"),
                        result.status_code,
                    )
                content = result.content
            if _source_identity(path) != stamps[index]:
                raise RuntimeError("The benchmark slide changed during rendering.")
            response.update(base64=base64.b64encode(content).decode("ascii"), bytes=len(content))
        except Exception as error:
            response["error"] = {
                "message": str(error),
                "code": getattr(error, "code", "BENCHMARK_READ_FAILED"),
                "status": getattr(error, "status_code", 500),
            }
        response["elapsedMs"] = (time.perf_counter() - start) * 1000
        respond(response)

    executor = ThreadPoolExecutor(max_workers=3)
    try:
        for raw in sys.stdin:
            request = json.loads(raw)
            if request.get("op") == "reset":
                # A reset is a barrier; retire existing work before clearing caches.
                executor.shutdown(wait=True)
                close_readers()
                SLIDE_IMAGES.clear()
                executor = ThreadPoolExecutor(max_workers=3)
                respond({"id": request["id"], "elapsedMs": 0, "reset": True})
            else:
                executor.submit(handle, request)
    finally:
        executor.shutdown(wait=True)
        close_readers()
        SLIDE_IMAGES.clear()
        if api is not None:
            api[-1]()


if __name__ == "__main__":
    main()
