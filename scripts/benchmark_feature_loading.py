#!/usr/bin/env python3
"""Compare identical float32 HDF5 and packed bags using OceanPath's real readers.

Run with OceanPath's Python environment. This is an opt-in benchmark, not a
HistoPilot runtime dependency. It never edits either input. The optional cold
phase evicts only the selected input files with POSIX_FADV_DONTNEED; Linux
mincore residency measurements disclose whether that advisory eviction worked.
Host/hardware caches are not flushed. Long runs should be launched inside tmux.

Example:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ../OceanPath-colon-development/.venv/bin/python scripts/benchmark_feature_loading.py \
      --oceanpath ../OceanPath-colon-development --features /path/features_uni_v1 \
      --pack /path/matched-float32-pack --output docs/evidence/loading.json

Every source feature bit and coordinate is checked against the pack before any
timing. Both readers receive exactly the same slide order, row indices, return
float32 features and int32 coords, and use the same DataLoader configuration.
Sampling is seeded and sorted on BOTH readers (OceanPath normally sorts only
the packed path). No augmentation, model, CUDA, or application LRU is involved.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import mmap
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oceanpath", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--bag-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--warm-repeats", type=int, default=7)
    parser.add_argument("--cold-repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional small smoke-test subset")
    parser.add_argument("--build-receipt", type=Path)
    return parser.parse_args()


ARGS = arguments()
sys.path.insert(0, str(ARGS.oceanpath.resolve() / "src"))
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from oceanpath.datasets.datamodule import SlideDataset  # noqa: E402
from oceanpath.datasets.packed import PackedFeatureStore  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class MatchedSlideDataset(SlideDataset):
    """Use the same deterministic row order on both existing reader methods."""

    def _random_indices(self, *args):
        return np.sort(super()._random_indices(*args))

    def _fill_indices(self, *args):
        return np.sort(super()._fill_indices(*args))


def single_bag(batch):
    return batch[0]


@dataclass
class WorkerReady:
    queue: object

    def __call__(self, worker_id):
        torch.set_num_threads(1)
        self.queue.put((worker_id, time.perf_counter()))


def event(**data):
    print(json.dumps(data, sort_keys=True), flush=True)


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def residency(paths):
    """Count Linux resident pages without faulting file data into RAM."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    resident = total = 0
    for path in paths:
        size = path.stat().st_size
        if not size:
            continue
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        with path.open("rb") as handle:
            # MAP_PRIVATE makes the writable buffer view safe for read-only files.
            with mmap.mmap(
                handle.fileno(), size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE
            ) as mapping:
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                vector = (ctypes.c_ubyte * pages)()
                if libc.mincore(address, size, vector):
                    raise OSError(ctypes.get_errno(), "mincore failed")
                resident += sum(value & 1 for value in vector)
                total += pages
    return {
        "residentPages": resident,
        "totalPages": total,
        "fraction": resident / total if total else 0,
    }


def evict(paths):
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    return residency(paths)


def input_stamps():
    """Pin mutable input files for the entire measurement, excluding access times."""
    paths = [
        *sorted(ARGS.features.glob("*.h5")),
        *sorted(path for path in ARGS.pack.iterdir() if path.is_file()),
    ]
    return {
        str(path.resolve()): {
            "sizeBytes": info.st_size,
            "mtimeNs": info.st_mtime_ns,
            "ctimeNs": info.st_ctime_ns,
            "deviceId": info.st_dev,
            "inode": info.st_ino,
        }
        for path in paths
        for info in [path.stat()]
    }


def verify_pair():
    start = time.perf_counter()
    store = PackedFeatureStore(ARGS.pack)
    paths = {p.stem: p for p in sorted(ARGS.features.glob("*.h5"))}
    if set(paths) != set(store.slide_ids):
        raise ValueError("Source and pack slide IDs differ; refusing unmatched benchmark")
    ids = sorted(paths)
    if ARGS.limit:
        ids = ids[: ARGS.limit]
    if store.meta.feat_dtype != "float32":
        raise ValueError("Pack must already store float32, not convert float16 at load time")
    source_hash = hashlib.sha256()
    pack_hash = hashlib.sha256()
    inventory = []
    patches = 0
    for number, slide_id in enumerate(ids):
        path = paths[slide_id]
        pos = store.position(slide_id)
        length = store.length_of(slide_id)
        with h5py.File(path, "r") as handle:
            features, coords = handle["features"], handle["coords"]
            if features.dtype != np.dtype("float32"):
                raise ValueError(f"Source {slide_id} is not float32")
            if features.shape != (length, store.feat_dim) or coords.shape != (length, 2):
                raise ValueError(
                    f"Shape mismatch for {slide_id}: source {features.shape}, pack {(length, store.feat_dim)}"
                )
            chunk_rows = max(1, (16 * 1024 * 1024) // (store.feat_dim * 4))
            for offset in range(0, length, chunk_rows):
                rows = np.arange(offset, min(offset + chunk_rows, length))
                original = features[offset : offset + len(rows)]
                packed = store.read_features(pos, rows)
                if not np.array_equal(original.view(np.uint32), packed.view(np.uint32)):
                    raise ValueError(f"Feature bytes differ for {slide_id} starting row {offset}")
                original_coords = coords[offset : offset + len(rows)].astype(np.int64)
                packed_coords = store.read_coords(pos, rows).astype(np.int64)
                if not np.array_equal(original_coords, packed_coords):
                    raise ValueError(f"Coordinates differ for {slide_id} starting row {offset}")
                for digest, values, xy in (
                    (source_hash, original, original_coords),
                    (pack_hash, packed, packed_coords),
                ):
                    digest.update(values.tobytes())
                    digest.update(xy.tobytes())
            inventory.append(
                {
                    "slideId": slide_id,
                    "patches": length,
                    "bytes": path.stat().st_size,
                    "chunks": features.chunks,
                    "compression": features.compression,
                }
            )
        patches += length
        if number % 20 == 0 or number == len(ids) - 1:
            event(stage="verifying_all_values", completed=number + 1, total=len(ids))
    result = {
        "allFeatureBitsEqual": True,
        "allCoordinatesEqual": True,
        "sourceTensorSha256": source_hash.hexdigest(),
        "packTensorSha256": pack_hash.hexdigest(),
        "slideCount": len(ids),
        "totalPatches": patches,
        "dimensions": store.feat_dim,
        "sourceDtype": "float32",
        "packDtype": "float32",
        "returnedCoordsDtype": "int32",
        "sourceBytes": sum(row["bytes"] for row in inventory),
        "packBytes": sum(p.stat().st_size for p in ARGS.pack.iterdir() if p.is_file()),
        "sourceDevice": ARGS.features.stat().st_dev,
        "packDevice": ARGS.pack.stat().st_dev,
        "seconds": time.perf_counter() - start,
        "inventory": inventory,
    }
    # Explicitly release maps before file-local cache eviction later.
    for attr in ("_features", "_coords"):
        value = getattr(store, attr, None)
        if value is not None:
            value._mmap.close()
    del store
    gc.collect()
    return ids, result


def make_loader(backend, bag_size, workers, ids):
    begin = time.perf_counter()
    store = PackedFeatureStore(ARGS.pack) if backend == "pack" else None
    dataset = MatchedSlideDataset(
        str(ARGS.features),
        slide_ids=ids,
        labels={sid: 0 for sid in ids},
        is_train=False,
        fixed_bag_size=bag_size,
        short_bag_policy="repeat",
        cap_strategy="random",
        eval_crop_seed=ARGS.seed,
        instance_dropout=0,
        feature_noise_std=0,
        cache_size_mb=0,
        return_coords=True,
        force_float32=True,
        store=store,
    )
    if dataset.slide_ids != ids:
        raise ValueError("Reader changed the exact requested slide order")
    context = mp.get_context("fork")
    queue = context.Queue() if workers else None
    options = (
        {
            "prefetch_factor": ARGS.prefetch,
            "persistent_workers": True,
            "multiprocessing_context": context,
            "worker_init_fn": WorkerReady(queue),
        }
        if workers
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        collate_fn=single_bag,
        generator=torch.Generator().manual_seed(ARGS.seed),
        **options,
    )
    return loader, queue, time.perf_counter() - begin


def close_loader(loader, queue):
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()
    store = loader.dataset.store
    if store is not None:
        for attr in ("_features", "_coords"):
            value = getattr(store, attr, None)
            if value is not None:
                value._mmap.close()
    if queue is not None:
        queue.close()
        queue.join_thread()
    del loader
    gc.collect()


def epoch(loader, queue, startup):
    begin = time.perf_counter()
    iterator = iter(loader)
    iterator_seconds = time.perf_counter() - begin
    ready_seconds = None
    if startup and queue is not None:
        ready = [queue.get(timeout=60) for _ in range(loader.num_workers)]
        ready_seconds = max(moment for _, moment in ready) - begin
    consume = wait = 0.0
    features_sum = 0.0
    coords_sum = rows = byte_count = slides = 0
    first = None
    while True:
        before_next = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            wait += time.perf_counter() - before_next
            break
        wait += time.perf_counter() - before_next
        if first is None:
            first = time.perf_counter() - begin
        before_consume = time.perf_counter()
        values, coords = batch["features"], batch["coords"]
        if values.dtype != torch.float32 or coords.dtype != torch.int32:
            raise ValueError("Reader output dtype changed")
        # Read every returned element; float32 reduction avoids casting/copying 5 GB
        # to float64 and obscuring loader differences with a consumer allocation.
        features_sum += values.sum().item()
        coords_sum += coords.sum().item()
        rows += values.shape[0]
        byte_count += values.nbytes + coords.nbytes
        slides += 1
        del batch, values, coords
        consume += time.perf_counter() - before_consume
    elapsed = time.perf_counter() - begin
    return {
        "seconds": elapsed,
        "firstBatchSeconds": first,
        "iteratorConstructionSeconds": iterator_seconds,
        "workersReadySeconds": ready_seconds,
        "mainThreadNextSeconds": wait,
        "consumeSeconds": consume,
        "slideCount": slides,
        "rows": rows,
        "returnedBytes": byte_count,
        "featureSumFloat32": features_sum,
        "coordinateSum": coords_sum,
        "slidesPerSecond": slides / elapsed,
        "returnedGiBPerSecond": byte_count / elapsed / 1024**3,
    }


def summarize(results, build_seconds):
    summaries = []
    keys = sorted({(v["workers"], v["workload"], v["cache"]) for v in results})
    for workers, workload, cache in keys:
        group = [
            v
            for v in results
            if (v["workers"], v["workload"], v["cache"]) == (workers, workload, cache)
        ]
        native = sorted((v for v in group if v["backend"] == "h5"), key=lambda v: v["repeat"])
        packed = sorted((v for v in group if v["backend"] == "pack"), key=lambda v: v["repeat"])
        ratios = [n["seconds"] / p["seconds"] for n, p in zip(native, packed, strict=True)]
        for n, p in zip(native, packed, strict=True):
            for field in ("featureSumFloat32", "coordinateSum", "rows", "returnedBytes"):
                if n[field] != p[field]:
                    raise ValueError(f"Matched loader outputs differ: {field}")
        rng = np.random.default_rng(ARGS.seed)
        boot = np.median(rng.choice(ratios, size=(20000, len(ratios)), replace=True), axis=1)
        n = statistics.median(v["seconds"] for v in native)
        p = statistics.median(v["seconds"] for v in packed)
        saved = n - p
        summaries.append(
            {
                "workers": workers,
                "workload": workload,
                "cache": cache,
                "pairs": len(ratios),
                "nativeMedianSeconds": n,
                "packMedianSeconds": p,
                "medianPairedSpeedup": statistics.median(ratios),
                "speedupBootstrap95Interval": np.quantile(boot, [0.025, 0.975]).tolist(),
                "minimumPairedSpeedup": min(ratios),
                "maximumPairedSpeedup": max(ratios),
                "secondsSavedPerEpoch": saved,
                "loaderTimeReductionPercent": 100 * saved / n,
                "packingBreakEvenEpochs": build_seconds / saved
                if build_seconds and saved > 0
                else None,
                "nativeMedianConsumeSeconds": statistics.median(
                    v["consumeSeconds"] for v in native
                ),
                "packMedianConsumeSeconds": statistics.median(v["consumeSeconds"] for v in packed),
                "nativeMedianFirstBatchSeconds": statistics.median(
                    v["firstBatchSeconds"] for v in native
                ),
                "packMedianFirstBatchSeconds": statistics.median(
                    v["firstBatchSeconds"] for v in packed
                ),
            }
        )
    return summaries


def save(report):
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARGS.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(ARGS.output)


def main():
    output = ARGS.output.resolve()
    if output.is_relative_to(ARGS.features.resolve()) or output.is_relative_to(ARGS.pack.resolve()):
        raise ValueError("Write benchmark results outside the immutable input directories")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {
        "state": "running",
        "startedAt": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (OceanPath Python 3.10)
        "command": sys.argv,
        "host": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "oceanPathRevision": git_revision(ARGS.oceanpath),
        "source": str(ARGS.features.resolve()),
        "pack": str(ARGS.pack.resolve()),
        "config": {
            "workers": ARGS.workers,
            "prefetchFactor": ARGS.prefetch,
            "batchSize": 1,
            "persistentWorkers": "true when workers > 0; same workers retained for warm repeats",
            "shuffle": False,
            "samplerSeed": ARGS.seed,
            "sampledBagSize": ARGS.bag_size,
            "pinMemory": False,
            "floatDtype": "float32",
            "coordsDtype": "int32",
            "torchThreadsPerProcess": 1,
            "returnCoords": True,
            "applicationCacheBytes": 0,
            "collation": "single bag passthrough, standard Torch multiprocessing tensor transfer",
            "consumption": "float32 sum touches every feature; int64 sum touches every coordinate",
        },
        "cacheMethod": "Warm: untimed epoch per reader before repeated persistent-worker epochs. Cold: fresh workers after input-only POSIX_FADV_DONTNEED; mincore verifies residency; no global cache drop. WSL host and hardware caches remain uncontrolled.",
        "timingScope": "Loader-only CPU delivery plus all-element consumption, no model/GPU. Setup, worker readiness, first-batch latency, and consumer time separately recorded. Cold epochs include worker startup; warm epochs exclude initial startup.",
        "statisticalLimit": "Paired alternating repeated measurements on one host; bootstrap interval describes repeat variability, not generalization to other machines/cohorts.",
        "results": [],
        "warmups": [],
    }
    if ARGS.build_receipt:
        report["packingBuild"] = json.loads(ARGS.build_receipt.read_text())
    report["inputStampsBefore"] = input_stamps()
    save(report)
    try:
        ids, report["verification"] = verify_pair()
    except Exception as error:
        report["state"] = "invalid-input-pair"
        report["error"] = str(error)
        save(report)
        raise
    event(stage="verified", **{k: v for k, v in report["verification"].items() if k != "inventory"})
    save(report)
    data_paths = {
        "h5": [ARGS.features / f"{sid}.h5" for sid in ids],
        "pack": [ARGS.pack / "features.bin", ARGS.pack / "coords.bin"],
    }
    for workers in ARGS.workers:
        for bag_size in (None, ARGS.bag_size):
            workload = "full_bag" if bag_size is None else f"fixed_{bag_size}"
            pools = {}
            try:
                for backend in ("h5", "pack"):
                    loader, queue, setup = make_loader(backend, bag_size, workers, ids)
                    pools[backend] = (loader, queue)
                    value = {
                        "backend": backend,
                        "workers": workers,
                        "workload": workload,
                        "setupSeconds": setup,
                        **epoch(loader, queue, startup=True),
                    }
                    report["warmups"].append(value)
                    event(stage="warmup", **value)
                for repeat in range(ARGS.warm_repeats):
                    order = ("h5", "pack") if repeat % 2 == 0 else ("pack", "h5")
                    for backend in order:
                        pages = residency(data_paths[backend])
                        loader, queue = pools[backend]
                        result = {
                            "backend": backend,
                            "workers": workers,
                            "workload": workload,
                            "cache": "warm",
                            "repeat": repeat,
                            "residencyBefore": pages,
                            **epoch(loader, queue, startup=False),
                        }
                        report["results"].append(result)
                        event(stage="timing", **result)
                        save(report)
            finally:
                for loader, queue in pools.values():
                    close_loader(loader, queue)
                pools.clear()
            for repeat in range(ARGS.cold_repeats):
                order = ("h5", "pack") if repeat % 2 == 0 else ("pack", "h5")
                for backend in order:
                    loader, queue, setup = make_loader(backend, bag_size, workers, ids)
                    try:
                        pages = evict(data_paths[backend])
                        result = {
                            "backend": backend,
                            "workers": workers,
                            "workload": workload,
                            "cache": "file_cache_cold",
                            "repeat": repeat,
                            "setupSeconds": setup,
                            "residencyBefore": pages,
                            **epoch(loader, queue, startup=True),
                        }
                        report["results"].append(result)
                        event(stage="timing", **result)
                        save(report)
                    finally:
                        close_loader(loader, queue)
    report["inputStampsAfter"] = input_stamps()
    report["inputsUnchanged"] = report["inputStampsBefore"] == report["inputStampsAfter"]
    if not report["inputsUnchanged"]:
        report["state"] = "invalid-inputs-changed"
        save(report)
        raise ValueError("An input changed during benchmarking; timings are invalid")
    report["summary"] = summarize(report["results"], report.get("packingBuild", {}).get("seconds"))
    report["state"] = "complete"
    report["finishedAt"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017 (Python 3.10)
    save(report)
    for value in report["summary"]:
        event(stage="summary", **value)


if __name__ == "__main__":
    main()
