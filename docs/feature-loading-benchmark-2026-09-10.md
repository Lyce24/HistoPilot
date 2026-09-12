# BLCA feature loading benchmark — 2026-09-10

Packing gives a reproducible loading benefit for these BLCA features and should remain optional. It is useful for repeated training or evaluation passes, but native HDF5 remains sufficient for occasional reads. These measurements concern CPU data delivery using the existing one-row-chunked HDF5 files and OceanPath readers; they do not establish the same speedup for other HDF5 layouts or complete GPU training.

## The supplied pack does not match the supplied features

The current source has **1,191,065 patches**; `/path/to/ext4-data/mmap/blca` has **1,191,038**. Both contain the same 138 slide IDs, but **29 slide patch counts differ**. The original and current creation records differ in feature and coordinate tensor hashes for all 138 slides. Direct reads of four sampled slides also differ in the first 100 feature rows and coordinates, including slides with equal row counts. This may reflect ordering or extraction changes; shape/size equality cannot establish content identity.

The provided pack records its source under `/path/to/research-data/features/blca/...`, whereas the requested source is under `/path/to/ext4-data/features/blca/...`. Initial diagnostic timings against that mismatched pack are excluded from the conclusions.

For a fair comparison, HistoPilot created a new, separate float32 pack on the **same ext4 volume**:

`/path/to/ext4-data/mmap/blca-loading-benchmark-20260910`

Neither the source nor the supplied pack was modified. Every feature bit and every coordinate in the new pack was independently compared with the current source before timing, and input stat stamps remained unchanged through the run.

## Results

Each time is the median seconds needed to deliver and consume one complete pass over 138 slides. Speedup is the median of paired native/packed ratios; parentheses show a 95% bootstrap interval across repetitions, not uncertainty across machines.

| Workers | Bag | Cache | Native HDF5 | Packed float32 | Paired speedup (95% interval) | Loader time saved |
|---:|---|---|---:|---:|---:|---:|
| 4 | All patches | Warm | 5.250 s | 2.316 s | 2.27× (2.17–2.33) | 55.9% |
| 4 | All patches | File cache cold | 6.644 s | 3.048 s | 2.16× (2.14–2.21) | 54.1% |
| 4 | 4,096 patches/slide | Warm | 4.984 s | 1.378 s | 3.67× (3.45–3.88) | 72.4% |
| 4 | 4,096 patches/slide | File cache cold | 5.861 s | 2.340 s | 2.57× (2.48–2.63) | 60.1% |
| 0 | All patches | Warm | 10.366 s | 1.461 s | 7.10× (6.99–7.27) | 85.9% |
| 0 | All patches | File cache cold | 13.510 s | 3.072 s | 4.40× (3.80–5.43) | 77.3% |
| 0 | 4,096 patches/slide | Warm | 10.612 s | 0.503 s | 21.06× (20.88–21.59) | 95.3% |
| 0 | 4,096 patches/slide | File cache cold | 13.951 s | 2.123 s | 6.66× (4.75–7.10) | 84.8% |

Warm results use seven alternating paired repetitions after an untimed warmup with persistent workers. File-cache-cold results use five alternating paired repetitions and fresh workers. Linux `mincore` confirmed all relevant pages resident before warm trials and zero resident pages before each cold trial. Only these input files received `POSIX_FADV_DONTNEED`; no system-wide cache drop occurred. WSL host and storage-controller caches were not flushed, so these are Linux file-cache-cold measurements, not guaranteed physical-device-cold measurements.

## Matched settings and timing scope

- **Data:** 138 slides, 1,191,065 patches, 1,024 feature dimensions; float32 on disk and after loading for both readers. Native HDF5 uses uncompressed one-row chunks. Source and packed feature/coordinate verification digests are identical: `17f57215277e10f2934eb8c596fdb9b8b1558857347ea88ec5db8efdfe1af25a`.
- **DataLoader:** batch size 1, identical slide order, 4 workers vs 4 workers with prefetch factor 2, persistent workers, pinning disabled, one Torch thread per process, no model, CUDA, augmentation, or application LRU. The zero-worker pairs are a separate matched single-process comparison; prefetch/persistent-worker options are inapplicable to both.
- **Sampling:** identical deterministic sorted indices with seed 20260910; large bags select 4,096 rows, small bags retain every row and repeat selected rows to reach 4,096. Both paths therefore return exactly 565,248 rows for the sampled workload. Full bags return every patch. This preserves OceanPath reader behavior while sorting the native indices to match its packed ordering.
- **Coordinates:** both existing OceanPath readers return int32 XY. Native HDF5 stores int64 XY while the packed format stores checked int32 XY; equality is checked numerically. No feature precision conversion contributes to the speedup.
- **Consumption:** a float32 reduction touches every returned feature and an integer reduction touches every coordinate. Consumer time, main-thread waiting, iterator startup, worker readiness, first-batch latency and dataset setup are recorded separately in the raw JSON. Warm epochs exclude initial worker startup; cold epochs include it. These are loading-plus-consumption timings, not bare disk throughput.
- **Reader implementations:** actual OceanPath `SlideDataset._read_h5` (read full slide, then select rows) and `PackedFeatureStore` (read only selected rows), with the same collation and Torch multiprocessing tensor transfer. No resident GPU or RAM store optimization is used.

## Cost and verdict

HistoPilot built the matching pack in **60.19 seconds**, including source tensor validation, source checksums, writes, fsync, readback verification and publication. The source was already cached from the preliminary diagnostic; this is not a cold-source packing-time estimate. At the measured four-worker loading savings, the one-time packing cost breaks even after roughly **17–21 complete data passes**. This assumes data loading is exposed in the workload; GPU work may hide some or all of that saved time behind prefetch.

Native files occupy 5,011,107,952 bytes (4.667 GiB); the matching pack occupies 4,888,502,913 bytes (4.553 GiB), only 2.45% smaller. Keeping both consumes an additional 4.553 GiB. Packing is primarily a loading/layout optimization, not compression.

Proceed with optional pack creation and verified existing-pack selection inside the PFM & features flow. Preserve native features as a usable selection. A pack must pass per-slide IDs/counts, expected payload byte lengths, dtype/dimension checks and full tensor/coordinate matching before being treated as interchangeable with a feature version. Folder byte totals should not be required to equal HDF5 totals, which include container metadata and a different coordinate storage dtype. A mismatch should be visible and block use of that pack for the selected features.

## Reproduction and evidence

Host: Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz; 36 logical CPUs, 188.7 GiB RAM; Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39. Python 3.10.19, Torch 2.10.0+cu128, NumPy 2.2.6, h5py 3.15.1, HDF5 1.14.6. OceanPath revision `9cd4b2d03c5cdcba6c18ee0c52ce31287c18b381`.

- Reusable script: [`scripts/benchmark_feature_loading.py`](../scripts/benchmark_feature_loading.py).
- Raw timings, settings, equality evidence and intervals: [`blca-loading-benchmark-20260910.json`](evidence/blca-loading-benchmark-20260910.json).
- Actual executed script snapshot: [`blca-loading-benchmark-20260910.executed.py`](evidence/blca-loading-benchmark-20260910.executed.py). The reusable script also includes automatic before/after input stamp validation, added while this run was active; this run records the equivalent checks in `inputStability`.
- Supplied-pack mismatch details: [`blca-existing-pack-mismatch-20260910.json`](evidence/blca-existing-pack-mismatch-20260910.json).
- Measured pack creation: [`blca-benchmark-pack-build.json`](evidence/blca-benchmark-pack-build.json), [`build script`](evidence/blca-benchmark-pack-build.py).

```bash
cd /path/to/HistoPilot
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../OceanPath-colon-development/.venv/bin/python -u scripts/benchmark_feature_loading.py \
  --oceanpath ../OceanPath-colon-development \
  --features /path/to/ext4-data/features/blca/20x_256px_0px_overlap/features_uni_v1 \
  --pack /path/to/ext4-data/mmap/blca-loading-benchmark-20260910 \
  --output docs/evidence/blca-loading-rerun.json \
  --workers 4 0 --prefetch 2 --warm-repeats 7 --cold-repeats 5
```

Run the command inside tmux for persistence. The original completed run used session `blca-load-bench`; its full log remains in `docs/evidence/blca-loading-benchmark-20260910.log`. No benchmark process needs to remain running after completion.
