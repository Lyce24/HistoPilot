#!/usr/bin/env python
"""
Write [N, D] float16 copies of UNI2-h release features for the slides of a dataset table.

MahmoodLab's UNI2-h feature release stores each slide as features (1, N, D) float32 and
coords (1, N, 2); HistoPilot attaches TRIDENT-style files with features [N, D] and
coords [N, 2]. The copy drops the leading axis, stores features as float16 (the
precision OceanPath's feature stores train on) and keeps the coordinate attributes.

    python convert_uni2h_release.py TABLE.csv SOURCE_ROOT OUT_ROOT [--workers 8]

TABLE.csv needs a Feature_Path column relative to SOURCE_ROOT; outputs keep that path.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def convert(args):
    src, dst = args
    if dst.is_file():
        return dst, "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(src, "r") as h:
        feats = h["features"][0] if h["features"].ndim == 3 else h["features"][()]
        coords_ds = h["coords"]
        coords = coords_ds[0] if coords_ds.ndim == 3 else coords_ds[()]
        attrs = dict(coords_ds.attrs)
    if feats.shape[0] != coords.shape[0] or not np.isfinite(feats).all():
        return dst, "invalid"
    with h5py.File(tmp, "w") as out:
        f = out.create_dataset("features", data=feats.astype(np.float16))
        f.attrs["encoder"] = "uni_v2"
        f.attrs["source"] = "MahmoodLab UNI2-h feature release, (1, N, D) float32 -> [N, D] float16"
        c = out.create_dataset("coords", data=coords.astype(np.int64))
        for k, v in attrs.items():
            c.attrs[k] = v
    tmp.rename(dst)
    return dst, "written"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("table")
    p.add_argument("source_root", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    paths = pd.read_csv(a.table)["Feature_Path"]
    jobs = [(a.source_root / rel, a.out_root / rel) for rel in paths]
    counts = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for i, (dst, status) in enumerate(pool.map(convert, jobs, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "invalid":
                print("invalid:", dst, file=sys.stderr)
            if i % 200 == 0:
                print(i, "of", len(jobs), counts, flush=True)
    print("done", counts)


if __name__ == "__main__":
    main()
