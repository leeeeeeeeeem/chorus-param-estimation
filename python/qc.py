"""Dump random dry excerpts per source to data/qc/ for listening, and print
clipping and silence stats.

python python/qc.py
Listen for reverb and baked-in modulation (vibrato, chorus, tremolo).
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import SR, frame_rms_db, write_int16
from prepare_dry import FRAME, SILENCE_DB


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--per-source", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = read_csv(args.data / "manifest.csv")
    by_source = defaultdict(list)
    for r in manifest:
        by_source[r["source"]].append(r)

    qc = args.data / "qc"
    shutil.rmtree(qc, ignore_errors=True)
    rng = random.Random(args.seed)
    for source, rows in sorted(by_source.items()):
        for r in rng.sample(rows, min(args.per_source, len(rows))):
            x = sf.read(args.data / "dry" / r["file"], dtype="float32")[0]
            n = int(args.seconds * SR)
            start = rng.randint(0, max(0, len(x) - n))
            write_int16(qc / source / f"{r['group_id']}__{Path(r['file']).name}", x[start : start + n])
    print(f"wrote excerpts to {qc}/\n")

    stats = defaultdict(list)
    for s in read_csv(args.data / "dry" / "stats.csv"):
        stats[s["source"]].append(s)
    print(f"{'source':12} {'files':>6} {'clipped':>8} {'worst':>7} {'raw h':>6} {'kept h':>6} "
          f"{'dropped':>8} {'silent':>7}  minutes per split")
    for source, rows in sorted(by_source.items()):
        st = stats[source]
        clipped = [int(s["raw_clipped"]) for s in st]
        raw_h = sum(float(s["duration_raw"]) for s in st) / 3600
        kept_h = sum(float(s["duration_kept"]) for s in st) / 3600
        silent = total = 0
        for r in rows:
            f = frame_rms_db(sf.read(args.data / "dry" / r["file"], dtype="float32")[0], FRAME)
            silent += int((f < SILENCE_DB).sum())
            total += len(f)
        per_split = defaultdict(float)
        for r in rows:
            per_split[r["split"]] += float(r["duration"]) / 60
        print(f"{source:12} {len(st):6d} {sum(c > 0 for c in clipped):8d} {max(clipped, default=0):7d} "
              f"{raw_h:6.2f} {kept_h:6.2f} {1 - kept_h / max(raw_h, 1e-9):8.0%} {silent / max(total, 1):7.1%}  "
              + ", ".join(f"{k} {v:.1f}" for k, v in sorted(per_split.items())))
    print("\nclipped: raw files with samples at full scale; worst: most in one file;")
    print("dropped: raw time removed as silence; silent: 50 ms frames under "
          f"{SILENCE_DB:.0f} dBFS left inside kept segments.")

    params = args.data / "dataset" / "params.csv"
    if params.exists():
        rows = read_csv(params)
        bad = [r for r in rows if int(r["wet_clipped"])]
        print(f"\ndataset: {len(rows)} pairs, {len(bad)} with clipped wet samples")
        counts = defaultdict(int)
        for r in rows:
            counts[(r["split"], r["source"])] += 1
        for (split, source), n in sorted(counts.items()):
            print(f"  {split:8} {source:12} {n:6d}")


if __name__ == "__main__":
    main()
