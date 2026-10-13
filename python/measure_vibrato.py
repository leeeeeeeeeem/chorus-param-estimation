"""Measure vibrato in the dry audio.

python python/measure_vibrato.py --calibrate   # VocalSet technique folders, incl. excluded ones
python python/measure_vibrato.py               # every segment in the manifest -> data/dry/vibrato.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import read, to_mono
from vibrato import measure

COLS = ["file", "source", "group_id", "split", "voiced_s", "extent_cents", "rate_hz", "periodicity"]


def measure_file(path: Path) -> dict:
    x, sr = read(path)
    return measure(to_mono(x, sr))


def measure_segment(path: Path) -> dict:
    return measure(sf.read(path, dtype="float32")[0])


def summary(ms: list[dict]) -> str:
    voiced = [m for m in ms if m["voiced_s"] > 0]
    if not voiced:
        return "no voiced audio"
    e = np.array([m["extent_cents"] for m in voiced])
    p = np.array([m["periodicity"] for m in voiced])
    return (f"n {len(voiced):4d}  extent p50 {np.median(e):5.1f}  p90 {np.percentile(e, 90):5.1f}  "
            f"periodicity p50 {np.median(p):.2f}  >20c {np.mean(e > 20):4.0%}  >30c {np.mean(e > 30):4.0%}")


def calibrate(data: Path, per_technique: int, pool: ProcessPoolExecutor) -> None:
    root = data / "raw" / "vocalset" / "FULL"
    by_tech = defaultdict(list)
    for p in sorted(root.rglob("*.wav")):
        if not p.name.startswith("."):
            by_tech[p.relative_to(root).parts[2]].append(p)
    rng = random.Random(0)
    picks = {t: rng.sample(fs, min(per_technique, len(fs))) for t, fs in by_tech.items()}
    flat = [(t, f) for t, fs in picks.items() for f in fs]
    measured = list(pool.map(measure_file, [f for _, f in flat]))
    results = defaultdict(list)
    for (t, _), m in zip(flat, measured):
        results[t].append(m)
    order = sorted(results, key=lambda t: np.median([m["extent_cents"] for m in results[t]]))
    print(f"VocalSet techniques, {per_technique} random files each, sorted by median extent:")
    for t in order:
        print(f"  {t:12} {summary(results[t])}")


def segments(data: Path, pool: ProcessPoolExecutor) -> None:
    with open(data / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    measured = list(pool.map(measure_segment, [data / "dry" / r["file"] for r in rows], chunksize=8))
    out = [{k: r[k] for k in COLS[:4]} | m for r, m in zip(rows, measured)]
    with open(data / "dry" / "vibrato.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    by_source = defaultdict(list)
    for r in out:
        by_source[r["source"]].append(r)
    print("per source (voiced segments only):")
    for s, ms in sorted(by_source.items()):
        print(f"  {s:12} {summary(ms)}")
    print("\nstrongest 15 (periodicity > 0.5):")
    strong = sorted((r for r in out if r["periodicity"] > 0.5), key=lambda r: -r["extent_cents"])
    for r in strong[:15]:
        print(f"  {r['extent_cents']:6.1f} c  {r['rate_hz']:4.1f} Hz  p {r['periodicity']:.2f}  {r['file']}")
    print(f"\nwrote {data / 'dry' / 'vibrato.csv'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--per-technique", type=int, default=15)
    args = ap.parse_args(argv)
    with ProcessPoolExecutor() as pool:
        if args.calibrate:
            calibrate(args.data, args.per_technique, pool)
        else:
            segments(args.data, pool)


if __name__ == "__main__":
    main()
