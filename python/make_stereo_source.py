"""data/dataset/ val pairs -> data/dataset_stereo/: the same pairs with a stereo dry.

python python/make_stereo_source.py
python python/make_stereo_source.py --split val --out data/dataset_stereo

Each pair keeps its original crop, gain and chorus settings; only the plugin's
input changes. Kinds, in turn per pair:
  mono  x on both inputs (the original pair, bit for bit)
  pan   cos(t)*x, sin(t)*x: 1 to 9 dB level difference, louder side random
  corr  x, rho*x + sqrt(1-rho^2)*y: rho 0.3 to 0.9
  wide  x, y
y is a second crop from the same group, level-matched to x. There is never a
time offset between L and R: one would look like a chorus delay.
"""

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import CLIP, WARMUP, db, write_int16
from make_dataset import COLS, NEED, build_render, load_crop, render_pair

KINDS = ("mono", "pan", "corr", "wide")
STEREO_COLS = [*COLS, "stereo_kind", "stereo_amount", "second_source", "second_offset"]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x[WARMUP : WARMUP + CLIP].astype(np.float64) ** 2)))


def second_crop(row: dict, segments: list[dict], data: Path, rng: np.random.Generator) -> tuple[str, int, np.ndarray]:
    """A crop from the same group that does not overlap the original crop."""
    own = int(row["crop_offset"])
    for _ in range(50):
        seg = segments[rng.integers(len(segments))]
        frames = sf.info(data / "dry" / seg["file"]).frames
        if frames < NEED:
            continue
        offset = int(rng.integers(0, frames - NEED + 1))
        if seg["file"] == row["dry_source"] and abs(offset - own) < NEED:
            continue
        y = sf.read(data / "dry" / seg["file"], start=offset, frames=NEED, dtype="float32")[0]
        if rms(y) > 0:
            return seg["file"], offset, y
    raise RuntimeError(f"{row['id']}: no second crop in group {row['group_id']}")


def stereo_dry(kind: str, x: np.ndarray, y: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """(NEED, 2) dry and the kind's amount (dB for pan, rho for corr)."""
    if kind == "pan":
        diff_db = float(rng.uniform(1.0, 9.0)) * rng.choice([-1, 1])
        t = np.arctan(10 ** (-diff_db / 20))  # cos/sin ratio gives the level difference
        return np.stack([np.cos(t) * x, np.sin(t) * x], axis=1), round(diff_db, 3)
    y = y * (rms(x) / rms(y))
    if kind == "corr":
        rho = float(rng.uniform(0.3, 0.9))
        return np.stack([x, rho * x + np.sqrt(1 - rho**2) * y], axis=1), round(rho, 3)
    if kind == "wide":
        return np.stack([x, y], axis=1), 0.0
    raise ValueError(kind)


def make(job: dict) -> dict:
    row, i = job["row"], job["index"]
    kind = KINDS[i % len(KINDS)]
    params = {k: float(row[k]) for k in ("rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase")}
    x = load_crop(job["data"] / "dry" / row["dry_source"], int(row["crop_offset"]), float(row["gain_db"]))
    extra = dict(stereo_kind=kind, stereo_amount=0.0, second_source="", second_offset="")
    if kind == "mono":
        dry_in = x
    else:
        rng = np.random.default_rng([job["seed"], i])
        y = None
        if kind != "pan":
            src, off, y = second_crop(row, job["groups"][(row["split"], row["group_id"])], job["data"], rng)
            extra.update(second_source=src, second_offset=off)
        dry_in, amount = stereo_dry(kind, x, y, rng)
        dry_in = dry_in * 10 ** ((float(row["gain_db"]) - db(np.abs(dry_in).max())) / 20)  # same peak as the original
        extra["stereo_amount"] = amount
    dry, wet = render_pair(dry_in, params, job["render"])
    out = job["out"] / row["split"]
    write_int16(out / f"{row['id']}_dry.wav", dry)
    write_int16(out / f"{row['id']}_wet.wav", wet)
    scaled = np.round(wet * 32768)
    return {**row, "wet_clipped": int(((scaled > 32767) | (scaled < -32768)).sum()), **extra}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--dataset", type=Path, default=None, help="default: <data>/dataset")
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/dataset_stereo")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    src = args.dataset or args.data / "dataset"
    out = args.out or args.data / "dataset_stereo"
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} is not empty, pass --overwrite to replace it")
        shutil.rmtree(out)
    (out / args.split).mkdir(parents=True)

    with open(src / "params.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    with open(args.data / "manifest.csv", newline="") as f:
        groups = {}
        for r in csv.DictReader(f):
            groups.setdefault((r["split"], r["group_id"]), []).append(r)
    render = build_render()
    jobs = [dict(row=r, index=i, seed=args.seed, groups=groups, data=args.data, out=out, render=render)
            for i, r in enumerate(rows)]
    with ThreadPoolExecutor(args.workers) as pool:
        made = list(pool.map(make, jobs))
    with open(out / "params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEREO_COLS)
        w.writeheader()
        w.writerows(made)
    counts = {k: sum(r["stereo_kind"] == k for r in made) for k in KINDS}
    print(f"wrote {len(made)} pairs to {out}: {counts}, {sum(r['wet_clipped'] > 0 for r in made)} with clipped wet")


if __name__ == "__main__":
    main()
