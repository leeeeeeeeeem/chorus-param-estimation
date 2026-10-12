"""data/manifest.csv + data/dry/ -> data/dataset/ (pairs, params.csv, log).

python python/make_dataset.py
python python/make_dataset.py --n-heldout 0     # before held-out audio exists
See SPEC.md for the rules.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import CLIP, SR, WARMUP, db, write_int16
from sources import DEMO_EXCLUDED

REPO = Path(__file__).resolve().parents[1]
SPLITS = ["train", "val", "test", "heldout"]
NEED = WARMUP + CLIP
MIN_CROP_RMS_DB = -40.0
MAX_TRIES = 50

COLS = ["id", "split", "source", "group_id", "dry_source", "crop_offset", "gain_db",
        "rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase", "wet_clipped", "demo_ok"]

log = logging.getLogger("make_dataset")


def sample_params(rng: np.random.Generator) -> dict:
    """Rounded before rendering, so the CSV holds exactly what was rendered."""
    p = dict(
        rate_hz=np.exp(rng.uniform(np.log(0.1), np.log(5.0))),
        depth_ms=rng.uniform(0.0, 5.0),
        delay_ms=rng.uniform(7.0, 25.0),
        mix=rng.uniform(0.0, 1.0),
        lfo_phase=rng.uniform(0.0, 1.0),
    )
    p = {k: round(float(v), 6) for k, v in p.items()}
    p["rate_hz"] = min(max(p["rate_hz"], 0.1), 5.0)
    p["lfo_phase"] %= 1.0
    return p


def make_pair(job: dict) -> dict:
    """Crops, renders and writes one pair. Returns its params.csv row."""
    rng = np.random.default_rng(job["seed"])
    segs, weights = job["segments"], job["weights"]
    for _ in range(MAX_TRIES):
        seg = segs[rng.choice(len(segs), p=weights)]
        path = job["dry_root"] / seg["file"]
        offset = int(rng.integers(0, sf.info(path).frames - NEED + 1))
        x, _ = sf.read(path, start=offset, frames=NEED, dtype="float32")
        if db(np.sqrt(np.mean(x[WARMUP:].astype(np.float64) ** 2))) > MIN_CROP_RMS_DB:
            break
    else:
        raise RuntimeError(f"{job['id']}: no crop above {MIN_CROP_RMS_DB} dBFS in {MAX_TRIES} tries")

    gain_db = round(float(rng.uniform(-12.0, -1.0)), 6)
    x = x * 10 ** ((gain_db - db(np.abs(x).max())) / 20)
    params = sample_params(rng)

    out = job["out"] / job["split"]
    dry_path, wet_path = out / f"{job['id']}_dry.wav", out / f"{job['id']}_wet.wav"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_int16(tmp / "in.wav", x)
        (tmp / "p.json").write_text(json.dumps(params | {"warmup_samples": WARMUP}))
        r = subprocess.run([job["render"], tmp / "in.wav", tmp / "p.json", wet_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{job['id']}: render failed: {r.stderr.strip()}")
    write_int16(dry_path, x[WARMUP:])

    wet = sf.read(wet_path, dtype="int16")[0]
    clipped = int(((wet == 32767) | (wet == -32768)).sum())
    return dict(id=job["id"], split=job["split"], source=seg["source"], group_id=seg["group_id"],
                dry_source=seg["file"], crop_offset=offset, gain_db=gain_db,
                **params,
                wet_clipped=clipped, demo_ok=seg["source"] not in DEMO_EXCLUDED)


def build_render() -> Path:
    subprocess.run(["cargo", "build", "--release", "--quiet", "--bin", "render"], cwd=REPO, check=True)
    return REPO / "target" / "release" / "render"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/dataset")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-heldout", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or args.data / "dataset"
    counts = dict(zip(SPLITS, [args.n_train, args.n_val, args.n_test, args.n_heldout]))

    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} is not empty, pass --overwrite to replace it")
        shutil.rmtree(out)
    for s in SPLITS:
        (out / s).mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(out / "make_dataset.log"), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)

    with open(args.data / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    render = build_render()
    log.info("args: %s", vars(args))

    jobs = []
    for split_idx, split in enumerate(SPLITS):
        segs = [r for r in manifest if r["split"] == split and float(r["duration"]) * SR >= NEED]
        if counts[split] and not segs:
            raise SystemExit(f"no dry audio for split '{split}'; pass --n-{split} 0 to skip it")
        if not counts[split]:
            continue
        w = np.array([float(r["duration"]) for r in segs])
        log.info("%s: %d pairs from %d segments, %.1f min", split, counts[split], len(segs), w.sum() / 60)
        for i in range(counts[split]):
            jobs.append(dict(id=f"{split}_{i:06d}", split=split, seed=[args.seed, split_idx, i],
                             segments=segs, weights=w / w.sum(), dry_root=args.data / "dry",
                             out=out, render=render))

    rows, start = [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for n, row in enumerate(pool.map(make_pair, jobs), 1):
            rows.append(row)
            if n % 500 == 0 or n == len(jobs):
                rate = n / (time.time() - start)
                log.info("%d/%d pairs, %.0f/s, eta %.0f s", n, len(jobs), rate, (len(jobs) - n) / rate)

    with open(out / "params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    clipped = [r for r in rows if r["wet_clipped"]]
    log.info("done: %d pairs, %d with clipped wet samples", len(rows), len(clipped))


if __name__ == "__main__":
    main()
