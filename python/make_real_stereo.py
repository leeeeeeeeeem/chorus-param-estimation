"""GuitarSet pickup + microphone, recorded at the same time -> data/dataset_realstereo/.

python python/make_real_stereo.py                 # the val player, 500 crops
python python/make_real_stereo.py --split test    # only for final numbers
python python/make_real_stereo.py --split train --kinds mic --n 4000 \
    --append-to data/dataset_mixed2                # real stereo for training

Each crop is rendered twice with the same gain and chorus settings:
  mono  pickup on both plugin inputs, as in training
  mic   pickup on the left input, microphone on the right: a real two-signal
        stereo source, with its own room sound and acoustic delay
The microphone is not time-aligned to the pickup; its lag is part of what real
stereo looks like. The measured lag (ms, mic after pickup) is recorded in
stereo_amount.
"""

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from audio import CLIP, SR, WARMUP, db, read, to_mono, write_int16
from make_dataset import NEED, RENDER_SR, build_render, render_pair, sample_params
from make_stereo_source import STEREO_COLS
from prepare_dry import find_segments
from sources import guitarset_group

MAX_LAG = int(0.025 * SR)


def mic_name(pickup: Path) -> str:
    return pickup.name.replace("_mix.wav", "_mic.wav")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x[WARMUP : WARMUP + CLIP].astype(np.float64) ** 2)))


def lag_samples(a: np.ndarray, b: np.ndarray) -> int:
    """Lag of b behind a, within +-25 ms, by GCC-PHAT: the cross-spectrum is
    whitened first, so the peaks of a pitched sound (one per pitch period) do not
    outshine the true lag, as they do in a plain cross-correlation."""
    n = min(len(a), len(b), 30 * SR)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    size = 1 << (2 * n - 1).bit_length()
    x = np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size))
    c = np.fft.irfft(x / (np.abs(x) + 1e-12), size)
    lags = np.r_[np.arange(0, MAX_LAG + 1), np.arange(-MAX_LAG, 0)]
    vals = np.r_[c[: MAX_LAG + 1], c[-MAX_LAG:]]
    return int(lags[np.argmax(np.abs(vals))])


def load_take(pickup: Path, mic_dir: Path) -> dict:
    p = to_mono(*read(pickup))
    m = to_mono(*read(mic_dir / mic_name(pickup)))
    n = min(len(p), len(m))
    p, m = p[:n], m[:n]
    runs = [(s, e) for s, e in find_segments(p / max(np.abs(p).max(), 1e-9)) if e - s >= NEED]
    return dict(pickup=p, mic=m, runs=runs, lag=lag_samples(p, m), name=pickup.name)


def make(job: dict) -> list[dict]:
    take, i = job["take"], job["index"]
    rng = np.random.default_rng([job["seed"], i])
    s, e = take["runs"][rng.integers(len(take["runs"]))]
    offset = int(rng.integers(s, e - NEED + 1))
    p = take["pickup"][offset : offset + NEED]
    m = take["mic"][offset : offset + NEED]
    m = m * (rms(p) / max(rms(m), 1e-12))
    gain_db = round(float(rng.uniform(-12.0, -1.0)), 6)
    scale = 10 ** ((gain_db - db(max(np.abs(p).max(), np.abs(m).max()))) / 20)
    params = sample_params(rng)
    rows = []
    inputs = {"mono": p * scale, "mic": np.stack([p * scale, m * scale], axis=1)}
    for kind in job["kinds"]:
        dry_in = inputs[kind]
        pid = f"{job['split']}_{i:06d}_{kind}"
        dry, wet = render_pair(dry_in.astype(np.float32), params, job["render"])
        write_int16(job["out"] / job["split"] / f"{pid}_dry.wav", dry)
        write_int16(job["out"] / job["split"] / f"{pid}_wet.wav", wet)
        scaled = np.round(wet * 32768)
        rows.append(dict(id=pid, split=job["split"], source="guitarset", group_id=job["group"],
                         dry_source=f"guitarset/{take['name']}", crop_offset=offset, gain_db=gain_db, **params,
                         render_sr=RENDER_SR, wet_clipped=int(((scaled > 32767) | (scaled < -32768)).sum()),
                         demo_ok=True, stereo_kind=kind,
                         stereo_amount=round(1000 * take["lag"] / SR, 3) if kind == "mic" else 0.0,
                         second_source=f"guitarset_mic/{mic_name(Path(take['name']))}" if kind == "mic" else "",
                         second_offset=offset if kind == "mic" else ""))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/dataset_realstereo")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=500, help="crops; each is rendered as mono and as mic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--kinds", default="mono,mic", help="which renders to make per crop")
    ap.add_argument("--append-to", type=Path, default=None,
                    help="add the pairs to this existing dataset (files and params.csv) instead")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    kinds = args.kinds.split(",")
    if set(kinds) - {"mono", "mic"}:
        raise SystemExit("--kinds takes mono and/or mic")
    if args.append_to:
        out = args.append_to
        if not (out / "params.csv").exists():
            raise SystemExit(f"{out} has no params.csv to append to")
        (out / args.split).mkdir(exist_ok=True)
    else:
        out = args.out or args.data / "dataset_realstereo"
        if out.exists() and any(out.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"{out} is not empty, pass --overwrite to replace it")
            shutil.rmtree(out)
        (out / args.split).mkdir(parents=True)

    raw, mic_dir = args.data / "raw" / "guitarset", args.data / "raw" / "guitarset_mic"
    groups = json.loads((args.data / "splits.json").read_text())["guitarset"]
    pickups = sorted(p for p in raw.glob("*_mix.wav") if groups.get(guitarset_group(p, raw)) == args.split)
    if not pickups:
        raise SystemExit(f"no GuitarSet takes for split '{args.split}'")
    missing = [p.name for p in pickups if not (mic_dir / mic_name(p)).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} microphone files missing in {mic_dir}, e.g. {missing[0]}")
    with ThreadPoolExecutor(args.workers) as pool:
        takes = [t for t in pool.map(lambda p: load_take(p, mic_dir), pickups) if t["runs"]]
    lags = [1000 * t["lag"] / SR for t in takes]
    print(f"{len(takes)} takes; mic lag behind pickup: median {np.median(lags):.2f} ms, "
          f"range {min(lags):.2f} to {max(lags):.2f} ms")

    rng = np.random.default_rng(args.seed)
    weights = np.array([sum(e - s for s, e in t["runs"]) for t in takes], dtype=float)
    chosen = rng.choice(len(takes), size=args.n, p=weights / weights.sum())
    render = build_render()
    jobs = [dict(take=takes[t], index=i, seed=args.seed, split=args.split, out=out, render=render, kinds=kinds,
                 group=guitarset_group(raw / takes[t]["name"], raw)) for i, t in enumerate(chosen)]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for pair in pool.map(make, jobs) for r in pair]
    if args.append_to:
        with open(out / "params.csv", newline="") as f:
            existing = {r["id"] for r in csv.DictReader(f)}
        if clash := existing & {r["id"] for r in rows}:
            raise SystemExit(f"{len(clash)} ids already in {out}, e.g. {sorted(clash)[0]}")
        with open(out / "params.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=STEREO_COLS).writerows(rows)
    else:
        with open(out / "params.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STEREO_COLS)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(rows)} pairs to {out}, {sum(r['wet_clipped'] > 0 for r in rows)} with clipped wet")


if __name__ == "__main__":
    main()
