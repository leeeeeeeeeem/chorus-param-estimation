"""data/raw/<source>/ -> data/dry/, data/splits.json, data/manifest.csv.

python python/prepare_dry.py               # every source that has raw data
python python/prepare_dry.py synth own     # only these
See SPEC.md for the rules.
"""

import argparse
import csv
import json
import random
import shutil
import zlib
from pathlib import Path

import numpy as np

from audio import SR, db, frame_rms_db, read, to_mono, write_int16
from sources import AUDIO_EXTS, SOURCES

FRAME = int(0.05 * SR)
SILENCE_DB = -45.0
MERGE_GAP = 0.3
MIN_SEGMENT = 2.2
TARGET_RMS_DB = -20.0
PEAK_DB = -1.0
VAL_FRAC = TEST_FRAC = 0.15

MANIFEST_COLS = ["file", "source", "group_id", "split", "role", "duration", "licence"]
STATS_COLS = ["raw_file", "source", "raw_peak_db", "raw_clipped", "duration_raw", "duration_kept"]


def find_segments(x: np.ndarray) -> list[tuple[int, int]]:
    """Active runs as (start, end) samples. x should be peak-normalised."""
    active = frame_rms_db(x, FRAME) > SILENCE_DB
    runs, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * FRAME < MERGE_GAP * SR:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(s * FRAME, e * FRAME) for s, e in merged if (e - s) * FRAME >= MIN_SEGMENT * SR]


def normalise(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    gain = min(10 ** ((TARGET_RMS_DB - db(rms)) / 20), 10 ** ((PEAK_DB - db(np.abs(x).max())) / 20))
    return (x * gain).astype(np.float32)


def source_seed(seed: int, source: str) -> int:
    return seed * 1_000_003 + zlib.crc32(source.encode())


def assign_splits(groups: list[str], role: str, seed: int) -> dict[str, str]:
    if role == "heldout":
        return {g: "heldout" for g in groups}
    groups = sorted(groups)
    if len(groups) < 3:
        raise ValueError(f"need at least 3 groups for train/val/test, got {groups}")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * VAL_FRAC))
    n_test = max(1, round(len(groups) * TEST_FRAC))
    split = {g: "val" for g in groups[:n_val]}
    split |= {g: "test" for g in groups[n_val : n_val + n_test]}
    split |= {g: "train" for g in groups[n_val + n_test :]}
    return split


def check_splits(rows: list[dict], splits: dict) -> None:
    """Every (source, group) is in splits.json and in exactly one split."""
    seen = {}
    for r in rows:
        key = (r["source"], r["group_id"])
        expected = splits[r["source"]][r["group_id"]]
        if r["split"] != expected or seen.setdefault(key, r["split"]) != r["split"]:
            raise AssertionError(f"{key} is in more than one split")


def prepare_source(source: str, cfg: dict, data: Path, seed: int) -> tuple[list[dict], list[dict], list[str]]:
    """Writes data/dry/<source>/ and returns (manifest rows without split, stats rows, groups)."""
    root = data / "raw" / source
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
    by_group: dict[str, list[Path]] = {}
    excluded = 0
    for p in files:
        g = cfg["group"](p, root)
        if g is None:
            excluded += 1
        else:
            by_group.setdefault(g, []).append(p)
    if not by_group:
        raise ValueError(f"{source}: no audio found in {root}")

    rng = random.Random(source_seed(seed, source))
    groups = sorted(by_group)
    if "max_groups" in cfg and len(groups) > cfg["max_groups"]:
        groups = sorted(rng.sample(groups, cfg["max_groups"]))
    queues = {g: rng.sample(by_group[g], len(by_group[g])) for g in groups}

    out = data / "dry" / source
    shutil.rmtree(out, ignore_errors=True)
    target = cfg["target_min"] * 60
    rows, stats, kept = [], [], 0.0
    order = rng.sample(groups, len(groups))
    # One file per group per pass until the target is reached.
    while kept < target and any(queues.values()):
        for g in order:
            if kept >= target or not queues[g]:
                continue
            p = queues[g].pop()
            raw, raw_sr = read(p)
            x = to_mono(raw, raw_sr)
            peak = np.abs(x).max()
            segs = find_segments(x / peak) if peak > 0 else []
            rel = p.relative_to(root)
            stem = "__".join(rel.with_suffix("").parts)
            for k, (s, e) in enumerate(segs):
                f = out / g / f"{stem}_{k:02d}.wav"
                write_int16(f, normalise(x[s:e]))
                rows.append(dict(file=str(f.relative_to(data / "dry")), source=source, group_id=g,
                                 role=cfg["role"], duration=round((e - s) / SR, 3), licence=cfg["licence"]))
                kept += (e - s) / SR
            stats.append(dict(raw_file=str(rel), source=source, raw_peak_db=round(db(np.abs(raw).max()), 2),
                              raw_clipped=int((np.abs(raw) >= 0.999).sum()),
                              duration_raw=round(len(raw) / raw_sr, 3),
                              duration_kept=round(sum(e - s for s, e in segs) / SR, 3)))
    print(f"{source}: {len(rows)} segments, {kept / 60:.1f} of {cfg['target_min']} min, "
          f"{len(stats)} files used, {excluded} excluded by parser, {len(groups)} groups")
    if kept < target:
        print(f"{source}: WARNING ran out of audio below the target")
    return rows, stats, groups


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv=None, sources=SOURCES) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    names = args.sources or [s for s in sources if (args.data / "raw" / s).is_dir()]
    unknown = set(names) - set(sources)
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")

    splits_path = args.data / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {}
    manifest = [r for r in read_csv(args.data / "manifest.csv") if r["source"] not in names]
    stats = [r for r in read_csv(args.data / "dry" / "stats.csv") if r["source"] not in names]

    for name in names:
        cfg = sources[name]
        rows, st, groups = prepare_source(name, cfg, args.data, args.seed)
        if name in splits:
            new = set(groups) - set(splits[name])
            if new:
                raise SystemExit(f"{name}: groups {sorted(new)} are not in splits.json. "
                                 f"Remove '{name}' from it to re-split on purpose.")
        else:
            splits[name] = assign_splits(groups, cfg["role"], source_seed(args.seed, name))
        for r in rows:
            r["split"] = splits[name][r["group_id"]]
        manifest += rows
        stats += st
        counts = {}
        for r in rows:
            counts[r["split"]] = counts.get(r["split"], 0) + float(r["duration"]) / 60
        print(f"{name}: minutes per split", {k: round(v, 1) for k, v in sorted(counts.items())})

    check_splits(manifest, splits)
    (args.data / "dry").mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n")
    write_csv(args.data / "manifest.csv", manifest, MANIFEST_COLS)
    write_csv(args.data / "dry" / "stats.csv", stats, STATS_COLS)


if __name__ == "__main__":
    main()
