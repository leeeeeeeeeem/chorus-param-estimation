"""Score predictions against data/dataset/params.csv.

python python/evaluate.py results/mean.csv
A predictions CSV has columns id, rate_hz, depth_ms, delay_ms, mix (real units).
Predictions are clipped to the label ranges first.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from chorus_data import LABELS, scale

# Error columns: rate in octaves (rate is a log knob), the rest in their units,
# and the mean error over all four labels on the 0..1 training scale.
ERRORS = ["rate_oct", "depth_ms", "delay_ms", "mix", "scaled"]
# Where some labels cannot be heard (see SPEC.md).
CONDITIONS = {
    "mix < 0.1": lambda r: float(r["mix"]) < 0.1,
    "mix > 0.9": lambda r: float(r["mix"]) > 0.9,  # wet mode: no dry left, delay is a pure time shift
    "depth < 0.5 ms": lambda r: float(r["depth_ms"]) < 0.5,
    "rate < 0.3 Hz": lambda r: float(r["rate_hz"]) < 0.3,
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def clip(name: str, v: float) -> float:
    lo, hi, _ = LABELS[name]
    return min(max(v, lo), hi)


def errors(pred: dict, true: dict) -> dict:
    p = {n: clip(n, float(pred[n])) for n in LABELS}
    t = {n: float(true[n]) for n in LABELS}
    return dict(
        rate_oct=abs(np.log2(p["rate_hz"] / t["rate_hz"])),
        depth_ms=abs(p["depth_ms"] - t["depth_ms"]),
        delay_ms=abs(p["delay_ms"] - t["delay_ms"]),
        mix=abs(p["mix"] - t["mix"]),
        scaled=float(np.mean([abs(scale(n, p[n]) - scale(n, t[n])) for n in LABELS])),
    )


def summarise(preds: list[dict], truth: list[dict]) -> list[dict]:
    """Mean absolute errors per split, overall, per source and per condition."""
    by_id = {r["id"]: r for r in truth}
    missing = [p["id"] for p in preds if p["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} predicted ids are not in params.csv, e.g. {missing[0]}")
    groups = defaultdict(list)
    for p in preds:
        t = by_id[p["id"]]
        e = errors(p, t)
        groups[(t["split"], "all")].append(e)
        groups[(t["split"], f"source {t['source']}")].append(e)
        for name, test in CONDITIONS.items():
            if test(t):
                groups[(t["split"], name)].append(e)
    order = {"train": 0, "val": 1, "test": 2, "heldout": 3}
    rows = []
    for (split, group), es in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1] != "all", kv[0][1])):
        rows.append(dict(split=split, group=group, n=len(es), **{k: float(np.mean([e[k] for e in es])) for k in ERRORS}))
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'split':8} {'group':22} {'n':>5}  {'rate oct':>8} {'depth ms':>8} {'delay ms':>8} {'mix':>6} {'scaled':>7}")
    for r in rows:
        print(f"{r['split']:8} {r['group']:22} {r['n']:5d}  {r['rate_oct']:8.3f} {r['depth_ms']:8.3f} "
              f"{r['delay_ms']:8.3f} {r['mix']:6.3f} {r['scaled']:7.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    args = ap.parse_args(argv)
    rows = summarise(read_csv(args.predictions), read_csv(args.dataset / "params.csv"))
    print_table(rows)
    out = args.predictions.with_name(args.predictions.stem + "_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "group", "n", *ERRORS])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
