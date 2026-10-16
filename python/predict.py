"""Score a trained run on another dataset, e.g. the stereo-source pairs.

python python/predict.py wet_ms_e30 --dataset data/dataset_stereo --split val
Loads runs/<name>/best.pt with the run's own settings (mode, view, width,
n_fft) and writes results/<name>_<suffix>.csv, then scores it with
evaluate.py against that dataset's params.csv.
"""

import argparse
import csv
from pathlib import Path

import torch

import evaluate
from chorus_data import LABELS, N_FFT, ChorusPairs
from train import ChorusCNN, pick_device, predict


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--suffix", default="stereosrc")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    run = args.runs / args.name
    s = torch.load(run / "last.pt", map_location="cpu", weights_only=False)["settings"]
    n_fft = int(s.get("n_fft", N_FFT))
    ds = ChorusPairs(args.dataset, args.split, s["mode"], s["view"], n_fft=n_fft)
    device = pick_device(args.device)
    model = ChorusCNN(ds.channels, int(s["width"])).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    preds = predict(model, ds, device, args.batch_size, args.workers)

    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{args.name}_{args.suffix}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", *LABELS])
        w.writeheader()
        w.writerows(preds)
    print(f"wrote {len(preds)} predictions to {out}")
    evaluate.main([str(out), "--dataset", str(args.dataset)])


if __name__ == "__main__":
    main()
