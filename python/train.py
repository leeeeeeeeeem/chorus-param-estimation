"""Train a CNN on the log-spectrograms and write predictions for evaluate.py.

python python/train.py --mode wet --view left
python python/train.py --mode dry_wet --view left --epochs 30
Picks the epoch with the lowest val error, then predicts val, test and heldout.
Writes runs/<name>/ (best.pt, log.csv) and results/<name>.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chorus_data import LABELS, MODES, WET_VIEWS, ChorusPairs, unscale

PRED_SPLITS = ("val", "test", "heldout")


class ChorusCNN(nn.Module):
    """Four conv blocks (each halves both axes), then pooling to a small
    frequency x time grid, so frequency position (comb notches) and slow
    changes over time (the LFO) both survive, then two dense layers."""

    def __init__(self, channels: int, width: int = 16):
        super().__init__()
        layers = [nn.BatchNorm2d(channels)]  # learns the input scale
        c = channels
        for out in (width, 2 * width, 4 * width, 8 * width):
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(),
                       nn.Conv2d(out, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
            c = out
        # 513 x 173 -> 32 x 10 after the blocks -> 4 x 5. MPS only pools to
        # sizes that divide the input evenly.
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 5)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 20, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, len(LABELS)), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x))


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def loader(ds, batch_size: int, workers: int, shuffle: bool, seed: int = 0):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       persistent_workers=workers > 0,
                                       generator=torch.Generator().manual_seed(seed))


def run_epoch(model, data, device, opt=None) -> tuple[float, float]:
    """(MSE loss, mean absolute error on the 0..1 scale). Trains if opt is given."""
    model.train(opt is not None)
    total_loss = total_abs = n = 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, y in data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            total_abs += (pred - y).abs().mean(dim=1).sum().item()
            n += len(y)
    return total_loss / n, total_abs / n


def predict(model, ds, device, batch_size: int, workers: int) -> list[dict]:
    model.eval()
    out = []
    with torch.no_grad():
        for start, (x, _) in zip(range(0, len(ds), batch_size), loader(ds, batch_size, workers, shuffle=False)):
            for row, p in zip(ds.rows[start : start + len(x)], model(x.to(device)).cpu().numpy()):
                out.append({"id": row["id"], **{n: round(unscale(n, float(v)), 6) for n, v in zip(LABELS, p)}})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="wet")
    ap.add_argument("--view", choices=list(WET_VIEWS), default="left")
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    ap.add_argument("--name", default=None, help="default: <mode>_<view>")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    name = args.name or f"{args.mode}_{args.view}"
    run_dir = args.runs / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_ds = ChorusPairs(args.dataset, "train", args.mode, args.view)
    if args.max_train:
        train_ds.rows = train_ds.rows[: args.max_train]
    val_ds = ChorusPairs(args.dataset, "val", args.mode, args.view)
    train_dl = loader(train_ds, args.batch_size, args.workers, shuffle=True, seed=args.seed)
    val_dl = loader(val_ds, args.batch_size, args.workers, shuffle=False)

    model = ChorusCNN(train_ds.channels, args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {len(train_ds)} train, {len(val_ds)} val pairs, {train_ds.channels} channels, "
          f"{n_params / 1e6:.2f} M parameters, on {device}")

    best = float("inf")
    with open(run_dir / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_mae = run_epoch(model, train_dl, device, opt)
            val_loss, val_mae = run_epoch(model, val_dl, device)
            sched.step()
            seconds = time.time() - start
            log.writerow([epoch, f"{train_loss:.5f}", f"{train_mae:.4f}", f"{val_loss:.5f}", f"{val_mae:.4f}",
                          f"{seconds:.1f}"])
            f.flush()
            mark = ""
            if val_mae < best:
                best, mark = val_mae, " *"
                torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"epoch {epoch:3d}  train {train_loss:.4f} / {train_mae:.3f}  val {val_loss:.4f} / "
                  f"{val_mae:.3f}  {seconds:5.1f} s{mark}")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    preds = []
    for split in PRED_SPLITS:
        preds += predict(model, ChorusPairs(args.dataset, split, args.mode, args.view), device,
                         args.batch_size, args.workers)
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", *LABELS])
        w.writeheader()
        w.writerows(preds)
    print(f"best val mae {best:.4f}; wrote {len(preds)} predictions to {out}")


if __name__ == "__main__":
    main()
