"""Train a CNN on the log-spectrograms and write predictions for evaluate.py.

python python/train.py --mode wet --view left
python python/train.py --mode wet --view left --resume    # continue a stopped run
Picks the epoch with the lowest val error, then predicts val, test and heldout.
Writes runs/<name>/ (last.pt, best.pt, log.csv) and results/<name>.csv, plus
results/<name>_monoin.csv and results/<name>_leftin.csv: the same model fed a
mono downmix, or the left channel on both sides, instead of the stereo wet.
A full checkpoint is saved after every epoch, so a run can be stopped at any
time (Ctrl-C, shutdown) and resumed; at most the current epoch is lost.
"""

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chorus_data import LABELS, MODES, N_FFT, WET_VIEWS, ChorusPairs, unscale

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


def loader(ds, batch_size: int, workers: int, shuffle: bool, generator: torch.Generator | None = None):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       persistent_workers=workers > 0, generator=generator)


# Settings that must match to resume a run.
RUN_KEYS = ("mode", "view", "dataset", "epochs", "batch_size", "lr", "width", "max_train", "seed", "n_fft")


def save(obj, path: Path) -> None:
    """Write then rename, so a kill during saving never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict, device: torch.device) -> None:
    # torch.load(map_location=device) moves these to the device; both setters need CPU tensors.
    torch.set_rng_state(state["cpu"].cpu())
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def write_log(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "seconds"])
        w.writeheader()
        w.writerows(history)


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
    ap.add_argument("--n-fft", type=int, default=N_FFT, help="STFT window; 2048 shows long-delay notches better")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/ from its last epoch")
    ap.add_argument("--fresh", action="store_true", help="delete runs/<name>/ and start over")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this epoch (resume later)")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    name = args.name or f"{args.mode}_{args.view}"
    run_dir = args.runs / name
    settings = {k: str(getattr(args, k)) for k in RUN_KEYS}
    if args.fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
    if (run_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{run_dir} already has a run: add --resume to continue it, or --fresh to start over")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_ds = ChorusPairs(args.dataset, "train", args.mode, args.view, n_fft=args.n_fft)
    if args.max_train:
        train_ds.rows = train_ds.rows[: args.max_train]
    val_ds = ChorusPairs(args.dataset, "val", args.mode, args.view, n_fft=args.n_fft)
    shuffle_rng = torch.Generator()
    train_dl = loader(train_ds, args.batch_size, args.workers, shuffle=True, generator=shuffle_rng)
    val_dl = loader(val_ds, args.batch_size, args.workers, shuffle=False)

    model = ChorusCNN(train_ds.channels, args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {len(train_ds)} train, {len(val_ds)} val pairs, {train_ds.channels} channels, "
          f"{n_params / 1e6:.2f} M parameters, on {device}")

    best, history, first = float("inf"), [], 1
    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        saved = {"n_fft": str(N_FFT), **ckpt["settings"]}  # runs from before --n-fft
        if saved != settings:
            changed = {k: (saved[k], v) for k, v in settings.items() if saved[k] != v}
            raise SystemExit(f"cannot resume {run_dir}: settings changed {changed}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        set_rng_state(ckpt["rng"], device)
        best, history, first = ckpt["best"], ckpt["history"], ckpt["epoch"] + 1
        print(f"resuming after epoch {ckpt['epoch']} (best val mae {best:.4f})")

    for epoch in range(first, args.epochs + 1):
        if args.stop_after is not None and epoch > args.stop_after:
            print(f"stopped after epoch {epoch - 1}; continue with --resume")
            return
        start = time.time()
        shuffle_rng.manual_seed(args.seed * 1000 + epoch)  # same order whether resumed or not
        train_loss, train_mae = run_epoch(model, train_dl, device, opt)
        val_loss, val_mae = run_epoch(model, val_dl, device)
        sched.step()
        seconds = time.time() - start
        history.append(dict(epoch=epoch, train_loss=f"{train_loss:.5f}", train_mae=f"{train_mae:.4f}",
                            val_loss=f"{val_loss:.5f}", val_mae=f"{val_mae:.4f}", seconds=f"{seconds:.1f}"))
        mark = ""
        if val_mae < best:
            best, mark = val_mae, " *"
            save(model.state_dict(), run_dir / "best.pt")
        save(dict(settings=settings, epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                  sched=sched.state_dict(), rng=rng_state(device), best=best, history=history),
             run_dir / "last.pt")
        write_log(run_dir / "log.csv", history)
        print(f"epoch {epoch:3d}  train {train_loss:.4f} / {train_mae:.3f}  val {val_loss:.4f} / "
              f"{val_mae:.3f}  {seconds:5.1f} s{mark}")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    args.results.mkdir(parents=True, exist_ok=True)
    print(f"best val mae {best:.4f}")
    for wet_input, suffix in (("stereo", ""), ("mono", "_monoin"), ("left", "_leftin")):
        preds = []
        for split in PRED_SPLITS:
            preds += predict(model, ChorusPairs(args.dataset, split, args.mode, args.view, wet_input, args.n_fft), device,
                             args.batch_size, args.workers)
        out = args.results / f"{name}{suffix}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", *LABELS])
            w.writeheader()
            w.writerows(preds)
        print(f"wrote {len(preds)} predictions ({wet_input} wet input) to {out}")


if __name__ == "__main__":
    main()
