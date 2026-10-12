"""PyTorch dataset over data/dataset/: log-magnitude STFT computed on CPU per item."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODES = {"dry_wet": ("dry", "wet"), "wet": ("wet",)}
N_FFT = 1024
HOP = 256

# name: (low, high, log scale)
LABELS = {
    "rate_hz": (0.1, 5.0, True),
    "depth_ms": (0.0, 5.0, False),
    "delay_ms": (7.0, 25.0, False),
    "mix": (0.0, 1.0, False),
}


def scale(name: str, v: float) -> float:
    lo, hi, log = LABELS[name]
    if log:
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (v - lo) / (hi - lo)


def unscale(name: str, u: float) -> float:
    lo, hi, log = LABELS[name]
    if log:
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return lo + u * (hi - lo)


def log_stft(audio: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """(C, samples) -> (C, N_FFT // 2 + 1, frames)."""
    spec = torch.stft(audio, N_FFT, HOP, window=window, return_complex=True)
    return torch.log(spec.abs() + 1e-5)


class ChorusPairs(torch.utils.data.Dataset):
    """Returns (features, labels): features (C, 513, 173), labels (4,) in [0, 1]."""

    def __init__(self, root: Path, split: str, mode: str = "dry_wet"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.root, self.kinds = Path(root), MODES[mode]
        with open(self.root / "params.csv", newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split '{split}' in {self.root / 'params.csv'}")
        self.window = torch.hann_window(N_FFT)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        audio = np.stack([sf.read(self.root / r["split"] / f"{r['id']}_{k}.wav", dtype="float32")[0]
                          for k in self.kinds])
        x = log_stft(torch.from_numpy(audio), self.window)
        y = torch.tensor([scale(n, float(r[n])) for n in LABELS], dtype=torch.float32)
        return x, y
