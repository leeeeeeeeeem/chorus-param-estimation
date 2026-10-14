"""PyTorch dataset over data/dataset/: log-magnitude STFT computed on CPU per item."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODES = ("dry_wet", "wet")
# How the stereo wet clip is shown to the model, and how many channels it takes.
WET_VIEWS = {
    "left": 1,  # left output only
    "lr": 2,  # left, right
    "ms": 2,  # mid (L+R)/2, side (L-R)/2
    "mono": 1,  # mid only: what a mono downmix hears
    "random_mix": 1,  # a*L + (1-a)*R, a uniform 0..1 per item
}
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


def wet_view(wet: np.ndarray, view: str) -> np.ndarray:
    """(samples, 2) left/right -> (channels, samples)."""
    left, right = wet[:, 0], wet[:, 1]
    if view == "left":
        return left[None]
    if view == "lr":
        return np.stack([left, right])
    if view == "ms":
        return np.stack([(left + right) / 2, (left - right) / 2])
    if view == "mono":
        return ((left + right) / 2)[None]
    if view == "random_mix":
        a = torch.rand(()).item()  # torch RNG: seeded per DataLoader worker
        return (a * left + (1 - a) * right)[None]
    raise ValueError(f"view must be one of {list(WET_VIEWS)}")


class ChorusPairs(torch.utils.data.Dataset):
    """Returns (features, labels): features (C, 513, 173), labels (4,) in [0, 1].
    C is the wet view's channels, plus 1 for the dry in dry_wet mode."""

    def __init__(self, root: Path, split: str, mode: str = "dry_wet", view: str = "left"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if view not in WET_VIEWS:
            raise ValueError(f"view must be one of {list(WET_VIEWS)}")
        self.root, self.mode, self.view = Path(root), mode, view
        self.channels = WET_VIEWS[view] + (mode == "dry_wet")
        with open(self.root / "params.csv", newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split '{split}' in {self.root / 'params.csv'}")
        self.window = torch.hann_window(N_FFT)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        path = self.root / r["split"] / r["id"]
        audio = wet_view(sf.read(f"{path}_wet.wav", dtype="float32", always_2d=True)[0], self.view)
        if self.mode == "dry_wet":
            audio = np.concatenate([sf.read(f"{path}_dry.wav", dtype="float32")[0][None], audio])
        x = log_stft(torch.from_numpy(np.ascontiguousarray(audio)), self.window)
        y = torch.tensor([scale(n, float(r[n])) for n in LABELS], dtype=torch.float32)
        return x, y
