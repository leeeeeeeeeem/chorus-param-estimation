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
# What the stored stereo wet is turned into before the view, to test a model on
# input it was not trained on: "mono" is a mono downmix, "left" one channel
# copied to both. Both leave no side signal.
WET_INPUTS = ("stereo", "mono", "left")
# What a side-dropped item's wet becomes: the mid, or a random blend a*L + (1-a)*R
# on both channels (covers the mid, a=0.5, and one channel alone, a=0 or 1).
DROP_INPUTS = ("mono", "blend")
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
    """(C, samples) -> (C, n_fft // 2 + 1, frames), n_fft = the window length."""
    spec = torch.stft(audio, len(window), HOP, window=window, return_complex=True)
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
    """Returns (features, labels): features (C, n_fft // 2 + 1, 173), labels (4,) in [0, 1].
    C is the wet view's channels, plus 1 for the dry in dry_wet mode."""

    def __init__(self, root: Path, split: str, mode: str = "dry_wet", view: str = "left",
                 wet_input: str = "stereo", n_fft: int = N_FFT, side_drop: float = 0.0,
                 drop_input: str = "mono"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if view not in WET_VIEWS:
            raise ValueError(f"view must be one of {list(WET_VIEWS)}")
        if wet_input not in WET_INPUTS:
            raise ValueError(f"wet_input must be one of {list(WET_INPUTS)}")
        if not 0 <= side_drop <= 1:
            raise ValueError("side_drop must be in 0..1")
        if drop_input not in DROP_INPUTS:
            raise ValueError(f"drop_input must be one of {list(DROP_INPUTS)}")
        self.drop_input = drop_input
        self.root, self.mode, self.view, self.wet_input = Path(root), mode, view, wet_input
        # Training only: with this probability an item's wet is replaced by its mono
        # downmix (side = 0), so one model also learns to cope with mono files.
        self.side_drop = side_drop
        self.channels = WET_VIEWS[view] + (mode == "dry_wet")
        with open(self.root / "params.csv", newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split '{split}' in {self.root / 'params.csv'}")
        self.window = torch.hann_window(n_fft)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        path = self.root / r["split"] / r["id"]
        wet = sf.read(f"{path}_wet.wav", dtype="float32", always_2d=True)[0]
        dropped = bool(self.side_drop) and torch.rand(()).item() < self.side_drop
        if dropped and self.drop_input == "blend":
            a = torch.rand(()).item()
            wet = np.repeat((a * wet[:, 0] + (1 - a) * wet[:, 1])[:, None], 2, axis=1)
        elif self.wet_input == "mono" or dropped:
            wet = np.repeat(wet.mean(axis=1, keepdims=True), 2, axis=1)
        elif self.wet_input == "left":
            wet = np.repeat(wet[:, :1], 2, axis=1)
        audio = wet_view(wet, self.view)
        if self.mode == "dry_wet":
            dry = sf.read(f"{path}_dry.wav", dtype="float32", always_2d=True)[0].mean(axis=1)  # stereo dry: its mid
            audio = np.concatenate([dry[None], audio])
        x = log_stft(torch.from_numpy(np.ascontiguousarray(audio)), self.window)
        y = torch.tensor([scale(n, float(r[n])) for n in LABELS], dtype=torch.float32)
        return x, y
