"""Audio helpers shared by the dataset scripts."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 22050
CLIP = 2 * SR
WARMUP = 2048


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def read(path: Path) -> tuple[np.ndarray, int]:
    """Returns (frames, channels) float32 and the sample rate."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x, sr


def to_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Downmix and resample to SR."""
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def frame_rms_db(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    frames = x[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-12))


def write_int16(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    sf.write(path, s, SR, subtype="PCM_16")
