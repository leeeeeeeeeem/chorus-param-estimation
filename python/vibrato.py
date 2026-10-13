"""Vibrato measurement: how much periodic pitch wobble a dry segment already has.

Pitch is tracked with pYIN. In each voiced run, a median filter removes note
changes and glides (it keeps steps but flattens oscillations shorter than its
window). What is left is searched for a periodic wobble at 3.5 to 9 Hz.

Extent is approximate: about 0.75 to 1.1 of the true value for 4 to 8 Hz
wobbles (the pitch window and the median filter both shrink it a little).
A linear filter would be flatter, but it lets fast scales through as vibrato.
"""

import librosa
import numpy as np
from scipy.ndimage import median_filter

from audio import SR

HOP = 256
FPS = SR / HOP  # pitch frames per second, about 86
MIN_RUN = 0.6  # seconds of continuous voicing needed
TREND = 59  # median filter length in frames, about 0.68 s
BAND = (3.5, 9.0)  # vibrato rates searched, Hz
NFFT = 1024


def pitch_cents(x: np.ndarray) -> np.ndarray:
    """Pitch in cents per frame (re 1 Hz), NaN where unvoiced."""
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=1100, sr=SR, frame_length=1024, hop_length=HOP)
    f0[~voiced] = np.nan
    return 1200 * np.log2(f0)


def voiced_runs(c: np.ndarray) -> list[np.ndarray]:
    ok = np.isfinite(c)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    return [c[s:e] for s, e in zip(edges[::2], edges[1::2]) if e - s >= MIN_RUN * FPS]


def run_wobble(c: np.ndarray) -> tuple[float, float, float]:
    """(extent in cents, rate in Hz, periodicity 0..1) of one voiced run.
    Extent is the wobble amplitude (half of peak to peak)."""
    res = c - median_filter(c, TREND, mode="nearest")
    res -= res.mean()
    w = np.hanning(len(res))
    spec = np.abs(np.fft.rfft(res * w, max(NFFT, len(res))))
    freqs = np.fft.rfftfreq(max(NFFT, len(res)), 1 / FPS)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    k = np.flatnonzero(band)[np.argmax(spec[band])]
    rate = freqs[k]
    power = spec**2
    wide = (freqs >= 1.5) & (freqs <= 20)
    near = np.abs(freqs - rate) <= 1.0
    periodicity = power[near].sum() / max(power[wide].sum(), 1e-12)
    return 2 * spec[k] / w.sum(), rate, periodicity


def measure(x: np.ndarray) -> dict:
    """Voicing-weighted vibrato summary of a mono 22.05 kHz segment."""
    runs = voiced_runs(pitch_cents(x))
    if not runs:
        return dict(voiced_s=0.0, extent_cents=0.0, rate_hz=0.0, periodicity=0.0)
    w = np.array([len(r) for r in runs], float)
    vals = np.array([run_wobble(r) for r in runs])
    extent, rate, periodicity = (w @ vals) / w.sum()
    return dict(voiced_s=round(w.sum() / FPS, 2), extent_cents=round(float(extent), 1),
                rate_hz=round(float(rate), 2), periodicity=round(float(periodicity), 3))
