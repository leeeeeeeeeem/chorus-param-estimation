"""Baselines that any model must beat.

python python/baselines.py mean    # always the training average -> results/mean.csv
python python/baselines.py xcorr   # hand-made dry_wet estimator  -> results/xcorr.csv
Predictions cover val, test and heldout. Score them with evaluate.py.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize
from scipy.signal import butter, correlate, lfilter, sosfiltfilt

from audio import SR
from chorus_data import LABELS, scale, unscale

SPLITS = ("val", "test", "heldout")
COLS = ["id", *LABELS]

# xcorr: frames of the whitened signals, lags covering every possible delay.
FRAME = 512  # 23 ms
HOP = 256
MIN_LAG = int(0.002 * SR)  # 7 ms delay - 5 ms depth
MAX_LAG = int(np.ceil(0.031 * SR))  # 25 ms + 5 ms, plus a margin
LPC_ORDER = 32
RATES = np.exp(np.linspace(np.log(0.1), np.log(5.0), 120))
OUTLIER_MS = 1.5
# Matching is done below this frequency: a moving delay shifts pitch, and high
# frequencies drift out of step within a frame first.
LOWPASS_HZ = 1000


def mean_prediction(train: list[dict]) -> dict:
    """Average of each label on its 0..1 training scale (rate on a log scale)."""
    return {n: unscale(n, float(np.mean([scale(n, float(r[n])) for r in train]))) for n in LABELS}


def frame_maps(dry: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame with signal: time (s) and cross-correlation over lags 0..MAX_LAG.
    Both signals are whitened with the dry's LPC filter and low-passed, so each
    row approximates the chorus impulse response: a peak at the current delay
    (the wet part), with copies at pitch periods. The dry part is removed."""
    noise = 1e-9 * np.random.default_rng(0).standard_normal(len(dry))
    a = librosa.lpc(dry.astype(np.float64) + noise, order=LPC_ORDER)
    lp = butter(4, LOWPASS_HZ, fs=SR, output="sos")
    ed, ew = sosfiltfilt(lp, lfilter(a, [1.0], dry)), sosfiltfilt(lp, lfilter(a, [1.0], wet))
    starts = np.arange(MAX_LAG, len(dry) - FRAME, HOP)
    energy = np.array([np.sum(ed[s : s + FRAME] ** 2) for s in starts])
    keep = energy > 1e-3 * energy.max() if energy.max() > 0 else np.zeros(len(starts), bool)
    maps = []
    for s, e in zip(starts[keep], energy[keep]):
        past = ed[s - MAX_LAG : s + FRAME]
        h = correlate(past, ew[s : s + FRAME], mode="valid", method="fft")[::-1] / e
        # The dry part of the wet adds (1 - mix) times the dry's own
        # autocorrelation, with copies at every pitch period. Fit its scale on
        # lags shorter than any delay and remove it.
        auto = correlate(past, ed[s : s + FRAME], mode="valid", method="fft")[::-1] / e
        scale = (h[:MIN_LAG] @ auto[:MIN_LAG]) / max(auto[:MIN_LAG] @ auto[:MIN_LAG], 1e-12)
        maps.append(h - scale * auto)
    return (starts[keep] + FRAME / 2) / SR, np.array(maps).reshape(-1, MAX_LAG + 1)


def coarse_candidates(norm: np.ndarray, t: np.ndarray, top: int) -> list[tuple]:
    """The `top` best (centre, depth, rate, phase) on a coarse grid."""
    rows = np.arange(len(t))
    centres, depths = np.arange(7.0, 25.01, 0.5), np.arange(0.0, 5.01, 0.5)
    found = []
    for f in RATES[::4]:
        for phase in np.linspace(0, 1, 16, endpoint=False):
            wave = np.sin(2 * np.pi * (f * t + phase))
            curve = centres[:, None, None] + depths[None, :, None] * wave
            score = norm[rows, np.clip(np.rint(curve * SR / 1000).astype(int), 0, MAX_LAG)].mean(axis=-1)
            for i in np.argsort(score, axis=None)[-top:]:
                ci, wi = np.unravel_index(i, score.shape)
                found.append((score[ci, wi], centres[ci], depths[wi], f, phase))
    return [c[1:] for c in sorted(found, reverse=True)[:top]]


def path_score(norm: np.ndarray, t: np.ndarray, centre: float, depth: float, rate: float, phase: float) -> float:
    """Mean correlation along a sine path, lags linearly interpolated."""
    lag = np.clip((centre + depth * np.sin(2 * np.pi * (rate * t + phase))) * SR / 1000, 0, MAX_LAG - 1)
    k = lag.astype(int)
    frac = lag - k
    rows = np.arange(len(t))
    return float(np.mean(norm[rows, k] * (1 - frac) + norm[rows, k + 1] * frac))


def search_path(t: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Delay curve (ms per frame) of the sine path that collects the most
    correlation. Pitch copies rarely follow a sine; the chorus delay always does.
    Coarse candidates on maps blurred by about 1 ms, each refined with Nelder-Mead
    on maps blurred by about 0.25 ms, then on the sharp maps."""
    norm = np.clip(maps / (np.abs(maps[:, MIN_LAG:]).max(axis=1, keepdims=True) + 1e-12), 0, None)
    blur = lambda ms: maximum_filter1d(norm, size=2 * int(ms / 1000 * SR) + 1, axis=1)  # noqa: E731
    coarse, mid = blur(1.0), blur(0.25)
    best = (-np.inf, None)
    for c, w, f, ph in coarse_candidates(coarse, t, top=10):
        x = np.array([c, w, np.log(f), ph])
        for m in (mid, norm):
            x = minimize(lambda v, m=m: -path_score(m, t, v[0], abs(v[1]), np.exp(v[2]), v[3]), x,
                         method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400)).x
        c, w, f, ph = x[0], abs(x[1]), float(np.clip(np.exp(x[2]), 0.1, 5.0)), x[3]
        score = path_score(norm, t, c, w, f, ph)
        if score > best[0]:
            best = (score, c + w * np.sin(2 * np.pi * (f * t + ph)))
    return best[1]


def refine(maps: np.ndarray, path_ms: np.ndarray) -> np.ndarray:
    """Per frame, the peak lag (ms, sub-sample) within 0.7 ms of the path."""
    out = []
    for h, centre in zip(maps, np.clip(np.rint(path_ms * SR / 1000).astype(int), MIN_LAG + 1, MAX_LAG - 2)):
        lo, hi = max(MIN_LAG, centre - 15), min(MAX_LAG - 1, centre + 15)
        k = lo + int(np.argmax(h[lo : hi + 1]))
        y0, y1, y2 = h[k - 1], h[k], h[k + 1]
        denom = y0 - 2 * y1 + y2
        out.append(1000 * (k + (0.5 * (y0 - y2) / denom if denom < 0 else 0.0)) / SR)
    return np.array(out)


def sine_design(t: np.ndarray, rate: float) -> np.ndarray:
    return np.stack([np.ones_like(t), np.sin(2 * np.pi * rate * t), np.cos(2 * np.pi * rate * t)], axis=1)


def fit_sine(t: np.ndarray, d: np.ndarray) -> tuple[float, np.ndarray]:
    """Best d(t) = c0 + c1 sin(2 pi rate t) + c2 cos(2 pi rate t) over the rate grid."""
    best = (np.inf, RATES[0], np.zeros(3))
    for f in RATES:
        A = sine_design(t, f)
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        sse = float(np.sum((A @ coef - d) ** 2))
        if sse < best[0]:
            best = (sse, f, coef)
    return best[1], best[2]


def estimate_mix(dry: np.ndarray, wet: np.ndarray, rate: float, coef: np.ndarray) -> float:
    """Least squares wet = a * dry + b * dry delayed along the fitted curve; mix = b / (a + b)."""
    lp = butter(4, LOWPASS_HZ, fs=SR, output="sos")
    dry, wet = sosfiltfilt(lp, dry), sosfiltfilt(lp, wet)
    n = np.arange(MAX_LAG, len(dry))
    delay = sine_design(n / SR, rate) @ coef * SR / 1000  # in samples
    delayed = np.interp(n - delay, np.arange(len(dry)), dry)
    (a, b), *_ = np.linalg.lstsq(np.stack([dry[n], delayed], axis=1), wet[n], rcond=None)
    a, b = max(a, 0.0), max(b, 0.0)
    return b / (a + b) if a + b > 0 else 0.0


def xcorr_estimate(dry: np.ndarray, wet: np.ndarray, fallback: dict) -> dict:
    """Search the best sine path, refine it with nearby peaks, fit a sine, then the mix."""
    t, maps = frame_maps(dry, wet)
    if len(t) < 10:
        return dict(fallback)
    d = refine(maps, search_path(t, maps))
    rate, coef = fit_sine(t, d)
    inliers = np.abs(sine_design(t, rate) @ coef - d) < OUTLIER_MS
    if inliers.sum() >= 10:
        rate, coef = fit_sine(t[inliers], d[inliers])
    return dict(rate_hz=rate, depth_ms=float(np.hypot(coef[1], coef[2])), delay_ms=float(coef[0]),
                mix=estimate_mix(dry, wet, rate, coef))


def _xcorr_row(args: tuple) -> dict:
    root, row, fallback = args
    path = root / row["split"] / row["id"]
    dry = sf.read(f"{path}_dry.wav", dtype="float32")[0]
    wet = sf.read(f"{path}_wet.wav", dtype="float32", always_2d=True)[0][:, 0]  # left
    est = xcorr_estimate(dry, wet, fallback)
    return {"id": row["id"], **{n: round(float(np.clip(est[n], *LABELS[n][:2])), 6) for n in LABELS}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", choices=["mean", "xcorr"])
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=None, help="default: results/<baseline>.csv")
    args = ap.parse_args(argv)
    with open(args.dataset / "params.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    mean = mean_prediction([r for r in rows if r["split"] == "train"])
    targets = [r for r in rows if r["split"] in SPLITS]

    if args.baseline == "mean":
        preds = [{"id": r["id"], **{n: round(v, 6) for n, v in mean.items()}} for r in targets]
    else:
        with ProcessPoolExecutor() as pool:
            preds = list(pool.map(_xcorr_row, [(args.dataset, r, mean) for r in targets], chunksize=16))

    out = args.out or Path("results") / f"{args.baseline}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(preds)
    print(f"wrote {len(preds)} predictions to {out}")


if __name__ == "__main__":
    main()
