import numpy as np
import pytest

from audio import SR
from vibrato import measure


def sing(cents_curve: np.ndarray, f_ref: float = 220.0) -> np.ndarray:
    """Harmonic tone following a pitch curve given in cents re f_ref."""
    f = f_ref * 2 ** (cents_curve / 1200)
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * x).astype(np.float32)


t = np.arange(4 * SR) / SR


@pytest.mark.parametrize("extent, rate", [(25, 4.0), (25, 5.0), (50, 6.0), (100, 7.0), (50, 8.0)])
@pytest.mark.parametrize("f_ref", [110.0, 440.0])
def test_measures_vibrato_extent_and_rate(extent, rate, f_ref):
    m = measure(sing(extent * np.sin(2 * np.pi * rate * t), f_ref))
    assert m["voiced_s"] > 3
    assert 0.7 * extent <= m["extent_cents"] <= 1.15 * extent
    assert m["rate_hz"] == pytest.approx(rate, abs=0.3)
    assert m["periodicity"] > 0.7


def test_steady_tone_has_no_vibrato():
    m = measure(sing(np.zeros_like(t)))
    assert m["extent_cents"] < 5


@pytest.mark.parametrize("note_s", [0.4, 0.2, 0.15])
def test_scales_are_not_vibrato(note_s):
    # A new note every note_s seconds, 200 cents apart. Fast scales step at
    # 5 to 7 Hz, right in the vibrato band.
    m = measure(sing(200.0 * np.floor(t / note_s)))
    assert m["extent_cents"] < 10


def test_short_runs_measure_like_long_ones():
    # 0.7 s voiced runs, barely longer than the minimum: edge handling matters.
    short_t = np.arange(int(0.7 * SR)) / SR
    x = sing(50 * np.sin(2 * np.pi * 6 * short_t))
    gap = np.zeros(int(0.3 * SR), np.float32)
    short = measure(np.concatenate([gap, x, gap, x, gap, x, gap]))
    long = measure(sing(50 * np.sin(2 * np.pi * 6 * t)))
    assert short["voiced_s"] > 1.5
    assert short["extent_cents"] == pytest.approx(long["extent_cents"], rel=0.2)


def test_silence_gives_zeros():
    m = measure(np.zeros(2 * SR, np.float32))
    assert m == dict(voiced_s=0.0, extent_cents=0.0, rate_hz=0.0, periodicity=0.0)
