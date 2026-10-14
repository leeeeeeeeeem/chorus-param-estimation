import csv

import numpy as np
import pytest

import baselines
import evaluate
import make_dataset
from audio import SR

RENDER = make_dataset.REPO / "target" / "release" / "render"
FALLBACK = dict(rate_hz=0.7, depth_ms=2.5, delay_ms=16.0, mix=0.5)


def estimate(x, **p):
    dry, wet = make_dataset.render_pair(x, p | dict(lfo_phase=0.3), RENDER)
    return baselines.xcorr_estimate(dry, wet[:, 0], FALLBACK)


def test_mean_prediction_averages_on_the_training_scale():
    train = [dict(rate_hz="0.1", depth_ms="0", delay_ms="7", mix="0"),
             dict(rate_hz="5", depth_ms="5", delay_ms="25", mix="1")]
    m = baselines.mean_prediction(train)
    assert m["rate_hz"] == pytest.approx(np.sqrt(0.5))  # middle of the log range
    assert (m["depth_ms"], m["delay_ms"], m["mix"]) == pytest.approx((2.5, 16.0, 0.5))


@pytest.mark.parametrize("rate, depth, delay, mix", [(2.0, 2.0, 15.0, 0.5), (0.8, 1.0, 22.0, 0.3),
                                                     (4.0, 4.5, 9.0, 0.8), (1.5, 3.0, 12.0, 0.15)])
def test_xcorr_recovers_settings_on_noise(rate, depth, delay, mix):
    x = (np.random.default_rng(0).standard_normal(make_dataset.NEED) * 0.1).astype(np.float32)
    e = estimate(x, rate_hz=rate, depth_ms=depth, delay_ms=delay, mix=mix)
    assert e["delay_ms"] == pytest.approx(delay, abs=0.2)
    assert e["depth_ms"] == pytest.approx(depth, abs=0.3)
    assert e["rate_hz"] == pytest.approx(rate, rel=0.1)
    assert e["mix"] == pytest.approx(mix, abs=0.1)


def test_xcorr_is_not_fooled_by_pitch_periods():
    # A harmonic tone gliding 150 -> 300 Hz: its period (6.7 to 3.3 ms) puts
    # copies of both the dry and the wet path inside the delay range.
    n = make_dataset.NEED
    f0 = 150 * 2 ** (np.arange(n) / n)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = (0.1 * sum(np.sin(k * phase) / k for k in range(1, 8))).astype(np.float32)
    e = estimate(x, rate_hz=1.0, depth_ms=2.0, delay_ms=14.0, mix=0.25)
    assert e["delay_ms"] == pytest.approx(14.0, abs=0.3)
    assert e["depth_ms"] == pytest.approx(2.0, abs=0.4)


def test_silence_falls_back():
    e = baselines.xcorr_estimate(np.zeros(2 * SR), np.zeros(2 * SR), FALLBACK)
    assert e == FALLBACK


@pytest.mark.parametrize("name", ["mean", "xcorr"])
def test_main_predicts_every_eval_pair(synthetic, tmp_path, name):
    out = tmp_path / f"{name}.csv"
    baselines.main([name, "--dataset", str(synthetic), "--out", str(out)])
    with open(out, newline="") as f:
        preds = list(csv.DictReader(f))
    with open(synthetic / "params.csv", newline="") as f:
        truth = list(csv.DictReader(f))
    assert sorted(p["id"] for p in preds) == sorted(r["id"] for r in truth if r["split"] != "train")
    rows = evaluate.summarise(preds, truth)
    assert {r["split"] for r in rows} == {"val", "test", "heldout"}
