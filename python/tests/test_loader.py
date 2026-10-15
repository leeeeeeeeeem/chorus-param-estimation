import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from audio import SR
from chorus_data import LABELS, N_FFT, ChorusPairs, log_stft, scale, unscale, wet_view
from conftest import REAL_DATA

SHAPE = (N_FFT // 2 + 1, 173)
REAL = REAL_DATA / "dataset"
has_real = pytest.mark.skipif(not (REAL / "params.csv").exists(), reason="real dataset not built yet")


class TinyCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        layers, c = [], channels
        for out in (16, 32, 64):
            layers += [nn.Conv2d(c, out, 3, stride=2, padding=1), nn.ReLU()]
            c = out
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, len(LABELS)))

    def forward(self, x):
        return self.net(x)


def overfit(x: torch.Tensor, y: torch.Tensor, steps: int = 800) -> tuple[float, float]:
    torch.manual_seed(0)
    model = TinyCNN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses[0], losses[-1]


def test_label_scaling_round_trips():
    for name, (lo, hi, _) in LABELS.items():
        assert scale(name, lo) == pytest.approx(0) and scale(name, hi) == pytest.approx(1)
        for v in np.linspace(lo, hi, 7)[1:]:
            assert unscale(name, scale(name, v)) == pytest.approx(v)
    assert scale("rate_hz", np.sqrt(0.1 * 5.0)) == pytest.approx(0.5), "rate is log scaled"


def test_log_stft_peaks_at_the_tone_bin():
    f = 40 * SR / N_FFT
    x = torch.sin(2 * torch.pi * f * torch.arange(2 * SR) / SR)[None].float()
    spec = log_stft(x, torch.hann_window(N_FFT))
    assert spec.shape == (1, *SHAPE)
    assert spec[0, :, 50:-50].mean(dim=1).argmax().item() == 40


def test_log_stft_window_sets_the_resolution():
    f = 80 * SR / 2048
    x = torch.sin(2 * torch.pi * f * torch.arange(2 * SR) / SR)[None].float()
    spec = log_stft(x, torch.hann_window(2048))
    assert spec.shape == (1, 1025, 173)
    assert spec[0, :, 50:-50].mean(dim=1).argmax().item() == 80


def expected_wet(wet, view):
    left, right = wet[:, 0], wet[:, 1]
    return {"left": [left], "lr": [left, right], "ms": [(left + right) / 2, (left - right) / 2],
            "mono": [(left + right) / 2]}[view]


@pytest.mark.parametrize("mode", ["dry_wet", "wet"])
@pytest.mark.parametrize("view", ["left", "lr", "ms", "mono"])
def test_items(synthetic, mode, view):
    ds = ChorusPairs(synthetic, "train", mode, view)
    assert len(ds) == 8
    x, y = ds[0]
    assert x.shape == (ds.channels, *SHAPE) and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert y.shape == (4,) and ((0 <= y) & (y <= 1)).all()
    r = ds.rows[0]
    wet = sf.read(synthetic / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
    dry = sf.read(synthetic / "train" / f"{r['id']}_dry.wav", dtype="float32")[0]
    signals = ([dry] if mode == "dry_wet" else []) + expected_wet(wet, view)
    expected = log_stft(torch.from_numpy(np.stack(signals).astype(np.float32)), ds.window)
    assert torch.allclose(x, expected, atol=1e-5), "channels are dry (if any), then the wet view"
    assert y[3].item() == pytest.approx(float(r["mix"]))


def test_random_mix_is_between_left_and_right():
    left, right = np.ones(100, np.float32), np.zeros(100, np.float32)
    mixes = {float(wet_view(np.stack([left, right], axis=1), "random_mix")[0, 0]) for _ in range(20)}
    assert len(mixes) > 10 and all(0 <= a <= 1 for a in mixes)
    with pytest.raises(ValueError):
        wet_view(np.stack([left, right], axis=1), "side")


def test_dataloader_with_workers(synthetic):
    dl = torch.utils.data.DataLoader(ChorusPairs(synthetic, "train"), batch_size=4, num_workers=2)
    xs = [x for x, _ in dl]
    assert [len(x) for x in xs] == [4, 4]


def test_bad_mode_view_and_empty_split(synthetic):
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "stereo")
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "wet", "side")
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "nope")


@pytest.mark.parametrize("mode, view", [("dry_wet", "left"), ("wet", "left"), ("wet", "lr"), ("wet", "ms")])
def test_tiny_model_overfits_one_batch(synthetic, mode, view):
    x, y = next(iter(torch.utils.data.DataLoader(ChorusPairs(synthetic, "train", mode, view), batch_size=8)))
    first, last = overfit(x, y)
    assert last < 1e-3 and last < first / 50, (first, last)


@has_real
@pytest.mark.parametrize("mode, view", [("dry_wet", "left"), ("wet", "lr")])
def test_real_dataset_batch_and_overfit(mode, view):
    channels = ChorusPairs(REAL, "train", mode, view).channels
    for split in ("train", "val", "test", "heldout"):
        x, y = ChorusPairs(REAL, split, mode, view)[0]
        assert x.shape == (channels, *SHAPE) and torch.isfinite(x).all()
    dl = torch.utils.data.DataLoader(ChorusPairs(REAL, "train", mode, view), batch_size=16, shuffle=True,
                                     num_workers=4, generator=torch.Generator().manual_seed(0))
    x, y = next(iter(dl))
    assert x.shape == (16, channels, *SHAPE) and ((0 <= y) & (y <= 1)).all()
    first, last = overfit(x, y, steps=1200)
    assert last < 1e-3 and last < first / 50, (first, last)


def test_mono_and_left_input_leave_no_side_signal(synthetic):
    r = ChorusPairs(synthetic, "train", "wet", "ms").rows[0]
    wet = sf.read(synthetic / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
    floor = np.log(1e-5)
    for wet_input, source in (("mono", wet.mean(axis=1)), ("left", wet[:, 0])):
        ms = ChorusPairs(synthetic, "train", "wet", "ms", wet_input)[0][0]
        assert torch.allclose(ms[1], torch.full_like(ms[1], floor)), f"{wet_input}: side must be silent"
        lr = ChorusPairs(synthetic, "train", "wet", "lr", wet_input)[0][0]
        assert torch.equal(lr[0], lr[1])
        expected = log_stft(torch.from_numpy(source.astype(np.float32))[None], torch.hann_window(N_FFT))[0]
        assert torch.allclose(lr[0], expected, atol=1e-5)
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "wet", "lr", "side")


def test_side_drop_gives_some_items_a_mono_wet(synthetic):
    stereo = ChorusPairs(synthetic, "train", "wet", "ms")
    mono = ChorusPairs(synthetic, "train", "wet", "ms", "mono")
    assert torch.equal(ChorusPairs(synthetic, "train", "wet", "ms", side_drop=1.0)[0][0], mono[0][0])
    torch.manual_seed(0)
    half = ChorusPairs(synthetic, "train", "wet", "ms", side_drop=0.5)
    dropped = [torch.equal(half[i % len(half)][0], mono[i % len(half)][0]) for i in range(200)]
    assert 60 < sum(dropped) < 140
    for i in range(min(5, len(stereo))):
        assert not torch.equal(stereo[i][0], mono[i][0]), "synthetic wet must have a side signal"
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "wet", "ms", side_drop=1.5)


def test_blend_drop_gives_one_signal_on_both_sides_between_l_and_r(synthetic):
    ds = ChorusPairs(synthetic, "train", "wet", "lr", side_drop=1.0, drop_input="blend")
    r = ds.rows[0]
    wet = sf.read(synthetic / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
    left = log_stft(torch.from_numpy(wet[:, 0].copy())[None], torch.hann_window(N_FFT))[0]
    seen = set()
    torch.manual_seed(1)
    for _ in range(20):
        x = ds[0][0]
        assert torch.equal(x[0], x[1]), "both channels carry the same blend"
        seen.add(round(float((x[0] - left).abs().mean()), 3))
    assert len(seen) > 5, "the blend changes between draws"
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "wet", "ms", side_drop=0.5, drop_input="side")
