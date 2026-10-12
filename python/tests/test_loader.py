import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

import make_dataset
import prepare_dry
from audio import SR
from chorus_data import LABELS, N_FFT, ChorusPairs, log_stft, scale, unscale
from conftest import REAL_DATA, TEST_SOURCES

SHAPE = (N_FFT // 2 + 1, 173)
REAL = REAL_DATA / "dataset"
has_real = pytest.mark.skipif(not (REAL / "params.csv").exists(), reason="real dataset not built yet")


@pytest.fixture
def synthetic(raw_data):
    prepare_dry.main(["--data", str(raw_data)], sources=TEST_SOURCES)
    out = raw_data / "dataset"
    make_dataset.main(["--data", str(raw_data), "--out", str(out), "--workers", "4",
                       "--n-train", "8", "--n-val", "2", "--n-test", "2", "--n-heldout", "2"])
    return out


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


@pytest.mark.parametrize("mode, channels", [("dry_wet", 2), ("wet", 1)])
def test_items(synthetic, mode, channels):
    ds = ChorusPairs(synthetic, "train", mode)
    assert len(ds) == 8
    x, y = ds[0]
    assert x.shape == (channels, *SHAPE) and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert y.shape == (4,) and ((0 <= y) & (y <= 1)).all()
    r = ds.rows[0]
    wet = sf.read(synthetic / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
    assert torch.equal(x[-1], log_stft(torch.from_numpy(wet)[None], ds.window)[0]), "last channel is wet"
    assert y[3].item() == pytest.approx(float(r["mix"]))


def test_dataloader_with_workers(synthetic):
    dl = torch.utils.data.DataLoader(ChorusPairs(synthetic, "train"), batch_size=4, num_workers=2)
    xs = [x for x, _ in dl]
    assert [len(x) for x in xs] == [4, 4]


def test_bad_mode_and_empty_split(synthetic):
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "train", "stereo")
    with pytest.raises(ValueError):
        ChorusPairs(synthetic, "nope")


@pytest.mark.parametrize("mode", ["dry_wet", "wet"])
def test_tiny_model_overfits_one_batch(synthetic, mode):
    x, y = next(iter(torch.utils.data.DataLoader(ChorusPairs(synthetic, "train", mode), batch_size=8)))
    first, last = overfit(x, y)
    assert last < 1e-3 and last < first / 50, (first, last)


@has_real
@pytest.mark.parametrize("mode", ["dry_wet", "wet"])
def test_real_dataset_batch_and_overfit(mode):
    channels = len(mode.split("_"))
    for split in ("train", "val", "test"):
        x, y = ChorusPairs(REAL, split, mode)[0]
        assert x.shape == (channels, *SHAPE) and torch.isfinite(x).all()
    dl = torch.utils.data.DataLoader(ChorusPairs(REAL, "train", mode), batch_size=16, shuffle=True,
                                     num_workers=4, generator=torch.Generator().manual_seed(0))
    x, y = next(iter(dl))
    assert x.shape == (16, channels, *SHAPE) and ((0 <= y) & (y <= 1)).all()
    first, last = overfit(x, y, steps=1200)
    assert last < 1e-3 and last < first / 50, (first, last)
