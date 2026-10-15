import csv

import pytest
import torch

import evaluate
import train
from chorus_data import LABELS


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_model_shapes():
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        for channels in (1, 2, 3):
            model = train.ChorusCNN(channels).to(device)
            for bins in (513, 1025):
                out = model(torch.randn(4, channels, bins, 173, device=device))
                assert out.shape == (4, len(LABELS)) and ((out >= 0) & (out <= 1)).all(), device


def test_train_writes_log_checkpoint_and_predictions(synthetic, tmp_path):
    train.main(["--dataset", str(synthetic), "--mode", "dry_wet", "--view", "lr", "--epochs", "2",
                "--batch-size", "4", "--workers", "0", "--device", "cpu",
                "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results")])
    log = read(tmp_path / "runs" / "dry_wet_lr" / "log.csv")
    assert [r["epoch"] for r in log] == ["1", "2"]
    assert (tmp_path / "runs" / "dry_wet_lr" / "best.pt").exists()
    preds = read(tmp_path / "results" / "dry_wet_lr.csv")
    truth = read(synthetic / "params.csv")
    assert sorted(p["id"] for p in preds) == sorted(r["id"] for r in truth if r["split"] != "train")
    for p in preds:
        for n, (lo, hi, _) in LABELS.items():
            assert lo - 1e-6 <= float(p[n]) <= hi + 1e-6
    assert {r["split"] for r in evaluate.summarise(preds, truth)} == {"val", "test", "heldout"}
    for suffix in ("_monoin", "_leftin"):
        other = read(tmp_path / "results" / f"dry_wet_lr{suffix}.csv")
        assert [p["id"] for p in other] == [p["id"] for p in preds]


def test_training_reduces_the_loss(synthetic, tmp_path):
    train.main(["--dataset", str(synthetic), "--mode", "wet", "--epochs", "25", "--batch-size", "8",
                "--lr", "3e-4", "--workers", "0", "--device", "cpu",
                "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results")])
    losses = [float(r["train_loss"]) for r in read(tmp_path / "runs" / "wet_left" / "log.csv")]
    assert min(losses[-5:]) < losses[0] / 3, losses


def run(synthetic, tmp_path, name, *extra):
    train.main(["--dataset", str(synthetic), "--mode", "wet", "--epochs", "4", "--batch-size", "4",
                "--workers", "0", "--device", "cpu", "--name", name,
                "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results"), *extra])


def test_stopped_and_resumed_run_matches_an_uninterrupted_one(synthetic, tmp_path):
    run(synthetic, tmp_path, "straight")
    run(synthetic, tmp_path, "resumed", "--stop-after", "2")
    assert len(read(tmp_path / "runs" / "resumed" / "log.csv")) == 2
    assert not (tmp_path / "results" / "resumed.csv").exists()
    run(synthetic, tmp_path, "resumed", "--resume")

    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert strip(read(tmp_path / "runs" / "resumed" / "log.csv")) == strip(read(tmp_path / "runs" / "straight" / "log.csv"))
    a = torch.load(tmp_path / "runs" / "straight" / "best.pt")
    b = torch.load(tmp_path / "runs" / "resumed" / "best.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert read(tmp_path / "results" / "resumed.csv") == read(tmp_path / "results" / "straight.csv")


def test_existing_run_needs_resume_or_fresh(synthetic, tmp_path):
    run(synthetic, tmp_path, "r", "--stop-after", "1")
    with pytest.raises(SystemExit, match="--resume"):
        run(synthetic, tmp_path, "r")
    with pytest.raises(SystemExit, match="settings changed"):
        run(synthetic, tmp_path, "r", "--resume", "--lr", "1e-3")
    run(synthetic, tmp_path, "r", "--fresh", "--stop-after", "1")
    assert len(read(tmp_path / "runs" / "r" / "log.csv")) == 1


def test_n_fft_reaches_the_model_and_resume_accepts_old_runs(synthetic, tmp_path):
    run(synthetic, tmp_path, "big", "--n-fft", "2048", "--stop-after", "1")
    ckpt = torch.load(tmp_path / "runs" / "big" / "last.pt", weights_only=False)
    assert ckpt["settings"]["n_fft"] == "2048"
    with pytest.raises(SystemExit, match="settings changed"):
        run(synthetic, tmp_path, "big", "--resume")
    run(synthetic, tmp_path, "old", "--stop-after", "1")
    path = tmp_path / "runs" / "old" / "last.pt"
    ckpt = torch.load(path, weights_only=False)
    del ckpt["settings"]["n_fft"]  # as saved before --n-fft existed
    torch.save(ckpt, path)
    run(synthetic, tmp_path, "old", "--resume", "--stop-after", "2")
    assert len(read(tmp_path / "runs" / "old" / "log.csv")) == 2
