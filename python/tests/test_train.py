import csv

import torch

import evaluate
import train
from chorus_data import LABELS


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_model_shapes():
    for channels in (1, 2, 3):
        out = train.ChorusCNN(channels)(torch.randn(4, channels, 513, 173))
        assert out.shape == (4, len(LABELS)) and ((out >= 0) & (out <= 1)).all()


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


def test_training_reduces_the_loss(synthetic, tmp_path):
    train.main(["--dataset", str(synthetic), "--mode", "wet", "--epochs", "25", "--batch-size", "8",
                "--lr", "3e-4", "--workers", "0", "--device", "cpu",
                "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results")])
    losses = [float(r["train_loss"]) for r in read(tmp_path / "runs" / "wet_left" / "log.csv")]
    assert min(losses[-5:]) < losses[0] / 3, losses
