import csv

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate
import make_stereo_source
import predict
import train
from chorus_data import ChorusPairs


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def stereo(synthetic):
    out = synthetic.parent / "dataset_stereo"
    make_stereo_source.main(["--data", str(synthetic.parent), "--dataset", str(synthetic), "--out", str(out),
                             "--split", "train", "--workers", "4"])
    return out


def test_every_kind_keeps_the_original_settings(synthetic, stereo):
    original = {r["id"]: r for r in read(synthetic / "params.csv")}
    rows = read(stereo / "params.csv")
    assert [r["stereo_kind"] for r in rows] == list(make_stereo_source.KINDS) * 2
    for r in rows:
        o = original[r["id"]]
        for k in ("dry_source", "crop_offset", "gain_db", "rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase"):
            assert r[k] == o[k], (r["id"], k)


def test_mono_kind_is_the_original_pair_bit_for_bit(synthetic, stereo):
    for r in read(stereo / "params.csv"):
        if r["stereo_kind"] != "mono":
            continue
        for part in ("dry", "wet"):
            a = sf.read(synthetic / "train" / f"{r['id']}_{part}.wav", dtype="int16")[0]
            b = sf.read(stereo / "train" / f"{r['id']}_{part}.wav", dtype="int16")[0]
            assert np.array_equal(a, b), (r["id"], part)


def test_stereo_kinds_change_only_the_input(stereo):
    for r in read(stereo / "params.csv"):
        dry = sf.read(stereo / "train" / f"{r['id']}_dry.wav", dtype="float32")[0]
        if r["stereo_kind"] == "mono":
            assert dry.ndim == 1
            continue
        assert dry.shape[1] == 2 and not np.allclose(dry[:, 0], dry[:, 1])
        if r["stereo_kind"] == "pan":
            ratio_db = 20 * np.log10(np.abs(dry[:, 0]).max() / np.abs(dry[:, 1]).max())
            assert ratio_db == pytest.approx(float(r["stereo_amount"]), abs=0.1)
        else:
            second = r["second_source"]
            assert second and second.split("/")[1] == r["group_id"], "second crop comes from the same group"
            if second == r["dry_source"]:
                assert abs(int(r["second_offset"]) - int(r["crop_offset"])) >= make_stereo_source.NEED


def test_loader_scoring_and_predict_on_stereo_pairs(synthetic, stereo, tmp_path):
    x, _ = ChorusPairs(stereo, "train", "dry_wet", "ms")[1]
    assert x.shape[0] == 3 and torch.isfinite(x).all()
    train.main(["--dataset", str(synthetic), "--mode", "wet", "--view", "ms", "--epochs", "1", "--batch-size", "4",
                "--workers", "0", "--device", "cpu", "--name", "m", "--runs", str(tmp_path / "runs"),
                "--results", str(tmp_path / "results")])
    predict.main(["m", "--dataset", str(stereo), "--split", "train", "--runs", str(tmp_path / "runs"),
                  "--results", str(tmp_path / "results"), "--workers", "0", "--device", "cpu"])
    preds = read(tmp_path / "results" / "m_stereosrc.csv")
    assert len(preds) == 8
    groups = {r["group"]: r["n"] for r in evaluate.summarise(preds, read(stereo / "params.csv"))}
    assert all(groups[f"stereo {k}"] == 2 for k in make_stereo_source.KINDS)
