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
                             "--splits", "train", "--workers", "4"])
    return out


def test_every_kind_keeps_the_original_settings(synthetic, stereo):
    original = {r["id"]: r for r in read(synthetic / "params.csv")}
    rows = read(stereo / "params.csv")
    kinds = make_stereo_source.KINDS
    assert [r["stereo_kind"] for r in rows] == [kinds[i % len(kinds)] for i in range(len(rows))]
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
        if r["stereo_kind"] in ("mono", "balance"):
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
    assert {k: groups[f"stereo {k}"] for k in make_stereo_source.KINDS} == dict(mono=2, pan=2, corr=2, wide=1, balance=1)


def test_balance_turns_one_output_down_after_the_chorus(synthetic, stereo):
    rows = [r for r in read(stereo / "params.csv") if r["stereo_kind"] == "balance"]
    assert rows
    for r in rows:
        original = sf.read(synthetic / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
        wet = sf.read(stereo / "train" / f"{r['id']}_wet.wav", dtype="float32")[0]
        diff = float(r["stereo_amount"])
        quiet, loud = (1, 0) if diff > 0 else (0, 1)
        assert np.array_equal(wet[:, loud], original[:, loud])
        g = 10 ** (-abs(diff) / 20)
        assert np.abs(wet[:, quiet] - original[:, quiet] * g).max() <= 2 / 32768, "quiet side scaled, int16 rounding only"


def test_weighted_kinds_and_several_splits(synthetic, tmp_path):
    out = tmp_path / "mixed"
    make_stereo_source.main(["--data", str(synthetic.parent), "--dataset", str(synthetic), "--out", str(out),
                             "--splits", "train,val", "--kinds", "mono=1,wide=1", "--workers", "4"])
    rows = read(out / "params.csv")
    assert sorted({r["split"] for r in rows}) == ["train", "val"] and len(rows) == 10
    assert {r["stereo_kind"] for r in rows} <= {"mono", "wide"} and len({r["stereo_kind"] for r in rows}) == 2
    again = tmp_path / "again"
    make_stereo_source.main(["--data", str(synthetic.parent), "--dataset", str(synthetic), "--out", str(again),
                             "--splits", "train,val", "--kinds", "mono=1,wide=1", "--workers", "2"])
    assert [r["stereo_kind"] for r in read(again / "params.csv")] == [r["stereo_kind"] for r in rows]
    with pytest.raises(SystemExit, match="unknown kinds"):
        make_stereo_source.main(["--data", str(synthetic.parent), "--dataset", str(synthetic),
                                 "--out", str(tmp_path / "bad"), "--kinds", "mono=1,haas=1"])


def test_eval_dataset_gives_predictions_for_the_standard_splits(synthetic, tmp_path):
    out = tmp_path / "mixed"
    make_stereo_source.main(["--data", str(synthetic.parent), "--dataset", str(synthetic), "--out", str(out),
                             "--splits", "train,val", "--kinds", "mono=1,pan=1", "--workers", "4"])
    train.main(["--dataset", str(out), "--eval-dataset", str(synthetic), "--mode", "wet", "--view", "ms",
                "--epochs", "1", "--batch-size", "4", "--workers", "0", "--device", "cpu", "--name", "mx",
                "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results")])
    preds = read(tmp_path / "results" / "mx.csv")
    assert sorted(p["id"] for p in preds) == sorted(r["id"] for r in read(synthetic / "params.csv") if r["split"] != "train")
