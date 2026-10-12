import csv
import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

import make_dataset
import prepare_dry
from audio import CLIP, SR, WARMUP, db, write_int16
from conftest import TEST_SOURCES

COUNTS = dict(train=12, val=4, test=4, heldout=4)


def build(data, out, seed=0):
    make_dataset.main(["--data", str(data), "--out", str(out), "--seed", str(seed), "--workers", "4",
                       *[a for s, n in COUNTS.items() for a in (f"--n-{s}", str(n))]])
    with open(out / "params.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def dataset(raw_data):
    prepare_dry.main(["--data", str(raw_data)], sources=TEST_SOURCES)
    out = raw_data / "dataset"
    return raw_data, out, build(raw_data, out)


def test_rows_files_and_ranges(dataset):
    data, out, rows = dataset
    assert list(rows[0]) == make_dataset.COLS
    assert {s: sum(r["split"] == s for r in rows) for s in COUNTS} == COUNTS
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        for kind in ("dry", "wet"):
            info = sf.info(out / r["split"] / f"{r['id']}_{kind}.wav")
            assert (info.samplerate, info.channels, info.frames, info.subtype) == (SR, 1, CLIP, "PCM_16")
        assert 0.1 <= float(r["rate_hz"]) <= 5 and 0 <= float(r["depth_ms"]) <= 5
        assert 7 <= float(r["delay_ms"]) <= 25 and 0 <= float(r["mix"]) <= 1
        assert 0 <= float(r["lfo_phase"]) < 1 and -12 <= float(r["gain_db"]) <= -1
        assert r["demo_ok"] == str(r["source"] != "idmt_guitar")
        assert int(r["wet_clipped"]) == 0
    assert "done: 24 pairs" in (out / "make_dataset.log").read_text()


def test_pairs_stay_in_their_group_split(dataset):
    data, _, rows = dataset
    splits = json.loads((data / "splits.json").read_text())
    for r in rows:
        assert splits[r["source"]][r["group_id"]] == r["split"], r["id"]


def test_dry_clip_is_the_recorded_crop(dataset):
    data, out, rows = dataset
    for r in rows:
        off = int(r["crop_offset"])
        x = sf.read(data / "dry" / r["dry_source"], dtype="float32")[0][off : off + WARMUP + CLIP]
        x = x * 10 ** ((float(r["gain_db"]) - db(np.abs(x).max())) / 20)
        dry = sf.read(out / r["split"] / f"{r['id']}_dry.wav", dtype="float32")[0]
        assert np.abs(dry - x[WARMUP:]).max() <= 1 / 32768, r["id"]


def test_wet_matches_a_rerender_from_csv_params(dataset, tmp_path):
    data, out, rows = dataset
    render = make_dataset.REPO / "target" / "release" / "render"
    for r in rows[:4]:
        off = int(r["crop_offset"])
        x = sf.read(data / "dry" / r["dry_source"], dtype="float32")[0][off : off + WARMUP + CLIP]
        write_int16(tmp_path / "in.wav", x * 10 ** ((float(r["gain_db"]) - db(np.abs(x).max())) / 20))
        p = {k: float(r[k]) for k in ("rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase")}
        (tmp_path / "p.json").write_text(json.dumps(p | {"warmup_samples": WARMUP}))
        subprocess.run([render, tmp_path / "in.wav", tmp_path / "p.json", tmp_path / "wet.wav"], check=True)
        again = sf.read(tmp_path / "wet.wav", dtype="int16")[0]
        stored = sf.read(out / r["split"] / f"{r['id']}_wet.wav", dtype="int16")[0]
        assert np.array_equal(again, stored), r["id"]


def test_same_seed_same_dataset(dataset):
    data, out, rows = dataset
    assert build(data, data / "again") == rows
    other = build(data, data / "other", seed=1)
    assert [r["rate_hz"] for r in other] != [r["rate_hz"] for r in rows]


def test_refuses_to_overwrite(dataset):
    data, out, _ = dataset
    with pytest.raises(SystemExit, match="--overwrite"):
        build(data, out)
