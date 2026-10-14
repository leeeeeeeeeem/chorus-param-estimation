import csv
import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

import make_dataset
import prepare_dry
from audio import CLIP, SR, WARMUP
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


RENDER = make_dataset.REPO / "target" / "release" / "render"
PARAM_KEYS = ("rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase")


def stored(out, r, kind, dtype="float32"):
    return sf.read(out / r["split"] / f"{r['id']}_{kind}.wav", dtype=dtype)[0]


def test_dry_clip_is_the_recorded_crop(dataset):
    data, out, rows = dataset
    for r in rows:
        x = make_dataset.load_crop(data / "dry" / r["dry_source"], int(r["crop_offset"]), float(r["gain_db"]))
        expected = make_dataset.down(make_dataset.up(x))[WARMUP : WARMUP + CLIP]
        assert np.abs(stored(out, r, "dry") - expected).max() <= 1 / 32768, r["id"]
        assert r["render_sr"] == str(2 * SR)


def test_wet_matches_a_rerender_from_csv_params(dataset):
    data, out, rows = dataset
    for r in rows[:4]:
        x = make_dataset.load_crop(data / "dry" / r["dry_source"], int(r["crop_offset"]), float(r["gain_db"]))
        _, wet = make_dataset.render_pair(x, {k: float(r[k]) for k in PARAM_KEYS}, RENDER)
        again = np.clip(np.round(wet * 32768), -32768, 32767).astype(np.int16)
        assert np.array_equal(again, stored(out, r, "wet", "int16")), r["id"]


def crop_from(data, rows):
    r = rows[0]
    return make_dataset.load_crop(data / "dry" / r["dry_source"], int(r["crop_offset"]), -3.0)


def test_mix_zero_pair_nulls(dataset):
    data, _, rows = dataset
    p = dict(rate_hz=3.3, depth_ms=5.0, delay_ms=20.0, mix=0.0, lfo_phase=0.7)
    dry, wet = make_dataset.render_pair(crop_from(data, rows), p, RENDER)
    assert len(dry) == CLIP and np.array_equal(dry, wet)


def test_pair_keeps_alignment():
    x = np.zeros(make_dataset.NEED, np.float32)
    x[WARMUP + 1000] = 0.5
    p = dict(rate_hz=1.0, depth_ms=0.0, delay_ms=20.0, mix=1.0, lfo_phase=0.0)
    dry, wet = make_dataset.render_pair(x, p, RENDER)
    assert np.argmax(np.abs(dry)) == 1000
    assert np.argmax(np.abs(wet)) == 1000 + 441  # 20 ms at 22.05 kHz


def test_lfo_phase_is_at_the_clip_start(dataset, tmp_path):
    # Same render with render's own warmup handling: the LFO must line up.
    data, _, rows = dataset
    x = crop_from(data, rows)
    p = dict(rate_hz=4.7, depth_ms=5.0, delay_ms=15.0, mix=1.0, lfo_phase=0.37)
    hi = make_dataset.up(x)
    sf.write(tmp_path / "in.wav", hi, 2 * SR, subtype="FLOAT")
    (tmp_path / "p.json").write_text(json.dumps(p | {"warmup_samples": 2 * WARMUP}))
    subprocess.run([RENDER, tmp_path / "in.wav", tmp_path / "p.json", tmp_path / "out.wav"], check=True)
    ref = make_dataset.down(np.concatenate([np.zeros(2 * WARMUP, np.float32), sf.read(tmp_path / "out.wav")[0]]))
    _, wet = make_dataset.render_pair(x, p, RENDER)
    # Away from the reference's artificial edge at the clip start.
    assert np.abs(wet[100:] - ref[WARMUP + 100 : WARMUP + CLIP]).max() < 1e-5


def test_same_seed_same_dataset(dataset):
    data, out, rows = dataset
    assert build(data, data / "again") == rows
    other = build(data, data / "other", seed=1)
    assert [r["rate_hz"] for r in other] != [r["rate_hz"] for r in rows]


def test_refuses_to_overwrite(dataset):
    data, out, _ = dataset
    with pytest.raises(SystemExit, match="--overwrite"):
        build(data, out)
