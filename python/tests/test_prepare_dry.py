import csv
import json
import random

import numpy as np
import pytest
import soundfile as sf

import prepare_dry
from audio import SR, db
from conftest import REAL_DATA, TEST_SOURCES, tone
from prepare_dry import MANIFEST_COLS, assign_splits, check_splits, find_segments, normalise
from sources import SOURCES


def run(data, *sources):
    prepare_dry.main([*sources, "--data", str(data)], sources=TEST_SOURCES)


def manifest(data):
    with open(data / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_find_segments_merges_short_gaps_and_drops_short_runs():
    rng = np.random.default_rng(0)
    z = lambda s: np.zeros(int(s * SR), np.float32)  # noqa: E731
    x = np.concatenate([z(0.5), tone(3, SR, 200, rng), z(0.2), tone(2, SR, 300, rng),
                        z(1), tone(1, SR, 400, rng), z(1)])
    segs = find_segments(x / np.abs(x).max())
    assert len(segs) == 1
    s, e = segs[0]
    assert abs(s / SR - 0.5) <= 0.05 and abs(e / SR - 5.7) <= 0.05


def test_normalise_hits_rms_or_peak_cap():
    rng = np.random.default_rng(0)
    x = tone(3, SR, 200, rng) * 0.01
    y = normalise(x)
    assert db(np.sqrt(np.mean(y.astype(np.float64) ** 2))) == pytest.approx(-20, abs=0.01)
    spiky = np.zeros(SR, np.float32)
    spiky[::1000] = 0.5
    assert db(np.abs(normalise(spiky)).max()) == pytest.approx(-1, abs=0.01)


def test_assign_splits():
    s = assign_splits([f"s{i}" for i in range(20)], "pool", 0)
    counts = {k: list(s.values()).count(k) for k in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 3, "test": 3}
    assert assign_splits([f"p{i}" for i in range(6)], "pool", 0) == assign_splits(
        [f"p{i}" for i in reversed(range(6))], "pool", 0), "order must not matter"
    assert set(assign_splits(["a", "b"], "heldout", 0).values()) == {"heldout"}
    with pytest.raises(ValueError):
        assign_splits(["a", "b"], "pool", 0)


def test_stratified_split_spreads_voices():
    groups = [f"female{i}" for i in range(1, 10)] + [f"male{i}" for i in range(1, 12)]
    for seed in range(5):
        s = assign_splits(groups, "pool", seed, stratum=lambda g: g.rstrip("0123456789"))
        count = lambda split, voice: sum(v == split and g.startswith(voice) for g, v in s.items())  # noqa: E731
        assert [count(sp, v) for sp in ("train", "val", "test") for v in ("female", "male")] == [6, 8, 1, 2, 2, 1]


def test_max_groups_takes_equal_numbers_per_gender(tmp_path):
    speakers = {f"p{100 + i}": "F" if i < 14 else "M" for i in range(20)}
    (tmp_path / "speaker-info.txt").write_text(
        "ID  AGE  GENDER  ACCENTS\n" + "".join(f"{s}  22  {g}    English\n" for s, g in speakers.items()))
    for seed in range(5):
        chosen = prepare_dry.select_groups(sorted(speakers), SOURCES["vctk"], random.Random(seed), tmp_path)
        assert len(chosen) == 12 and sum(speakers[s] == "F" for s in chosen) == 6


def test_check_splits_catches_a_group_in_two_splits():
    splits = {"synth": {"g0": "train"}}
    rows = [dict(source="synth", group_id="g0", split="train"), dict(source="synth", group_id="g0", split="val")]
    with pytest.raises(AssertionError):
        check_splits(rows, splits)


def test_end_to_end(raw_data):
    run(raw_data)
    rows = manifest(raw_data)
    assert list(rows[0]) == MANIFEST_COLS
    assert {r["source"] for r in rows} == set(TEST_SOURCES)

    splits = json.loads((raw_data / "splits.json").read_text())
    check_splits(rows, splits)
    assert set(splits["synth"].values()) == {"train", "val", "test"}
    assert set(splits["own"].values()) == {"heldout"}
    assert {r["role"] for r in rows if r["source"] == "own"} == {"heldout"}

    for r in rows:
        x, sr = sf.read(raw_data / "dry" / r["file"], dtype="int16")
        info = sf.info(raw_data / "dry" / r["file"])
        assert sr == SR and info.channels == 1 and info.subtype == "PCM_16"
        assert len(x) / SR == pytest.approx(float(r["duration"]), abs=1e-3)
        # 4 s tone, silence trimmed to within a frame or two.
        assert 3.8 <= len(x) / SR <= 4.2
        assert r["file"].startswith(f"{r['source']}/{r['group_id']}/")

    # Target 0.5 min per pool source: stops shortly after 30 s.
    synth_min = sum(float(r["duration"]) for r in rows if r["source"] == "synth") / 60
    assert 0.5 <= synth_min < 0.5 + 5 * 4.2 / 60

    stats = list(csv.DictReader(open(raw_data / "dry" / "stats.csv")))
    assert all(int(s["raw_clipped"]) == 0 for s in stats)


def test_heavily_clipped_files_are_skipped(raw_data):
    t = np.arange(4 * 44100) / 44100
    sf.write(raw_data / "raw" / "synth" / "g0" / "clipped.wav", np.clip(2 * np.sin(2 * np.pi * 220 * t), -1, 1), 44100)
    cfg = TEST_SOURCES["synth"] | dict(target_min=10)  # big enough to read every file
    rows, stats, _ = prepare_dry.prepare_source("synth", cfg, raw_data, 0)
    clipped = [st for st in stats if st["raw_file"] == "g0/clipped.wav"]
    assert len(clipped) == 1 and clipped[0]["raw_clipped"] > 100 and clipped[0]["duration_kept"] == 0
    assert not any("clipped" in r["file"] for r in rows)
    assert all(st["duration_kept"] > 0 for st in stats if st["raw_file"] != "g0/clipped.wav")


def test_rerun_keeps_splits_and_rejects_new_groups(raw_data):
    run(raw_data)
    first = (raw_data / "splits.json").read_text()
    run(raw_data, "synth")
    assert (raw_data / "splits.json").read_text() == first
    assert {r["source"] for r in manifest(raw_data)} == set(TEST_SOURCES), "other sources kept"

    (raw_data / "raw" / "synth" / "g0").rename(raw_data / "raw" / "synth" / "g_new")
    with pytest.raises(SystemExit, match="not in splits.json"):
        run(raw_data, "synth")


# Paths copied from the real downloads.
@pytest.mark.parametrize("source, rel, group", [
    ("vocalset", "FULL/female2/scales/straight/f2_scales_straight_a.wav", "female2"),
    ("vocalset", "FULL/male10/scales/fast_piano/scales_c_fast_piano_e.wav", "male10"),
    ("vocalset", "FULL/male7/arpeggios/vibrato/m7_arpeggios_vibrato_e.wav", None),
    ("vocalset", "FULL/female1/arpeggios/vibrado/f1_arpeggios_vibrado_a.wav", None),
    ("vocalset", "FULL/male11/long_tones/trillo/m11_long_trillo_u.wav", None),
    ("vocalset", "FULL/male7/scales/lip_trill/m7_scales_lip_trill_e.wav", None),
    ("vocalset", "FULL/female7/excerpts/spoken/f7_dona_spoken.wav", None),
    ("idmt_guitar", "IDMT-SMT-GUITAR_V2/dataset4/Career SG/slow/reggae_ska/audio/reggae_4_100BPM.wav", "reggae_4"),
    ("idmt_guitar", "IDMT-SMT-GUITAR_V2/dataset4/acoustic_pickup/fast/country_folk/audio/folk_2_75BPM.wav", "folk_2"),
    ("idmt_guitar", "IDMT-SMT-GUITAR_V2/dataset4/acoustic_mic/slow/metal/audio/metal_2_150BPM.wav", None),
    ("idmt_guitar", "IDMT-SMT-GUITAR_V2/dataset2/audio/AR_Lick11_FN.wav", None),
    ("guitarset", "04_Funk1-114-Ab_solo_mix.wav", "player04"),
    ("guitarset", "00_SS1-100-C#_comp_mix.wav", "player00"),
    ("vctk", "wav48_silence_trimmed/p236/p236_433_mic2.flac", "p236"),
    ("vctk", "wav48_silence_trimmed/s5/s5_012_mic2.flac", "s5"),
    ("vctk", "wav48_silence_trimmed/p236/p236_433_mic1.flac", None),
])
def test_parsers_on_real_names(tmp_path, source, rel, group):
    assert SOURCES[source]["group"](tmp_path / rel, tmp_path) == group


@pytest.mark.parametrize("source, rel", [
    ("vocalset", "FULL/female2/scales/f2_scales_a.wav"),
    ("idmt_guitar", "IDMT-SMT-GUITAR_V2/dataset4/Career SG/slow/pop/audio/pop_1.wav"),
    ("guitarset", "player4_Funk1_mix.wav"),
    ("vctk", "wav48_silence_trimmed/p236/p237_433_mic2.flac"),
    ("vctk", "wav48/p236/p236_433_mic2.flac"),
])
def test_parsers_reject_unexpected_names(tmp_path, source, rel):
    with pytest.raises(ValueError):
        SOURCES[source]["group"](tmp_path / rel, tmp_path)


@pytest.mark.skipif(not (REAL_DATA / "splits.json").exists(), reason="real data not prepared yet")
def test_real_data_has_no_group_in_two_splits():
    splits = json.loads((REAL_DATA / "splits.json").read_text())
    check_splits(manifest(REAL_DATA), splits)
