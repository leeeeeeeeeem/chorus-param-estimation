import csv
import json

import numpy as np
import pytest
import soundfile as sf

import make_real_stereo
from audio import read as read_audio, to_mono
from conftest import REAL_DATA

SR_RAW = 44100
LAG = 44  # 1 ms at 44.1 kHz


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def raw(tmp_path):
    """Two val takes of player01: pickup, and a microphone that hears it 1 ms later, duller, with room noise."""
    data = tmp_path / "data"
    (data / "raw" / "guitarset").mkdir(parents=True)
    (data / "raw" / "guitarset_mic").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("01_BN1-129-Eb_comp", "01_Funk2-108-Eb_solo", "02_Rock1-90-C_comp"):
        t = np.arange(8 * SR_RAW) / SR_RAW
        x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((196.0, 247.0, 294.0, 392.0), 1))
        x = (0.3 * x * (t > 0.5) + rng.normal(0, 0.002, len(t))).astype(np.float32)
        mic = np.convolve(np.r_[np.zeros(LAG), x[:-LAG]], np.ones(8) / 8, "same") + rng.normal(0, 0.003, len(t))
        sf.write(data / "raw" / "guitarset" / f"{name}_mix.wav", x, SR_RAW)
        sf.write(data / "raw" / "guitarset_mic" / f"{name}_mic.wav", mic.astype(np.float32), SR_RAW)
    (data / "splits.json").write_text(json.dumps({"guitarset": {"player01": "val", "player02": "train"}}))
    return data


def test_pairs_share_crop_and_settings_and_differ_only_in_the_right_input(raw):
    out = raw / "rs"
    make_real_stereo.main(["--data", str(raw), "--out", str(out), "--n", "6", "--workers", "2"])
    rows = read(out / "params.csv")
    assert len(rows) == 12 and {r["group_id"] for r in rows} == {"player01"}, "only the val player"
    for mono, mic in zip(rows[::2], rows[1::2]):
        assert (mono["stereo_kind"], mic["stereo_kind"]) == ("mono", "mic")
        for k in ("dry_source", "crop_offset", "gain_db", "rate_hz", "depth_ms", "delay_ms", "mix", "lfo_phase"):
            assert mono[k] == mic[k], k
        assert float(mic["stereo_amount"]) == pytest.approx(1.0, abs=0.05), "measured mic lag"
        d_mono = sf.read(out / "val" / f"{mono['id']}_dry.wav", dtype="int16")[0]
        d_mic = sf.read(out / "val" / f"{mic['id']}_dry.wav", dtype="int16")[0]
        assert d_mono.ndim == 1 and d_mic.shape[1] == 2
        assert np.array_equal(d_mic[:, 0], d_mono), "left input is the same pickup crop"
        assert not np.array_equal(d_mic[:, 1], d_mono)


def test_missing_microphone_files_are_an_error(raw):
    next((raw / "raw" / "guitarset_mic").glob("01_*")).unlink()
    with pytest.raises(SystemExit, match="microphone files missing"):
        make_real_stereo.main(["--data", str(raw), "--out", str(raw / "rs"), "--n", "2"])



TAKE = "01_BN1-129-Eb_comp"


@pytest.mark.skipif(not (REAL_DATA / "raw" / "guitarset_mic" / f"{TAKE}_mic.wav").exists(),
                    reason="GuitarSet microphone files not downloaded")
def test_real_mic_lag_is_acoustic_not_a_pitch_copy():
    """Plain cross-correlation reads 8.7 ms on this take (a pitch copy); the
    microphone is really about 2 ms behind the pickup."""
    a = to_mono(*read_audio(REAL_DATA / "raw" / "guitarset" / f"{TAKE}_mix.wav"))
    b = to_mono(*read_audio(REAL_DATA / "raw" / "guitarset_mic" / f"{TAKE}_mic.wav"))
    assert 1.0 <= 1000 * make_real_stereo.lag_samples(a, b) / make_real_stereo.SR <= 4.0
