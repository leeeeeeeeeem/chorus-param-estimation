from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sources import group_from_folder

REPO = Path(__file__).resolve().parents[2]
REAL_DATA = REPO / "data"

# Small test config: folder-per-group layout for every source.
TEST_SOURCES = {
    "synth": dict(role="pool", licence="own", target_min=0.5, group=group_from_folder),
    "idmt_guitar": dict(role="pool", licence="CC BY-NC-ND 4.0", target_min=0.5, group=group_from_folder),
    "own": dict(role="heldout", licence="own", target_min=0.3, group=group_from_folder),
}


def tone(seconds: float, sr: int, f0: float, rng) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 6))
    return (x * 0.3 + rng.normal(0, 0.003, len(t))).astype(np.float32)


def write_raw(data: Path, source: str, groups: int, files: int, sr=44100, seed=0) -> None:
    """raw/<source>/<group>/take_N.wav: stereo, tone with silence around it."""
    rng = np.random.default_rng(seed)
    for g in range(groups):
        for n in range(files):
            x = np.concatenate([np.zeros(sr // 2), tone(4.0, sr, rng.uniform(80, 600), rng), np.zeros(sr)])
            path = data / "raw" / source / f"g{g}" / f"take_{n}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.stack([x, x * 0.8], axis=1), sr)


@pytest.fixture
def raw_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    write_raw(data, "synth", groups=5, files=3, seed=1)
    write_raw(data, "idmt_guitar", groups=4, files=2, sr=48000, seed=2)
    write_raw(data, "own", groups=2, files=2, seed=3)
    return data
