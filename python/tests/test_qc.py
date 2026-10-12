import soundfile as sf

import prepare_dry
import qc
from audio import SR
from conftest import TEST_SOURCES


def test_qc_dumps_excerpts_and_prints_stats(raw_data, capsys):
    prepare_dry.main(["--data", str(raw_data)], sources=TEST_SOURCES)
    qc.main(["--data", str(raw_data), "--per-source", "3", "--seconds", "2"])
    out = capsys.readouterr().out
    for source in TEST_SOURCES:
        files = list((raw_data / "qc" / source).glob("*.wav"))
        assert len(files) == 3
        assert all(sf.info(f).frames == 2 * SR for f in files)
        assert source in out
