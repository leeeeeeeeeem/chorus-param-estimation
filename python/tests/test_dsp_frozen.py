import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DSP = REPO / "crazy-chorus-dsp"


def dsp_checksum(root: Path) -> str:
    """sha256 over "<file sha256>  ./<path>" lines, files sorted by path. Same as
    `cd crazy-chorus-dsp && find . -type f | sort | xargs shasum -a 256 | shasum -a 256`."""
    lines = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = "./" + f.relative_to(root).as_posix()
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def test_frozen_dsp_matches_the_checksum_in_spec():
    m = re.search(r"Checksum: `([0-9a-f]{64})`", (REPO / "SPEC.md").read_text())
    assert m, "SPEC.md must record the frozen DSP checksum"
    assert dsp_checksum(DSP) == m.group(1), (
        "crazy-chorus-dsp/ differs from the frozen copy recorded in SPEC.md. "
        "The DSP is frozen on purpose; see SPEC.md before changing it.")
