"""Source config. See SPEC.md."""

import re
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac"}

# NC-ND: never in demo outputs.
DEMO_EXCLUDED = {"idmt_guitar"}


def group_from_folder(path: Path, root: Path) -> str:
    """raw/<source>/<group>/.../file.wav -> <group>."""
    return path.relative_to(root).parts[0]


# Takes with baked-in pitch or amplitude modulation, and speech (kept unseen
# for the VCTK held-out set). "vibrado" is a typo folder in VocalSet.
VOCALSET_EXCLUDED = {"vibrato", "vibrado", "trill", "trillo", "lip_trill", "spoken"}


def vocalset_group(path: Path, root: Path) -> str | None:
    """FULL/<singer>/<category>/<technique>/<file>.wav -> <singer>."""
    parts = path.relative_to(root).parts
    if parts[0] != "FULL":
        return None
    if len(parts) != 5:
        raise ValueError(f"unexpected VocalSet path: {path}")
    return None if parts[3] in VOCALSET_EXCLUDED else parts[1]


def idmt_group(path: Path, root: Path) -> str | None:
    """dataset4/<guitar>/<tempo>/<genre>/audio/<piece>_<bpm>BPM.wav -> <piece>.
    Subset 4 only, without acoustic_mic (a microphone, not DI)."""
    parts = path.relative_to(root).parts
    if len(parts) < 2 or parts[1] != "dataset4":
        return None
    m = re.fullmatch(r"(.+)_\d+BPM\.wav", parts[-1])
    if len(parts) != 7 or not m:
        raise ValueError(f"unexpected IDMT path: {path}")
    return None if parts[2] == "acoustic_mic" else m.group(1)


def guitarset_group(path: Path, root: Path) -> str:
    """<player>_<song>-<bpm>-<key>_<comp|solo>_mix.wav -> player<NN>."""
    m = re.fullmatch(r"(\d\d)_.+_(comp|solo)_mix\.wav", path.name)
    if not m:
        raise ValueError(f"unexpected GuitarSet name: {path}")
    return f"player{m.group(1)}"


def vctk_group(path: Path, root: Path) -> str | None:
    """wav48_silence_trimmed/<speaker>/<speaker>_<NNN>_mic<1|2>.flac -> <speaker>, mic2 only."""
    parts = path.relative_to(root).parts
    m = re.fullmatch(r"([ps]\d+)_\d+_(mic[12])\.flac", parts[-1])
    if len(parts) != 3 or parts[0] != "wav48_silence_trimmed" or not m or m.group(1) != parts[1]:
        raise ValueError(f"unexpected VCTK path: {path}")
    return parts[1] if m.group(2) == "mic2" else None


def vctk_gender(speaker: str, root: Path) -> str:
    """F or M, from speaker-info.txt."""
    for line in (root / "speaker-info.txt").read_text().splitlines()[1:]:
        cols = line.split()
        if cols and cols[0] == speaker:
            return cols[2]
    raise ValueError(f"{speaker} is not in speaker-info.txt")


# group(path, root) returns the group_id, or None to exclude the file.
# stratum(group, root) labels groups that splits and max_groups keep balanced.
SOURCES = {
    "vocalset": dict(role="pool", licence="CC BY 4.0", target_min=60, group=vocalset_group,
                     stratum=lambda g, root: g.rstrip("0123456789")),  # female / male
    "idmt_guitar": dict(role="pool", licence="CC BY-NC-ND 4.0", target_min=45, group=idmt_group),
    "guitarset": dict(role="pool", licence="MIT (to confirm)", target_min=30, group=guitarset_group),
    "synth": dict(role="pool", licence="own", target_min=45, group=group_from_folder),
    "own": dict(role="heldout", licence="own", target_min=20, group=group_from_folder),
    "vctk": dict(role="heldout", licence="CC BY 4.0", target_min=30, max_groups=12, group=vctk_group,
                 stratum=vctk_gender),
}
