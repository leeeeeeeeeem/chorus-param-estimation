"""Source config. See SPEC.md."""

from pathlib import Path

AUDIO_EXTS = {".wav", ".flac"}

# NC-ND: never in demo outputs.
DEMO_EXCLUDED = {"idmt_guitar"}


def group_from_folder(path: Path, root: Path) -> str:
    """raw/<source>/<group>/.../file.wav -> <group>."""
    return path.relative_to(root).parts[0]


def not_inspected(source: str):
    def parse(path: Path, root: Path) -> str | None:
        raise NotImplementedError(
            f"{source}: group_id parsing is not written yet. "
            f"Run python/inspect_raw.py {source} and agree on it first."
        )

    return parse


# group(path, root) returns the group_id, or None to exclude the file.
SOURCES = {
    "vocalset": dict(role="pool", licence="CC BY 4.0", target_min=60, group=not_inspected("vocalset")),
    "idmt_guitar": dict(role="pool", licence="CC BY-NC-ND 4.0", target_min=45, group=not_inspected("idmt_guitar")),
    "guitarset": dict(role="pool", licence="MIT (to confirm)", target_min=30, group=not_inspected("guitarset")),
    "synth": dict(role="pool", licence="own", target_min=45, group=group_from_folder),
    "own": dict(role="heldout", licence="own", target_min=20, group=group_from_folder),
    "vctk": dict(role="heldout", licence="CC BY 4.0", target_min=30, max_groups=12, group=not_inspected("vctk")),
}
