"""Show what is in data/raw/<source>/ before writing its group_id parser.

python python/inspect_raw.py vocalset           # layout and sample names
python python/inspect_raw.py vocalset --parse   # also the parsed groups
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import soundfile as sf

from sources import AUDIO_EXTS, SOURCES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=SOURCES)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()

    root = args.data / "raw" / args.source
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    audio = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
    print(f"{root}: {len(files)} files, {len(audio)} audio")
    print("extensions:", dict(Counter(p.suffix.lower() for p in files).most_common()))

    print("\nfolders (depth 3, audio files below each):")
    dirs = Counter(str(Path(*p.relative_to(root).parts[:-1][:3])) for p in audio)
    for d, n in sorted(dirs.items()):
        print(f"  {n:6d}  {d}")

    print(f"\n{args.samples} random audio paths:")
    for p in sorted(random.Random(0).sample(audio, min(args.samples, len(audio)))):
        print("  ", p.relative_to(root))

    infos = Counter()
    total = 0.0
    for p in audio:
        i = sf.info(p)
        infos[(i.samplerate, i.channels, i.subtype)] += 1
        total += i.duration
    print(f"\ntotal audio: {total / 3600:.2f} h")
    print("rate, channels, format:", dict(infos))

    if args.parse:
        parse = SOURCES[args.source]["group"]
        groups = Counter()
        print(f"\nparsed groups for {args.samples} random files:")
        for p in sorted(random.Random(1).sample(audio, min(args.samples, len(audio)))):
            print(f"  {parse(p, root)!s:20}  {p.relative_to(root)}")
        for p in audio:
            groups[parse(p, root)] += 1
        print(f"\n{len(groups)} groups (None = excluded):")
        for g, n in sorted(groups.items(), key=lambda kv: str(kv[0])):
            print(f"  {n:6d}  {g}")


if __name__ == "__main__":
    main()
