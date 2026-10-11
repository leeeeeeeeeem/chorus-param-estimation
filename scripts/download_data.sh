#!/usr/bin/env bash
# Downloads and unpacks the public sources into data/raw/<source>/.
# Each zip is deleted after a successful unzip (disk is tight); set
# KEEP_ZIPS=1 to keep them. A finished source is skipped on re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
    local source=$1 url=$2 zip=$3
    local dir=data/raw/$source
    if [[ -e $dir/.unpacked ]]; then
        echo "$source: already unpacked, skipping"
        return
    fi
    mkdir -p "$dir"
    echo "$source: downloading"
    curl -L -C - -o "$dir/$zip" "$url"
    echo "$source: unzipping"
    unzip -q -n "$dir/$zip" -d "$dir"
    touch "$dir/.unpacked"
    if [[ ${KEEP_ZIPS:-0} != 1 ]]; then
        rm "$dir/$zip"
    fi
}

fetch vocalset "https://zenodo.org/records/1442513/files/VocalSet11.zip?download=1" VocalSet11.zip
fetch idmt_guitar "https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip?download=1" IDMT-SMT-GUITAR_V2.zip
fetch guitarset "https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1" audio_mono-pickup_mix.zip

# VCTK v0.92 (manual): https://datashare.ed.ac.uk/handle/10283/3443
# Download VCTK-Corpus-0.92.zip (about 11 GB) by hand. To save disk, list it
# first with `unzip -l`, then extract only the mic2 audio into data/raw/vctk/.

mkdir -p data/raw/synth data/raw/own
echo "done. Put synth renders in data/raw/synth/<patch>/ and own recordings in data/raw/own/<session>/"
