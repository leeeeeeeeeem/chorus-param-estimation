# Chorus parameter estimation

Train a small CNN to recover chorus settings from 2 s audio clips.

Two input modes:
- `dry_wet`: dry clip and wet clip, stacked as 2 channels.
- `wet`: wet clip only, 1 channel.

## DSP

The chorus is `Chorus` from `crazy-chorus-dsp` (`../crazy-chorus/dsp`, a path
dependency, read-only here).

Per sample: `delay = delay_ms + depth_ms * sin(2π·phase)`, read with cubic
Hermite, then `out = dry·(1-mix) + wet·mix`. The right LFO is fixed at
+0.25 cycle from the left.

Smoothing lives in the plugin wrapper, not in `Chorus`. Setters apply
immediately, so `render` has no smoothers to settle. It sets all parameters
and the LFO phase once on a fresh `Chorus` before the first sample.

## Parameters

Ranges are the plugin's, which match the DSP clamps.

| Name       | Range        | Sampling    | Role     |
|------------|--------------|-------------|----------|
| rate_hz    | 0.1 to 5     | log-uniform | label    |
| depth_ms   | 0 to 5       | uniform     | label    |
| delay_ms   | 7 to 25      | uniform     | label    |
| mix        | 0 to 1       | uniform     | label    |
| lfo_phase  | 0 to 1 cycle | uniform     | nuisance |
| source, group, crop offset | random | | nuisance |
| gain_db    | -12 to -1 dBFS peak | uniform | nuisance |

`render` rejects values outside these ranges instead of letting the DSP clamp
them, so labels always match what was rendered.

`lfo_phase` is the left LFO phase at the first output sample. It is saved in
the CSV but is not a training target.

Training targets are scaled to [0, 1]: rate on a log scale, the rest linear.

Known blind spots, kept in the data on purpose:
- mix near 0: nothing else is observable.
- depth near 0: rate is not observable.
- rate near 0.1 Hz: a 2 s clip sees only 0.2 of a cycle.

## Mono / stereo

Mono. Dry audio is mono. It goes into both DSP inputs (as the plugin does for
a mono input) and only the left output is kept. The right output only adds a
fixed 90° LFO offset, so it is left out for now.

## Audio

- 22.05 kHz mono. Clips are 2 s (44100 samples), int16 WAV.
- Each clip is rendered with 2048 samples (93 ms) of real dry audio before
  it, longer than the 30 ms max delay, then the warmup is cut. So the wet
  clip has no silent start.
- Measured latency of the DSP is 0 samples (tested).

## Sources

All sources must be dry: no reverb, delay, chorus or other effects.

| Source       | Role    | Target  | group_id | Licence |
|--------------|---------|---------|----------|---------|
| vocalset     | pool    | ~1 h    | singer   | CC BY 4.0 |
| idmt_guitar  | pool    | ~45 min | piece    | CC BY-NC-ND 4.0 |
| guitarset    | pool    | ~30 min | player   | MIT (to confirm) |
| synth        | pool    | ~45 min | patch    | own |
| own          | heldout | 15-20 min | session | own |
| vctk         | heldout | ~30 min | speaker  | CC BY 4.0 |

- vocalset: group is the singer folder. Technique folders `vibrato`,
  `vibrado` (typo), `trill`, `trillo` and `lip_trill` are excluded (baked-in
  modulation), and `spoken` too (speech stays unseen until VCTK).
- idmt_guitar: subset 4 only, without `acoustic_mic` (a microphone, not DI).
  That leaves 2 tempi x 3 guitars. Group is the piece name (`rock_1`), so all
  six takes of a piece share a group. Never used in demo outputs.
- guitarset: `audio_mono-pickup_mix` only. Group is the player (`player04`).
  All players play the same 30 lead sheets.
- vctk: v0.92, mic2 only, 12 speakers.
- synth and own: my files, laid out as `data/raw/synth/<patch>/*.wav` and
  `data/raw/own/<session>/*.wav`. The folder name is the group.

group_id parsing for downloaded sources is written only after looking at the
real file names (`python/inspect_raw.py`). Unexpected names are an error.

## Dry preparation (`python/prepare_dry.py`)

1. Read, downmix to mono, resample to 22.05 kHz. Raw files with more than
   100 full-scale samples (real clipping) are skipped.
2. Peak-normalise the file, then mark 50 ms frames below -45 dBFS RMS as
   silent. Active runs closer than 0.3 s are merged. Runs shorter than 2.2 s
   are dropped. Each run becomes one segment.
3. Normalise each segment to -20 dBFS RMS, capped at -1 dBFS peak.
4. Select segments round-robin across groups, seeded, until the source target
   duration is reached.
5. Write `data/dry/<source>/<group_id>/<name>.wav` (int16),
   `data/manifest.csv` (file, source, group_id, split, role, duration,
   licence) and `data/dry/stats.csv` (raw peak, clipped samples, silence
   dropped).

## Splits

- `data/splits.json` maps source -> group -> split. It is written once,
  seeded. A source that is already in it is never re-split; a new group in an
  existing source is an error.
- Pool sources: 70/15/15 of groups, at least one group in val and test.
- VocalSet is stratified by voice (female/male), so each split gets both:
  train 6 F + 8 M, val 1 F + 2 M, test 2 F + 1 M.
- Held-out sources: every group is `heldout`.
- A test asserts that no group appears in two splits.

## Dataset (`python/make_dataset.py`)

- 20000 train, 2000 val, 2000 test, 2000 heldout pairs.
- Per pair: pick a segment from the split weighted by duration, a random crop
  offset, random parameters and LFO phase, and a random peak gain.
- Output in `data/dataset/`: `{split}/{id}_dry.wav`, `{split}/{id}_wet.wav`,
  `params.csv` and `make_dataset.log`.
- `params.csv` has a `demo_ok` column, false for idmt_guitar.

Size: about 4.6 GB for all 26k pairs.

## Features

Log-magnitude STFT computed on CPU in the loader: n_fft 1024, hop 256, Hann.
1024 points at 22.05 kHz gives 21.5 Hz bins, enough to see comb notches
from a 25 ms delay (40 Hz apart).
