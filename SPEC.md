# Chorus parameter estimation

Train a small CNN to recover chorus settings from 2 s audio clips.

Two input modes:
- `dry_wet`: dry clip and wet clip, stacked as 2 channels.
- `wet`: wet clip only, 1 channel.

## DSP

The chorus is `Chorus` in `../crazy-chorus/src/dsp/` (read-only). It is a
private module inside the plugin crate, not a separate crate, so a Cargo path
dependency cannot reach it. `render` compiles those files directly with
`#[path]`. No copy is kept here.

Per sample: `delay = delay_ms + depth_ms * sin(2π·phase)`, read with cubic
Hermite, then `out = dry·(1-mix) + wet·mix`. The right LFO is fixed at
+0.25 cycle from the left.

Smoothing lives in the plugin wrapper, not in `Chorus`. Setters apply
immediately, so `render` has no smoothers to settle. It sets all parameters
once on a fresh `Chorus` before the first sample.

## Parameters

Ranges are the plugin's (`src/lib.rs`), which match the DSP clamps.

| Name       | Range        | Sampling    | Role     |
|------------|--------------|-------------|----------|
| rate_hz    | 0.1 to 5     | log-uniform | label    |
| depth_ms   | 0 to 5       | uniform     | label    |
| delay_ms   | 7 to 25      | uniform     | label    |
| mix        | 0 to 1       | uniform     | label    |
| lfo_phase  | 0 to 1 cycle | uniform     | nuisance |
| source     | dry recording and offset | random | nuisance |
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

- 48 kHz. Recordings at other rates are resampled.
- Clips are 2 s (96000 samples), int16 WAV.
- Each clip is rendered with 2048 samples of real dry audio before it
  (warmup), longer than the 30 ms max delay, then the warmup is cut. So the
  wet clip has no silent start.
- Measured latency of the DSP is 0 samples (tested).

## Dataset

- Dry recordings in `data/dry/`, split 80/10/10 by recording with a fixed seed.
- About 20k train, 2k val, 2k test pairs.
- Output in `data/dataset/`: `{split}/{id}_dry.wav`, `{split}/{id}_wet.wav`,
  `params.csv`, and `make_dataset.log`.

## Features

Log-magnitude STFT computed on CPU in the loader: n_fft 2048, hop 512, Hann.
2048 points gives 23 Hz bins, enough to see comb notches from a 25 ms delay
(40 Hz apart).
