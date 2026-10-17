# Chorus Parameter Estimation

Can a small neural network listen to a clip processed by my chorus plugin
(crazy-chorus) and tell which knob settings were used? The knobs are rate,
depth, delay and mix.

This is an educational project, I did it to learn/practice audio ML. It is not meant to be a useful tool since it only knows this one plugin.

## What I did

- Took dry recordings (VocalSet singing, IDMT and GuitarSet guitar, VCTK
  speech as a held-out test) and cleaned them up: silence removed, clipped
  files and vibrato dropped, split by singer/player so nobody is in two splits.
- Ran them through the plugin's real DSP with random settings, rendering at
  44.1 kHz. 26000 pairs of 2 s clips, each with its settings.
- Turned the audio into spectrograms and trained a small CNN to predict the
  four knobs, in two modes: `wet`, where it hears only the processed clip (the
  real use case), and `dry_wet`, where it also hears the original and can
  compare the two. `dry_wet` is the easy case and mostly a reference. Both
  final models are `wet`.
- Compared against two baselines: always guessing the average, and a
  handmade cross-correlation method that also gets to hear the dry signal.
- Tried different ways of showing the stereo output to the model: left
  channel only, left and right, or mid and side. Mid/side won by far (val
  overall error 0.033 against 0.060 for left/right and 0.147 for left only,
  see Results). With a mono source the dry cancels in L minus R, so the side
  is pure chorus, and the model needs to be given it directly: a magnitude
  spectrogram drops phase, so it can't rebuild the side from L and R itself.
- Made it robust to mono files and to stereo sources (panned, doubled, a real
  guitar recorded with pickup and microphone at the same time).

The DSP is a frozen copy of the plugin's `dsp/` crate in `crazy-chorus-dsp/`,
so this repo builds on its own. See `SPEC.md` for where it comes from.

## Results

Errors on the test set, model hearing only the processed clip. All numbers
are mean absolute errors (MAE) over the 2000 test clips. Rate is a ratio
(x1.16 means off by a factor of 1.16 on average). Overall scales each knob's
MAE to 0..1 over its range (rate on a log axis) and averages the four: 0 is
perfect, 0.25 is what always guessing the middle gets.

| | rate | depth | delay | mix | overall |
|---|---|---|---|---|---|
| guessing the average | x2.6 | 1.27 ms | 4.5 ms | 0.25 | 0.251 |
| stereo file | x1.16 | 0.20 ms | 1.3 ms | 0.03 | 0.044 |
| mono file | x1.6 | 0.77 ms | 2.5 ms | 0.09 | 0.125 |

The stereo row is the average of three models trained the same way; the mono
row is a separate model trained on mono-style input.

- The side channel (L minus R) is the main cue. With it the model beats the
  cross-correlation method on rate, depth and mix without hearing the dry
  signal. It is still worse on delay.
- Without a side channel (mono files) it is about three times less accurate.
- At mix close to 1 the delay can't be heard without the dry signal, so the
  model can't get it.
- Models trained only on mono sources fail completely on stereo sources.
  Training on panned, doubled and real two-mic material fixes most of it.
- A wider model helped. Training longer, a bigger spectrogram and adding phase
  did not.

## The final models

Both are the same CNN: the log spectrogram (513 frequency bins x 173 frames
of 11.6 ms) goes through four blocks of two 3x3 convolutions and a max-pool
(32 to 256 channels), is averaged down to a 4x5 grid of frequency x time, so
where the comb notches sit and how they move over the 2 s both survive, and a
small dense head outputs the four knobs on a 0..1 scale. About 2.5 M
parameters, trained with squared error.

- Stereo files: mid and side as two input channels, trained on mono sources
  plus panned, doubled and real pickup + microphone stereo. Three of these are
  trained and their predictions averaged.
- Mono files: one channel, a random blend of L and R per clip, so it has no
  side channel to depend on and works on a mono downmix or one channel alone.

The split exists because the side channel only means "pure chorus" when the
plugin's input is mono. A model trained only on that case reads a silent side
(mono file) as no chorus, and extra side from a stereo source as more chorus.

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/download_data.sh
.venv/bin/python python/prepare_dry.py
.venv/bin/python python/make_dataset.py
.venv/bin/python python/train.py --mode wet --view ms --width 32 --epochs 30 --lr 2e-4
cargo test --workspace && .venv/bin/pytest
```
