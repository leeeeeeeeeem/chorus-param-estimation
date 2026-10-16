use std::path::PathBuf;
use std::process::Command;

use chorus_render::dsp::chorus::{Chorus, MAX_DELAY_MS, MAX_DEPTH_MS};
use chorus_render::dsp::delay_line::DelayLine;
use chorus_render::dsp::lfo::Lfo;
use chorus_render::{LATENCY_SAMPLES, Params, render, render_stereo, start_phase};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

const SR: f32 = 22050.0;
/// 20 ms, a whole number of samples at 22.05 kHz.
const DELAY_20MS: usize = 441;

/// Deterministic noise in [-amp, amp).
fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1 << 23) as f32 - 1.0) * amp
        })
        .collect()
}

fn params(rate_hz: f32, depth_ms: f32, delay_ms: f32, mix: f32, lfo_phase: f32) -> Params {
    Params { rate_hz, depth_ms, delay_ms, mix, lfo_phase, warmup_samples: 0 }
}

fn argmax_abs(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs())).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn write_i16(path: &PathBuf, samples: &[i16]) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SR as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
}

/// Reads a stereo int16 WAV as (left, right).
fn read_stereo_i16(path: &PathBuf) -> (Vec<i16>, Vec<i16>) {
    let mut reader = WavReader::open(path).unwrap();
    assert_eq!(reader.spec().channels, 2);
    let s: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    s.chunks(2).map(|f| (f[0], f[1])).unzip()
}

/// Runs the `render` binary on int16 samples. Returns (left, right).
fn run_bin(name: &str, dry: &[i16], json: &str) -> (Vec<i16>, Vec<i16>) {
    let (dry_path, json_path, wet_path) =
        (tmp(&format!("{name}_dry.wav")), tmp(&format!("{name}.json")), tmp(&format!("{name}_wet.wav")));
    write_i16(&dry_path, dry);
    std::fs::write(&json_path, json).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_render"))
        .args([&dry_path, &json_path, &wet_path])
        .status()
        .unwrap();
    assert!(status.success(), "render failed for {name}");
    read_stereo_i16(&wet_path)
}

// Raw DSP: an impulse comes out at the same index at mix 0, and exactly the
// set delay later at mix 1 with no depth, on both channels. Anything else is
// latency.
#[test]
fn measured_latency_matches_constant() {
    let n = 4000;
    let at = 1000;
    for (mix, extra) in [(0.0, 0), (1.0, DELAY_20MS)] {
        let mut chorus = Chorus::new(SR);
        chorus.set_depth_ms(0.0);
        chorus.set_delay_ms(20.0);
        chorus.set_mix(mix);
        let (left, right): (Vec<f32>, Vec<f32>) = (0..n)
            .map(|i| {
                let x = if i == at { 1.0 } else { 0.0 };
                chorus.process(x, x)
            })
            .unzip();
        for (name, out) in [("left", &left), ("right", &right)] {
            let latency = argmax_abs(out) as isize - (at + extra) as isize;
            assert_eq!(latency, LATENCY_SAMPLES as isize, "{name}, mix {mix}");
        }
    }
}

#[test]
fn mix_zero_nulls_against_dry_through_binary() {
    let dry: Vec<i16> = noise(20000, 5, 1.0).iter().map(|&x| (x * 30000.0) as i16).collect();
    let warmup = 2048;
    let json = format!(
        r#"{{"rate_hz": 3.3, "depth_ms": 5.0, "delay_ms": 20.0, "mix": 0.0,
            "lfo_phase": 0.7, "warmup_samples": {warmup}}}"#
    );
    let (left, right) = run_bin("null", &dry, &json);
    assert_eq!(left.len(), dry.len() - warmup);
    assert_eq!(left, dry[warmup..], "mix 0 does not null on the left");
    assert_eq!(right, dry[warmup..], "mix 0 does not null on the right");
}

// After warmup trimming, the wet impulse is exactly delay_ms after the dry one.
#[test]
fn binary_keeps_alignment_with_warmup() {
    let mut dry = vec![0i16; 10000];
    dry[5000] = 20000;
    let warmup = 2048;
    let json = format!(
        r#"{{"rate_hz": 1.0, "depth_ms": 0.0, "delay_ms": 20.0, "mix": 1.0,
            "lfo_phase": 0.0, "warmup_samples": {warmup}}}"#
    );
    let (left, right) = run_bin("align", &dry, &json);
    for wet in [left, right] {
        let peak = (0..wet.len()).max_by_key(|&i| wet[i].unsigned_abs()).unwrap();
        assert_eq!(peak, 5000 - warmup + DELAY_20MS);
        assert_eq!(wet[peak], 20000);
    }
}

/// One channel of the chorus, built from the DSP's own parts with the LFO
/// phase set directly.
fn reference(dry: &[f32], p: &Params, phase: f32) -> Vec<f32> {
    let mut dl = DelayLine::new((SR * ((MAX_DELAY_MS + MAX_DEPTH_MS) / 1000.0)).ceil() as usize);
    let mut lfo = Lfo::new(SR);
    lfo.set_rate(p.rate_hz);
    lfo.set_phase(phase);
    let delay = SR * (p.delay_ms / 1000.0);
    let depth = SR * (p.depth_ms / 1000.0);
    dry.iter()
        .map(|&x| {
            let d = delay + depth * lfo.next();
            dl.write(x);
            let wet = dl.read_frac(d);
            x * (1.0 - p.mix) + wet * p.mix
        })
        .collect()
}

/// Both channels: the right LFO runs a quarter cycle ahead of the left.
fn stereo_reference(dry: &[f32], p: &Params, phase: f32) -> Vec<[f32; 2]> {
    let left = reference(dry, p, phase);
    let right = reference(dry, p, phase + 0.25);
    left.into_iter().zip(right).map(|(l, r)| [l, r]).collect()
}

#[test]
fn lfo_starts_at_requested_phase() {
    let dry = noise(30000, 11, 0.9);
    for (rate, phase) in [(0.1, 0.0), (0.1, 0.93), (0.8, 0.25), (3.7, 0.5), (5.0, 0.999)] {
        let p = params(rate, 5.0, 15.0, 0.8, phase);
        assert_eq!(start_phase(&p, SR), phase, "no warmup, no shift");
        let got = render(&dry, SR, &p).unwrap();
        assert_eq!(got, stereo_reference(&dry, &p, phase), "rate {rate} phase {phase}");
    }
}

// With warmup, the phase must be the target at the first *output* sample.
#[test]
fn lfo_phase_counts_from_after_warmup() {
    let warmup = 2048;
    let dry = noise(warmup + 20000, 13, 0.9);
    for (rate, target) in [(0.1, 0.0), (2.0, 0.6), (5.0, 0.95)] {
        let mut p = params(rate, 5.0, 12.0, 1.0, target);
        p.warmup_samples = warmup;
        let start = start_phase(&p, SR);
        let expected = stereo_reference(&dry, &p, start)[warmup..].to_vec();
        assert_eq!(render(&dry, SR, &p).unwrap(), expected, "rate {rate}");

        // An LFO started at `start` matches one started at the target once
        // the warmup has passed.
        let mut from_start = Lfo::new(SR);
        let mut at_target = Lfo::new(SR);
        for lfo in [&mut from_start, &mut at_target] {
            lfo.set_rate(rate);
        }
        from_start.set_phase(start);
        at_target.set_phase(target);
        for _ in 0..warmup {
            from_start.next();
        }
        for n in 0..1000 {
            let (a, b) = (from_start.next(), at_target.next());
            assert!((a - b).abs() < 1e-3, "rate {rate}, sample {n}: {a} vs {b}");
        }
    }
}

#[test]
fn rejects_out_of_range_params() {
    let dry = vec![0.0; 100];
    for p in [
        params(0.05, 2.0, 15.0, 0.5, 0.0),
        params(6.0, 2.0, 15.0, 0.5, 0.0),
        params(1.0, -0.1, 15.0, 0.5, 0.0),
        params(1.0, 5.1, 15.0, 0.5, 0.0),
        params(1.0, 2.0, 6.9, 0.5, 0.0),
        params(1.0, 2.0, 25.1, 0.5, 0.0),
        params(1.0, 2.0, 15.0, 1.1, 0.0),
        params(1.0, 2.0, 15.0, 0.5, 1.0),
        params(1.0, 2.0, 15.0, f32::NAN, 0.0),
    ] {
        assert!(render(&dry, SR, &p).is_err(), "{p:?}");
    }
    let mut p = params(1.0, 2.0, 15.0, 0.5, 0.0);
    p.warmup_samples = 101;
    assert!(render(&dry, SR, &p).is_err(), "warmup longer than input");
}

#[test]
fn fresh_instance_gives_identical_output() {
    let dry = noise(10000, 17, 0.9);
    let p = params(1.3, 3.0, 9.0, 0.6, 0.4);
    assert_eq!(render(&dry, SR, &p).unwrap(), render(&dry, SR, &p).unwrap());
}

#[test]
fn channels_differ_only_with_depth() {
    let dry = noise(10000, 19, 0.9);
    let same = render(&dry, SR, &params(1.3, 0.0, 9.0, 0.6, 0.4)).unwrap();
    assert!(same.iter().all(|[l, r]| l == r), "depth 0 must give identical channels");
    let wide = render(&dry, SR, &params(1.3, 3.0, 9.0, 0.6, 0.4)).unwrap();
    assert!(wide.iter().any(|[l, r]| (l - r).abs() > 0.01), "depth 3 ms must give different channels");
}

#[test]
fn stereo_input_feeds_each_channel_its_own_signal() {
    let x = noise(4000, 11, 0.5);
    let p = params(1.3, 3.0, 9.0, 0.6, 0.4);
    let mono = render(&x, SR, &p).unwrap();
    let left_only: Vec<[f32; 2]> = x.iter().map(|&v| [v, 0.0]).collect();
    let out = render_stereo(&left_only, SR, &p).unwrap();
    assert!(out.iter().all(|f| f[1] == 0.0), "silent right input must give a silent right output");
    assert!(out.iter().zip(&mono).all(|(a, b)| a[0] == b[0]), "left output must not depend on the right input");
    let swapped: Vec<[f32; 2]> = x.iter().map(|&v| [0.0, v]).collect();
    let out = render_stereo(&swapped, SR, &p).unwrap();
    assert!(out.iter().all(|f| f[0] == 0.0));
    assert!(out.iter().zip(&mono).all(|(a, b)| a[1] == b[1]));
}

#[test]
fn binary_gives_the_same_output_for_mono_and_dual_mono_files() {
    let dry: Vec<i16> = noise(3000, 5, 0.4).iter().map(|v| (v * 32767.0) as i16).collect();
    let json = r#"{"rate_hz": 2.0, "depth_ms": 4.0, "delay_ms": 12.0, "mix": 0.5, "lfo_phase": 0.1}"#;
    let mono = run_bin("dual_mono_ref", &dry, json);
    let path = tmp("dual_mono_in.wav");
    let spec = WavSpec { channels: 2, sample_rate: SR as u32, bits_per_sample: 16, sample_format: SampleFormat::Int };
    let mut w = WavWriter::create(&path, spec).unwrap();
    for &s in &dry {
        w.write_sample(s).unwrap();
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
    std::fs::write(tmp("dual_mono.json"), json).unwrap();
    let out = tmp("dual_mono_out.wav");
    let status = Command::new(env!("CARGO_BIN_EXE_render")).args([&path, &tmp("dual_mono.json"), &out]).status().unwrap();
    assert!(status.success());
    assert_eq!(read_stereo_i16(&out), mono);
}
