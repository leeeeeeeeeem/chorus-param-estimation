// The chorus DSP from the plugin repo, compiled from its source files.
// It is a private module of the plugin crate, so it can't be a path dependency.
#[allow(dead_code)]
#[path = "../../crazy-chorus/src/dsp/mod.rs"]
pub mod dsp;

use dsp::chorus::Chorus;
use serde::Deserialize;

/// Measured offset between input and output at mix 0 (see tests/render.rs).
pub const LATENCY_SAMPLES: usize = 0;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub rate_hz: f32,
    pub depth_ms: f32,
    pub delay_ms: f32,
    pub mix: f32,
    /// Left LFO phase in cycles at the first output sample.
    pub lfo_phase: f32,
    /// Input samples processed before the output starts, then dropped.
    #[serde(default)]
    pub warmup_samples: usize,
}

impl Params {
    /// Same ranges as the plugin. Rejected rather than clamped, so the
    /// label always matches the audio.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("rate_hz", self.rate_hz, 0.1, 5.0),
            ("depth_ms", self.depth_ms, 0.0, 5.0),
            ("delay_ms", self.delay_ms, 7.0, 25.0),
            ("mix", self.mix, 0.0, 1.0),
        ];
        for (name, v, lo, hi) in checks {
            if !(lo..=hi).contains(&v) {
                return Err(format!("{name} = {v} is outside [{lo}, {hi}]"));
            }
        }
        if !(0.0..1.0).contains(&self.lfo_phase) {
            return Err(format!("lfo_phase = {} is outside [0, 1)", self.lfo_phase));
        }
        Ok(())
    }
}

/// `Chorus` has no phase setter, so the LFO is moved by running silence first.
/// Returns how many silent samples to run so the left LFO is closest to
/// `target` after those plus `warmup` steps, and the phase it lands on.
/// Mirrors the f32 accumulator in `dsp::lfo` so the result is exact.
pub fn preroll(rate_hz: f32, sample_rate: f32, warmup: usize, target: f32) -> (usize, f32) {
    let inc = (rate_hz / sample_rate).max(0.0).min(0.5);
    let period = (1.0 / inc).ceil() as usize + 1;
    let mut phase: f32 = 0.0;
    let mut best = (0, 0.0, f32::INFINITY);
    for step in 0..=warmup + period {
        if step >= warmup {
            let d = (phase - target).abs();
            let d = d.min(1.0 - d);
            if d < best.2 {
                best = (step - warmup, phase, d);
            }
        }
        phase += inc;
        if phase >= 1.0 {
            phase -= 1.0;
        }
    }
    (best.0, best.1)
}

/// Renders the left output of a fresh `Chorus` with mono `dry` on both inputs.
/// Output is `dry.len() - warmup_samples` long.
pub fn render(dry: &[f32], sample_rate: f32, p: &Params) -> Result<Vec<f32>, String> {
    p.validate()?;
    if p.warmup_samples > dry.len() {
        return Err(format!(
            "warmup_samples {} is longer than the input ({})",
            p.warmup_samples,
            dry.len()
        ));
    }
    let mut chorus = Chorus::new(sample_rate);
    chorus.set_rate(p.rate_hz);
    chorus.set_depth_ms(p.depth_ms);
    chorus.set_delay_ms(p.delay_ms);
    chorus.set_mix(p.mix);

    let (silence, _) = preroll(p.rate_hz, sample_rate, p.warmup_samples, p.lfo_phase);
    for _ in 0..silence {
        chorus.process(0.0, 0.0);
    }

    let mut out: Vec<f32> = dry.iter().map(|&x| chorus.process(x, x).0).collect();
    for _ in 0..LATENCY_SAMPLES {
        out.push(chorus.process(0.0, 0.0).0);
    }
    Ok(out.split_off(p.warmup_samples + LATENCY_SAMPLES))
}
