pub use crazy_chorus_dsp as dsp;

use dsp::chorus::{Chorus, MAX_DELAY_MS, MAX_DEPTH_MS, MIN_DELAY_MS};
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
            ("depth_ms", self.depth_ms, 0.0, MAX_DEPTH_MS),
            ("delay_ms", self.delay_ms, MIN_DELAY_MS, MAX_DELAY_MS),
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

/// LFO phase at the first input sample, so it reaches `lfo_phase` after the
/// warmup. Off by under 1e-4 cycle from f32 rounding in the LFO.
pub fn start_phase(p: &Params, sample_rate: f32) -> f32 {
    let advance = p.warmup_samples as f64 * p.rate_hz as f64 / sample_rate as f64;
    (p.lfo_phase as f64 - advance).rem_euclid(1.0) as f32
}

/// Renders a fresh `Chorus` with mono `dry` on both inputs. Returns
/// `[left, right]` frames, `dry.len() - warmup_samples` long.
pub fn render(dry: &[f32], sample_rate: f32, p: &Params) -> Result<Vec<[f32; 2]>, String> {
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
    chorus.set_lfo_phase(start_phase(p, sample_rate));

    let input = dry.iter().copied().chain(std::iter::repeat_n(0.0, LATENCY_SAMPLES));
    let mut out: Vec<[f32; 2]> = input
        .map(|x| {
            let (l, r) = chorus.process(x, x);
            [l, r]
        })
        .collect();
    Ok(out.split_off(p.warmup_samples + LATENCY_SAMPLES))
}
