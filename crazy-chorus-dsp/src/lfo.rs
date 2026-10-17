use std::f32::consts::TAU;

pub struct Lfo {
    phase: f32,
    phase_inc: f32,
    sample_rate: f32,
}

impl Lfo {
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.0);
        Lfo {
            phase: 0.0,
            phase_inc: 0.0,
            sample_rate,
        }
    }

    pub fn set_rate(&mut self, rate_hz: f32) {
        debug_assert!(!rate_hz.is_nan());
        self.phase_inc = (rate_hz / self.sample_rate).max(0.0).min(0.5);
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        assert!(sample_rate > 0.0);
        let rate_hz = self.sample_rate * self.phase_inc;
        self.sample_rate = sample_rate;
        self.set_rate(rate_hz);
    }

    pub fn set_phase(&mut self, phase: f32) {
        debug_assert!(!phase.is_nan());
        self.phase = phase.rem_euclid(1.0);
    }

    /// returns current value then advances (always [-1, 1])
    pub fn next(&mut self) -> f32 {
        let cur = (self.phase * TAU).sin();
        self.phase += self.phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0
        }
        cur
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn set_rate_computes_phase_inc() {
        let mut lfo = Lfo::new(44100.0);
        for rate in [0.1, 1.0, 5.0] {
            lfo.set_rate(rate);
            assert_close(lfo.phase_inc, rate / 44100.0, &format!("rate {rate}"));
        }
    }

    #[test]
    fn set_rate_clamps_to_zero_and_nyquist() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_rate(-3.0);
        assert_eq!(lfo.phase_inc, 0.0, "negative rate");
        lfo.set_rate(22050.0);
        assert_eq!(lfo.phase_inc, 0.5, "exactly nyquist");
        lfo.set_rate(1e9);
        assert_eq!(lfo.phase_inc, 0.5, "above nyquist");
        lfo.set_rate(f32::INFINITY);
        assert_eq!(lfo.phase_inc, 0.5, "infinite rate");
    }

    // The debug_assert in set_rate fires on NaN in debug builds, so this only
    // runs under `cargo test --release`, where the release fallback applies.
    #[test]
    #[cfg(not(debug_assertions))]
    fn set_rate_nan_stops_lfo_in_release() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_rate(1.0);
        lfo.set_rate(f32::NAN);
        assert_eq!(lfo.phase_inc, 0.0);
    }

    #[test]
    fn set_sample_rate_keeps_rate_in_hz() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_rate(1.0);
        lfo.set_sample_rate(96000.0);
        assert_eq!(lfo.sample_rate, 96000.0);
        assert_close(lfo.phase_inc, 1.0 / 96000.0, "44.1k -> 96k");
        lfo.set_sample_rate(48000.0);
        assert_close(lfo.phase_inc, 1.0 / 48000.0, "96k -> 48k");
    }

    #[test]
    fn set_sample_rate_lowering_keeps_nyquist_bound() {
        let mut lfo = Lfo::new(96000.0);
        lfo.set_rate(48000.0);
        assert_eq!(lfo.phase_inc, 0.5);
        lfo.set_sample_rate(44100.0);
        assert!(lfo.phase_inc <= 0.5, "phase_inc {}", lfo.phase_inc);
    }

    #[test]
    fn set_phase_wraps_into_unit_range() {
        let mut lfo = Lfo::new(44100.0);
        for (input, expected) in [(0.0, 0.0), (0.25, 0.25), (1.25, 0.25), (-0.25, 0.75), (3.0, 0.0)] {
            lfo.set_phase(input);
            assert_close(lfo.phase, expected, &format!("set_phase({input})"));
        }
    }

    // Looser than `assert_close`: f32 `TAU` is not exactly 2π, so sin at a
    // "zero" phase lands around 1e-7, not 0.
    fn assert_within(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{msg}: expected {expected} ± {tol}, got {actual}"
        );
    }

    #[test]
    fn next_returns_value_at_current_phase_before_advancing() {
        for (phase, expected) in [(0.0, 0.0), (0.25, 1.0), (0.5, 0.0), (0.75, -1.0)] {
            let mut lfo = Lfo::new(48000.0);
            lfo.set_rate(1.0);
            lfo.set_phase(phase);
            assert_within(lfo.next(), expected, 1e-6, &format!("phase {phase}"));
        }
    }

    #[test]
    fn next_at_phase_one_matches_phase_zero() {
        let mut lfo = Lfo::new(48000.0);
        lfo.set_rate(1.0);
        lfo.phase = 1.0; // reachable via set_phase's rem_euclid edge case
        assert_within(lfo.next(), 0.0, 1e-6, "phase 1.0");
        assert!(lfo.phase < 1.0, "phase not wrapped: {}", lfo.phase);
    }

    #[test]
    fn next_output_and_phase_stay_in_range() {
        for rate in [0.1, 1.0, 5.0, 20.0, 1000.0, 24000.0, 1e9] {
            let mut lfo = Lfo::new(48000.0);
            lfo.set_rate(rate);
            for i in 0..200_000 {
                let v = lfo.next();
                assert!(v.is_finite() && (-1.0..=1.0).contains(&v), "rate {rate}, sample {i}: {v}");
                assert!(
                    (0.0..1.0).contains(&lfo.phase),
                    "rate {rate}, sample {i}: phase {}",
                    lfo.phase
                );
            }
        }
    }

    #[test]
    fn next_period_matches_rate() {
        let sample_rate = 48000.0;
        for rate in [1.0, 5.0] {
            let mut lfo = Lfo::new(sample_rate);
            lfo.set_rate(rate);

            // Indices of rising zero crossings (previous < 0, current >= 0).
            let mut crossings = Vec::new();
            let mut prev = lfo.next();
            for i in 1..(sample_rate as usize * 4) {
                let cur = lfo.next();
                if prev < 0.0 && cur >= 0.0 {
                    crossings.push(i);
                }
                prev = cur;
            }

            let expected = sample_rate / rate;
            assert!(crossings.len() >= 2, "rate {rate}: only {} crossings", crossings.len());
            for pair in crossings.windows(2) {
                let period = (pair[1] - pair[0]) as f32;
                // The f32 phase accumulator rounds every add, which skews the
                // rate by up to ~0.15% at 1 Hz / 48 kHz. 0.5% still catches real
                // bugs (wrong formula, double advance, missing wrap).
                assert_within(period, expected, expected * 0.005, &format!("rate {rate}"));
            }
        }
    }

    #[test]
    fn zero_rate_gives_constant_output() {
        let mut lfo = Lfo::new(48000.0);
        lfo.set_rate(0.0);
        lfo.set_phase(0.3);
        let first = lfo.next();
        for i in 0..1000 {
            assert_eq!(lfo.next(), first, "sample {i}");
        }
    }

    #[test]
    fn reset_returns_phase_to_zero() {
        let mut lfo = Lfo::new(48000.0);
        lfo.set_rate(3.0);
        lfo.set_phase(0.6);
        for _ in 0..1234 {
            lfo.next();
        }
        lfo.reset();
        assert_eq!(lfo.phase, 0.0);
        assert_within(lfo.next(), 0.0, 1e-6, "first sample after reset");
    }

    #[test]
    fn reset_keeps_rate_and_repeats_sequence() {
        let mut lfo = Lfo::new(48000.0);
        lfo.set_rate(3.0);
        for _ in 0..5000 {
            lfo.next();
        }
        lfo.reset();

        let mut fresh = Lfo::new(48000.0);
        fresh.set_rate(3.0);
        for i in 0..5000 {
            assert_eq!(lfo.next(), fresh.next(), "sample {i}");
        }
    }
}
