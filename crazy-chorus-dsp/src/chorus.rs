use super::delay_line::DelayLine;
use super::lfo::Lfo;

pub const MIN_DELAY_MS: f32 = 7.0;
pub const MAX_DELAY_MS: f32 = 25.0;
pub const MAX_DEPTH_MS: f32 = 5.0;
const STEREO_OFFSET: f32 = 0.25;

const DEFAULT_RATE: f32 = 0.8;
const DEFAULT_DELAY: f32 = 15.;
const DEFAULT_DEPTH: f32 = 2.;
const DEFAULT_MIX: f32 = 0.5;

pub struct Chorus {
    delay_l: DelayLine,
    delay_r: DelayLine,
    lfo_l: Lfo,
    lfo_r: Lfo,
    sample_rate: f32,
    delay_ms: f32,
    delay_samples: f32,
    depth_ms: f32,
    depth_samples: f32,
    mix: f32,
}

fn ms_to_samples(ms: f32, sample_rate: f32) -> f32 {
    sample_rate * (ms / 1000.)
}

fn max_delay_samples(sample_rate: f32) -> usize {
    ms_to_samples(MAX_DELAY_MS + MAX_DEPTH_MS, sample_rate).ceil() as usize
}

impl Chorus {
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.);
        let samples = max_delay_samples(sample_rate);
        let mut chorus = Chorus {
            delay_l: DelayLine::new(samples),
            delay_r: DelayLine::new(samples),
            lfo_l: Lfo::new(sample_rate),
            lfo_r: Lfo::new(sample_rate),
            sample_rate,
            delay_ms: DEFAULT_DELAY,
            delay_samples: ms_to_samples(DEFAULT_DELAY, sample_rate),
            depth_ms: DEFAULT_DEPTH,
            depth_samples: ms_to_samples(DEFAULT_DEPTH, sample_rate),
            mix: DEFAULT_MIX,
        };
        chorus.set_rate(DEFAULT_RATE);
        chorus.reset();
        chorus
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        assert!(sample_rate > 0.);
        self.sample_rate = sample_rate;
        let samples = max_delay_samples(sample_rate);
        self.delay_l = DelayLine::new(samples);
        self.delay_r = DelayLine::new(samples);
        self.lfo_l.set_sample_rate(sample_rate);
        self.lfo_r.set_sample_rate(sample_rate);
        self.set_delay_ms(self.delay_ms);
        self.set_depth_ms(self.depth_ms);
    }

    pub fn set_rate(&mut self, rate_hz: f32) {
        self.lfo_l.set_rate(rate_hz);
        self.lfo_r.set_rate(rate_hz);
    }

    pub fn set_delay_ms(&mut self, delay_ms: f32) {
        debug_assert!(!delay_ms.is_nan());
        let delay_ms = delay_ms.max(MIN_DELAY_MS).min(MAX_DELAY_MS);
        let delay_samples = ms_to_samples(delay_ms, self.sample_rate);
        self.delay_ms = delay_ms;
        self.delay_samples = delay_samples;
    }

    pub fn set_depth_ms(&mut self, depth_ms: f32) {
        debug_assert!(!depth_ms.is_nan());
        let depth_ms = depth_ms.max(0.).min(MAX_DEPTH_MS);
        let depth_samples = ms_to_samples(depth_ms, self.sample_rate);
        self.depth_ms = depth_ms;
        self.depth_samples = depth_samples;
    }

    pub fn set_mix(&mut self, mix: f32) {
        debug_assert!(!mix.is_nan());
        self.mix = mix.max(0.).min(1.)
    }

    /// Sets the left LFO phase in cycles; the right stays STEREO_OFFSET ahead.
    pub fn set_lfo_phase(&mut self, phase: f32) {
        self.lfo_l.set_phase(phase);
        self.lfo_r.set_phase(phase + STEREO_OFFSET);
    }

    pub fn reset(&mut self) {
        self.delay_l.reset();
        self.delay_r.reset();
        self.lfo_l.reset();
        self.lfo_r.reset();
        self.lfo_r.set_phase(STEREO_OFFSET);
    }

    pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let l_lfo = self.lfo_l.next();
        let r_lfo = self.lfo_r.next();
        let l_delay = self.delay_samples + self.depth_samples * l_lfo;
        let r_delay = self.delay_samples + self.depth_samples * r_lfo;
        self.delay_l.write(left);
        self.delay_r.write(right);
        let l_wet = self.delay_l.read_frac(l_delay);
        let r_wet = self.delay_r.read_frac(r_delay);
        let l_out = left * (1. - self.mix) + l_wet * self.mix;
        let r_out = right * (1. - self.mix) + r_wet * self.mix;
        if self.mix == 0. {
            (left, right)
        } else {
            (l_out, r_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn assert_within(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{msg}: expected {expected} ± {tol}, got {actual}"
        );
    }

    // Writes an impulse, pushes it back `delay` samples, and checks it is still
    // there. Integer delays make Hermite return the tap exactly.
    fn assert_delay_reachable(dl: &mut DelayLine, delay: usize, msg: &str) {
        dl.write(1.0);
        for _ in 0..delay {
            dl.write(0.0);
        }
        assert_eq!(dl.read_frac(delay as f32), 1.0, "{msg}: delay {delay}");
    }

    #[test]
    fn new_sizes_delay_lines_for_max_delay_plus_depth() {
        for sample_rate in [44100.0, 48000.0, 96000.0] {
            let mut chorus = Chorus::new(sample_rate);
            let longest = (sample_rate * (MAX_DELAY_MS + MAX_DEPTH_MS) / 1000.0).ceil() as usize;
            assert_delay_reachable(&mut chorus.delay_l, longest, &format!("left @ {sample_rate}"));
            assert_delay_reachable(&mut chorus.delay_r, longest, &format!("right @ {sample_rate}"));
        }
    }

    #[test]
    fn new_applies_stereo_offset() {
        let mut chorus = Chorus::new(48000.0);
        // sin(2π · 0) = 0 on the left, sin(2π · 0.25) = 1 on the right.
        assert_within(chorus.lfo_l.next(), 0.0, 1e-6, "left starts at phase 0");
        assert_within(chorus.lfo_r.next(), 1.0, 1e-6, "right starts 90° ahead");
    }

    #[test]
    fn set_lfo_phase_keeps_stereo_offset() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_lfo_phase(0.25);
        // sin(2π · 0.25) = 1 on the left, sin(2π · 0.5) = 0 on the right.
        assert_within(chorus.lfo_l.next(), 1.0, 1e-6, "left at phase 0.25");
        assert_within(chorus.lfo_r.next(), 0.0, 1e-6, "right stays 90° ahead");
    }

    #[test]
    fn new_runs_lfos_at_default_rate() {
        let sample_rate = 48000.0;
        let mut chorus = Chorus::new(sample_rate);
        let n = 1000;
        for _ in 0..n {
            chorus.lfo_l.next();
        }
        let expected = (TAU * n as f32 * DEFAULT_RATE / sample_rate).sin();
        assert_within(chorus.lfo_l.next(), expected, 1e-3, &format!("left LFO after {n} samples"));
    }

    #[test]
    fn reset_clears_delay_lines() {
        let mut chorus = Chorus::new(48000.0);
        for i in 0..500 {
            chorus.delay_l.write(i as f32 + 1.0);
            chorus.delay_r.write(-(i as f32) - 1.0);
        }
        chorus.reset();
        for d in 0..500 {
            assert_eq!(chorus.delay_l.read(d), 0.0, "left delay {d}");
            assert_eq!(chorus.delay_r.read(d), 0.0, "right delay {d}");
        }
    }

    #[test]
    fn reset_restores_stereo_offset() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_rate(3.0);
        for _ in 0..12345 {
            chorus.lfo_l.next();
            chorus.lfo_r.next();
        }
        chorus.reset();
        assert_within(chorus.lfo_l.next(), 0.0, 1e-6, "left after reset");
        assert_within(chorus.lfo_r.next(), 1.0, 1e-6, "right after reset");
    }

    #[test]
    fn set_sample_rate_resizes_delay_lines() {
        let mut chorus = Chorus::new(44100.0);
        chorus.set_sample_rate(96000.0);
        let longest = (96000.0_f32 * (MAX_DELAY_MS + MAX_DEPTH_MS) / 1000.0).ceil() as usize;
        assert_delay_reachable(&mut chorus.delay_l, longest, "left @ 96k");
        assert_delay_reachable(&mut chorus.delay_r, longest, "right @ 96k");
    }

    #[test]
    fn set_sample_rate_keeps_delay_and_depth_in_ms() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_delay_ms(20.0);
        chorus.set_depth_ms(4.0);
        chorus.set_sample_rate(96000.0);

        assert_eq!(chorus.sample_rate, 96000.0);
        assert_eq!(chorus.delay_ms, 20.0);
        assert_eq!(chorus.depth_ms, 4.0);
        // 20 ms and 4 ms at 96 kHz.
        assert_within(chorus.delay_samples, 1920.0, 1e-2, "delay samples");
        assert_within(chorus.depth_samples, 384.0, 1e-2, "depth samples");
    }

    #[test]
    fn set_delay_ms_converts_and_clamps() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_delay_ms(10.0);
        assert_eq!(chorus.delay_ms, 10.0);
        assert_within(chorus.delay_samples, 480.0, 1e-2, "10 ms @ 48k");

        for (input, expected) in [
            (0.0, MIN_DELAY_MS),
            (-5.0, MIN_DELAY_MS),
            (f32::NEG_INFINITY, MIN_DELAY_MS),
            (100.0, MAX_DELAY_MS),
            (f32::INFINITY, MAX_DELAY_MS),
        ] {
            chorus.set_delay_ms(input);
            assert_eq!(chorus.delay_ms, expected, "set_delay_ms({input})");
        }
    }

    #[test]
    fn set_depth_ms_converts_and_clamps() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_depth_ms(3.0);
        assert_eq!(chorus.depth_ms, 3.0);
        assert_within(chorus.depth_samples, 144.0, 1e-2, "3 ms @ 48k");

        for (input, expected) in [
            (-1.0, 0.0),
            (f32::NEG_INFINITY, 0.0),
            (10.0, MAX_DEPTH_MS),
            (f32::INFINITY, MAX_DEPTH_MS),
        ] {
            chorus.set_depth_ms(input);
            assert_eq!(chorus.depth_ms, expected, "set_depth_ms({input})");
        }
    }

    #[test]
    fn set_mix_clamps_to_unit_range() {
        let mut chorus = Chorus::new(48000.0);
        for (input, expected) in [
            (0.0, 0.0),
            (0.3, 0.3),
            (1.0, 1.0),
            (-0.5, 0.0),
            (2.0, 1.0),
            (f32::INFINITY, 1.0),
        ] {
            chorus.set_mix(input);
            assert_eq!(chorus.mix, expected, "set_mix({input})");
        }
    }

    // The debug_asserts fire on NaN in debug builds, so these only run under
    // `cargo test --release`, where the release fallback (max then min) applies.
    #[test]
    #[cfg(not(debug_assertions))]
    fn nan_setters_fall_back_to_lower_bound_in_release() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_delay_ms(f32::NAN);
        assert_eq!(chorus.delay_ms, MIN_DELAY_MS, "delay");
        chorus.set_depth_ms(f32::NAN);
        assert_eq!(chorus.depth_ms, 0.0, "depth");
        chorus.set_mix(f32::NAN);
        assert_eq!(chorus.mix, 0.0, "mix");
    }

    // --- process ---

    /// Deterministic pseudo-random values in [-1, 1) (no `rand` dependency).
    struct Noise(u32);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1 << 23) as f32 - 1.0
        }
    }

    #[test]
    fn silence_in_gives_silence_out() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_depth_ms(MAX_DEPTH_MS);
        chorus.set_mix(1.0);
        for n in 0..10_000 {
            assert_eq!(chorus.process(0.0, 0.0), (0.0, 0.0), "sample {n}");
        }
    }

    #[test]
    fn mix_zero_is_bit_exact_passthrough() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_rate(5.0);
        chorus.set_depth_ms(MAX_DEPTH_MS);
        chorus.set_mix(0.0);

        let specials = [0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, f32::MIN_POSITIVE, 1e6];
        let mut noise = Noise(1);
        let inputs = specials.into_iter().chain((0..5000).map(|_| noise.next()));
        for (n, x) in inputs.enumerate() {
            // Different values per channel so swapped channels are caught too.
            let (l, r) = chorus.process(x, -x);
            assert_eq!(l.to_bits(), x.to_bits(), "left, sample {n}: in {x}, out {l}");
            assert_eq!(r.to_bits(), (-x).to_bits(), "right, sample {n}: in {}, out {r}", -x);
        }
    }

    // Running at mix 0 must still feed the delay lines and advance the LFOs,
    // so turning the mix up afterwards sounds the same as if it had always
    // been up: no stale audio, no frozen modulation.
    #[test]
    fn mix_zero_keeps_state_running() {
        let mut was_dry = Chorus::new(48000.0);
        let mut always_wet = Chorus::new(48000.0);
        for chorus in [&mut was_dry, &mut always_wet] {
            chorus.set_rate(2.0);
            chorus.set_depth_ms(3.0);
        }
        was_dry.set_mix(0.0);
        always_wet.set_mix(1.0);

        let mut noise = Noise(7);
        for _ in 0..5000 {
            let x = noise.next();
            was_dry.process(x, x);
            always_wet.process(x, x);
        }

        was_dry.set_mix(1.0);
        for n in 0..5000 {
            let x = noise.next();
            assert_eq!(was_dry.process(x, x), always_wet.process(x, x), "sample {n} after mix up");
        }
    }

    #[test]
    fn pure_wet_without_depth_is_plain_delay() {
        for sample_rate in [44100.0, 48000.0, 96000.0] {
            let mut chorus = Chorus::new(sample_rate);
            chorus.set_delay_ms(10.0);
            chorus.set_depth_ms(0.0);
            chorus.set_mix(1.0);
            // 10 ms is a whole number of samples at all three rates.
            let delay = (sample_rate / 100.0) as usize;

            for n in 0..delay * 2 {
                let x = if n == 0 { 1.0 } else { 0.0 };
                let (l, r) = chorus.process(x, x);
                let expected = if n == delay { 1.0 } else { 0.0 };
                let msg = format!("{sample_rate} Hz, sample {n}");
                assert_within(l, expected, 1e-4, &format!("left, {msg}"));
                assert_within(r, expected, 1e-4, &format!("right, {msg}"));
            }
        }
    }

    #[test]
    fn half_mix_is_average_of_dry_and_wet() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_delay_ms(10.0);
        chorus.set_depth_ms(0.0);
        chorus.set_mix(0.5);
        let delay = 480;

        for n in 0..delay * 2 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (l, _) = chorus.process(x, x);
            let expected = if n == 0 || n == delay { 0.5 } else { 0.0 };
            assert_within(l, expected, 1e-4, &format!("sample {n}"));
        }
    }

    #[test]
    fn stereo_offset_makes_channels_differ() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_depth_ms(MAX_DEPTH_MS);
        chorus.set_mix(1.0);

        // Identical input on both channels: any difference comes from the LFOs.
        let mut max_diff: f32 = 0.0;
        for n in 0..48000 {
            let x = (TAU * 440.0 * n as f32 / 48000.0).sin();
            let (l, r) = chorus.process(x, x);
            max_diff = max_diff.max((l - r).abs());
        }
        assert!(max_diff > 0.1, "channels nearly identical: max diff {max_diff}");
    }

    #[test]
    fn modulation_moves_the_delay() {
        // With depth, a constant-pitch sine comes out pitch-shifted back and
        // forth, so the wet signal no longer matches a fixed delay of the input.
        let sample_rate = 48000.0;
        let mut modulated = Chorus::new(sample_rate);
        let mut fixed = Chorus::new(sample_rate);
        for (chorus, depth) in [(&mut modulated, MAX_DEPTH_MS), (&mut fixed, 0.0)] {
            chorus.set_rate(2.0);
            chorus.set_depth_ms(depth);
            chorus.set_mix(1.0);
        }

        let mut max_diff: f32 = 0.0;
        for n in 0..48000 {
            let x = (TAU * 440.0 * n as f32 / sample_rate).sin();
            let (m, _) = modulated.process(x, x);
            let (f, _) = fixed.process(x, x);
            max_diff = max_diff.max((m - f).abs());
        }
        assert!(max_diff > 0.1, "depth has no effect: max diff {max_diff}");
    }

    #[test]
    fn extreme_and_jumping_params_stay_finite_and_bounded() {
        let mut noise = Noise(42);
        for sample_rate in [44100.0, 48000.0, 96000.0] {
            let mut chorus = Chorus::new(sample_rate);
            for block in 0..200 {
                // Jump every param to a new value, including out-of-range ones.
                chorus.set_rate(noise.next() * 50.0);
                chorus.set_delay_ms(noise.next() * 40.0);
                chorus.set_depth_ms(noise.next() * 10.0);
                chorus.set_mix(noise.next() * 2.0);
                for n in 0..64 {
                    let (l, r) = chorus.process(noise.next(), noise.next());
                    // Cubic Hermite can overshoot full-scale input a little.
                    for y in [l, r] {
                        assert!(
                            y.is_finite() && y.abs() <= 1.5,
                            "{sample_rate} Hz, block {block}, sample {n}: {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reset_silences_the_tail() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_mix(1.0);
        let mut noise = Noise(3);
        for _ in 0..5000 {
            chorus.process(noise.next(), noise.next());
        }
        chorus.reset();
        for n in 0..5000 {
            assert_eq!(chorus.process(0.0, 0.0), (0.0, 0.0), "sample {n} after reset");
        }
    }

    // Release-only for the same reason as the NaN setter test above.
    #[test]
    #[cfg(not(debug_assertions))]
    fn nan_params_keep_output_finite_in_release() {
        let mut chorus = Chorus::new(48000.0);
        chorus.set_rate(f32::NAN);
        chorus.set_delay_ms(f32::NAN);
        chorus.set_depth_ms(f32::NAN);
        chorus.set_mix(f32::NAN);
        let mut noise = Noise(9);
        for n in 0..5000 {
            let (l, r) = chorus.process(noise.next(), noise.next());
            assert!(l.is_finite() && r.is_finite(), "sample {n}: ({l}, {r})");
        }
    }
}
