pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    mask: usize,
    max_delay_samples: usize
}

impl DelayLine {
    pub fn new(max_delay_samples: usize) -> Self {
        assert!(max_delay_samples > 0);
        let len = (max_delay_samples + 3).next_power_of_two();
        DelayLine {
            buffer: vec![0.0; len],
            write_pos: 0,
            mask: len - 1,
            max_delay_samples: max_delay_samples
        }
    }

    pub fn write(&mut self, x: f32) {
        self.buffer[self.write_pos] = x;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    /// read(0) is the most recent write
    pub fn read(&self, delay: usize) -> f32 {
        debug_assert!(delay <= self.mask);
        self.buffer[self.write_pos.wrapping_sub(1 + delay) & self.mask]
    }

    pub fn read_frac(&self, delay: f32) -> f32 {
        debug_assert!(!delay.is_nan());
        let delay = delay.clamp(1.0, self.max_delay_samples as f32);
        let d = delay as usize;
        let frac = delay - d as f32;
        let x0 = self.read(d - 1);
        let x1 = self.read(d);
        let x2 = self.read(d + 1);
        let x3 = self.read(d + 2);
        let c1 = 0.5 * (x2 - x0);
        let c2 = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3;
        let c3 = 0.5 * (x3 - x0) + 1.5 * (x1 - x2);
        ((c3 * frac + c2) * frac + c1) * frac + x1
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_comes_out_after_delay() {
        let mut dl = DelayLine::new(16);
        for n in 0..32 {
            dl.write(if n == 0 { 1.0 } else { 0.0 });
            let expected = if n == 5 { 1.0 } else { 0.0 };
            assert_eq!(dl.read(5), expected, "sample {n}");
        }
    }

    #[test]
    fn delay_is_correct_after_wraparound() {
        let mut dl = DelayLine::new(10);
        let len = dl.mask + 1;
        for i in 0..len * 3 + 5 {
            dl.write(i as f32);
            for d in 0..=dl.mask.min(i) {
                assert_eq!(dl.read(d), (i - d) as f32, "write {i}, delay {d}");
            }
        }
    }

    #[test]
    fn reset_clears_history() {
        let mut dl = DelayLine::new(10);
        for i in 0..dl.mask * 2 {
            dl.write(i as f32 + 1.0);
        }
        dl.reset();
        for d in 0..=dl.mask {
            assert_eq!(dl.read(d), 0.0, "delay {d}");
        }

        dl.write(1.0);
        assert_eq!(dl.read(0), 1.0);
        assert_eq!(dl.read(1), 0.0);
    }

    #[test]
    fn max_delay_is_readable() {
        for max in [1, 61, 62, 100, 1000] {
            for delay in [max, max + 1, max + 2] {
                let mut dl = DelayLine::new(max);
                dl.write(1.0);
                for _ in 0..delay {
                    dl.write(0.0);
                }
                assert_eq!(dl.read(delay), 1.0, "max {max}, delay {delay}");
            }
        }
    }

    fn assert_close(actual: f32, expected: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn read_frac_dc_input_returns_constant() {
        let mut dl = DelayLine::new(16);
        for _ in 0..32 {
            dl.write(0.7);
        }
        for delay in [1.0, 1.25, 3.5, 7.9, 16.0] {
            assert_close(dl.read_frac(delay), 0.7, &format!("delay {delay}"));
        }
    }

    #[test]
    fn read_frac_integer_delay_matches_read() {
        let mut dl = DelayLine::new(16);
        for i in 0..40 {
            dl.write((i as f32 * 0.37).sin());
        }
        for d in 1..=16 {
            assert_eq!(dl.read_frac(d as f32), dl.read(d), "delay {d}");
        }
    }

    #[test]
    fn read_frac_interpolates_ramp_across_wraparound() {
        let mut dl = DelayLine::new(10);
        let len = dl.mask + 1;
        for i in 0..len * 3 + 5 {
            dl.write(i as f32);
            for delay in [1.0, 1.5, 2.25, 4.75, 9.1, 10.0] {
                if (delay as usize) + 2 <= i {
                    assert_close(
                        dl.read_frac(delay),
                        i as f32 - delay,
                        &format!("write {i}, delay {delay}"),
                    );
                }
            }
        }
    }

    #[test]
    fn read_frac_clamps_out_of_range_delays() {
        let max = 10;
        let mut dl = DelayLine::new(max);
        for i in 0..32 {
            dl.write((i as f32 * 0.37).sin());
        }
        let at_min = dl.read_frac(1.0);
        let at_max = dl.read_frac(max as f32);
        assert_eq!(dl.read_frac(0.0), at_min);
        assert_eq!(dl.read_frac(0.5), at_min);
        assert_eq!(dl.read_frac(-5.0), at_min);
        assert_eq!(dl.read_frac(10.5), at_max);
        assert_eq!(dl.read_frac(1e9), at_max);
        assert_eq!(dl.read_frac(f32::INFINITY), at_max);
    }
}
