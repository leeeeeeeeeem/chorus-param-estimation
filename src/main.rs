//! render <dry.wav> <params.json> <wet.wav>
//!
//! Mono or stereo WAV in, stereo WAV out (left, right) in the same format
//! (int16 or float32). A mono input goes to both chorus inputs.

use std::process::ExitCode;

use chorus_render::{Params, render_stereo};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

fn run(dry_path: &str, params_path: &str, out_path: &str) -> Result<(), String> {
    let json = std::fs::read_to_string(params_path).map_err(|e| format!("{params_path}: {e}"))?;
    let params: Params = serde_json::from_str(&json).map_err(|e| format!("{params_path}: {e}"))?;

    let mut reader = WavReader::open(dry_path).map_err(|e| format!("{dry_path}: {e}"))?;
    let spec = reader.spec();
    if spec.channels != 1 && spec.channels != 2 {
        return Err(format!("{dry_path}: expected mono or stereo, got {} channels", spec.channels));
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>(),
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>(),
        (f, b) => return Err(format!("{dry_path}: unsupported format {f:?} {b}-bit")),
    }
    .map_err(|e| format!("{dry_path}: {e}"))?;

    let dry: Vec<[f32; 2]> = if spec.channels == 1 {
        samples.iter().map(|&x| [x, x]).collect()
    } else {
        samples.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    };
    let wet = render_stereo(&dry, spec.sample_rate as f32, &params)?;

    let out_spec = WavSpec { channels: 2, ..spec };
    let mut writer = WavWriter::create(out_path, out_spec).map_err(|e| format!("{out_path}: {e}"))?;
    let mut clipped = 0;
    for y in wet.into_iter().flatten() {
        let r = if spec.sample_format == SampleFormat::Int {
            let s = (y * 32768.0).round();
            if !(-32768.0..=32767.0).contains(&s) {
                clipped += 1;
            }
            writer.write_sample(s.clamp(-32768.0, 32767.0) as i16)
        } else {
            writer.write_sample(y)
        };
        r.map_err(|e| format!("{out_path}: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("{out_path}: {e}"))?;
    if clipped > 0 {
        eprintln!("warning: {clipped} samples clipped in {out_path}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: render <dry.wav> <params.json> <wet.wav>");
        return ExitCode::from(2);
    }
    match run(&args[1], &args[2], &args[3]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
