import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf


DEFAULT_PATTERNS = [
    "latent_blend*.wav",
    "latent_generated*.wav",
    "waveform_generated*.wav",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce static/hiss in generated WAV files and normalize them to a more audible loudness."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional WAV files or directories. If omitted, scans the current folder for generated WAVs.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for WAV discovery inside input directories. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to writing beside each input file.",
    )
    parser.add_argument(
        "--suffix",
        default="_clean_loud",
        help="Suffix added before the .wav extension.",
    )
    parser.add_argument(
        "--noise-seconds",
        type=float,
        default=0.5,
        help="Use the first N seconds as the noise profile when possible.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.25,
        help="How aggressively to suppress the estimated noise floor.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.08,
        help="Minimum retained mask value. Higher values preserve more ambience.",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=2048,
        help="FFT size used for spectral gating.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Hop size used for spectral gating.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.14,
        help="Target RMS loudness after denoising.",
    )
    parser.add_argument(
        "--target-peak",
        type=float,
        default=0.98,
        help="Peak ceiling after normalization.",
    )
    parser.add_argument(
        "--max-gain-db",
        type=float,
        default=18.0,
        help="Maximum gain boost allowed during normalization, in dB.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only show what would be processed.",
    )
    return parser.parse_args()


def discover_files(inputs: List[str], patterns: List[str]) -> List[Path]:
    effective_patterns = patterns or DEFAULT_PATTERNS
    discovered: List[Path] = []

    if not inputs:
        inputs = ["."]

    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".wav":
            discovered.append(path)
            continue

        if path.is_dir():
            for pattern in effective_patterns:
                discovered.extend(sorted(path.glob(pattern)))

    unique = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def frame_audio(channel: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if channel.size == 0:
        return np.zeros((0, frame_size), dtype=np.float32)

    if channel.size < frame_size:
        pad_width = frame_size - channel.size
    else:
        remainder = (channel.size - frame_size) % hop_size
        pad_width = 0 if remainder == 0 else hop_size - remainder

    padded = np.pad(channel, (0, pad_width), mode="constant")
    frames = []
    for start in range(0, padded.size - frame_size + 1, hop_size):
        frames.append(padded[start:start + frame_size])
    return np.stack(frames, axis=0) if frames else np.zeros((0, frame_size), dtype=np.float32)


def overlap_add(frames: np.ndarray, frame_size: int, hop_size: int, original_length: int) -> np.ndarray:
    if frames.size == 0:
        return np.zeros(original_length, dtype=np.float32)

    total_length = hop_size * (frames.shape[0] - 1) + frame_size
    output = np.zeros(total_length, dtype=np.float32)
    norm = np.zeros(total_length, dtype=np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    window_sq = window * window

    for index, frame in enumerate(frames):
        start = index * hop_size
        output[start:start + frame_size] += frame * window
        norm[start:start + frame_size] += window_sq

    output /= np.maximum(norm, 1e-8)
    return output[:original_length]


def moving_average(matrix: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or matrix.shape[0] == 0:
        return matrix

    padded = np.pad(matrix, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    smoothed = np.empty_like(matrix)
    for bin_index in range(matrix.shape[1]):
        smoothed[:, bin_index] = np.convolve(padded[:, bin_index], kernel, mode="valid")
    return smoothed


def reduce_static_channel(
    channel: np.ndarray,
    sample_rate: int,
    noise_seconds: float,
    strength: float,
    floor: float,
    fft_size: int,
    hop_size: int,
) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    original_length = channel.shape[0]
    frames = frame_audio(channel, fft_size, hop_size)
    if frames.shape[0] == 0:
        return channel

    window = np.hanning(fft_size).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=1)
    magnitude = np.abs(spectrum)

    noise_frame_count = int(round(max(0.0, noise_seconds) * sample_rate / hop_size))
    if noise_frame_count >= 3 and frames.shape[0] >= noise_frame_count:
        noise_profile = magnitude[:noise_frame_count].mean(axis=0)
    else:
        frame_energy = magnitude.mean(axis=1)
        quiet_count = max(4, min(frames.shape[0], max(1, frames.shape[0] // 10)))
        quiet_indices = np.argsort(frame_energy)[:quiet_count]
        noise_profile = magnitude[quiet_indices].mean(axis=0)

    raw_mask = 1.0 - (strength * noise_profile[None, :] / np.maximum(magnitude, 1e-8))
    mask = np.clip(raw_mask, floor, 1.0).astype(np.float32)
    mask = moving_average(mask, radius=2)

    cleaned_spectrum = spectrum * mask
    cleaned_frames = np.fft.irfft(cleaned_spectrum, n=fft_size, axis=1).astype(np.float32)
    cleaned = overlap_add(cleaned_frames, fft_size, hop_size, original_length)
    return np.clip(cleaned, -1.0, 1.0)


def reduce_static(
    audio: np.ndarray,
    sample_rate: int,
    noise_seconds: float,
    strength: float,
    floor: float,
    fft_size: int,
    hop_size: int,
) -> np.ndarray:
    if audio.ndim == 1:
        return reduce_static_channel(audio, sample_rate, noise_seconds, strength, floor, fft_size, hop_size)

    channels = []
    for channel_index in range(audio.shape[1]):
        channels.append(
            reduce_static_channel(
                audio[:, channel_index],
                sample_rate,
                noise_seconds,
                strength,
                floor,
                fft_size,
                hop_size,
            )
        )
    return np.stack(channels, axis=1)


def rms_level(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def peak_level(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)))


def normalize_audio(audio: np.ndarray, target_rms: float, target_peak: float, max_gain_db: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current_rms = rms_level(audio)
    current_peak = peak_level(audio)

    if current_peak <= 1e-8:
        return audio

    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain_from_rms = target_rms / max(current_rms, 1e-8) if target_rms > 0 else 1.0
    gain = min(max_gain, gain_from_rms)

    if current_peak * gain > target_peak:
        gain = target_peak / current_peak

    normalized = audio * gain
    normalized_peak = peak_level(normalized)
    if normalized_peak > target_peak:
        normalized = normalized * (target_peak / max(normalized_peak, 1e-8))

    return np.clip(normalized, -1.0, 1.0)


def build_output_path(input_path: Path, output_dir: str, suffix: str) -> Path:
    if output_dir:
        destination_dir = Path(output_dir)
    else:
        destination_dir = input_path.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def process_file(input_path: Path, output_path: Path, args):
    audio, sample_rate = sf.read(str(input_path), always_2d=False)
    cleaned = reduce_static(
        audio,
        sample_rate,
        noise_seconds=args.noise_seconds,
        strength=args.strength,
        floor=args.floor,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
    )
    normalized = normalize_audio(
        cleaned,
        target_rms=args.target_rms,
        target_peak=args.target_peak,
        max_gain_db=args.max_gain_db,
    )
    sf.write(str(output_path), normalized, sample_rate)
    print(f"Processed {input_path} -> {output_path}")


def main():
    args = parse_args()
    files = discover_files(args.inputs, args.pattern)
    if not files:
        raise RuntimeError("No WAV files found to process.")

    for file_path in files:
        output_path = build_output_path(file_path, args.output_dir, args.suffix)
        if args.preview:
            print(f"Would process {file_path} -> {output_path}")
            continue
        process_file(file_path, output_path, args)


if __name__ == "__main__":
    main()
