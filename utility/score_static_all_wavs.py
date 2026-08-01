import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute a static/hiss score for WAV files. Higher score means more static-like content."
    )
    parser.add_argument(
        "--directory",
        default=".",
        help="Directory containing WAV files (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for WAV files.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=6000.0,
        help="Frequency boundary used for high-frequency energy analysis.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=2048,
        help="STFT frame size.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="STFT hop size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to print (0 = all).",
    )
    return parser.parse_args()


def discover_wavs(directory: str, recursive: bool) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1, dtype=np.float32)


def frame_audio(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size), mode="constant")

    remainder = (signal.size - frame_size) % hop_size
    if remainder != 0:
        signal = np.pad(signal, (0, hop_size - remainder), mode="constant")

    frame_count = 1 + (signal.size - frame_size) // hop_size
    shape = (frame_count, frame_size)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def normalize_metric(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def analyze_static(audio: np.ndarray, sample_rate: int, highpass_hz: float, frame_size: int, hop_size: int) -> Dict[str, float]:
    mono = to_mono(audio)
    if mono.size == 0:
        return {
            "score": 0.0,
            "high_ratio": 0.0,
            "flatness": 0.0,
            "zcr": 0.0,
            "centroid_norm": 0.0,
        }

    frames = frame_audio(mono, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = np.abs(spectrum) ** 2

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    high_mask = freqs >= float(highpass_hz)

    total_power = np.sum(power, axis=1) + 1e-12
    high_power = np.sum(power[:, high_mask], axis=1) + 1e-12
    high_ratio = float(np.median(high_power / total_power))

    if np.any(high_mask):
        high_band = power[:, high_mask] + 1e-12
        gm = np.exp(np.mean(safe_log(high_band), axis=1))
        am = np.mean(high_band, axis=1)
        flatness = float(np.mean(gm / np.maximum(am, 1e-12)))
    else:
        flatness = 0.0

    signs = np.sign(frames)
    zc = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    zcr = float(np.mean(zc))

    centroid = np.sum(power * freqs[None, :], axis=1) / np.maximum(total_power, 1e-12)
    centroid_norm = float(np.mean(centroid) / max(sample_rate * 0.5, 1e-12))

    # Composite static score: emphasize broadband high-frequency and noise-like texture.
    n_high = normalize_metric(high_ratio, low=0.05, high=0.45)
    n_flat = normalize_metric(flatness, low=0.10, high=0.75)
    n_zcr = normalize_metric(zcr, low=0.02, high=0.18)
    n_cent = normalize_metric(centroid_norm, low=0.08, high=0.40)

    score_0_1 = (0.45 * n_high) + (0.30 * n_flat) + (0.15 * n_zcr) + (0.10 * n_cent)
    score = 100.0 * clamp01(score_0_1)

    return {
        "score": score,
        "high_ratio": high_ratio,
        "flatness": flatness,
        "zcr": zcr,
        "centroid_norm": centroid_norm,
    }


def main():
    args = parse_args()
    wavs = discover_wavs(args.directory, args.recursive)
    if not wavs:
        raise RuntimeError("No WAV files found.")

    rows: List[Tuple[Path, Dict[str, float]]] = []
    for path in wavs:
        try:
            audio, sample_rate = sf.read(str(path), always_2d=False)
            metrics = analyze_static(
                audio,
                sample_rate,
                highpass_hz=args.highpass_hz,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
            )
            rows.append((path, metrics))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    rows.sort(key=lambda item: item[1]["score"], reverse=True)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Scored {len(rows)} WAV files in {Path(args.directory).resolve()}")
    print("Higher static_score means more static-like / hiss-like content.")
    print()
    print("rank  static_score  high_ratio  flatness  zcr      centroid  created              file")

    for idx, (path, m) in enumerate(rows, start=1):
        created = datetime.fromtimestamp(path.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{idx:>4}  {m['score']:>11.2f}  {m['high_ratio']:>10.4f}  "
            f"{m['flatness']:>8.4f}  {m['zcr']:>7.4f}  {m['centroid_norm']:>8.4f}  {created}  {path.name}"
        )


if __name__ == "__main__":
    main()
