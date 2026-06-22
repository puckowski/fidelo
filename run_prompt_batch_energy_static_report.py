import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


DEFAULT_PROMPTS = [
    "instrumental pop",
    "indie pop instrumental",
    "synth pop instrumental",
    "electropop instrumental",
    "dream pop instrumental",
    "dance pop instrumental",
    "folk pop instrumental",
    "experimental pop instrumental",
    "lofi pop instrumental",
    "acoustic pop instrumental",
    "cinematic pop instrumental",
    "uplifting pop instrumental",
    "melancholic pop instrumental",
    "retro pop instrumental",
    "ambient pop instrumental",
    "orchestral pop instrumental",
    "minimal pop instrumental",
    "art pop instrumental",
    "modern pop instrumental",
    "radio pop instrumental",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run themed sticky crossfade generation with many prompts and report energy gate events plus static scores."
    )
    parser.add_argument(
        "--generator-script",
        default="generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade.py",
        help="Generator script to run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="How many generations to run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the full prompt list this many times. If > 1, total runs become repeat * number_of_prompts.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=88,
        help="Starting seed; each run increments by 1.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--source-strength", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rank-choice-top", type=int, default=2)
    parser.add_argument("--window-energy-check-top", type=int, default=12)
    parser.add_argument("--min-window-rms", type=float, default=0.012)
    parser.add_argument("--min-window-peak", type=float, default=0.04)
    parser.add_argument("--theme-repeat-window", type=int, default=6)
    parser.add_argument("--theme-crossfade-ms", type=int, default=1000)
    parser.add_argument("--source-overlap", type=int, default=1024)
    parser.add_argument("--theme-repeat-bonus", type=float, default=3.5)
    parser.add_argument(
        "--output-root",
        default="batch_reports",
        help="Root directory under which a timestamped run folder is created.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=6000.0,
        help="Static-score high-frequency split used in analysis.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=2048,
        help="Static-score STFT frame size.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Static-score STFT hop size.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    value = text.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:60] if value else "prompt"


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
            "static_score": 0.0,
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

    n_high = normalize_metric(high_ratio, low=0.05, high=0.45)
    n_flat = normalize_metric(flatness, low=0.10, high=0.75)
    n_zcr = normalize_metric(zcr, low=0.02, high=0.18)
    n_cent = normalize_metric(centroid_norm, low=0.08, high=0.40)

    score_0_1 = (0.45 * n_high) + (0.30 * n_flat) + (0.15 * n_zcr) + (0.10 * n_cent)
    static_score = 100.0 * clamp01(score_0_1)

    return {
        "static_score": static_score,
        "high_ratio": high_ratio,
        "flatness": flatness,
        "zcr": zcr,
        "centroid_norm": centroid_norm,
    }


def count_energy_gate_instances(output_text: str) -> Dict[str, int]:
    theme_rejects = output_text.count("Rejected low-energy theme window")
    clip_rejects = output_text.count("Rejected low-energy clip")
    return {
        "theme_gate_rejects": theme_rejects,
        "clip_gate_rejects": clip_rejects,
        "total_gate_instances": theme_rejects + clip_rejects,
    }


def build_run_folder(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"energy_static_report_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_generation(args, run_index: int, prompt: str, seed: int, wav_path: Path) -> subprocess.CompletedProcess:
    command = [
        "python",
        args.generator_script,
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--duration-seconds",
        str(args.duration_seconds),
        "--source-strength",
        str(args.source_strength),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--rank-choice-top",
        str(args.rank_choice_top),
        "--window-energy-check-top",
        str(args.window_energy_check_top),
        "--min-window-rms",
        str(args.min_window_rms),
        "--min-window-peak",
        str(args.min_window_peak),
        "--theme-repeat-window",
        str(args.theme_repeat_window),
        "--theme-crossfade-ms",
        str(args.theme_crossfade_ms),
        "--source-overlap",
        str(args.source_overlap),
        "--theme-repeat-bonus",
        str(args.theme_repeat_bonus),
        "--output",
        str(wav_path),
    ]

    print(f"[{run_index:02d}] prompt='{prompt}' seed={seed}")
    return subprocess.run(command, capture_output=True, text=True)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    run_dir = build_run_folder(args.output_root)
    logs_dir = run_dir / "logs"
    audio_dir = run_dir / "audio"
    logs_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompts = DEFAULT_PROMPTS
    rows: List[Dict] = []

    if args.repeat > 1:
        total_runs = max(1, int(args.repeat)) * len(prompts)
    else:
        total_runs = max(1, int(args.runs))

    for idx in range(total_runs):
        prompt = prompts[idx % len(prompts)]
        seed = args.seed_start + idx
        wav_name = f"{idx + 1:02d}_{seed}_{slugify(prompt)}.wav"
        wav_path = audio_dir / wav_name

        log_path = logs_dir / f"{idx + 1:02d}_{seed}_{slugify(prompt)}.log"
        proc = None
        combined_output = ""
        retry_used = 0
        max_attempts = 3  # initial attempt + up to 2 retries

        for attempt in range(max_attempts):
            proc = run_generation(args, idx + 1, prompt, seed, wav_path)
            combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            write_text(log_path, combined_output)
            if combined_output.strip():
                retry_used = attempt
                break
            if attempt < (max_attempts - 1):
                print(
                    f"[{idx + 1:02d}] empty inference log/output; retrying "
                    f"({attempt + 1}/{max_attempts - 1})"
                )
            retry_used = attempt + 1

        energy_counts = count_energy_gate_instances(combined_output)
        static_metrics = {
            "static_score": float("nan"),
            "high_ratio": float("nan"),
            "flatness": float("nan"),
            "zcr": float("nan"),
            "centroid_norm": float("nan"),
        }

        if proc.returncode == 0 and wav_path.is_file():
            audio, sr = sf.read(str(wav_path), always_2d=False)
            static_metrics = analyze_static(
                audio,
                sr,
                highpass_hz=args.highpass_hz,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
            )

        row = {
            "run_index": idx + 1,
            "prompt": prompt,
            "seed": seed,
            "retries_used": retry_used,
            "exit_code": proc.returncode,
            "wav_file": str(wav_path),
            **energy_counts,
            **static_metrics,
        }
        rows.append(row)

    csv_path = run_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_index",
            "prompt",
            "seed",
            "retries_used",
            "exit_code",
            "wav_file",
            "theme_gate_rejects",
            "clip_gate_rejects",
            "total_gate_instances",
            "static_score",
            "high_ratio",
            "flatness",
            "zcr",
            "centroid_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["exit_code"] == 0]
    summary = {
        "run_folder": str(run_dir.resolve()),
        "runs_requested": total_runs,
        "runs_successful": len(successful),
        "runs_failed": total_runs - len(successful),
        "runs_with_log_retries": int(sum(1 for row in rows if row["retries_used"] > 0)),
        "total_log_retries_used": int(sum(int(row["retries_used"]) for row in rows)),
        "total_theme_gate_rejects": int(sum(row["theme_gate_rejects"] for row in rows)),
        "total_clip_gate_rejects": int(sum(row["clip_gate_rejects"] for row in rows)),
        "total_gate_instances": int(sum(row["total_gate_instances"] for row in rows)),
    }

    if successful:
        summary["avg_static_score_successful"] = float(np.mean([row["static_score"] for row in successful]))
        summary["max_static_score_successful"] = float(np.max([row["static_score"] for row in successful]))
        summary["min_static_score_successful"] = float(np.min([row["static_score"] for row in successful]))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Batch complete. Run folder: {run_dir.resolve()}")
    print(f"CSV report: {csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(
        f"Energy gates total={summary['total_gate_instances']} "
        f"(theme={summary['total_theme_gate_rejects']}, clip={summary['total_clip_gate_rejects']})"
    )


if __name__ == "__main__":
    main()
