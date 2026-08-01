import argparse
from pathlib import Path
from typing import List, Tuple
import re

import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print MAE and MSE for all original/reconstructed WAV pairs in a directory, including numbered variants like *_original2.wav."
    )
    parser.add_argument("--input-dir", default="reconstruction_test_out")
    parser.add_argument("--sort-by", choices=["name", "mae", "mse"], default="name")
    return parser.parse_args()


def load_waveform(path: Path) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, sample_rate


def align_length(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    target = min(a.shape[-1], b.shape[-1])
    return a[..., :target], b[..., :target]


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Tuple[float, float]:
    original, reconstructed = align_length(original, reconstructed)
    mae = torch.mean(torch.abs(reconstructed - original)).item()
    mse = torch.mean((reconstructed - original) ** 2).item()
    return mae, mse


ORIGINAL_RE = re.compile(r"^(?P<base>.+)_original(?P<tag>\d*)$")


def parse_original_name(path: Path):
    match = ORIGINAL_RE.match(path.stem)
    if not match:
        return None
    return match.group("base"), match.group("tag")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    original_files = sorted([p for p in input_dir.glob("*.wav") if parse_original_name(p) is not None])
    if not original_files:
        raise RuntimeError(f"No *_original*.wav files found in {input_dir}")

    rows: List[Tuple[str, float, float, int]] = []
    missing: List[Path] = []
    all_reconstructed = {p.stem: p for p in input_dir.glob("*.wav") if "_reconstructed" in p.stem}

    for original_path in original_files:
        parsed = parse_original_name(original_path)
        if parsed is None:
            continue
        stem, tag = parsed

        preferred_stem = f"{stem}_reconstructed{tag}"
        reconstructed_path = all_reconstructed.get(preferred_stem)

        if reconstructed_path is None and tag:
            fallback_stem = f"{stem}_reconstructed"
            reconstructed_path = all_reconstructed.get(fallback_stem)

        if reconstructed_path is None:
            missing.append(input_dir / f"{preferred_stem}.wav")
            continue

        original_waveform, original_sr = load_waveform(original_path)
        reconstructed_waveform, reconstructed_sr = load_waveform(reconstructed_path)

        if original_sr != reconstructed_sr:
            raise RuntimeError(
                f"Sample rate mismatch for {stem}: original={original_sr}, reconstructed={reconstructed_sr}"
            )

        mae, mse = compute_metrics(original_waveform, reconstructed_waveform)
        samples = min(original_waveform.shape[-1], reconstructed_waveform.shape[-1])
        rows.append((original_path.stem, mae, mse, samples))

    if not rows:
        raise RuntimeError("No valid original/reconstructed pairs were found.")

    if args.sort_by == "mae":
        rows.sort(key=lambda row: row[1], reverse=True)
    elif args.sort_by == "mse":
        rows.sort(key=lambda row: row[2], reverse=True)
    else:
        rows.sort(key=lambda row: row[0])

    print(f"Input directory: {input_dir}")
    print(f"Pairs found: {len(rows)}")
    if missing:
        print(f"Missing reconstructed files: {len(missing)}")

    print("\nPer-file metrics")
    print("file\tMAE\tMSE\tsamples")
    for stem, mae, mse, samples in rows:
        print(f"{stem}\t{mae:.6f}\t{mse:.6f}\t{samples}")

    maes = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    mses = torch.tensor([row[2] for row in rows], dtype=torch.float32)

    print("\nSummary")
    print(f"MAE mean:   {maes.mean().item():.6f}")
    print(f"MAE median: {maes.median().item():.6f}")
    print(f"MAE min:    {maes.min().item():.6f}")
    print(f"MAE max:    {maes.max().item():.6f}")
    print(f"MSE mean:   {mses.mean().item():.6f}")
    print(f"MSE median: {mses.median().item():.6f}")
    print(f"MSE min:    {mses.min().item():.6f}")
    print(f"MSE max:    {mses.max().item():.6f}")


if __name__ == "__main__":
    main()