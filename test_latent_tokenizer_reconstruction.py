import argparse
import os
import random
from pathlib import Path

import torch
import torchaudio

from latent_audio_token_pipeline import (
    crop_or_pad,
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test latent tokenizer reconstruction on a real training clip.")
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--input-audio", default="", help="Optional path to a specific audio file. If omitted, a training file is chosen.")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--audio-dir", default="dataset/audio")
    parser.add_argument("--output-dir", default="reconstruction_test_out")
    parser.add_argument(
        "--output-seconds",
        type=float,
        default=0.0,
        help="Optional positive clip length override in seconds. Uses tokenizer training clip length when omitted or <= 0.",
    )
    parser.add_argument("--random-sample", action="store_true", help="Pick a random training file when --input-audio is not provided.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for this script. Re-run with --allow-cpu to override.")


def choose_input_audio(args, sample_rate: int) -> str:
    if args.input_audio:
        if not os.path.isfile(args.input_audio):
            raise FileNotFoundError(f"Input audio not found: {args.input_audio}")
        load_audio_mono(args.input_audio, sample_rate)
        return args.input_audio

    items = load_dataset_items(args.metadata_csv, args.audio_dir)
    if not items:
        raise RuntimeError("No valid dataset items found.")

    candidates = [item["path"] for item in items]
    if args.random_sample:
        random.shuffle(candidates)

    for path in candidates:
        try:
            load_audio_mono(path, sample_rate)
            return path
        except Exception as exc:
            print(f"Skipping invalid audio file: {path} ({exc})")

    raise RuntimeError("No decodable audio files found in the selected dataset inputs.")


def print_error_bucket_stats(metric_name: str, values: list[float]):
    if not values:
        return

    bucket_definitions = [
        ("<= 0.00001", lambda value: value <= 0.00001),
        ("0.00001 to 0.0001", lambda value: 0.00001 < value <= 0.0001),
        ("0.0001 to 0.001", lambda value: 0.0001 < value <= 0.001),
        ("0.001 to 0.005", lambda value: 0.001 < value <= 0.005),
        ("0.005 to 0.01", lambda value: 0.005 < value <= 0.01),
        ("0.01 to 0.02", lambda value: 0.01 < value <= 0.02),
        ("0.02 to 0.05", lambda value: 0.02 < value < 0.05),
        (">= 0.05", lambda value: value >= 0.05),
    ]

    total = len(values)
    print(f"{metric_name} bucket stats:")
    for label, predicate in bucket_definitions:
        count = sum(1 for value in values if predicate(value))
        pct = (count / total) * 100.0
        print(f"  {label}: {count}/{total} ({pct:.2f}%)")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.allow_cpu)

    tokenizer_model, config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    input_audio = choose_input_audio(args, config.sample_rate)
    target_samples = config.clip_samples
    if args.output_seconds > 0:
        target_samples = max(1, int(round(args.output_seconds * config.sample_rate)))

    waveform = load_audio_mono(input_audio, config.sample_rate)
    waveform = crop_or_pad(waveform, target_samples, random_crop=False)
    waveform_batch = waveform.unsqueeze(0).to(device)

    with torch.no_grad():
        codes = tokenizer_model.encode_codes(
            waveform_batch,
            return_all_codes=(getattr(tokenizer_model.config, "num_quantizers", 1) > 1),
        )
        reconstructed = tokenizer_model.decode_codes(codes, target_length=target_samples).squeeze(0).cpu()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = Path(input_audio).stem
    original_path = os.path.join(args.output_dir, f"{stem}_original.wav")
    recon_path = os.path.join(args.output_dir, f"{stem}_reconstructed.wav")

    torchaudio.save(original_path, waveform.cpu(), config.sample_rate)
    torchaudio.save(recon_path, reconstructed, config.sample_rate)

    mae = torch.mean(torch.abs(reconstructed - waveform.cpu())).item()
    mse = torch.mean((reconstructed - waveform.cpu()) ** 2).item()

    print(f"Input clip: {input_audio}")
    print(f"Sample rate: {config.sample_rate}")
    print(f"Latent steps: {codes.shape[-1]}")
    print(f"Saved original clip to: {original_path}")
    print(f"Saved reconstructed clip to: {recon_path}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print_error_bucket_stats("MAE", [mae])
    print_error_bucket_stats("MSE", [mse])
    print("Listen to both files. If the reconstructed clip already sounds poor, the tokenizer is the main bottleneck.")


if __name__ == "__main__":
    main()
