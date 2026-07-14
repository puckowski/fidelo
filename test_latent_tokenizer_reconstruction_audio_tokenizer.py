import argparse
import json
import os
import random
from pathlib import Path

import torch
import torchaudio

from latent_audio_token_pipeline import (
    LatentAudioConfig,
    VQAudioAutoencoder,
    crop_or_pad,
    load_audio_mono,
    load_dataset_items,
    load_tokenizer_state_dict,
    safe_torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test latent tokenizer reconstruction using audio_tokenizer.pt directly."
    )
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path. Defaults to <tokenizer-dir>/audio_tokenizer.pt.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional config path. Defaults to <tokenizer-dir>/config.json.",
    )
    parser.add_argument("--input-audio", default="", help="Optional path to a specific audio file. If omitted, a training file is chosen.")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--audio-dir", default="dataset/audio")
    parser.add_argument("--output-dir", default="reconstruction_test_out")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="How many unique audio samples to reconstruct. Each sample produces one original and one reconstructed output file.",
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


def choose_input_audios(args, sample_rate: int) -> list[str]:
    sample_count = max(1, int(args.sample_count))
    if args.input_audio:
        if not os.path.isfile(args.input_audio):
            raise FileNotFoundError(f"Input audio not found: {args.input_audio}")
        load_audio_mono(args.input_audio, sample_rate)
        if sample_count > 1:
            raise ValueError("--sample-count cannot exceed 1 when --input-audio is provided.")
        return [args.input_audio]

    items = load_dataset_items(args.metadata_csv, args.audio_dir)
    if not items:
        raise RuntimeError("No valid dataset items found.")

    unique_paths: list[str] = []
    seen_paths = set()
    for item in items:
        path = item["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)

    if sample_count > len(unique_paths):
        raise ValueError(
            f"Requested {sample_count} unique samples, but only {len(unique_paths)} unique dataset items are available."
        )

    if args.random_sample:
        random.shuffle(unique_paths)

    selected_paths: list[str] = []
    invalid_count = 0
    for path in unique_paths:
        try:
            load_audio_mono(path, sample_rate)
            selected_paths.append(path)
        except Exception as exc:
            invalid_count += 1
            print(f"Skipping invalid audio file: {path} ({exc})")
        if len(selected_paths) >= sample_count:
            break

    if len(selected_paths) < sample_count:
        raise RuntimeError(
            f"Only found {len(selected_paths)} decodable audio files out of {len(unique_paths)} unique dataset items"
            f" while requesting {sample_count}. Skipped {invalid_count} invalid files."
        )

    return selected_paths


def load_audio_tokenizer_from_checkpoint(tokenizer_dir: str, checkpoint_path: str, config_path: str, device: torch.device):
    checkpoint = checkpoint_path or os.path.join(tokenizer_dir, "audio_tokenizer.pt")
    config_file = config_path or os.path.join(tokenizer_dir, "config.json")

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = LatentAudioConfig.from_dict(json.load(f))

    model = VQAudioAutoencoder(config)
    state = safe_torch_load(checkpoint, device)
    load_tokenizer_state_dict(model, state)
    model.to(device).eval()
    return model, config, checkpoint


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

    tokenizer_model, config, checkpoint_used = load_audio_tokenizer_from_checkpoint(
        args.tokenizer_dir,
        args.checkpoint,
        args.config,
        device,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    input_audios = choose_input_audios(args, config.sample_rate)
    total_mae = 0.0
    total_mse = 0.0
    mae_values: list[float] = []
    mse_values: list[float] = []

    print(f"Checkpoint: {checkpoint_used}")
    print(f"Sample rate: {config.sample_rate}")
    print(f"Reconstructing {len(input_audios)} unique sample(s)")

    for sample_idx, input_audio in enumerate(input_audios, start=1):
        waveform = load_audio_mono(input_audio, config.sample_rate)
        waveform = crop_or_pad(waveform, config.clip_samples, random_crop=False)
        waveform_batch = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            codes = tokenizer_model.encode_codes(waveform_batch)
            reconstructed = tokenizer_model.decode_codes(codes, target_length=config.clip_samples).squeeze(0).cpu()

        stem = Path(input_audio).stem
        output_stem = f"{sample_idx:03d}_{stem}"
        original_path = os.path.join(args.output_dir, f"{output_stem}_original.wav")
        recon_path = os.path.join(args.output_dir, f"{output_stem}_reconstructed.wav")

        torchaudio.save(original_path, waveform.cpu(), config.sample_rate)
        torchaudio.save(recon_path, reconstructed, config.sample_rate)

        mae = torch.mean(torch.abs(reconstructed - waveform.cpu())).item()
        mse = torch.mean((reconstructed - waveform.cpu()) ** 2).item()
        total_mae += mae
        total_mse += mse
        mae_values.append(mae)
        mse_values.append(mse)

        print(f"Sample {sample_idx}: {input_audio}")
        print(f"Latent steps: {codes.shape[1]}")
        print(f"Saved original clip to: {original_path}")
        print(f"Saved reconstructed clip to: {recon_path}")
        print(f"MAE: {mae:.6f}")
        print(f"MSE: {mse:.6f}")

    if len(input_audios) > 1:
        print(f"Average MAE: {total_mae / len(input_audios):.6f}")
        print(f"Average MSE: {total_mse / len(input_audios):.6f}")
    print_error_bucket_stats("MAE", mae_values)
    print_error_bucket_stats("MSE", mse_values)
    print("Listen to both files. If the reconstructed clip already sounds poor, the tokenizer is the main bottleneck.")


if __name__ == "__main__":
    main()
