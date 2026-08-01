import argparse
from pathlib import Path
from typing import List

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the most recent WAV files in a directory against a text prompt using LAION-CLAP."
    )
    parser.add_argument("prompt", help="Text prompt used for CLAP similarity scoring.")
    parser.add_argument(
        "--directory",
        default=".",
        help="Directory to scan for WAV files (default: current directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many most-recent WAV files to score (default: 5).",
    )
    parser.add_argument(
        "--amodel",
        default="HTSAT-tiny",
        help="LAION-CLAP audio backbone model name (default: HTSAT-tiny).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional local CLAP checkpoint path. If omitted, laion_clap loads its default checkpoint.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU if CUDA is unavailable.",
    )
    return parser.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for CLAP scoring. Re-run with --allow-cpu to override.")


def find_recent_wavs(directory: str, limit: int) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    wavs = [path for path in root.glob("*.wav") if path.is_file()]
    wavs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return wavs[: max(1, limit)]


def load_clap_model(device: torch.device, amodel: str, checkpoint: str):
    try:
        import laion_clap
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'laion_clap'. Install with: pip install laion-clap"
        ) from exc

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=str(device))
    try:
        if checkpoint:
            model.load_ckpt(checkpoint)
        else:
            model.load_ckpt()
    except RuntimeError as exc:
        message = str(exc)
        if "Error(s) in loading state_dict for CLAP" in message:
            raise RuntimeError(
                "CLAP checkpoint/model mismatch. Try --amodel HTSAT-tiny (or the model architecture used by your checkpoint)."
            ) from exc
        raise
    model.eval()
    return model


def to_2d_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def cosine_similarities(audio_embeds: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    audio_norm = torch.nn.functional.normalize(audio_embeds, dim=1)
    text_norm = torch.nn.functional.normalize(text_embed, dim=1)
    return torch.matmul(audio_norm, text_norm.transpose(0, 1)).squeeze(1)


def main():
    args = parse_args()
    device = get_device(args.allow_cpu)
    wav_files = find_recent_wavs(args.directory, args.limit)
    if not wav_files:
        raise RuntimeError(f"No .wav files found in directory: {args.directory}")

    model = load_clap_model(device, args.amodel, args.checkpoint)
    file_list = [str(path.resolve()) for path in wav_files]

    with torch.no_grad():
        audio_embeds = model.get_audio_embedding_from_filelist(x=file_list, use_tensor=True)
        text_embed = model.get_text_embedding([args.prompt], use_tensor=True)

    audio_embeds = to_2d_tensor(audio_embeds)
    text_embed = to_2d_tensor(text_embed)
    scores = cosine_similarities(audio_embeds, text_embed)

    ranked = sorted(zip(wav_files, scores.tolist()), key=lambda item: item[1], reverse=True)

    print(f"Prompt: {args.prompt}")
    print(f"Scored {len(ranked)} most recent wav files in: {Path(args.directory).resolve()}")
    print()
    for idx, (path, score) in enumerate(ranked, start=1):
        print(f"{idx}. score={score:.4f}  file={path.name}")


if __name__ == "__main__":
    main()