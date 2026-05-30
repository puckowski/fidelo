import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from latent_audio_token_pipeline import (
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
)


TARGET_SR = 24000
MAX_AUDIO_COMPARE_SECONDS = 20.0
DEFAULT_TOP_K = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the most likely dataset audio originator for a generated audio file using tokenizer-code overlap and audio similarity."
    )
    parser.add_argument("input_audio", help="Generated/inference output audio file to analyze.")
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-source-seconds", type=float, default=30.0, help="Trim dataset candidates before encoding codes.")
    parser.add_argument("--max-input-seconds", type=float, default=30.0, help="Trim the input audio before encoding codes.")
    parser.add_argument("--scan-step", type=int, default=8, help="Sliding-window step in latent tokens for local code matching.")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for tokenizer-based matching. Re-run with --allow-cpu to override.")


def resolve_input_audio(input_audio: str) -> Path:
    candidates = [
        Path(input_audio),
        Path("inference_output") / input_audio,
        Path.cwd() / input_audio,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Input audio not found: {input_audio}")


def trim_waveform(waveform: torch.Tensor, sample_rate: int, max_seconds: float) -> torch.Tensor:
    if max_seconds <= 0:
        return waveform
    max_samples = int(round(sample_rate * max_seconds))
    return waveform[..., :max_samples]


@torch.no_grad()
def encode_codes(path: str, tokenizer_model, sample_rate: int, device: torch.device, max_seconds: float) -> torch.Tensor:
    waveform = load_audio_mono(path, sample_rate)
    waveform = trim_waveform(waveform, sample_rate, max_seconds)
    codes = tokenizer_model.encode_codes(waveform.unsqueeze(0).to(device))
    return codes.squeeze(0).cpu()


def _resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_length = max(1, int(round(waveform.shape[-1] * float(target_rate) / float(source_rate))))
    return F.interpolate(
        waveform.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def load_audio_feature(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    try:
        import soundfile as sf

        waveform_np, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np).transpose(0, 1)
    except Exception:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load audio with soundfile or torchaudio: {exc}") from exc

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.float().clamp(-1.0, 1.0)
    if sr != target_sr:
        waveform = _resample_waveform(waveform, sr, target_sr)

    max_samples = int(round(target_sr * MAX_AUDIO_COMPARE_SECONDS))
    waveform = waveform[..., :max_samples]
    audio = waveform.squeeze(0)
    if audio.numel() < 1024:
        audio = F.pad(audio, (0, max(0, 1024 - audio.numel())))

    window = torch.hann_window(1024)
    spec = torch.stft(
        audio,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-5)
    log_mag = torch.log(mag)

    band_mean = log_mag.mean(dim=1)
    band_std = log_mag.std(dim=1)
    frame_energy = log_mag.mean(dim=0)
    rms = torch.sqrt(torch.mean(audio.pow(2))).unsqueeze(0)
    zcr = ((audio[1:] * audio[:-1]) < 0).float().mean().unsqueeze(0)
    dyn = (audio.abs().quantile(0.95) - audio.abs().quantile(0.50)).unsqueeze(0)

    feature = torch.cat(
        [
            band_mean,
            band_std,
            frame_energy.mean().unsqueeze(0),
            frame_energy.std().unsqueeze(0),
            rms,
            zcr,
            dyn,
        ]
    )
    return F.normalize(feature, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


def token_set_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a_set = set(a.tolist())
    b_set = set(b.tolist())
    if not a_set or not b_set:
        return 0.0
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return intersection / max(1, union)


def best_local_match_ratio(query_codes: torch.Tensor, candidate_codes: torch.Tensor, scan_step: int) -> Tuple[float, int]:
    if query_codes.numel() == 0 or candidate_codes.numel() == 0:
        return 0.0, 0

    query = query_codes.view(-1)
    candidate = candidate_codes.view(-1)

    if candidate.shape[0] < query.shape[0]:
        window = candidate.shape[0]
        query_window = query[:window]
        ratio = (query_window == candidate).float().mean().item()
        return ratio, 0

    window = query.shape[0]
    best_ratio = -1.0
    best_start = 0
    step = max(1, scan_step)
    for start in range(0, candidate.shape[0] - window + 1, step):
        ratio = (query == candidate[start:start + window]).float().mean().item()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start

    final_start = candidate.shape[0] - window
    if final_start >= 0 and final_start != best_start and final_start % step != 0:
        ratio = (query == candidate[final_start:final_start + window]).float().mean().item()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = final_start

    return max(0.0, best_ratio), best_start


def prefix_match_ratio(query_codes: torch.Tensor, candidate_codes: torch.Tensor) -> float:
    overlap = min(query_codes.shape[0], candidate_codes.shape[0])
    if overlap <= 0:
        return 0.0
    return (query_codes[:overlap] == candidate_codes[:overlap]).float().mean().item()


def combined_origin_score(code_local: float, code_prefix: float, code_jaccard: float, audio_cosine: float) -> float:
    return (
        0.50 * code_local
        + 0.20 * code_prefix
        + 0.15 * code_jaccard
        + 0.15 * max(0.0, audio_cosine)
    )


def main():
    args = parse_args()
    device = get_device(args.allow_cpu)
    input_path = resolve_input_audio(args.input_audio)

    tokenizer_model, tokenizer_config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    items = load_dataset_items(tokenizer_config.metadata_csv, tokenizer_config.audio_dir)
    if not items:
        raise RuntimeError("No dataset audio files found from tokenizer config.")

    print(f"Input audio: {input_path}")
    print(f"Dataset candidates: {len(items)}")
    print(f"Encoding input with tokenizer on {device}...")
    query_codes = encode_codes(str(input_path), tokenizer_model, tokenizer_config.sample_rate, device, args.max_input_seconds)
    query_feature = load_audio_feature(str(input_path))

    ranked: List[Tuple[float, Dict[str, object]]] = []
    for item in items:
        try:
            candidate_codes = encode_codes(item["path"], tokenizer_model, tokenizer_config.sample_rate, device, args.max_source_seconds)
            candidate_feature = load_audio_feature(item["path"])

            code_local, best_start = best_local_match_ratio(query_codes, candidate_codes, args.scan_step)
            code_prefix = prefix_match_ratio(query_codes, candidate_codes)
            code_jaccard = token_set_jaccard(query_codes, candidate_codes)
            audio_cos = cosine_similarity(query_feature, candidate_feature)
            total_score = combined_origin_score(code_local, code_prefix, code_jaccard, audio_cos)

            ranked.append(
                (
                    total_score,
                    {
                        "file": item["file"],
                        "path": item["path"],
                        "text": item["text"],
                        "score": total_score,
                        "code_local": code_local,
                        "code_prefix": code_prefix,
                        "code_jaccard": code_jaccard,
                        "audio_cosine": audio_cos,
                        "best_start": best_start,
                    },
                )
            )
        except Exception as exc:
            print(f"Skipping unreadable candidate {item['file']} ({exc})")

    ranked.sort(key=lambda pair: (pair[0], pair[1]["code_local"], pair[1]["audio_cosine"]), reverse=True)

    if not ranked:
        raise RuntimeError("No readable dataset candidates could be compared.")

    best = ranked[0][1]
    print()
    print("Most likely originator:")
    print(f"- file: {best['file']}")
    print(f"- path: {best['path']}")
    print(f"- combined_score: {best['score']:.4f}")
    print(f"- local_code_match: {best['code_local']:.4f}")
    print(f"- prefix_code_match: {best['code_prefix']:.4f}")
    print(f"- token_jaccard: {best['code_jaccard']:.4f}")
    print(f"- audio_cosine: {best['audio_cosine']:.4f}")
    print(f"- best_alignment_start_token: {best['best_start']}")
    print(f"- metadata: {best['text']}")
    print()
    print(f"Top {min(args.top_k, len(ranked))} matches:")
    for index, (_, row) in enumerate(ranked[:args.top_k], start=1):
        print(
            f"{index:>2}. {row['file']} | score={row['score']:.4f} | local={row['code_local']:.4f} | "
            f"prefix={row['code_prefix']:.4f} | jaccard={row['code_jaccard']:.4f} | audio={row['audio_cosine']:.4f}"
        )


if __name__ == "__main__":
    main()
