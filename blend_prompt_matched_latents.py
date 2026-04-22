import argparse
import hashlib
import math
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple

import torch

from latent_audio_token_pipeline import (
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
    match_audio_length,
    save_audio_waveform,
)


WORD_RE = re.compile(r"[a-z0-9']+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blend top prompt-matched latent windows from multiple dataset files into a single generated audio file."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--num-sources", type=int, default=4)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--window-steps", type=int, default=256)
    parser.add_argument("--overlap-steps", type=int, default=64)
    parser.add_argument("--max-source-seconds", type=float, default=30.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required. Re-run with --allow-cpu to override.")


def make_output_name(prompt: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"blend_{prompt}_{datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
    return f"latent_blend_{timestamp}_{digest}.wav"


def prompt_tokens(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def score_prompt_match(prompt: str, text: str) -> float:
    prompt_set = prompt_tokens(prompt)
    text_set = prompt_tokens(text)
    if not prompt_set or not text_set:
        return 0.0
    overlap = len(prompt_set & text_set)
    contains_bonus = 2.0 if prompt.lower().strip() and prompt.lower().strip() in (text or "").lower() else 0.0
    return contains_bonus + overlap + (overlap / max(1, len(prompt_set)))


@torch.no_grad()
def encode_source_codes(item: Dict[str, str], tokenizer_model, config, device: torch.device, max_source_seconds: float) -> torch.Tensor:
    waveform = load_audio_mono(item["path"], config.sample_rate)
    if max_source_seconds > 0:
        max_samples = int(round(max_source_seconds * config.sample_rate))
        waveform = waveform[..., :max_samples]
    codes = tokenizer_model.encode_codes(waveform.unsqueeze(0).to(device))
    return codes.squeeze(0).cpu()


@torch.no_grad()
def lookup_quantized_window(tokenizer_model, codes: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tokenizer_model.quantizer.lookup(codes.unsqueeze(0).to(device))


def sample_code_window(codes: torch.Tensor, window_steps: int, rng: random.Random) -> torch.Tensor:
    if codes.numel() == 0:
        raise ValueError("Cannot sample from empty code sequence")
    if codes.shape[0] >= window_steps:
        start = rng.randint(0, codes.shape[0] - window_steps)
        return codes[start:start + window_steps].clone()

    repeats = math.ceil(window_steps / max(1, codes.shape[0]))
    tiled = codes.repeat(repeats)
    return tiled[:window_steps].clone()


def blend_quantized_chunks(chunks: List[torch.Tensor], overlap_steps: int) -> torch.Tensor:
    if not chunks:
        raise ValueError("No latent chunks to blend")
    out = chunks[0]
    for nxt in chunks[1:]:
        effective = min(overlap_steps, out.shape[-1] // 2, nxt.shape[-1] // 2)
        if effective <= 0:
            out = torch.cat([out, nxt], dim=-1)
            continue
        fade = torch.linspace(0.0, 1.0, effective, device=out.device).view(1, 1, effective)
        mixed = out[..., -effective:] * (1.0 - fade) + nxt[..., :effective] * fade
        out = torch.cat([out[..., :-effective], mixed, nxt[..., effective:]], dim=-1)
    return out


def choose_sources(prompt: str, items: List[Dict[str, str]], num_sources: int, candidate_pool: int) -> List[Tuple[float, Dict[str, str]]]:
    scored = [(score_prompt_match(prompt, item["text"]), item) for item in items]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    pool = scored[:max(num_sources, candidate_pool)]
    chosen = [pair for pair in pool if pair[0] > 0][:num_sources]
    if len(chosen) < num_sources:
        chosen = pool[:num_sources]
    return chosen


@torch.no_grad()
def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.allow_cpu)

    tokenizer_model, config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    items = load_dataset_items(config.metadata_csv, config.audio_dir)
    if not items:
        raise RuntimeError("No dataset items found.")

    selected = choose_sources(args.prompt, items, args.num_sources, args.candidate_pool)
    if not selected:
        raise RuntimeError("No source files available for blending.")

    source_entries = []
    print("Selected sources:")
    for match_score, item in selected:
        codes = encode_source_codes(item, tokenizer_model, config, device, args.max_source_seconds)
        source_entries.append({
            "match_score": match_score,
            "item": item,
            "codes": codes,
        })
        print(f"- {item['file']} | score={match_score:.2f} | {item['text']}")

    target_samples = int(round(args.duration_seconds * config.sample_rate))
    target_steps = math.ceil(target_samples / config.total_stride)
    window_steps = max(16, min(args.window_steps, target_steps))
    overlap_steps = max(0, min(args.overlap_steps, window_steps // 2))

    quantized_chunks: List[torch.Tensor] = []
    latent_steps_written = 0
    source_index = 0

    while latent_steps_written < target_steps:
        source = source_entries[source_index % len(source_entries)]
        code_window = sample_code_window(source["codes"], window_steps, rng)
        quantized = lookup_quantized_window(tokenizer_model, code_window, device)
        quantized_chunks.append(quantized)

        if latent_steps_written == 0:
            latent_steps_written += code_window.shape[0]
        else:
            latent_steps_written += max(1, code_window.shape[0] - overlap_steps)
        source_index += 1

    blended_latents = blend_quantized_chunks(quantized_chunks, overlap_steps)
    blended_latents = blended_latents[..., :target_steps]

    decoded = tokenizer_model.decoder(tokenizer_model.post_quant(blended_latents))
    decoded = match_audio_length(decoded, target_samples).cpu()

    peak = decoded.abs().max().item()
    if peak > 0:
        decoded = decoded / max(1.0, peak / 0.98)

    output_path = args.output or make_output_name(args.prompt)
    save_audio_waveform(output_path, decoded.squeeze(0), config.sample_rate)
    print(f"Saved blended latent audio to {output_path}")


if __name__ == "__main__":
    main()
