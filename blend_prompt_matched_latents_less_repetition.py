import argparse
import hashlib
import math
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

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
        description="Blend prompt-matched latent windows with diversity-aware source selection and anti-repetition window scheduling."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--num-sources", type=int, default=4)
    parser.add_argument("--candidate-pool", type=int, default=32)
    parser.add_argument("--diversity-weight", type=float, default=0.75)
    parser.add_argument("--window-steps", type=int, default=256)
    parser.add_argument("--window-jitter", type=float, default=0.25)
    parser.add_argument("--overlap-steps", type=int, default=64)
    parser.add_argument("--max-source-seconds", type=float, default=30.0)
    parser.add_argument("--recent-source-history", type=int, default=2)
    parser.add_argument("--min-window-separation", type=int, default=192)
    parser.add_argument("--candidate-starts", type=int, default=32)
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
    digest = hashlib.sha1(f"less_repeat_blend_{prompt}_{datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
    return f"latent_blend_less_repeat_{timestamp}_{digest}.wav"


def prompt_tokens(text: str) -> Set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def score_prompt_match(prompt: str, text: str) -> float:
    prompt_set = prompt_tokens(prompt)
    text_set = prompt_tokens(text)
    if not prompt_set or not text_set:
        return 0.0
    overlap = len(prompt_set & text_set)
    contains_bonus = 2.0 if prompt.lower().strip() and prompt.lower().strip() in (text or "").lower() else 0.0
    return contains_bonus + overlap + (overlap / max(1, len(prompt_set)))


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


@torch.no_grad()
def encode_source_codes(
    item: Dict[str, str],
    tokenizer_model,
    config,
    device: torch.device,
    max_source_seconds: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, float]:
    waveform = load_audio_mono(item["path"], config.sample_rate)
    excerpt_start_seconds = 0.0

    if max_source_seconds > 0:
        max_samples = int(round(max_source_seconds * config.sample_rate))
        total_samples = waveform.shape[-1]
        if total_samples > max_samples:
            max_start = total_samples - max_samples
            start = rng.randint(0, max_start)
            end = start + max_samples
            excerpt_start_seconds = start / config.sample_rate
            waveform = waveform[..., start:end]

    codes = tokenizer_model.encode_codes(waveform.unsqueeze(0).to(device))
    return codes.squeeze(0).cpu(), excerpt_start_seconds


@torch.no_grad()
def lookup_quantized_window(tokenizer_model, codes: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tokenizer_model.quantizer.lookup(codes.unsqueeze(0).to(device))


def choose_sources(
    prompt: str,
    items: List[Dict[str, str]],
    num_sources: int,
    candidate_pool: int,
    diversity_weight: float,
) -> List[Tuple[float, Dict[str, str]]]:
    candidates = []
    for item in items:
        text = item.get("text", "")
        token_set = prompt_tokens(text)
        prompt_score = score_prompt_match(prompt, text)
        candidates.append({
            "prompt_score": prompt_score,
            "item": item,
            "tokens": token_set,
        })

    candidates.sort(key=lambda entry: entry["prompt_score"], reverse=True)
    pool = candidates[:max(num_sources, candidate_pool)]
    positive_pool = [entry for entry in pool if entry["prompt_score"] > 0]
    working_pool = positive_pool if len(positive_pool) >= num_sources else pool

    selected = []
    remaining = working_pool.copy()
    while remaining and len(selected) < num_sources:
        best_index = 0
        best_value = None
        for index, entry in enumerate(remaining):
            if not selected:
                adjusted = entry["prompt_score"]
            else:
                max_similarity = max(
                    jaccard_similarity(entry["tokens"], chosen["tokens"])
                    for chosen in selected
                )
                adjusted = entry["prompt_score"] - (diversity_weight * max_similarity)

            if best_value is None or adjusted > best_value:
                best_value = adjusted
                best_index = index

        chosen = remaining.pop(best_index)
        selected.append(chosen)

    return [(entry["prompt_score"], entry["item"]) for entry in selected]


def sample_window_length(target_steps: int, args, rng: random.Random) -> int:
    jitter_ratio = max(0.0, args.window_jitter)
    base = max(16, args.window_steps)
    min_steps = max(16, int(round(base * max(0.5, 1.0 - jitter_ratio))))
    max_steps = max(min_steps, int(round(base * (1.0 + jitter_ratio))))
    return min(target_steps, rng.randint(min_steps, max_steps))


def circular_window(codes: torch.Tensor, start: int, window_steps: int) -> torch.Tensor:
    total_steps = codes.shape[0]
    if total_steps == 0:
        raise ValueError("Cannot sample from empty code sequence")
    if total_steps >= window_steps:
        return codes[start:start + window_steps].clone()

    indices = (torch.arange(window_steps) + start) % total_steps
    return codes.index_select(0, indices).clone()


def score_start_candidate(start: int, previous_starts: List[int], min_window_separation: int) -> float:
    if not previous_starts:
        return float(min_window_separation)
    nearest_distance = min(abs(start - previous) for previous in previous_starts)
    return float(nearest_distance)


def sample_code_window_less_repetition(
    codes: torch.Tensor,
    window_steps: int,
    rng: random.Random,
    previous_starts: List[int],
    min_window_separation: int,
    candidate_starts: int,
) -> Tuple[torch.Tensor, int]:
    if codes.numel() == 0:
        raise ValueError("Cannot sample from empty code sequence")

    total_steps = codes.shape[0]
    if total_steps <= window_steps:
        start = rng.randint(0, max(0, total_steps - 1))
        return circular_window(codes, start, window_steps), start

    max_start = total_steps - window_steps
    if max_start <= 0:
        return codes[:window_steps].clone(), 0

    if not previous_starts:
        start = rng.randint(0, max_start)
        return codes[start:start + window_steps].clone(), start

    best_start: Optional[int] = None
    best_score: Optional[float] = None
    tries = max(4, candidate_starts)
    for _ in range(tries):
        candidate = rng.randint(0, max_start)
        distance_score = score_start_candidate(candidate, previous_starts, min_window_separation)
        spacing_bonus = min(distance_score / max(1, min_window_separation), 1.5)
        random_bonus = rng.random() * 0.25
        candidate_score = distance_score + spacing_bonus + random_bonus
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_start = candidate

    assert best_start is not None
    return codes[best_start:best_start + window_steps].clone(), best_start


def pick_next_source_index(
    source_entries: List[Dict[str, object]],
    recent_history: List[int],
    rng: random.Random,
    recent_source_history: int,
) -> int:
    excluded = set(recent_history[-recent_source_history:]) if recent_source_history > 0 else set()
    candidate_indices = [index for index in range(len(source_entries)) if index not in excluded]
    if not candidate_indices:
        candidate_indices = list(range(len(source_entries)))

    weights = []
    for index in candidate_indices:
        entry = source_entries[index]
        prompt_score = float(entry["match_score"])
        usage_count = int(entry["usage_count"])
        weight = max(0.05, (prompt_score + 1.0) / (1.0 + usage_count * 0.35))
        weights.append(weight)

    total_weight = sum(weights)
    pick = rng.random() * total_weight
    running = 0.0
    for index, weight in zip(candidate_indices, weights):
        running += weight
        if pick <= running:
            return index
    return candidate_indices[-1]


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


@torch.no_grad()
def main():
    args = parse_args()
    start_time = time.perf_counter()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.allow_cpu)

    tokenizer_model, config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    items = load_dataset_items(config.metadata_csv, config.audio_dir)
    if not items:
        raise RuntimeError("No dataset items found.")

    selected = choose_sources(
        args.prompt,
        items,
        args.num_sources,
        args.candidate_pool,
        args.diversity_weight,
    )
    if not selected:
        raise RuntimeError("No source files available for blending.")

    source_entries: List[Dict[str, object]] = []
    print("Selected sources:")
    for match_score, item in selected:
        codes, excerpt_start_seconds = encode_source_codes(
            item,
            tokenizer_model,
            config,
            device,
            args.max_source_seconds,
            rng,
        )
        entry = {
            "match_score": match_score,
            "item": item,
            "codes": codes,
            "excerpt_start_seconds": excerpt_start_seconds,
            "usage_count": 0,
            "window_starts": [],
        }
        source_entries.append(entry)
        print(
            f"- {item['file']} | score={match_score:.2f} | excerpt_start={excerpt_start_seconds:.2f}s | {item['text']}"
        )

    target_samples = int(round(args.duration_seconds * config.sample_rate))
    target_steps = math.ceil(target_samples / config.total_stride)
    overlap_steps = max(0, min(args.overlap_steps, max(16, args.window_steps) // 2))

    quantized_chunks: List[torch.Tensor] = []
    latent_steps_written = 0
    recent_source_history: List[int] = []

    while latent_steps_written < target_steps:
        source_index = pick_next_source_index(
            source_entries,
            recent_source_history,
            rng,
            args.recent_source_history,
        )
        source = source_entries[source_index]

        remaining_steps = target_steps - latent_steps_written
        window_steps = sample_window_length(remaining_steps, args, rng)
        codes = source["codes"]
        window_starts = source["window_starts"]
        code_window, start = sample_code_window_less_repetition(
            codes,
            window_steps,
            rng,
            window_starts,
            args.min_window_separation,
            args.candidate_starts,
        )
        quantized = lookup_quantized_window(tokenizer_model, code_window, device)
        quantized_chunks.append(quantized)

        window_starts.append(start)
        source["usage_count"] = int(source["usage_count"]) + 1
        recent_source_history.append(source_index)

        if latent_steps_written == 0:
            latent_steps_written += code_window.shape[0]
        else:
            latent_steps_written += max(1, code_window.shape[0] - overlap_steps)

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
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
