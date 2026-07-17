import argparse
import hashlib
import math
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from latent_audio_token_pipeline import (
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
    load_latent_prior_bundle,
    save_audio_waveform,
    stitch_waveforms,
)


WORD_RE = re.compile(r"[a-z0-9']+")


def code_step_count(codes: torch.Tensor) -> int:
    return int(codes.shape[-1])


def empty_code_sequence(num_quantizers: int) -> torch.Tensor:
    if num_quantizers > 1:
        return torch.empty((num_quantizers, 0), dtype=torch.long)
    return torch.empty(0, dtype=torch.long)


def extract_code_tail(codes: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 0:
        return codes[..., :0]
    return codes[..., -length:]


def ensure_batched_codes(codes: torch.Tensor) -> torch.Tensor:
    return codes.unsqueeze(0)


def concat_code_sequences(*parts: torch.Tensor) -> torch.Tensor:
    valid_parts = [part for part in parts if part is not None and part.numel() > 0]
    if not valid_parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(valid_parts, dim=-1)


def slice_code_steps(codes: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return codes[..., start:end]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prompt-conditioned audio that stays close to source tokens but preserves some prior-driven creativity."
    )
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--prior-dir", default="latent_audio_prior_out")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-count", type=int, default=1)
    parser.add_argument("--beginning-bos-clips", type=int, default=1, help="Use beginning BOS for the first N generated clips. Set to 0 to disable it.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Target output duration in seconds. If set, overrides --clip-count.",
    )
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--source-candidates", type=int, default=8)
    parser.add_argument("--source-window", type=int, default=256)
    parser.add_argument("--source-overlap", type=int, default=64)
    parser.add_argument("--max-source-seconds", type=float, default=30.0)
    parser.add_argument("--proposal-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=3.0)
    parser.add_argument("--match-weight", type=float, default=0.5)
    parser.add_argument("--scan-step-divisor", type=int, default=4)
    parser.add_argument("--source-strength", type=float, default=0.8, help="Higher means closer to source tokens.")
    parser.add_argument("--window-choice-top", type=int, default=4, help="Choose among the top-N matching source windows.")
    parser.add_argument("--window-choice-temperature", type=float, default=0.35, help="Lower means more likely to choose the best source window.")
    parser.add_argument("--creative-span-count", type=int, default=3, help="How many prior-driven spans to inject per window.")
    parser.add_argument("--creative-span-min", type=int, default=8)
    parser.add_argument("--creative-span-max", type=int, default=32)
    parser.add_argument("--creative-token-mix", type=float, default=0.08, help="Small per-token chance to keep proposal tokens outside creative spans.")
    parser.add_argument("--fade-ms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for latent inference. Re-run with --allow-cpu to override.")


def make_output_name(prompt: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"source_creative_{prompt}_{datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
    return f"latent_generated_source_creative_{timestamp}_{digest}.wav"


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
    codes = tokenizer_model.encode_codes(
        waveform.unsqueeze(0).to(device),
        return_all_codes=(getattr(tokenizer_model.config, "num_quantizers", 1) > 1),
    )
    return codes.squeeze(0).cpu()


@torch.no_grad()
def build_source_entries(
    prompt: str,
    tokenizer_model,
    config,
    device: torch.device,
    limit: int,
    max_source_seconds: float,
    target_num_quantizers: Optional[int] = None,
) -> List[Dict]:
    if limit <= 0:
        return []

    items = load_dataset_items(config.metadata_csv, config.audio_dir)
    if not items:
        return []

    scored = []
    for item in items:
        score = score_prompt_match(prompt, item["text"])
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    chosen = [pair for pair in scored[:limit] if pair[0] > 0]
    if not chosen:
        chosen = scored[:limit]

    entries: List[Dict] = []
    for score, item in chosen:
        try:
            codes = encode_source_codes(item, tokenizer_model, config, device, max_source_seconds)
            if target_num_quantizers == 1 and codes.dim() == 2:
                codes = codes[0]
            if codes.numel() == 0:
                continue
            entries.append(
                {
                    "codes": codes,
                    "text": item["text"],
                    "path": item["path"],
                    "file": item["file"],
                    "match_score": score,
                }
            )
        except Exception as exc:
            print(f"Skipping source candidate {item['path']} ({exc})")
    return entries


def find_source_window_candidates(
    proposal_full: torch.Tensor,
    prefix_codes: Optional[torch.Tensor],
    candidate_entries: List[Dict],
    overlap_size: int,
    proposal_weight: float,
    continuity_weight: float,
    match_weight: float,
    scan_step_divisor: int,
) -> List[Dict]:
    proposal_full = proposal_full.cpu()
    prefix_len = 0 if prefix_codes is None else min(overlap_size, code_step_count(prefix_codes))
    prefix_tail = None if prefix_codes is None else extract_code_tail(prefix_codes, prefix_len).cpu()
    window_size = code_step_count(proposal_full)
    step = max(1, window_size // max(1, scan_step_divisor))
    candidates: List[Dict] = []

    for entry in candidate_entries:
        seq = entry["codes"]
        if code_step_count(seq) < window_size:
            continue
        for start in range(0, code_step_count(seq) - window_size + 1, step):
            window = slice_code_steps(seq, start, start + window_size)
            proposal_match = (window == slice_code_steps(proposal_full, 0, window_size)).float().mean().item()
            continuity = 0.0
            if prefix_tail is not None and prefix_len > 0:
                continuity = (slice_code_steps(window, 0, prefix_len) == prefix_tail).float().mean().item()
            score = (
                (continuity_weight * continuity)
                + (proposal_weight * proposal_match)
                + (match_weight * entry["match_score"])
            )
            candidates.append(
                {
                    "score": score,
                    "window": window.clone(),
                    "entry": entry,
                    "start": start,
                    "proposal_match": proposal_match,
                    "continuity": continuity,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def choose_source_window_creatively(candidates: List[Dict], top_n: int, temperature: float, rng: random.Random) -> Optional[Dict]:
    if not candidates:
        return None
    working = candidates[:max(1, top_n)]
    if len(working) == 1 or temperature <= 1e-6:
        return working[0]

    max_score = max(candidate["score"] for candidate in working)
    weights = []
    for candidate in working:
        weights.append(math.exp((candidate["score"] - max_score) / max(temperature, 1e-6)))

    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for candidate, weight in zip(working, weights):
        running += weight
        if pick <= running:
            return candidate
    return working[-1]


def inject_creative_spans(
    mixed_new: torch.Tensor,
    proposal_new: torch.Tensor,
    args,
    rng: random.Random,
) -> torch.Tensor:
    result = mixed_new.clone()
    total = code_step_count(result)
    if total <= 0:
        return result

    span_count = max(0, args.creative_span_count)
    min_span = max(1, args.creative_span_min)
    max_span = max(min_span, args.creative_span_max)
    for _ in range(span_count):
        if total <= 0:
            break
        span_len = min(total, rng.randint(min_span, max_span))
        max_start = max(0, total - span_len)
        start = rng.randint(0, max_start)
        end = start + span_len
        result[..., start:end] = proposal_new[..., start:end]

    token_mix = max(0.0, min(1.0, args.creative_token_mix))
    if token_mix > 0:
        mask = torch.rand(total) < token_mix
        result[..., mask] = proposal_new[..., mask]
    return result


def fuse_source_and_proposal_window(
    proposal_full: torch.Tensor,
    source_window: torch.Tensor,
    prefix_len: int,
    args,
    rng: random.Random,
) -> torch.Tensor:
    if prefix_len <= 0:
        source_new = source_window.clone()
        proposal_new = proposal_full.clone()
        anchored = source_new.clone()
    else:
        anchored = source_window.clone()
        anchored[..., :prefix_len] = source_window[..., :prefix_len]
        source_new = source_window[..., prefix_len:].clone()
        proposal_new = proposal_full[..., prefix_len:].clone()

    source_strength = max(0.0, min(1.0, args.source_strength))
    creative = inject_creative_spans(source_new, proposal_new, args, rng)

    if source_strength >= 0.999:
        fused_new = source_new
    elif source_strength <= 0.001:
        fused_new = creative
    else:
        fused_new = source_new.clone()
        keep_source_mask = torch.rand(code_step_count(source_new)) < source_strength
        fused_new[..., ~keep_source_mask] = creative[..., ~keep_source_mask]

    if prefix_len > 0:
        anchored[..., prefix_len:] = fused_new
        return anchored
    return fused_new


@torch.no_grad()
def generate_source_creative_codes(args, prior_model, text_tokens, text_mask, config, candidate_entries, device: torch.device) -> torch.Tensor:
    total_steps = config.latent_steps
    window_size = max(32, min(args.source_window, total_steps))
    overlap_size = max(0, min(args.source_overlap, window_size // 2))
    generated = empty_code_sequence(getattr(prior_model, "num_quantizers", 1))
    rng = random.Random(args.seed)

    while code_step_count(generated) < total_steps:
        prefix_codes = None
        prefix_len = 0
        if overlap_size > 0 and code_step_count(generated) > 0:
            prefix_codes = extract_code_tail(generated, overlap_size)
            prefix_len = code_step_count(prefix_codes)

        new_steps = min(window_size if prefix_len == 0 else (window_size - prefix_len), total_steps - code_step_count(generated))
        generated_new = prior_model.generate(
            text_tokens=text_tokens,
            text_mask=text_mask,
            num_steps=new_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            prefix_codes=(None if prefix_codes is None else ensure_batched_codes(prefix_codes)),
            song_beginning=bool(getattr(args, "song_beginning", False) and prefix_codes is None),
            device=device,
        ).squeeze(0).cpu()

        proposal_full = generated_new if prefix_codes is None else concat_code_sequences(prefix_codes.cpu(), generated_new)
        candidates = find_source_window_candidates(
            proposal_full,
            prefix_codes,
            candidate_entries,
            overlap_size,
            args.proposal_weight,
            args.continuity_weight,
            args.match_weight,
            args.scan_step_divisor,
        )
        chosen = choose_source_window_creatively(
            candidates,
            args.window_choice_top,
            args.window_choice_temperature,
            rng,
        )

        if chosen is not None:
            fused_window = fuse_source_and_proposal_window(
                proposal_full,
                chosen["window"],
                prefix_len,
                args,
                rng,
            )
            chosen_new = slice_code_steps(fused_window, prefix_len, prefix_len + new_steps)
        else:
            chosen_new = generated_new

        generated = concat_code_sequences(generated, chosen_new)

    return generated.unsqueeze(0).to(device)


def main():
    args = parse_args()
    start_time = time.perf_counter()
    device = get_device(args.allow_cpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer_model, tokenizer_config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    prior_model, text_tokenizer, prior_config = load_latent_prior_bundle(args.prior_dir, device)

    if tokenizer_config.codebook_size != prior_config.codebook_size:
        raise RuntimeError("Tokenizer and prior codebook sizes do not match.")

    clip_count = args.clip_count
    target_samples = None
    if args.duration_seconds > 0:
        target_samples = int(round(args.duration_seconds * tokenizer_config.sample_rate))
        clip_duration = tokenizer_config.clip_seconds
        clip_count = max(1, math.ceil(args.duration_seconds / clip_duration))

    text_tokens = text_tokenizer.encode(args.prompt, prior_config.max_text_tokens).unsqueeze(0)
    text_mask = text_tokenizer.attention_mask(text_tokens)

    candidate_entries = build_source_entries(
        args.prompt,
        tokenizer_model,
        prior_config,
        device,
        args.source_candidates,
        args.max_source_seconds,
        target_num_quantizers=getattr(prior_model, "num_quantizers", 1),
    )
    if not candidate_entries:
        raise RuntimeError("No source candidates found for source-creative generation.")

    print(f"Loaded {len(candidate_entries)} source candidates")
    for entry in candidate_entries:
        print(f"- {entry['file']} | score={entry['match_score']:.2f} | {entry['text']}")

    clips = []
    for clip_idx in range(clip_count):
        args.song_beginning = clip_idx < max(0, args.beginning_bos_clips)
        print(f"Generating source-creative latent clip {clip_idx + 1}/{clip_count} on {device}...")
        codes = generate_source_creative_codes(
            args,
            prior_model,
            text_tokens,
            text_mask,
            prior_config,
            candidate_entries,
            device,
        )
        codes = codes.to(device=device, dtype=torch.long)
        waveform = tokenizer_model.decode_codes(codes, target_length=tokenizer_config.clip_samples)
        clips.append(waveform.squeeze(0).cpu())

    output = stitch_waveforms(clips, tokenizer_config.sample_rate, fade_ms=args.fade_ms)
    if target_samples is not None:
        output = output[..., :target_samples]
    output = output - output.mean(dim=-1, keepdim=True)
    peak = output.abs().max().item()
    if peak > 0:
        output = output / max(1.0, peak / 0.98)
    rms = output.pow(2).mean().sqrt().item()
    target_rms = 0.14
    if rms > 1e-6:
        output = output * min(1.5, target_rms / rms)
        peak = output.abs().max().item()
        if peak > 0.98:
            output = output * (0.98 / peak)

    output_path = args.output or make_output_name(args.prompt)
    save_audio_waveform(output_path, output, tokenizer_config.sample_rate)
    print(f"Saved source-creative latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
