import argparse
import hashlib
import heapq
import math
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from latent_audio_token_pipeline import (
    latent_bos_token,
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
    load_latent_prior_bundle,
    save_audio_waveform,
    stitch_waveforms,
)


WORD_RE = re.compile(r"[a-z0-9']+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prompt-conditioned audio with source guidance and rank-relaxed token sampling that can prefer 2nd/3rd most likely tokens."
    )
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--prior-dir", default="latent_audio_prior_out")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-count", type=int, default=1)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Target output duration in seconds. If set, overrides --clip-count.",
    )
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--rank-choice-prob", type=float, default=0.35, help="Chance to force selection from the 2nd/3rd most likely tokens instead of the top token.")
    parser.add_argument("--rank-choice-top", type=int, default=3, help="Consider non-top choices up to this rank.")
    parser.add_argument("--rank-choice-temperature", type=float, default=0.75, help="Higher values flatten preference across 2nd/3rd ranked choices.")
    parser.add_argument("--source-candidates", type=int, default=10)
    parser.add_argument("--source-window", type=int, default=256)
    parser.add_argument("--source-overlap", type=int, default=64)
    parser.add_argument("--max-source-seconds", type=float, default=30.0)
    parser.add_argument("--proposal-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=2.75)
    parser.add_argument("--match-weight", type=float, default=0.45)
    parser.add_argument("--scan-step-divisor", type=int, default=4)
    parser.add_argument("--source-strength", type=float, default=0.65, help="Higher means closer to source tokens.")
    parser.add_argument("--window-choice-top", type=int, default=6, help="Choose among the top-N matching source windows.")
    parser.add_argument("--window-choice-temperature", type=float, default=0.7, help="Higher means more diverse source window choice.")
    parser.add_argument("--creative-span-count", type=int, default=5, help="How many prior-driven spans to inject per window.")
    parser.add_argument("--creative-span-min", type=int, default=8)
    parser.add_argument("--creative-span-max", type=int, default=40)
    parser.add_argument("--creative-token-mix", type=float, default=0.16, help="Per-token chance to keep proposal tokens outside creative spans.")
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
    digest = hashlib.sha1(f"source_creative_ranked_{prompt}_{datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
    return f"latent_generated_source_creative_ranked_{timestamp}_{digest}.wav"


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


def code_step_count(codes: torch.Tensor) -> int:
    return int(codes.shape[-1])


def empty_code_sequence(num_quantizers: int) -> torch.Tensor:
    if num_quantizers > 1:
        return torch.empty((num_quantizers, 0), dtype=torch.long)
    return torch.empty(0, dtype=torch.long)


def slice_code_steps(codes: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return codes[..., start:end]


def concat_code_sequences(*parts: torch.Tensor) -> torch.Tensor:
    valid_parts = [part for part in parts if part is not None and part.numel() > 0]
    if not valid_parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(valid_parts, dim=-1)


def extract_code_tail(codes: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 0:
        return codes[..., :0]
    return codes[..., -length:]


def ensure_batched_codes(codes: torch.Tensor) -> torch.Tensor:
    return codes.unsqueeze(0)


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
    max_candidates: Optional[int] = None,
) -> List[Dict]:
    proposal_full = proposal_full.cpu()
    prefix_len = 0 if prefix_codes is None else min(overlap_size, code_step_count(prefix_codes))
    prefix_tail = None if prefix_codes is None else extract_code_tail(prefix_codes, prefix_len).cpu()
    window_size = code_step_count(proposal_full)
    step = max(1, window_size // max(1, scan_step_divisor))
    candidate_limit = None if max_candidates is None else max(1, int(max_candidates))
    candidates = []
    candidate_index = 0

    for entry in candidate_entries:
        seq = entry["codes"]
        if code_step_count(seq) < window_size:
            continue
        windows = seq.unfold(-1, window_size, step).movedim(-2, 0)
        proposal_matches = (
            (windows == slice_code_steps(proposal_full, 0, window_size))
            .float()
            .mean(dim=tuple(range(1, windows.dim())))
            .tolist()
        )
        if prefix_tail is not None and prefix_len > 0:
            continuities = (
                (windows[..., :prefix_len] == prefix_tail)
                .float()
                .mean(dim=tuple(range(1, windows.dim())))
                .tolist()
            )
        else:
            continuities = [0.0] * len(proposal_matches)

        for window_idx, (proposal_match, continuity) in enumerate(zip(proposal_matches, continuities)):
            start = window_idx * step
            score = (
                (continuity_weight * continuity)
                + (proposal_weight * proposal_match)
                + (match_weight * entry["match_score"])
            )
            candidate = (score, -candidate_index, candidate_index, entry, start)
            if candidate_limit is None:
                candidates.append(candidate)
            elif len(candidates) < candidate_limit:
                heapq.heappush(candidates, candidate)
            elif candidate[:2] > candidates[0][:2]:
                heapq.heapreplace(candidates, candidate)
            candidate_index += 1

    candidates.sort(key=lambda item: (-item[0], item[2]))
    return [
        {
            "score": score,
            "window": slice_code_steps(entry["codes"], start, start + window_size).clone(),
            "entry": entry,
            "start": start,
        }
        for score, _, _, entry, start in candidates
    ]


def choose_source_window_creatively(candidates: List[Dict], top_n: int, temperature: float, rng: random.Random) -> Optional[Dict]:
    if not candidates:
        return None
    working = candidates[:max(1, top_n)]
    if len(working) == 1 or temperature <= 1e-6:
        return working[0]

    max_score = max(candidate["score"] for candidate in working)
    weights = [math.exp((candidate["score"] - max_score) / max(temperature, 1e-6)) for candidate in working]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for candidate, weight in zip(working, weights):
        running += weight
        if pick <= running:
            return candidate
    return working[-1]


def apply_repetition_penalty(logits: torch.Tensor, history: List[torch.Tensor], repetition_penalty: float, repetition_window: int) -> torch.Tensor:
    if repetition_penalty <= 1.0 or not history:
        return logits
    adjusted_logits = logits.clone()
    if logits.dim() == 2:
        recent = torch.cat(history[-max(1, repetition_window):], dim=1)
        for batch_idx in range(recent.shape[0]):
            unique_tokens = torch.unique(recent[batch_idx])
            token_logits = adjusted_logits[batch_idx, unique_tokens]
            adjusted = torch.where(
                token_logits >= 0,
                token_logits / repetition_penalty,
                token_logits * repetition_penalty,
            )
            adjusted_logits[batch_idx, unique_tokens] = adjusted
        return adjusted_logits

    recent = torch.stack(history[-max(1, repetition_window):], dim=-1)
    for batch_idx in range(recent.shape[0]):
        for quantizer_idx in range(recent.shape[1]):
            unique_tokens = torch.unique(recent[batch_idx, quantizer_idx])
            token_logits = adjusted_logits[batch_idx, quantizer_idx, unique_tokens]
            adjusted = torch.where(
                token_logits >= 0,
                token_logits / repetition_penalty,
                token_logits * repetition_penalty,
            )
            adjusted_logits[batch_idx, quantizer_idx, unique_tokens] = adjusted
    return adjusted_logits


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None and 0 < top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1].unsqueeze(-1)
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= top_p
        keep_mask[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def sample_rank_relaxed_next_code(logits: torch.Tensor, args, rng: random.Random) -> torch.Tensor:
    if logits.dim() == 3:
        next_codes = []
        for quantizer_idx in range(logits.shape[1]):
            sampled = sample_rank_relaxed_next_code(logits[:, quantizer_idx, :], args, rng)
            next_codes.append(sampled.squeeze(-1))
        return torch.stack(next_codes, dim=1)

    logits = logits / max(args.temperature, 1e-5)
    next_codes = []

    for batch_idx in range(logits.shape[0]):
        row = logits[batch_idx]
        if args.top_k is not None and 0 < args.top_k < row.shape[-1]:
            top_values = torch.topk(row, k=args.top_k, dim=-1).values
            valid_indices = torch.nonzero(
                torch.isfinite(row) & (row >= top_values[-1]),
                as_tuple=False,
            ).squeeze(-1)
        else:
            valid_indices = torch.nonzero(torch.isfinite(row), as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            next_codes.append(torch.argmax(logits[batch_idx]).view(1))
            continue

        valid_logits = row[valid_indices]
        order = torch.argsort(valid_logits, descending=True, stable=True)
        sorted_logits = valid_logits[order]
        sorted_indices = valid_indices[order]
        if args.top_p is not None and 0.0 < args.top_p < 1.0:
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            keep_mask = cumulative <= args.top_p
            keep_mask[0] = True
            sorted_logits = sorted_logits[keep_mask]
            sorted_indices = sorted_indices[keep_mask]
        probs = torch.softmax(sorted_logits, dim=-1)

        alt_cap = min(max(1, args.rank_choice_top), sorted_indices.numel())
        use_alt_rank = alt_cap >= 2 and (rng.random() < max(0.0, min(1.0, args.rank_choice_prob)))

        if use_alt_rank:
            alt_logits = sorted_logits[1:alt_cap]
            alt_indices = sorted_indices[1:alt_cap]
            alt_probs = torch.softmax(alt_logits / max(args.rank_choice_temperature, 1e-5), dim=-1)
            choice_pos = torch.multinomial(alt_probs, num_samples=1)
            next_code = alt_indices.gather(0, choice_pos)
        else:
            choice_pos = torch.multinomial(probs, num_samples=1)
            next_code = sorted_indices.gather(0, choice_pos)

        next_codes.append(next_code.view(1))

    return torch.stack(next_codes, dim=0)


@torch.no_grad()
def generate_rank_relaxed_window(args, prior_model, text_tokens, text_mask, num_steps: int, prefix_codes: Optional[torch.Tensor], device: torch.device, rng: random.Random) -> torch.Tensor:
    prior_model.eval()
    text_tokens = text_tokens.to(device)
    text_mask = text_mask.to(device)
    hidden = None
    text_cond = prior_model.encode_text(text_tokens, text_mask)
    current = prior_model.bos_codes(text_tokens.shape[0], device)
    outputs: List[torch.Tensor] = []
    history: List[torch.Tensor] = []

    if prefix_codes is not None and prefix_codes.numel() > 0:
        prefix_codes = prefix_codes.to(device=device, dtype=torch.long)
        if prefix_codes.dim() == 2 and getattr(prior_model, "num_quantizers", 1) == 1:
            prefix_codes = prefix_codes.unsqueeze(1)
        for step_idx in range(prefix_codes.shape[-1]):
            _, hidden = prior_model.forward_step(current, text_cond, hidden)
            current = prefix_codes[..., step_idx]
            history.append(current)

    for _ in range(num_steps):
        logits, hidden = prior_model.forward_step(current, text_cond, hidden)
        logits = apply_repetition_penalty(logits, history, args.repetition_penalty, args.repetition_window)

        if args.temperature <= 0:
            next_code = torch.argmax(logits, dim=-1)
        else:
            next_code = sample_rank_relaxed_next_code(logits, args, rng)

        outputs.append(next_code)
        history.append(next_code)
        current = next_code

    generated = torch.stack(outputs, dim=-1)
    if getattr(prior_model, "num_quantizers", 1) == 1:
        return generated[:, 0, :]
    return generated


def inject_creative_spans(mixed_new: torch.Tensor, proposal_new: torch.Tensor, args, rng: random.Random) -> torch.Tensor:
    result = mixed_new.clone()
    total = code_step_count(result)
    if total <= 0:
        return result

    span_count = max(0, args.creative_span_count)
    min_span = max(1, args.creative_span_min)
    max_span = max(min_span, args.creative_span_max)
    for _ in range(span_count):
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

    creative = inject_creative_spans(source_new, proposal_new, args, rng)
    source_strength = max(0.0, min(1.0, args.source_strength))

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
def generate_source_creative_ranked_codes(args, prior_model, text_tokens, text_mask, config, candidate_entries, device: torch.device) -> torch.Tensor:
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
        generated_new = generate_rank_relaxed_window(
            args,
            prior_model,
            text_tokens,
            text_mask,
            new_steps,
            prefix_codes=(None if prefix_codes is None else ensure_batched_codes(prefix_codes)),
            device=device,
            rng=rng,
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
            max_candidates=args.window_choice_top,
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
    if torch.cuda.is_available():
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
        tokenizer_config,
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
        print(f"Generating source-creative-ranked latent clip {clip_idx + 1}/{clip_count} on {device}...")
        clip_args = argparse.Namespace(**vars(args))
        clip_args.seed = args.seed + clip_idx * 1009
        codes = generate_source_creative_ranked_codes(
            clip_args,
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
    print(f"Saved source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
