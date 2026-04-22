import argparse
import hashlib
import math
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch

from latent_audio_token_pipeline import (
    crop_or_pad,
    load_audio_mono,
    load_audio_tokenizer_bundle,
    load_dataset_items,
    load_latent_prior_bundle,
    save_audio_waveform,
    stitch_waveforms,
)


WORD_RE = re.compile(r"[a-z0-9']+")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompt-conditioned audio using learned latent audio tokens on CUDA.")
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--prior-dir", default="latent_audio_prior_out")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-count", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Target output duration in seconds. If set, overrides --clip-count.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--guidance-candidates", type=int, default=8)
    parser.add_argument("--guidance-window", type=int, default=256)
    parser.add_argument("--guidance-overlap", type=int, default=64)
    parser.add_argument("--disable-retrieval-guidance", action="store_true")
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
    digest = hashlib.sha1(f"{prompt}_{datetime.now().timestamp()}".encode("utf-8")).hexdigest()[:8]
    return f"latent_generated_{timestamp}_{digest}.wav"


def prompt_tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def score_prompt_match(prompt: str, text: str) -> float:
    prompt_set = prompt_tokens(prompt)
    text_set = prompt_tokens(text)
    if not prompt_set or not text_set:
        return 0.0
    overlap = len(prompt_set & text_set)
    return overlap + (overlap / max(1, len(prompt_set)))


@torch.no_grad()
def build_guidance_entries(prompt: str, tokenizer_model, config, device: torch.device, limit: int) -> List[Dict]:
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
            waveform = load_audio_mono(item["path"], config.sample_rate)
            waveform = crop_or_pad(waveform, config.clip_samples, random_crop=False)
            codes = tokenizer_model.encode_codes(waveform.unsqueeze(0).to(device)).squeeze(0).cpu()
            entries.append({
                "codes": codes,
                "text": item["text"],
                "path": item["path"],
                "match_score": score,
            })
        except Exception as exc:
            print(f"Skipping guidance candidate {item['path']} ({exc})")
    return entries


def choose_guided_window(
    proposal_full: torch.Tensor,
    prefix_codes: Optional[torch.Tensor],
    candidate_entries: List[Dict],
    overlap_size: int,
) -> Optional[torch.Tensor]:
    if not candidate_entries:
        return None

    proposal_full = proposal_full.cpu()
    prefix_len = 0 if prefix_codes is None else min(overlap_size, prefix_codes.shape[0])
    prefix_tail = None if prefix_codes is None else prefix_codes[-prefix_len:].cpu()
    window_size = proposal_full.shape[0]
    step = max(1, window_size // 4)
    best_score = None
    best_window = None

    for entry in candidate_entries:
        seq = entry["codes"]
        if seq.shape[0] < window_size:
            continue
        for start in range(0, seq.shape[0] - window_size + 1, step):
            window = seq[start:start + window_size]
            proposal_match = (window == proposal_full[:window_size]).float().mean().item()
            continuity = 0.0
            if prefix_tail is not None and prefix_len > 0:
                continuity = (window[:prefix_len] == prefix_tail).float().mean().item()
            diversity = torch.unique(window).numel() / max(1, window.numel())
            score = (2.5 * continuity) + (1.5 * proposal_match) + (0.2 * diversity) + (0.35 * entry["match_score"])
            if best_score is None or score > best_score:
                best_score = score
                best_window = window.clone()
    return best_window


@torch.no_grad()
def generate_guided_codes(args, prior_model, text_tokens, text_mask, config, candidate_entries, device: torch.device) -> torch.Tensor:
    total_steps = config.latent_steps
    window_size = max(32, min(args.guidance_window, total_steps))
    overlap_size = max(0, min(args.guidance_overlap, window_size // 2))
    generated = torch.empty(0, dtype=torch.long)

    while generated.shape[0] < total_steps:
        prefix_codes = None
        prefix_len = 0
        if overlap_size > 0 and generated.shape[0] > 0:
            prefix_codes = generated[-overlap_size:]
            prefix_len = prefix_codes.shape[0]

        new_steps = min(window_size if prefix_len == 0 else (window_size - prefix_len), total_steps - generated.shape[0])
        generated_new = prior_model.generate(
            text_tokens=text_tokens,
            text_mask=text_mask,
            num_steps=new_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            prefix_codes=(None if prefix_codes is None else prefix_codes.unsqueeze(0)),
            device=device,
        ).squeeze(0).cpu()

        if prefix_codes is not None:
            proposal_full = torch.cat([prefix_codes.cpu(), generated_new], dim=0)
        else:
            proposal_full = generated_new

        guided_window = choose_guided_window(proposal_full, prefix_codes, candidate_entries, overlap_size)
        if guided_window is not None:
            chosen_new = guided_window[prefix_len:prefix_len + new_steps]
        else:
            chosen_new = generated_new
        generated = torch.cat([generated, chosen_new], dim=0)

    return generated.unsqueeze(0).to(device)


def main():
    args = parse_args()
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

    candidate_entries = []
    if not args.disable_retrieval_guidance:
        candidate_entries = build_guidance_entries(
            args.prompt,
            tokenizer_model,
            tokenizer_config,
            device,
            args.guidance_candidates,
        )
        if candidate_entries:
            print(f"Loaded {len(candidate_entries)} latent guidance candidates")

    clips = []
    for clip_idx in range(clip_count):
        print(f"Generating latent clip {clip_idx + 1}/{clip_count} on {device}...")
        if candidate_entries:
            codes = generate_guided_codes(
                args,
                prior_model,
                text_tokens,
                text_mask,
                prior_config,
                candidate_entries,
                device,
            )
        else:
            codes = prior_model.generate(
                text_tokens=text_tokens,
                text_mask=text_mask,
                num_steps=prior_config.latent_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                device=device,
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
    print(f"Saved latent generated audio to {output_path}")


if __name__ == "__main__":
    main()
