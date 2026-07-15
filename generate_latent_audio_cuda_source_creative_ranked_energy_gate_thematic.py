import argparse
import math
import random
import time
from typing import Dict, List, Optional

import torch

import generate_latent_audio_cuda_source_creative_ranked as base
import generate_latent_audio_cuda_source_creative_ranked_energy_gate as energy_gate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prompt-conditioned audio with fixed-duration source themes, rank-relaxed token sampling, and energy gates for windows and clips."
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
    parser.add_argument("--window-choice-top", type=int, default=6, help="Retained for CLI compatibility but not used in themed mode.")
    parser.add_argument("--window-choice-temperature", type=float, default=0.7, help="Retained for CLI compatibility but not used in themed mode.")
    parser.add_argument("--window-energy-check-top", type=int, default=8, help="How many scored source windows to inspect before falling back to a low-energy candidate.")
    parser.add_argument("--min-window-rms", type=float, default=0.008, help="Reject a source window if its decoded RMS is below this threshold.")
    parser.add_argument("--min-window-peak", type=float, default=0.03, help="Reject a source window if its decoded peak is below this threshold.")
    parser.add_argument("--clip-energy-check-seconds", type=float, default=1.0, help="Analyze generated clips in chunks of this length when checking for silence.")
    parser.add_argument("--min-clip-rms", type=float, default=0.02, help="Reject a generated clip if any analysis chunk falls below this RMS threshold.")
    parser.add_argument("--min-clip-peak", type=float, default=0.06, help="Reject a generated clip if any analysis chunk falls below this peak threshold.")
    parser.add_argument("--clip-retry-count", type=int, default=4, help="How many times to retry a clip with new seeds if it fails the audibility check.")
    parser.add_argument("--theme-seconds", type=float, default=1.66, help="How long to keep one randomly chosen source theme before switching to another top-N source file.")
    parser.add_argument("--theme-top-n", type=int, default=4, help="Randomly choose each theme from the top-N prompt-matched source files.")
    parser.add_argument("--creative-span-count", type=int, default=5, help="How many prior-driven spans to inject per window.")
    parser.add_argument("--creative-span-min", type=int, default=8)
    parser.add_argument("--creative-span-max", type=int, default=40)
    parser.add_argument("--creative-token-mix", type=float, default=0.16, help="Per-token chance to keep proposal tokens outside creative spans.")
    parser.add_argument("--fade-ms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


get_device = base.get_device
make_output_name = base.make_output_name
build_source_entries = base.build_source_entries
find_source_window_candidates = base.find_source_window_candidates
generate_rank_relaxed_window = base.generate_rank_relaxed_window
fuse_source_and_proposal_window = base.fuse_source_and_proposal_window
measure_window_energy = energy_gate.measure_window_energy
measure_audio_chunk_energies = energy_gate.measure_audio_chunk_energies
clip_has_sufficient_energy = energy_gate.clip_has_sufficient_energy
code_step_count = base.code_step_count
empty_code_sequence = base.empty_code_sequence
extract_code_tail = base.extract_code_tail
ensure_batched_codes = base.ensure_batched_codes
concat_code_sequences = base.concat_code_sequences
slice_code_steps = base.slice_code_steps


def choose_theme_entry(candidate_entries: List[Dict], top_n: int, window_size: int, rng: random.Random) -> Optional[Dict]:
    if not candidate_entries:
        return None
    pool = candidate_entries[:max(1, min(top_n, len(candidate_entries)))]
    viable = [entry for entry in pool if code_step_count(entry["codes"]) >= window_size]
    working = viable or pool
    if not working:
        return None
    return working[rng.randrange(len(working))]


def select_top_window_with_energy_gate(
    candidates: List[Dict],
    args,
    tokenizer_model,
    device: torch.device,
) -> Optional[Dict]:
    if not candidates:
        return None

    scored_candidates = candidates[:max(1, args.window_energy_check_top)]
    min_rms = max(0.0, float(args.min_window_rms))
    min_peak = max(0.0, float(args.min_window_peak))

    for candidate in scored_candidates:
        energy = measure_window_energy(candidate["window"], tokenizer_model, device)
        candidate["decoded_rms"] = energy["rms"]
        candidate["decoded_peak"] = energy["peak"]
        if energy["rms"] >= min_rms or energy["peak"] >= min_peak:
            return candidate
        print(
            "Rejected low-energy theme window "
            f"{candidate['entry']['file']} start={candidate['start']} rms={energy['rms']:.4f} peak={energy['peak']:.4f}"
        )

    return scored_candidates[0]


@torch.no_grad()
def generate_source_creative_ranked_codes(
    args,
    prior_model,
    tokenizer_model,
    text_tokens,
    text_mask,
    config,
    candidate_entries,
    device: torch.device,
) -> torch.Tensor:
    total_steps = config.latent_steps
    window_size = max(32, min(args.source_window, total_steps))
    overlap_size = max(0, min(args.source_overlap, window_size // 2))
    generated = empty_code_sequence(getattr(prior_model, "num_quantizers", 1))
    rng = random.Random(args.seed)
    steps_per_second = total_steps / max(config.clip_seconds, 1e-6)
    theme_steps = max(1, int(round(max(0.1, args.theme_seconds) * steps_per_second)))
    current_theme_entry: Optional[Dict] = None
    remaining_theme_steps = 0

    while code_step_count(generated) < total_steps:
        if current_theme_entry is None or remaining_theme_steps <= 0:
            current_theme_entry = choose_theme_entry(candidate_entries, args.theme_top_n, window_size, rng)
            remaining_theme_steps = theme_steps
            if current_theme_entry is not None:
                print(
                    "Selected theme "
                    f"{current_theme_entry['file']} for up to {min(args.theme_seconds, config.clip_seconds):.2f}s"
                )

        prefix_codes = None
        prefix_len = 0
        if overlap_size > 0 and code_step_count(generated) > 0:
            prefix_codes = extract_code_tail(generated, overlap_size)
            prefix_len = code_step_count(prefix_codes)

        max_new_steps = window_size if prefix_len == 0 else (window_size - prefix_len)
        new_steps = min(max_new_steps, total_steps - code_step_count(generated), remaining_theme_steps)
        if new_steps <= 0:
            current_theme_entry = None
            remaining_theme_steps = 0
            continue

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
        themed_entries = [current_theme_entry] if current_theme_entry is not None else candidate_entries
        candidates = find_source_window_candidates(
            proposal_full,
            prefix_codes,
            themed_entries,
            overlap_size,
            args.proposal_weight,
            args.continuity_weight,
            args.match_weight,
            args.scan_step_divisor,
            max_candidates=args.window_energy_check_top,
        )
        chosen = select_top_window_with_energy_gate(
            candidates,
            args,
            tokenizer_model,
            device,
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
        remaining_theme_steps -= new_steps

    return generated.unsqueeze(0).to(device)


def main():
    args = parse_args()
    start_time = time.perf_counter()
    device = get_device(args.allow_cpu)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer_model, tokenizer_config = base.load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    prior_model, text_tokenizer, prior_config = base.load_latent_prior_bundle(args.prior_dir, device)

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
        print(f"Generating themed source-creative latent clip {clip_idx + 1}/{clip_count} on {device}...")
        accepted_waveform = None
        retry_count = max(1, int(args.clip_retry_count))
        candidate_waveform = None

        for retry_idx in range(retry_count):
            clip_args = argparse.Namespace(**vars(args))
            clip_args.seed = args.seed + clip_idx * 1009 + retry_idx * 7919
            codes = generate_source_creative_ranked_codes(
                clip_args,
                prior_model,
                tokenizer_model,
                text_tokens,
                text_mask,
                prior_config,
                candidate_entries,
                device,
            )
            codes = codes.to(device=device, dtype=torch.long)
            waveform = tokenizer_model.decode_codes(codes, target_length=tokenizer_config.clip_samples)
            candidate_waveform = waveform.squeeze(0).cpu()
            if clip_has_sufficient_energy(candidate_waveform, tokenizer_config.sample_rate, args):
                accepted_waveform = candidate_waveform
                if retry_idx > 0:
                    print(f"Accepted clip {clip_idx + 1} after {retry_idx + 1} attempts")
                break

            chunk_energies = measure_audio_chunk_energies(
                candidate_waveform,
                tokenizer_config.sample_rate,
                args.clip_energy_check_seconds,
            )
            worst_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
            print(
                "Rejected low-energy clip "
                f"{clip_idx + 1} attempt {retry_idx + 1}/{retry_count} "
                f"worst_chunk_rms={worst_chunk['rms']:.4f} worst_chunk_peak={worst_chunk['peak']:.4f}"
            )

        if accepted_waveform is None:
            accepted_waveform = candidate_waveform
            print(f"Using last attempt for clip {clip_idx + 1} after {retry_count} failed energy checks")

        clips.append(accepted_waveform)

    output = base.stitch_waveforms(clips, tokenizer_config.sample_rate, fade_ms=args.fade_ms)
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
    base.save_audio_waveform(output_path, output, tokenizer_config.sample_rate)
    print(f"Saved themed source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()