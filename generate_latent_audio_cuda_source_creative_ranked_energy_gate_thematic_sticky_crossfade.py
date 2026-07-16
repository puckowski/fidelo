import argparse
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch

import generate_latent_audio_cuda_source_creative_ranked as base
import generate_latent_audio_cuda_source_creative_ranked_energy_gate as energy_gate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prompt-conditioned audio with fixed-duration source themes, rank-relaxed token sampling, and energy gates for windows and clips, with sticky source selection and crossfaded theme changes."
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
    parser.add_argument("--source-candidates", type=int, default=16)
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
    parser.add_argument("--min-clip-median-rms", type=float, default=0.0, help="Reject a generated clip if the median analysis-chunk RMS is below this threshold. Use this to require at least medium overall intensity.")
    parser.add_argument("--clip-retry-count", type=int, default=4, help="How many times to retry a clip with new seeds if it fails the audibility check.")
    parser.add_argument("--theme-seconds", type=float, default=1.66, help="How long to keep one randomly chosen source theme before switching to another top-N source file.")
    parser.add_argument("--theme-top-n", type=int, default=8, help="Randomly choose each theme from a broader top-N prompt-matched source file pool.")
    parser.add_argument("--theme-temperature", type=float, default=0.9, help="Higher values make source theme selection more varied within the top-N pool.")
    parser.add_argument("--theme-repeat-window", type=int, default=3, help="How many previous theme selections to bias toward repeating.")
    parser.add_argument("--theme-repeat-bonus", type=float, default=2.0, help="How strongly to prefer a source file that appeared recently.")
    parser.add_argument("--theme-repeat-decay", type=float, default=0.6, help="How quickly repeat preference decays for older entries in the sliding window.")
    parser.add_argument("--theme-crossfade-ms", type=int, default=180, help="Crossfade length used when stitching decoded theme segments together.")
    parser.add_argument("--creative-span-count", type=int, default=5, help="How many prior-driven spans to inject per window.")
    parser.add_argument("--creative-span-min", type=int, default=8)
    parser.add_argument("--creative-span-max", type=int, default=40)
    parser.add_argument("--creative-token-mix", type=float, default=0.16, help="Per-token chance to keep proposal tokens outside creative spans.")
    parser.add_argument("--fade-ms", type=int, default=40, help="Crossfade used when stitching separate output clips.")
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


def summarize_clip_loudness(chunk_energies: List[Dict[str, float]]) -> Dict[str, float]:
    if not chunk_energies:
        return {"median_rms": 0.0, "mean_rms": 0.0, "median_peak": 0.0}

    ordered_rms = sorted(item["rms"] for item in chunk_energies)
    ordered_peak = sorted(item["peak"] for item in chunk_energies)
    middle = len(chunk_energies) // 2
    if len(chunk_energies) % 2 == 1:
        median_rms = ordered_rms[middle]
        median_peak = ordered_peak[middle]
    else:
        median_rms = 0.5 * (ordered_rms[middle - 1] + ordered_rms[middle])
        median_peak = 0.5 * (ordered_peak[middle - 1] + ordered_peak[middle])
    mean_rms = sum(ordered_rms) / len(ordered_rms)
    return {"median_rms": median_rms, "mean_rms": mean_rms, "median_peak": median_peak}


def clip_has_sufficient_loudness(chunk_energies: List[Dict[str, float]], args) -> bool:
    min_clip_median_rms = max(0.0, float(getattr(args, "min_clip_median_rms", 0.0)))
    if min_clip_median_rms <= 0.0:
        return True
    loudness = summarize_clip_loudness(chunk_energies)
    return loudness["median_rms"] >= min_clip_median_rms


def choose_theme_entry(
    candidate_entries: List[Dict],
    top_n: int,
    window_size: int,
    rng: random.Random,
    temperature: float,
    recent_entries: List[Dict],
    repeat_window: int,
    repeat_bonus: float,
    repeat_decay: float,
) -> Optional[Dict]:
    if not candidate_entries:
        return None

    pool = candidate_entries[:max(1, min(top_n, len(candidate_entries)))]
    viable = [entry for entry in pool if code_step_count(entry["codes"]) >= window_size]
    working = viable or pool
    if not working:
        return None

    if len(working) == 1:
        return working[0]

    recent_entries = recent_entries[-max(0, repeat_window):]
    scores = []
    for entry in working:
        base_score = float(entry.get("match_score", 0.0)) + 0.25
        repeat_score = 0.0
        for age, previous in enumerate(reversed(recent_entries)):
            if previous.get("file") == entry.get("file"):
                repeat_score += repeat_decay ** age
        weight = max(0.05, base_score) * (1.0 + max(0.0, repeat_bonus) * repeat_score)
        scores.append(weight)

    max_score = max(scores)
    weights = [math.exp((score - max_score) / max(1e-6, float(temperature))) for score in scores]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for entry, weight in zip(working, weights):
        running += weight
        if pick <= running:
            return entry
    return working[-1]


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

    print("No theme window passed the energy gate; falling back to prior-only tokens for this step")
    return None


def choose_theme_window(
    proposal_full: torch.Tensor,
    prefix_codes: Optional[torch.Tensor],
    theme_entries: List[Dict],
    overlap_size: int,
    args,
    tokenizer_model,
    device: torch.device,
) -> Optional[Dict]:
    candidates = find_source_window_candidates(
        proposal_full,
        prefix_codes,
        theme_entries,
        overlap_size,
        args.proposal_weight,
        args.continuity_weight,
        args.match_weight,
        args.scan_step_divisor,
    )
    return select_top_window_with_energy_gate(candidates, args, tokenizer_model, device)


def blend_theme_windows(
    old_window: torch.Tensor,
    new_window: torch.Tensor,
    prefix_len: int,
    new_steps: int,
    transition_steps_total: int,
    transition_steps_done: int,
    rng: random.Random,
) -> torch.Tensor:
    blended = new_window.clone()
    active_steps = min(new_steps, max(0, transition_steps_total - transition_steps_done))
    if active_steps <= 0:
        return blended

    for idx in range(active_steps):
        absolute_progress = (transition_steps_done + idx + 1) / max(1, transition_steps_total)
        take_new_probability = max(0.0, min(1.0, absolute_progress))
        source_idx = prefix_len + idx
        if rng.random() > take_new_probability:
            blended[..., source_idx] = old_window[..., source_idx]
        else:
            blended[..., source_idx] = new_window[..., source_idx]
    return blended


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
    source_rng: random.Random,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    total_steps = config.latent_steps
    window_size = max(32, min(args.source_window, total_steps))
    overlap_size = max(0, min(args.source_overlap, window_size // 2))
    generated = empty_code_sequence(getattr(prior_model, "num_quantizers", 1))
    rng = random.Random(args.seed)
    steps_per_second = total_steps / max(config.clip_seconds, 1e-6)
    theme_steps = max(1, int(round(max(0.1, args.theme_seconds) * steps_per_second)))
    current_theme_entry: Optional[Dict] = None
    previous_theme_entry: Optional[Dict] = None
    remaining_theme_steps = 0
    recent_theme_entries: List[Dict] = []
    segment_ranges: List[Tuple[int, int]] = []
    segment_start = 0
    transition_steps_total = max(0, int(round((max(0, args.theme_crossfade_ms) / 1000.0) * steps_per_second)))
    transition_steps_done = transition_steps_total

    while code_step_count(generated) < total_steps:
        if current_theme_entry is None or remaining_theme_steps <= 0:
            if code_step_count(generated) > segment_start:
                segment_ranges.append((segment_start, code_step_count(generated)))
                segment_start = code_step_count(generated)

            previous_theme_entry = current_theme_entry
            current_theme_entry = choose_theme_entry(
                candidate_entries,
                args.theme_top_n,
                window_size,
                source_rng,
                args.theme_temperature,
                recent_theme_entries,
                args.theme_repeat_window,
                args.theme_repeat_bonus,
                args.theme_repeat_decay,
            )
            remaining_theme_steps = theme_steps
            if current_theme_entry is not None:
                if previous_theme_entry is not None and previous_theme_entry.get("file") != current_theme_entry.get("file"):
                    transition_steps_done = 0
                else:
                    transition_steps_done = transition_steps_total
                recent_theme_entries.append(current_theme_entry)
                recent_theme_entries = recent_theme_entries[-max(1, args.theme_repeat_window):]
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
        chosen = choose_theme_window(
            proposal_full,
            prefix_codes,
            ([current_theme_entry] if current_theme_entry is not None else candidate_entries),
            overlap_size,
            args,
            tokenizer_model,
            device,
        )

        if (
            chosen is not None
            and previous_theme_entry is not None
            and current_theme_entry is not None
            and previous_theme_entry.get("file") != current_theme_entry.get("file")
            and transition_steps_done < transition_steps_total
        ):
            previous_chosen = choose_theme_window(
                proposal_full,
                prefix_codes,
                [previous_theme_entry],
                overlap_size,
                args,
                tokenizer_model,
                device,
            )
            if previous_chosen is not None:
                previous_fused = fuse_source_and_proposal_window(
                    proposal_full,
                    previous_chosen["window"],
                    prefix_len,
                    args,
                    rng,
                )
                current_fused = fuse_source_and_proposal_window(
                    proposal_full,
                    chosen["window"],
                    prefix_len,
                    args,
                    rng,
                )
                blended_window = blend_theme_windows(
                    previous_fused,
                    current_fused,
                    prefix_len,
                    new_steps,
                    transition_steps_total,
                    transition_steps_done,
                    rng,
                )
                chosen_new = slice_code_steps(blended_window, prefix_len, prefix_len + new_steps)
            else:
                fused_window = fuse_source_and_proposal_window(
                    proposal_full,
                    chosen["window"],
                    prefix_len,
                    args,
                    rng,
                )
                chosen_new = slice_code_steps(fused_window, prefix_len, prefix_len + new_steps)
        elif chosen is not None:
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
        transition_steps_done = min(transition_steps_total, transition_steps_done + new_steps)

    if code_step_count(generated) > segment_start:
        segment_ranges.append((segment_start, code_step_count(generated)))

    return generated.unsqueeze(0).to(device), segment_ranges


def decode_and_stitch_segments(
    codes: torch.Tensor,
    segment_ranges: List[Tuple[int, int]],
    tokenizer_model,
    tokenizer_config,
    source_overlap: int,
    theme_crossfade_ms: int,
) -> torch.Tensor:
    waveforms: List[torch.Tensor] = []
    total_steps = max(1, int(codes.shape[-1]))
    segment_count = len(segment_ranges)
    samples_per_step = tokenizer_config.clip_samples / float(total_steps)
    fade_samples = int(tokenizer_config.sample_rate * max(0, theme_crossfade_ms) / 1000)
    context_steps = max(
        1,
        min(
            total_steps // 4,
            max(int(round(fade_samples / max(samples_per_step, 1e-6))), int(source_overlap)),
        ),
    )

    for segment_idx, (start, end) in enumerate(segment_ranges):
        left_context = min(start, context_steps) if segment_idx > 0 else 0
        right_context = min(total_steps - end, context_steps) if segment_idx < (segment_count - 1) else 0
        segment_codes = codes[..., start - left_context:end + right_context]
        expanded_steps = max(1, segment_codes.shape[-1])
        expanded_target_length = max(1, int(round(tokenizer_config.clip_samples * (expanded_steps / total_steps))))
        waveform = tokenizer_model.decode_codes(segment_codes, target_length=expanded_target_length).squeeze(0).cpu()
        waveforms.append(waveform)

    if not waveforms:
        raise RuntimeError("No waveform segments were generated for stitching.")
    if len(waveforms) == 1:
        return waveforms[0]
    return crossfade_theme_waveforms(waveforms, tokenizer_config.sample_rate, theme_crossfade_ms)


def crossfade_theme_waveforms(
    waveforms: List[torch.Tensor],
    sample_rate: int,
    fade_ms: int,
) -> torch.Tensor:
    if not waveforms:
        raise ValueError("No waveforms provided for theme crossfade")

    requested_fade_samples = int(sample_rate * max(0, fade_ms) / 1000)
    out = waveforms[0].clone()
    for idx, nxt in enumerate(waveforms[1:], start=1):
        effective = min(requested_fade_samples, max(0, out.shape[-1] - 1), max(0, nxt.shape[-1] - 1))
        if effective <= 0:
            out = torch.cat([out, nxt], dim=-1)
            print(f"Theme crossfade {idx}: no overlap available")
            continue

        fade_shape = [1] * out.dim()
        fade_shape[-1] = effective
        t = torch.linspace(0.0, 1.0, effective, device=out.device).view(*fade_shape)
        fade_out = torch.cos(t * (math.pi / 2.0))
        fade_in = torch.sin(t * (math.pi / 2.0))
        mixed = out[..., -effective:] * fade_out + nxt[..., :effective] * fade_in
        out = torch.cat([out[..., :-effective], mixed, nxt[..., effective:]], dim=-1)
        print(
            f"Theme crossfade {idx}: requested={fade_ms}ms effective={1000.0 * effective / sample_rate:.1f}ms"
        )
    return out


def main():
    args = parse_args()
    start_time = time.perf_counter()
    device = get_device(args.allow_cpu)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_rng = random.Random(args.seed + 918273)

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
        print(f"Generating sticky crossfaded source-creative latent clip {clip_idx + 1}/{clip_count} on {device}...")
        accepted_waveform = None
        retry_count = max(1, int(args.clip_retry_count))

        for retry_idx in range(retry_count):
            clip_args = argparse.Namespace(**vars(args))
            clip_args.seed = args.seed + clip_idx * 1009 + retry_idx * 7919
            codes, segment_ranges = generate_source_creative_ranked_codes(
                clip_args,
                prior_model,
                tokenizer_model,
                text_tokens,
                text_mask,
                prior_config,
                candidate_entries,
                device,
                source_rng,
            )
            codes = codes.to(device=device, dtype=torch.long)
            waveform = decode_and_stitch_segments(
                codes,
                segment_ranges,
                tokenizer_model,
                tokenizer_config,
                args.source_overlap,
                args.theme_crossfade_ms,
            )
            candidate_waveform = waveform.cpu()
            chunk_energies = measure_audio_chunk_energies(
                candidate_waveform,
                tokenizer_config.sample_rate,
                args.clip_energy_check_seconds,
            )
            loudness_summary = summarize_clip_loudness(chunk_energies)
            if clip_has_sufficient_energy(candidate_waveform, tokenizer_config.sample_rate, args) and clip_has_sufficient_loudness(chunk_energies, args):
                accepted_waveform = candidate_waveform
                quietest_accepted_chunk = min(
                    chunk_energies,
                    key=lambda item: (item["rms"], item["peak"]),
                )
                if retry_idx > 0:
                    print(f"Accepted clip {clip_idx + 1} after {retry_idx + 1} attempts")
                print(
                    f"Added clip {clip_idx + 1} to output "
                    f"quietest_chunk_rms={quietest_accepted_chunk['rms']:.4f} "
                    f"quietest_chunk_peak={quietest_accepted_chunk['peak']:.4f} "
                    f"median_chunk_rms={loudness_summary['median_rms']:.4f}"
                )
                break

            worst_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
            if not clip_has_sufficient_energy(candidate_waveform, tokenizer_config.sample_rate, args):
                print(
                    "Rejected low-energy clip "
                    f"{clip_idx + 1} attempt {retry_idx + 1}/{retry_count} "
                    f"worst_chunk_rms={worst_chunk['rms']:.4f} worst_chunk_peak={worst_chunk['peak']:.4f}"
                )
            else:
                print(
                    "Rejected low-loudness clip "
                    f"{clip_idx + 1} attempt {retry_idx + 1}/{retry_count} "
                    f"median_chunk_rms={loudness_summary['median_rms']:.4f} "
                    f"mean_chunk_rms={loudness_summary['mean_rms']:.4f}"
                )

        if accepted_waveform is None:
            print(f"Skipping clip {clip_idx + 1} after {retry_count} failed energy checks")
            continue

        clips.append(accepted_waveform)

    if not clips:
        raise RuntimeError("All generated clips failed the energy gate. Relax the thresholds or increase --clip-retry-count.")

    print(f"Accepted {len(clips)} clip(s) for final output")

    final_fade_ms = max(int(args.fade_ms), int(args.theme_crossfade_ms))
    output = base.stitch_waveforms(clips, tokenizer_config.sample_rate, fade_ms=final_fade_ms)
    if len(clips) > 1:
        print(f"Final clip stitch crossfade: {final_fade_ms}ms across {len(clips)} clips")
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
    print(f"Saved sticky crossfaded source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()