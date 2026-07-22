import argparse
import math
import random
import time
from typing import Dict, List

import torch

import generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured as structured


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prompt-conditioned audio with regular-BOS-only sticky source themes, energy gates, and intro/body structure."
    )
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--prior-dir", default="latent_audio_prior_out")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-count", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Target output duration in seconds. If set, overrides --clip-count.")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--rank-choice-prob", type=float, default=0.35)
    parser.add_argument("--rank-choice-top", type=int, default=3)
    parser.add_argument("--rank-choice-temperature", type=float, default=0.75)
    parser.add_argument("--source-candidates", type=int, default=16)
    parser.add_argument("--source-window", type=int, default=256)
    parser.add_argument("--source-overlap", type=int, default=64)
    parser.add_argument("--max-source-seconds", type=float, default=30.0)
    parser.add_argument("--proposal-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=2.75)
    parser.add_argument("--match-weight", type=float, default=0.45)
    parser.add_argument("--scan-step-divisor", type=int, default=4)
    parser.add_argument("--source-strength", type=float, default=0.65)
    parser.add_argument("--window-choice-top", type=int, default=6, help="Retained for CLI compatibility but not used in themed mode.")
    parser.add_argument("--window-choice-temperature", type=float, default=0.7, help="Retained for CLI compatibility but not used in themed mode.")
    parser.add_argument("--window-energy-check-top", type=int, default=8)
    parser.add_argument("--min-window-rms", type=float, default=0.008)
    parser.add_argument("--min-window-peak", type=float, default=0.03)
    parser.add_argument("--clip-energy-check-seconds", type=float, default=1.0)
    parser.add_argument("--min-clip-rms", type=float, default=0.02)
    parser.add_argument("--min-clip-peak", type=float, default=0.06)
    parser.add_argument("--dropout-check-seconds", type=float, default=0.25)
    parser.add_argument("--dropout-hop-seconds", type=float, default=0.125)
    parser.add_argument("--min-dropout-rms", type=float, default=None)
    parser.add_argument("--min-dropout-peak", type=float, default=None)
    parser.add_argument("--min-clip-median-rms", type=float, default=0.0)
    parser.add_argument("--clip-retry-count", type=int, default=4)
    parser.add_argument("--theme-seconds", type=float, default=1.66)
    parser.add_argument("--theme-top-n", type=int, default=8)
    parser.add_argument("--theme-temperature", type=float, default=0.9)
    parser.add_argument("--theme-repeat-window", type=int, default=3)
    parser.add_argument("--theme-repeat-bonus", type=float, default=2.0)
    parser.add_argument("--theme-repeat-decay", type=float, default=0.6)
    parser.add_argument("--theme-crossfade-ms", type=int, default=180)
    parser.add_argument("--intro-ratio", type=float, default=0.2, help="Approximate fraction of clips reserved for the intro section.")
    parser.add_argument("--intro-theme-top-n", type=int, default=1)
    parser.add_argument("--intro-theme-seconds", type=float, default=2.5)
    parser.add_argument("--intro-theme-temperature", type=float, default=0.35)
    parser.add_argument("--intro-repeat-bonus", type=float, default=5.0)
    parser.add_argument("--intro-source-strength", type=float, default=0.92)
    parser.add_argument("--intro-creative-token-mix", type=float, default=0.08)
    parser.add_argument("--intro-rank-choice-prob", type=float, default=0.12)
    parser.add_argument("--song-intro-fade-ms", type=int, default=220)
    parser.add_argument("--creative-span-count", type=int, default=5)
    parser.add_argument("--creative-span-min", type=int, default=8)
    parser.add_argument("--creative-span-max", type=int, default=40)
    parser.add_argument("--creative-token-mix", type=float, default=0.16)
    parser.add_argument("--fade-ms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def build_song_sections(clip_count: int, intro_ratio: float) -> List[str]:
    if clip_count <= 0:
        return []
    if clip_count == 1:
        return ["body"]

    intro_count = max(1, int(round(clip_count * structured.clamp_ratio(intro_ratio))))
    intro_count = min(intro_count, clip_count - 1)
    return ["intro"] * intro_count + ["body"] * (clip_count - intro_count)


def build_section_args(args, section_name: str):
    clip_args = argparse.Namespace(**vars(args))
    clip_args.song_beginning = False
    if section_name == "intro":
        clip_args.theme_top_n = max(1, int(args.intro_theme_top_n))
        clip_args.theme_seconds = max(0.1, float(args.intro_theme_seconds))
        clip_args.theme_temperature = max(1e-6, float(args.intro_theme_temperature))
        clip_args.theme_repeat_bonus = max(0.0, float(args.intro_repeat_bonus))
        clip_args.source_strength = max(0.0, min(1.0, float(args.intro_source_strength)))
        clip_args.creative_token_mix = max(0.0, min(1.0, float(args.intro_creative_token_mix)))
        clip_args.rank_choice_prob = max(0.0, min(1.0, float(args.intro_rank_choice_prob)))
    return clip_args


def build_section_candidate_entries(
    regular_entries: List[Dict],
    section_name: str,
    intro_theme_top_n: int,
) -> List[Dict]:
    if section_name == "intro":
        return regular_entries[:max(1, int(intro_theme_top_n))]
    return regular_entries


def apply_song_intro(audio: torch.Tensor, sample_rate: int, intro_fade_ms: int) -> torch.Tensor:
    shaped = audio.clone()
    intro_samples = min(shaped.shape[-1], max(0, int(sample_rate * max(0, intro_fade_ms) / 1000)))
    if intro_samples > 1:
        intro_curve = torch.sin(torch.linspace(0.0, math.pi / 2.0, intro_samples, device=shaped.device))
        shaped[..., :intro_samples] = shaped[..., :intro_samples] * intro_curve
    return shaped


def main():
    args = parse_args()
    start_time = time.perf_counter()
    device = structured.get_device(args.allow_cpu)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_rng = random.Random(args.seed + 918273)
    tokenizer_model, tokenizer_config = structured.base.load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    prior_model, text_tokenizer, prior_config = structured.base.load_latent_prior_bundle(args.prior_dir, device)
    if tokenizer_config.codebook_size != prior_config.codebook_size:
        raise RuntimeError("Tokenizer and prior codebook sizes do not match.")

    clip_count = args.clip_count
    target_samples = None
    if args.duration_seconds > 0:
        target_samples = int(round(args.duration_seconds * tokenizer_config.sample_rate))
        clip_count = max(1, math.ceil(args.duration_seconds / tokenizer_config.clip_seconds))

    text_tokens = text_tokenizer.encode(args.prompt, prior_config.max_text_tokens).unsqueeze(0)
    text_mask = text_tokenizer.attention_mask(text_tokens)
    candidate_entries = structured.build_source_entries(
        args.prompt,
        tokenizer_model,
        prior_config,
        device,
        args.source_candidates,
        args.max_source_seconds,
        target_num_quantizers=getattr(prior_model, "num_quantizers", 1),
    )
    regular_entries = [entry for entry in candidate_entries if not entry.get("song_beginning", False)]
    if not regular_entries:
        raise RuntimeError("No regular source candidates found. Sources marked song_beginning=1 are excluded.")

    excluded_count = len(candidate_entries) - len(regular_entries)
    print(f"Loaded {len(regular_entries)} regular source candidates; excluded {excluded_count} song-beginning candidate(s)")
    for entry in regular_entries:
        print(f"- {entry['file']} | source_type=regular | score={entry['match_score']:.2f} | {entry['text']}")

    section_names = build_song_sections(clip_count, args.intro_ratio)
    clips = []
    recent_theme_entries: List[Dict] = []
    for clip_idx in range(clip_count):
        section_name = section_names[clip_idx]
        section_entries = build_section_candidate_entries(regular_entries, section_name, args.intro_theme_top_n)
        effective_clip_args = build_section_args(args, section_name)
        print(
            f"Generating {section_name} regular-BOS sticky crossfaded source-creative latent clip "
            f"{clip_idx + 1}/{clip_count} on {device} "
            f"window_rms={effective_clip_args.min_window_rms:.4f} "
            f"window_peak={effective_clip_args.min_window_peak:.4f} "
            f"clip_rms={effective_clip_args.min_clip_rms:.4f} "
            f"clip_peak={effective_clip_args.min_clip_peak:.4f}..."
        )
        accepted_waveform = None
        retry_count = max(1, int(args.clip_retry_count))

        for retry_idx in range(retry_count):
            clip_args = argparse.Namespace(**vars(effective_clip_args))
            clip_args.seed = args.seed + clip_idx * 1009 + retry_idx * 7919
            codes, segment_ranges, attempted_theme_entries = structured.generate_source_creative_ranked_codes(
                clip_args,
                prior_model,
                tokenizer_model,
                text_tokens,
                text_mask,
                prior_config,
                section_entries,
                device,
                source_rng,
                recent_theme_entries,
            )
            codes = codes.to(device=device, dtype=torch.long)
            waveform = structured.decode_and_stitch_segments(
                codes,
                segment_ranges,
                tokenizer_model,
                tokenizer_config,
                args.source_overlap,
                args.theme_crossfade_ms,
            )
            candidate_waveform = waveform.cpu()
            chunk_energies = structured.measure_audio_chunk_energies(
                candidate_waveform,
                tokenizer_config.sample_rate,
                clip_args.clip_energy_check_seconds,
            )
            loudness_summary = structured.summarize_clip_loudness(chunk_energies)
            quiet_window = structured.find_quiet_audio_window(candidate_waveform, tokenizer_config.sample_rate, clip_args)
            has_energy = structured.clip_has_sufficient_energy(candidate_waveform, tokenizer_config.sample_rate, clip_args)
            has_loudness = structured.clip_has_sufficient_loudness(chunk_energies, clip_args)
            if quiet_window is None and has_energy and has_loudness:
                accepted_waveform = candidate_waveform
                recent_theme_entries = attempted_theme_entries
                quietest_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
                if retry_idx > 0:
                    print(f"Accepted clip {clip_idx + 1} after {retry_idx + 1} attempts")
                print(
                    f"Added clip {clip_idx + 1} to output section={section_name} "
                    f"quietest_chunk_rms={quietest_chunk['rms']:.4f} "
                    f"quietest_chunk_peak={quietest_chunk['peak']:.4f} "
                    f"median_chunk_rms={loudness_summary['median_rms']:.4f}"
                )
                break

            worst_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
            if quiet_window is not None:
                print(
                    f"Rejected clip with quiet portion {clip_idx + 1} section={section_name} "
                    f"attempt {retry_idx + 1}/{retry_count} at={quiet_window['start_seconds']:.3f}s "
                    f"short_rms={quiet_window['rms']:.4f} short_peak={quiet_window['peak']:.4f}"
                )
            elif not has_energy:
                print(
                    f"Rejected low-energy clip {clip_idx + 1} section={section_name} "
                    f"attempt {retry_idx + 1}/{retry_count} "
                    f"worst_chunk_rms={worst_chunk['rms']:.4f} worst_chunk_peak={worst_chunk['peak']:.4f}"
                )
            else:
                print(
                    f"Rejected low-loudness clip {clip_idx + 1} section={section_name} "
                    f"attempt {retry_idx + 1}/{retry_count} "
                    f"median_chunk_rms={loudness_summary['median_rms']:.4f} "
                    f"mean_chunk_rms={loudness_summary['mean_rms']:.4f}"
                )

        if accepted_waveform is None:
            print(f"Skipping clip {clip_idx + 1} section={section_name} after {retry_count} failed energy checks")
            continue
        clips.append(accepted_waveform)

    if not clips:
        raise RuntimeError("All generated clips failed the energy gate. Relax the thresholds or increase --clip-retry-count.")

    print(f"Accepted {len(clips)} clip(s) for final output")
    final_fade_ms = max(int(args.fade_ms), int(args.theme_crossfade_ms))
    output = structured.base.stitch_waveforms(
        clips,
        tokenizer_config.sample_rate,
        fade_ms=final_fade_ms,
    )
    if len(clips) > 1:
        print(f"Final clip stitch crossfade: {final_fade_ms}ms across {len(clips)} regular-BOS clips")
    if target_samples is not None:
        output = output[..., :target_samples]
    output = output - output.mean(dim=-1, keepdim=True)
    peak = output.abs().max().item()
    if peak > 0:
        output = output / max(1.0, peak / 0.98)
    rms = output.pow(2).mean().sqrt().item()
    if rms > 1e-6:
        output = output * min(1.5, 0.14 / rms)
        peak = output.abs().max().item()
        if peak > 0.98:
            output = output * (0.98 / peak)

    output = apply_song_intro(output, tokenizer_config.sample_rate, args.song_intro_fade_ms)
    output_path = args.output or structured.make_output_name(args.prompt)
    structured.base.save_audio_waveform(output_path, output, tokenizer_config.sample_rate)
    print(f"Saved regular-only intro/body sticky crossfaded source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()