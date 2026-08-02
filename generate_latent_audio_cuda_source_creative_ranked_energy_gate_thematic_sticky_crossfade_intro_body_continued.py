import argparse
from dataclasses import replace
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

import torch

import generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_intro_body as intro_body
import generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_structured as structured
from latent_audio_token_pipeline import match_audio_length


def format_console_text(value: object) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return text
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def parse_args(configure_parser=None):
    def add_continuation_args(parser):
        parser.description = (
            "Generate prompt-conditioned intro/body audio with accepted token-tail replay "
            "and a continuous-latent blend at the actual intro-source to regular-source boundary."
        )
        parser.add_argument(
            "--continuation-prefix-seconds",
            type=float,
            default=2.0,
            help="Accepted token history used to condition each later clip; it is not copied into the final output.",
        )
        parser.add_argument(
            "--disable-persistent-gru-state",
            action="store_true",
            help="Reconstruct state by replaying prefix tokens instead of carrying the exact accepted GRU hidden state.",
        )
        parser.add_argument(
            "--intro-body-prior-seconds",
            type=float,
            default=2.0,
            help="Use one coherent source-to-prior handoff across this interval at the first intro-to-body boundary. Set to 0 to disable.",
        )
        parser.add_argument(
            "--intro-body-source-strength",
            type=float,
            default=1.0,
            help="Fraction of the handoff interval kept as contiguous source tokens before switching once to the prior continuation.",
        )
        parser.add_argument(
            "--allow-raw-prior-transition",
            action="store_true",
            help="Allow direct raw-prior tokens at the intro/body boundary. Disabled by default because unstable priors can produce blaring audio.",
        )
        parser.add_argument(
            "--intro-body-average-seconds",
            type=float,
            default=0.4,
            help="Duration that smoothly moves from the intro tail to a 50/50 intro/regular VQ blend.",
        )
        parser.add_argument(
            "--intro-body-overlap-seconds",
            type=float,
            default=1.5,
            help="After the fixed average, gradually release from 50/50 into 100% regular VQ embeddings over this duration.",
        )
        parser.add_argument(
            "--disable-intro-source-continuation",
            action="store_true",
            help="Do not keep the accepted intro source eligible for the first body clip.",
        )
        if configure_parser is not None:
            configure_parser(parser)

    return intro_body.parse_args(configure_parser=add_continuation_args)


def build_continuation_prefix(
    accepted_codes: Optional[torch.Tensor],
    prefix_steps: int,
) -> Optional[torch.Tensor]:
    if accepted_codes is None or prefix_steps <= 0:
        return None
    return structured.base.extract_code_tail(accepted_codes, prefix_steps)


def average_intro_body_latents(
    latent_clips: List[torch.Tensor],
    section_names: List[str],
    average_steps: int,
    overlap_steps: int,
) -> torch.Tensor:
    if not latent_clips:
        raise ValueError("No latent clips provided")
    if len(latent_clips) != len(section_names):
        raise ValueError("Latent clip and section counts do not match")

    combined = latent_clips[0]
    for clip_idx, latent_clip in enumerate(latent_clips[1:], start=1):
        is_intro_body_boundary = section_names[clip_idx - 1] == "intro" and section_names[clip_idx] == "body"
        requested_transition_steps = average_steps + overlap_steps
        if is_intro_body_boundary and requested_transition_steps > 0:
            transition_steps = min(
                requested_transition_steps,
                combined.shape[-1],
                latent_clip.shape[-1],
            )
            fixed_steps = min(average_steps, transition_steps)
            release_steps = transition_steps - fixed_steps
            intro_transition = combined[..., -transition_steps:]
            regular_transition = latent_clip[..., :transition_steps]
            transition_parts = []
            if fixed_steps > 0:
                progress = torch.linspace(
                    0.0,
                    1.0,
                    fixed_steps + 1,
                    device=combined.device,
                    dtype=combined.dtype,
                )[1:]
                weight_scale = 0.5 if release_steps > 0 else 1.0
                regular_weight = (weight_scale * 0.5 * (1.0 - torch.cos(math.pi * progress))).view(1, 1, -1)
                transition_parts.append(
                    intro_transition[..., :fixed_steps] * (1.0 - regular_weight)
                    + regular_transition[..., :fixed_steps] * regular_weight
                )
            if release_steps > 0:
                progress = torch.linspace(
                    0.0,
                    1.0,
                    release_steps + 1,
                    device=combined.device,
                    dtype=combined.dtype,
                )[1:]
                start_weight = 0.5 if fixed_steps > 0 else 0.0
                regular_weight = (
                    start_weight + (1.0 - start_weight) * 0.5 * (1.0 - torch.cos(math.pi * progress))
                ).view(1, 1, -1)
                transition_parts.append(
                    intro_transition[..., fixed_steps:] * (1.0 - regular_weight)
                    + regular_transition[..., fixed_steps:] * regular_weight
                )
            transition_latent = torch.cat(transition_parts, dim=-1)
            distance_from_intro = (transition_latent - intro_transition).pow(2).mean().sqrt().item()
            distance_from_regular = (transition_latent - regular_transition).pow(2).mean().sqrt().item()
            blend_verified = distance_from_intro > 1e-7 and distance_from_regular > 1e-7
            combined = torch.cat(
                [
                    combined[..., :-transition_steps],
                    transition_latent,
                    latent_clip[..., transition_steps:],
                ],
                dim=-1,
            )
            print(
                f"Applied intro-to-regular VQ transition: intro_to_midpoint_steps={fixed_steps} "
                f"midpoint_to_regular_steps={release_steps} "
                f"at accepted clip boundary {clip_idx}/{clip_idx + 1}"
            )
            print(
                f"Verified blended VQ latents: distinct_from_both_sources={blend_verified} "
                f"distance_from_intro={distance_from_intro:.6f} "
                f"distance_from_regular={distance_from_regular:.6f}"
            )
        else:
            combined = torch.cat([combined, latent_clip], dim=-1)
    return combined


@torch.no_grad()
def decode_latent_averaged_song(
    accepted_codes: List[torch.Tensor],
    accepted_sections: List[str],
    tokenizer_model,
    tokenizer_config,
    average_seconds: float,
    overlap_seconds: float,
) -> torch.Tensor:
    latent_clips = [
        tokenizer_model.lookup_codes(codes.to(next(tokenizer_model.parameters()).device))
        for codes in accepted_codes
    ]
    average_steps = max(
        0,
        int(round(max(0.0, average_seconds) * tokenizer_config.sample_rate / tokenizer_config.total_stride)),
    )
    overlap_steps = max(
        0,
        int(round(max(0.0, overlap_seconds) * tokenizer_config.sample_rate / tokenizer_config.total_stride)),
    )
    combined_latents = average_intro_body_latents(
        latent_clips,
        accepted_sections,
        average_steps,
        overlap_steps,
    )
    decoded = tokenizer_model.decoder(tokenizer_model.post_quant(combined_latents))
    decoded_samples = max(1, int(round(combined_latents.shape[-1] * tokenizer_config.total_stride)))
    return match_audio_length(decoded, decoded_samples).squeeze(0).cpu()


def main(configure_parser=None, decode_song=None, prepare_clip_args=None):
    args = parse_args(configure_parser=configure_parser)
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
    runtime_clip_seconds = getattr(args, "clip_duration_seconds", None)
    if runtime_clip_seconds is not None:
        runtime_clip_seconds = max(0.1, float(runtime_clip_seconds))
        tokenizer_config = replace(tokenizer_config, clip_seconds=runtime_clip_seconds)
        prior_config = replace(prior_config, clip_seconds=runtime_clip_seconds)
        print(
            f"Runtime clip duration: {runtime_clip_seconds:.3f}s "
            f"({prior_config.latent_steps} latent steps, {tokenizer_config.clip_samples} samples)"
        )
    prior_checkpoint = "best_latent_prior.pt"
    if not os.path.exists(os.path.join(args.prior_dir, prior_checkpoint)):
        prior_checkpoint = "latent_prior.pt"
    print(f"Loaded prior: {os.path.abspath(args.prior_dir)}\\{prior_checkpoint}")

    clip_count = args.clip_count
    target_samples = None
    if args.duration_seconds > 0:
        target_samples = int(round(args.duration_seconds * tokenizer_config.sample_rate))
        clip_count = max(1, math.ceil(args.duration_seconds / tokenizer_config.clip_seconds))
        if hasattr(args, "clip_crossfade_ms"):
            clip_fade_seconds = max(0.0, float(args.clip_crossfade_ms) / 1000.0)
            section_fade_seconds = max(
                clip_fade_seconds,
                float(getattr(args, "section_crossfade_ms", args.clip_crossfade_ms)) / 1000.0,
            )
            section_alignment_skip_seconds = max(
                0.0,
                0.5 * (
                    float(getattr(args, "latent_transition_seconds", 0.0))
                    - section_fade_seconds
                ),
            )
            while True:
                section_overlap = (
                    section_fade_seconds
                    if clip_count > 1 and 0.0 < float(args.intro_ratio) < 1.0
                    else (clip_fade_seconds if clip_count > 1 else 0.0)
                )
                estimated_seconds = (
                    clip_count * tokenizer_config.clip_seconds
                    - max(0, clip_count - 2) * clip_fade_seconds
                    - section_overlap
                    - (section_alignment_skip_seconds if clip_count > 1 else 0.0)
                )
                if estimated_seconds >= args.duration_seconds:
                    break
                clip_count += 1
            print(
                f"Crossfade-aware clip plan: clips={clip_count} "
                f"estimated_output={estimated_seconds:.2f}s target={args.duration_seconds:.2f}s"
            )

    prefix_steps = min(
        prior_config.latent_steps,
        max(
            0,
            int(
                round(
                    args.continuation_prefix_seconds
                    * prior_config.latent_steps
                    / max(prior_config.clip_seconds, 1e-6)
                )
            ),
        ),
    )
    effective_prefix_seconds = prefix_steps * prior_config.total_stride / prior_config.sample_rate
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
    if not candidate_entries:
        raise RuntimeError("No source candidates found for source-creative generation.")

    print(f"Loaded {len(candidate_entries)} source candidates")
    print(
        f"Continuation prefix: requested={args.continuation_prefix_seconds:.2f}s "
        f"effective={effective_prefix_seconds:.2f}s "
        f"({prefix_steps} latent steps)"
    )
    for entry in candidate_entries:
        source_type = "beginning" if entry.get("song_beginning", False) else "regular"
        print(
            f"- {entry['file']} | source_type={source_type} | score={entry['match_score']:.2f} | "
            f"{format_console_text(entry['text'])}"
        )

    section_names = intro_body.build_song_sections(clip_count, args.intro_ratio)
    accepted_code_clips: List[torch.Tensor] = []
    accepted_sections: List[str] = []
    accepted_clip_metadata: List[Dict] = []
    accepted_codes: Optional[torch.Tensor] = None
    accepted_recurrent_state = None
    sequence_bos_is_beginning = False
    previous_accepted_section: Optional[str] = None
    recent_theme_entries: List[Dict] = []
    persistent_gru_enabled = not args.disable_persistent_gru_state
    print(
        "GRU continuation mode: "
        + ("persistent accepted hidden state" if persistent_gru_enabled else "prefix replay")
    )

    for clip_idx in range(clip_count):
        args.song_beginning = accepted_codes is None and clip_idx < max(0, args.beginning_bos_clips)
        section_name = section_names[clip_idx]
        section_entries = intro_body.build_section_candidate_entries(
            candidate_entries,
            section_name,
            args.intro_theme_top_n,
        )
        if not section_entries:
            print(f"Skipping clip {clip_idx + 1} section={section_name}: no eligible source entries")
            continue

        continuation_prefix = build_continuation_prefix(accepted_codes, prefix_steps)
        effective_clip_args = intro_body.build_section_args(args, section_name)
        effective_clip_args.prefix_song_beginning = sequence_bos_is_beginning
        blend_section_name = section_name
        is_intro_body_transition = previous_accepted_section == "intro" and section_name == "body"
        if is_intro_body_transition:
            if not args.disable_intro_source_continuation and recent_theme_entries:
                intro_sources = [
                    entry for entry in recent_theme_entries if entry.get("song_beginning", False)
                ]
                existing_files = {entry.get("file") for entry in section_entries}
                continued_sources = [
                    entry for entry in intro_sources if entry.get("file") not in existing_files
                ]
                if continued_sources:
                    section_entries = continued_sources[-1:] + section_entries
                    effective_clip_args.forced_initial_theme_file = continued_sources[-1]["file"]
                    blend_section_name = "intro"
                    print(
                        "Continuing first body theme from accepted intro source: "
                        f"{continued_sources[-1]['file']}"
                    )
                    print(
                        "Deferring final intro-to-regular latent blend until the next accepted "
                        "regular-source clip boundary"
                    )
            effective_clip_args.initial_handoff_steps = max(
                0,
                int(
                    round(
                        args.intro_body_prior_seconds
                        * prior_config.latent_steps
                        / max(prior_config.clip_seconds, 1e-6)
                    )
                ),
            )
            requested_source_fraction = max(0.0, min(1.0, args.intro_body_source_strength))
            effective_clip_args.initial_source_fraction = (
                requested_source_fraction if args.allow_raw_prior_transition else 1.0
            )
            if not args.allow_raw_prior_transition and requested_source_fraction < 1.0:
                print(
                    f"Ignoring --intro-body-source-strength {requested_source_fraction:.2f}: "
                    "raw-prior transition audio requires --allow-raw-prior-transition."
                )
            print(
                f"Intro-to-body coherent handoff: {args.intro_body_prior_seconds:.2f}s "
                f"source_fraction={effective_clip_args.initial_source_fraction:.2f} "
                f"prior_suffix_fraction={1.0 - effective_clip_args.initial_source_fraction:.2f}"
            )
            if effective_clip_args.initial_source_fraction < 0.8:
                print(
                    "Warning: low intro-body source strength exposes a long raw-prior suffix; "
                    "use 1.0 when the prior produces blaring or out-of-dataset audio."
                )
        if prepare_clip_args is not None:
            effective_clip_args = prepare_clip_args(
                effective_clip_args,
                section_name,
                previous_accepted_section,
                accepted_code_clips,
                section_entries,
                tokenizer_model,
                tokenizer_config,
                prior_config,
                device,
            )
        energy_profile = "beginning" if args.song_beginning else "regular"
        prefix_length = 0 if continuation_prefix is None else structured.base.code_step_count(continuation_prefix)
        print(
            f"Generating continued {section_name} clip {clip_idx + 1}/{clip_count} on {device} "
            f"energy_profile={energy_profile} replay_prefix_steps={prefix_length}..."
        )

        accepted_new_codes = None
        accepted_attempt_state = None
        accepted_segment_ranges = None
        retry_count = max(1, int(args.clip_retry_count))
        for retry_idx in range(retry_count):
            clip_args = argparse.Namespace(**vars(effective_clip_args))
            clip_args.seed = args.seed + clip_idx * 1009 + retry_idx * 7919
            clip_args.generation_retry_index = retry_idx
            codes, segment_ranges, attempted_theme_entries, attempted_recurrent_state = structured.generate_source_creative_ranked_codes(
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
                continuation_prefix_codes=continuation_prefix,
                recurrent_state=(accepted_recurrent_state if persistent_gru_enabled else None),
                return_recurrent_state=True,
            )
            codes = codes.to(device=device, dtype=torch.long)
            energy_waveform = structured.decode_and_stitch_segments(
                codes,
                segment_ranges,
                tokenizer_model,
                tokenizer_config,
                args.source_overlap,
                args.theme_crossfade_ms,
            ).cpu()
            chunk_energies = structured.measure_audio_chunk_energies(
                energy_waveform,
                tokenizer_config.sample_rate,
                clip_args.clip_energy_check_seconds,
            )
            loudness_summary = structured.summarize_clip_loudness(chunk_energies)
            quiet_window = structured.find_quiet_audio_window(
                energy_waveform,
                tokenizer_config.sample_rate,
                clip_args,
            )
            has_energy = structured.clip_has_sufficient_energy(
                energy_waveform,
                tokenizer_config.sample_rate,
                clip_args,
            )
            has_loudness = structured.clip_has_sufficient_loudness(chunk_energies, clip_args)
            if quiet_window is None and has_energy and has_loudness:
                accepted_new_codes = codes.detach().cpu()
                accepted_attempt_state = attempted_recurrent_state
                accepted_segment_ranges = list(segment_ranges)
                recent_theme_entries = attempted_theme_entries
                quietest_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
                print(
                    f"Accepted clip {clip_idx + 1} section={section_name} attempt={retry_idx + 1} "
                    f"quietest_chunk_rms={quietest_chunk['rms']:.4f} "
                    f"median_chunk_rms={loudness_summary['median_rms']:.4f}"
                )
                break

            worst_chunk = min(chunk_energies, key=lambda item: (item["rms"], item["peak"]))
            reason = "quiet portion" if quiet_window is not None else ("low energy" if not has_energy else "low loudness")
            quiet_detail = ""
            if quiet_window is not None:
                quiet_detail = (
                    f" quiet_start={quiet_window['start_seconds']:.3f}s"
                    f" quiet_ac_rms={quiet_window.get('ac_rms', quiet_window['rms']):.4f}"
                )
            print(
                f"Rejected clip {clip_idx + 1} section={section_name} attempt "
                f"{retry_idx + 1}/{retry_count}: {reason}; "
                f"worst_rms={worst_chunk['rms']:.4f} worst_peak={worst_chunk['peak']:.4f}"
                f"{quiet_detail}"
            )

        if accepted_new_codes is None:
            print(f"Skipping clip {clip_idx + 1} section={section_name} after {retry_count} failed energy checks")
            continue

        if accepted_codes is None:
            sequence_bos_is_beginning = bool(args.song_beginning)
        accepted_codes = accepted_new_codes
        if persistent_gru_enabled:
            accepted_recurrent_state = accepted_attempt_state
            hidden_shape = tuple(accepted_recurrent_state["hidden"].shape)
            print(f"Committed persistent GRU state for accepted clip: hidden_shape={hidden_shape}")
        previous_accepted_section = section_name
        accepted_code_clips.append(accepted_new_codes)
        accepted_sections.append(blend_section_name)
        accepted_clip_metadata.append(
            {
                "generated_section": section_name,
                "segment_ranges": accepted_segment_ranges,
                "latent_transition_first_motif_step": getattr(
                    effective_clip_args,
                    "latent_transition_first_motif_step",
                    None,
                )
            }
        )
        print(
            f"Final latent stitch role for accepted clip {len(accepted_code_clips)}: "
            f"{blend_section_name}"
        )

    if not accepted_code_clips:
        raise RuntimeError("All generated clips failed the energy gate. Relax the thresholds or increase --clip-retry-count.")

    print(f"Accepted {len(accepted_code_clips)} clip(s) for final output")
    if decode_song is None:
        output = decode_latent_averaged_song(
            accepted_code_clips,
            accepted_sections,
            tokenizer_model,
            tokenizer_config,
            args.intro_body_average_seconds,
            args.intro_body_overlap_seconds,
        )
    else:
        args.accepted_clip_metadata = accepted_clip_metadata
        output = decode_song(
            accepted_code_clips,
            accepted_sections,
            tokenizer_model,
            tokenizer_config,
            args,
        )
    if decode_song is None:
        print("Final accepted clips decoded as one latent sequence")
    else:
        print("Final accepted clips decoded with the configured final decoder")
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

    output = intro_body.apply_song_intro(output, tokenizer_config.sample_rate, args.song_intro_fade_ms)
    output_path = args.output or structured.make_output_name(args.prompt)
    structured.base.save_audio_waveform(output_path, output, tokenizer_config.sample_rate)
    print(f"Saved continued intro/body source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()