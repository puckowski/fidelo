import torch

import generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_intro_body_continued as continued
from latent_audio_token_pipeline import match_audio_length


def configure_parser(parser):
    parser.description = (
        "Generate continued intro/body audio and join the sections with a bounded thematic "
        "midpoint plus smooth latent-space entry and exit ramps."
    )
    parser.add_argument(
        "--clip-duration-seconds",
        type=float,
        default=7.0,
        help="Duration generated for each latent clip. Defaults to 7 seconds.",
    )
    parser.add_argument(
        "--thematic-blend-in-seconds",
        type=float,
        default=0.75,
        help="Duration of the zero-slope latent ramp from intro into the thematic midpoint.",
    )
    parser.add_argument(
        "--thematic-average-seconds",
        type=float,
        default=0.25,
        help="Duration for which the bounded 50/50 thematic midpoint is held.",
    )
    parser.add_argument(
        "--thematic-blend-out-seconds",
        type=float,
        default=0.75,
        help="Duration of the zero-slope latent ramp from the thematic midpoint into regular audio.",
    )
    parser.add_argument(
        "--thematic-midpoint-max-scale",
        type=float,
        default=1.5,
        help="Maximum magnitude correction applied to the arithmetic latent midpoint.",
    )
    parser.add_argument(
        "--thematic-max-peak-ratio",
        type=float,
        default=2.5,
        help="Reject output when transition peak exceeds this multiple of adjacent intro/regular peak.",
    )
    parser.add_argument(
        "--disable-thematic-quality-gate",
        action="store_true",
        help="Report thematic transition diagnostics without rejecting excessive decoded peaks.",
    )


def smootherstep(progress: torch.Tensor) -> torch.Tensor:
    return progress * progress * progress * (progress * (progress * 6.0 - 15.0) + 10.0)


def bounded_thematic_midpoint(
    intro_latent: torch.Tensor,
    regular_latent: torch.Tensor,
    max_scale: float,
) -> torch.Tensor:
    midpoint = 0.5 * (intro_latent + regular_latent)
    midpoint_norm = midpoint.norm(dim=1, keepdim=True)
    target_norm = 0.5 * (
        intro_latent.norm(dim=1, keepdim=True)
        + regular_latent.norm(dim=1, keepdim=True)
    )
    safe_scale = target_norm / midpoint_norm.clamp_min(1e-6)
    safe_scale = safe_scale.clamp(min=1.0, max=max(1.0, float(max_scale)))
    return midpoint * safe_scale


def smooth_thematic_transition(
    intro_latent: torch.Tensor,
    regular_latent: torch.Tensor,
    blend_in_steps: int,
    average_steps: int,
    blend_out_steps: int,
    midpoint_max_scale: float,
) -> torch.Tensor:
    total_steps = blend_in_steps + average_steps + blend_out_steps
    if total_steps <= 0:
        return regular_latent[..., :0]
    if intro_latent.shape[-1] != total_steps or regular_latent.shape[-1] != total_steps:
        raise ValueError("Transition latent lengths do not match the requested phase lengths")

    midpoint = bounded_thematic_midpoint(intro_latent, regular_latent, midpoint_max_scale)
    parts = []
    offset = 0
    if blend_in_steps > 0:
        progress = torch.linspace(
            0.0,
            1.0,
            blend_in_steps,
            device=intro_latent.device,
            dtype=intro_latent.dtype,
        )
        weight = smootherstep(progress).view(1, 1, -1)
        parts.append(
            intro_latent[..., offset:offset + blend_in_steps] * (1.0 - weight)
            + midpoint[..., offset:offset + blend_in_steps] * weight
        )
        offset += blend_in_steps
    if average_steps > 0:
        parts.append(midpoint[..., offset:offset + average_steps])
        offset += average_steps
    if blend_out_steps > 0:
        progress = torch.linspace(
            0.0,
            1.0,
            blend_out_steps,
            device=intro_latent.device,
            dtype=intro_latent.dtype,
        )
        weight = smootherstep(progress).view(1, 1, -1)
        parts.append(
            midpoint[..., offset:offset + blend_out_steps] * (1.0 - weight)
            + regular_latent[..., offset:offset + blend_out_steps] * weight
        )
    return torch.cat(parts, dim=-1)


def describe_thematic_latents(
    intro_latent: torch.Tensor,
    regular_latent: torch.Tensor,
    transition: torch.Tensor,
    blend_in_steps: int,
    average_steps: int,
) -> None:
    midpoint_start = min(blend_in_steps, transition.shape[-1] - 1)
    midpoint_end = min(transition.shape[-1], midpoint_start + max(1, average_steps))
    thematic_region = transition[..., midpoint_start:midpoint_end]
    intro_region = intro_latent[..., midpoint_start:midpoint_end]
    regular_region = regular_latent[..., midpoint_start:midpoint_end]
    source_distance = (intro_region - regular_region).pow(2).mean().sqrt().item()
    intro_distance = (thematic_region - intro_region).pow(2).mean().sqrt().item()
    regular_distance = (thematic_region - regular_region).pow(2).mean().sqrt().item()

    entry_jump = (transition[..., 0] - intro_latent[..., 0]).pow(2).mean().sqrt().item()
    exit_jump = (transition[..., -1] - regular_latent[..., -1]).pow(2).mean().sqrt().item()
    if source_distance > 1e-8:
        intro_fraction = intro_distance / source_distance
        regular_fraction = regular_distance / source_distance
    else:
        intro_fraction = 0.0
        regular_fraction = 0.0
    print(
        "Verified thematic latent blend: "
        f"midpoint_distance_from_intro={intro_fraction:.3f} "
        f"midpoint_distance_from_regular={regular_fraction:.3f} "
        f"entry_endpoint_jump={entry_jump:.6f} exit_endpoint_jump={exit_jump:.6f}"
    )
    if source_distance > 1e-8 and (intro_distance <= 1e-8 or regular_distance <= 1e-8):
        raise RuntimeError("The thematic midpoint collapsed onto one source instead of creating a new latent trajectory")


def assemble_thematic_latent_song(
    latent_clips,
    section_names,
    blend_in_steps: int,
    average_steps: int,
    blend_out_steps: int,
    midpoint_max_scale: float,
    return_transition_spans: bool = False,
):
    if not latent_clips:
        raise ValueError("No latent clips provided")
    if len(latent_clips) != len(section_names):
        raise ValueError("Latent clip and section counts do not match")

    combined = latent_clips[0]
    transition_applied = False
    transition_spans = []
    requested_steps = blend_in_steps + average_steps + blend_out_steps
    for clip_idx, latent_clip in enumerate(latent_clips[1:], start=1):
        is_intro_body_boundary = (
            section_names[clip_idx - 1] == "intro"
            and section_names[clip_idx] == "body"
        )
        if not is_intro_body_boundary or requested_steps <= 0:
            combined = torch.cat([combined, latent_clip], dim=-1)
            continue

        transition_steps = min(requested_steps, combined.shape[-1], latent_clip.shape[-1])
        if transition_steps < requested_steps:
            ratio = transition_steps / requested_steps
            actual_blend_in = int(round(blend_in_steps * ratio))
            actual_average = int(round(average_steps * ratio))
            actual_blend_out = transition_steps - actual_blend_in - actual_average
        else:
            actual_blend_in = blend_in_steps
            actual_average = average_steps
            actual_blend_out = blend_out_steps

        transition = smooth_thematic_transition(
            combined[..., -transition_steps:],
            latent_clip[..., :transition_steps],
            actual_blend_in,
            actual_average,
            actual_blend_out,
            midpoint_max_scale,
        )
        describe_thematic_latents(
            combined[..., -transition_steps:],
            latent_clip[..., :transition_steps],
            transition,
            actual_blend_in,
            actual_average,
        )
        transition_start = combined.shape[-1] - transition_steps
        combined = torch.cat(
            [combined[..., :-transition_steps], transition, latent_clip[..., transition_steps:]],
            dim=-1,
        )
        transition_spans.append((transition_start, transition_start + transition_steps))
        transition_applied = True
        print(
            "Applied smooth thematic latent transition: "
            f"blend_in_steps={actual_blend_in} average_steps={actual_average} "
            f"blend_out_steps={actual_blend_out} boundary={clip_idx}/{clip_idx + 1}"
        )

    if not transition_applied:
        print("No accepted intro-to-body boundary was available for thematic latent blending")
    if return_transition_spans:
        return combined, transition_spans
    return combined


def verify_decoded_transition_quality(
    decoded: torch.Tensor,
    transition_spans,
    tokenizer_config,
    max_peak_ratio: float,
    enforce_gate: bool,
) -> None:
    if not torch.isfinite(decoded).all():
        raise RuntimeError("The thematic decoder output contains NaN or infinite samples")

    context_samples = max(1, int(round(0.5 * tokenizer_config.sample_rate)))
    total_samples = decoded.shape[-1]
    for transition_idx, (start_step, end_step) in enumerate(transition_spans, start=1):
        start = max(0, int(round(start_step * tokenizer_config.total_stride)))
        end = min(total_samples, int(round(end_step * tokenizer_config.total_stride)))
        transition_audio = decoded[..., start:end]
        intro_context = decoded[..., max(0, start - context_samples):start]
        regular_context = decoded[..., end:min(total_samples, end + context_samples)]
        references = [chunk for chunk in (intro_context, regular_context) if chunk.numel() > 0]
        if transition_audio.numel() == 0 or not references:
            continue

        transition_rms = transition_audio.pow(2).mean().sqrt().item()
        transition_peak = transition_audio.abs().max().item()
        reference_rms = max(chunk.pow(2).mean().sqrt().item() for chunk in references)
        reference_peak = max(chunk.abs().max().item() for chunk in references)
        rms_ratio = transition_rms / max(reference_rms, 1e-6)
        peak_ratio = transition_peak / max(reference_peak, 1e-6)
        print(
            f"Verified decoded thematic transition {transition_idx}: "
            f"rms={transition_rms:.4f} adjacent_rms={reference_rms:.4f} rms_ratio={rms_ratio:.3f} "
            f"peak={transition_peak:.4f} adjacent_peak={reference_peak:.4f} peak_ratio={peak_ratio:.3f}"
        )
        if enforce_gate and peak_ratio > max(1.0, max_peak_ratio):
            raise RuntimeError(
                f"Decoded thematic transition peak ratio {peak_ratio:.3f} exceeds "
                f"--thematic-max-peak-ratio {max_peak_ratio:.3f}"
            )


@torch.no_grad()
def decode_thematic_song(
    accepted_codes,
    accepted_sections,
    tokenizer_model,
    tokenizer_config,
    args,
):
    device = next(tokenizer_model.parameters()).device
    latent_clips = [tokenizer_model.lookup_codes(codes.to(device)) for codes in accepted_codes]
    steps_per_second = tokenizer_config.sample_rate / tokenizer_config.total_stride
    blend_in_steps = max(0, int(round(args.thematic_blend_in_seconds * steps_per_second)))
    average_steps = max(0, int(round(args.thematic_average_seconds * steps_per_second)))
    blend_out_steps = max(0, int(round(args.thematic_blend_out_seconds * steps_per_second)))
    combined_latents, transition_spans = assemble_thematic_latent_song(
        latent_clips,
        accepted_sections,
        blend_in_steps,
        average_steps,
        blend_out_steps,
        args.thematic_midpoint_max_scale,
        return_transition_spans=True,
    )
    decoded = tokenizer_model.decoder(tokenizer_model.post_quant(combined_latents))
    decoded_samples = max(1, int(round(combined_latents.shape[-1] * tokenizer_config.total_stride)))
    decoded = match_audio_length(decoded, decoded_samples).squeeze(0).cpu()
    verify_decoded_transition_quality(
        decoded,
        transition_spans,
        tokenizer_config,
        args.thematic_max_peak_ratio,
        not args.disable_thematic_quality_gate,
    )
    return decoded


if __name__ == "__main__":
    continued.main(configure_parser=configure_parser, decode_song=decode_thematic_song)