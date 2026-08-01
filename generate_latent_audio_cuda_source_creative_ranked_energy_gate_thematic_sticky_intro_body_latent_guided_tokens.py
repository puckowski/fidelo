import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_crossfade_intro_body_continued as continued
from latent_audio_token_pipeline import match_audio_length


def configure_parser(parser):
    parser.description = (
        "Generate a continued song with an intro/body overlap quantized only to complete "
        "observed residual-VQ pairs from the intro tail or a quality-checked body motif."
    )
    parser.add_argument(
        "--latent-transition-seconds",
        type=float,
        default=5.0,
        help="Duration of first-body token generation guided by the evolving latent target.",
    )
    parser.add_argument(
        "--latent-motif-match-seconds",
        type=float,
        default=0.75,
        help="Intro-tail duration used to retrieve the closest regular body motif.",
    )
    parser.add_argument(
        "--latent-motif-scan-step-divisor",
        type=int,
        default=12,
        help="Motif scan density; larger values inspect source windows more densely.",
    )
    parser.add_argument(
        "--latent-motif-match-points",
        type=int,
        default=64,
        help="Evenly spaced latent points used for efficient batched motif comparison.",
    )
    parser.add_argument(
        "--latent-motif-prompt-weight",
        type=float,
        default=0.10,
        help="Weight of prompt-match score when selecting the body motif.",
    )
    parser.add_argument(
        "--latent-motif-choice-top",
        type=int,
        default=3,
        help="Sample among this many highest-ranked energy-valid body motifs. Set to 1 for the exact best match.",
    )
    parser.add_argument(
        "--latent-motif-choice-temperature",
        type=float,
        default=0.75,
        help="Motif selection randomness among the top matches; lower values favor the best match.",
    )
    parser.add_argument(
        "--latent-guidance-strength",
        type=float,
        default=4.0,
        help="Compatibility option retained for older commands; valid-pair overlap does not use arbitrary prior candidates.",
    )
    parser.add_argument(
        "--latent-guidance-magnitude-weight",
        type=float,
        default=0.5,
        help="Compatibility option retained for older candidate-guided commands.",
    )
    parser.add_argument(
        "--latent-guidance-repeat-penalty",
        type=float,
        default=0.25,
        help="Compatibility option retained for older candidate-guided commands.",
    )
    parser.add_argument(
        "--latent-guidance-candidate-top-k",
        type=int,
        default=4,
        help="Compatibility option retained for older candidate-guided commands.",
    )
    parser.add_argument(
        "--latent-guidance-temperature",
        type=float,
        default=0.0,
        help="Compatibility option retained for older candidate-guided commands.",
    )
    parser.add_argument(
        "--latent-guidance-retry-temperature",
        type=float,
        default=0.2,
        help="Minimum candidate temperature after a guided attempt fails its audio-quality gate.",
    )
    parser.add_argument(
        "--latent-guidance-fallback-after-retry",
        type=int,
        default=1,
        help="Use exact retrieved motif tokens after this many rejected guided attempts.",
    )
    parser.add_argument(
        "--latent-transition-min-ac-rms",
        type=float,
        default=0.008,
        help="Reject a guided body clip containing DC-like windows below this mean-centered RMS.",
    )
    parser.add_argument(
        "--latent-overlap-neighborhood-seconds",
        type=float,
        default=0.0,
        help="Nearby intro and motif range searched for complete observed residual-VQ pairs.",
    )
    parser.add_argument(
        "--latent-overlap-neighborhood-candidates",
        type=int,
        default=1,
        help="Evenly spaced candidate positions drawn from each source neighborhood.",
    )
    parser.add_argument(
        "--latent-overlap-continuity-weight",
        type=float,
        default=2.0,
        help="Penalty for latent jumps from the previously selected complete pair.",
    )
    parser.add_argument(
        "--latent-overlap-offset-weight",
        type=float,
        default=0.05,
        help="Penalty for selecting a pair far from the aligned source position.",
    )
    parser.add_argument(
        "--latent-overlap-progression-weight",
        type=float,
        default=5.0,
        help="Penalty for repeats or skips instead of advancing one observed source step.",
    )
    parser.add_argument(
        "--clip-crossfade-ms",
        type=int,
        default=1500,
        help="Equal-power waveform crossfade at every accepted clip/sequence boundary.",
    )
    parser.add_argument(
        "--section-crossfade-ms",
        type=int,
        default=3500,
        help="Equal-power intro-to-regular fade centered on the selected regular-motif token handoff.",
    )
    parser.set_defaults(
        disable_intro_source_continuation=True,
        intro_body_prior_seconds=0.0,
        intro_body_average_seconds=0.4,
        intro_body_overlap_seconds=1.5,
        intro_theme_top_n=3,
        intro_theme_temperature=0.75,
    )


def smootherstep(progress: torch.Tensor) -> torch.Tensor:
    return progress * progress * progress * (progress * (progress * 6.0 - 15.0) + 10.0)


def _batched_codes(codes: torch.Tensor, tokenizer_model) -> torch.Tensor:
    num_quantizers = len(tokenizer_model.quantizers)
    if codes.dim() == 3 and codes.shape[1] == num_quantizers:
        return codes
    if num_quantizers == 1:
        if codes.dim() == 1:
            return codes.unsqueeze(0)
        if codes.dim() == 2:
            return codes
    else:
        if codes.dim() == 2 and codes.shape[0] == num_quantizers:
            return codes.unsqueeze(0)
    raise ValueError(
        f"Cannot normalize code shape {tuple(codes.shape)} for {num_quantizers} quantizer streams"
    )


def choose_ranked_motif(valid_motifs, args):
    choice_count = min(
        len(valid_motifs),
        max(1, int(getattr(args, "latent_motif_choice_top", 1))),
    )
    choices = valid_motifs[:choice_count]
    if len(choices) == 1:
        return choices[0]

    temperature = max(1e-6, float(getattr(args, "latent_motif_choice_temperature", 0.0)))
    scores = [item[0] for item in choices]
    best_score = max(scores)
    weights = [math.exp((score - best_score) / temperature) for score in scores]
    pick = random.Random(int(getattr(args, "seed", 0)) + 641197).random() * sum(weights)
    running = 0.0
    for motif, weight in zip(choices, weights):
        running += weight
        if pick <= running:
            return motif
    return choices[-1]


@torch.no_grad()
def retrieve_body_motif(
    intro_codes: torch.Tensor,
    body_entries: List[Dict],
    tokenizer_model,
    tokenizer_config,
    args,
    device: torch.device,
) -> Optional[Tuple[Dict, int, torch.Tensor, float]]:
    steps_per_second = tokenizer_config.sample_rate / tokenizer_config.total_stride
    requested_steps = max(1, int(round(max(0.0, args.latent_transition_seconds) * steps_per_second)))
    match_steps = max(1, int(round(max(0.0, args.latent_motif_match_seconds) * steps_per_second)))
    match_steps = min(match_steps, intro_codes.shape[-1], requested_steps)
    intro_tail = intro_codes[..., -match_steps:]
    match_points = min(match_steps, max(1, int(args.latent_motif_match_points)))
    sample_positions = torch.linspace(0, match_steps - 1, match_points).round().long()
    intro_latent = tokenizer_model.lookup_codes(
        _batched_codes(intro_tail[..., sample_positions], tokenizer_model).to(device)
    )
    intro_vector = F.normalize(intro_latent.float().flatten(), dim=0, eps=1e-8)

    ranked_results = []
    scan_divisor = max(1, int(args.latent_motif_scan_step_divisor))
    scan_step = max(1, requested_steps // scan_divisor)
    prompt_weight = max(0.0, float(args.latent_motif_prompt_weight))

    for entry in body_entries:
        if entry.get("song_beginning", False):
            continue
        source_codes = entry["codes"]
        if source_codes.shape[-1] < requested_steps:
            continue
        last_start = source_codes.shape[-1] - requested_steps
        starts = list(range(0, last_start + 1, scan_step))
        if starts[-1] != last_start:
            starts.append(last_start)
        sampled_heads = torch.stack(
            [source_codes[..., start + sample_positions] for start in starts],
            dim=0,
        )
        motif_latents = tokenizer_model.lookup_codes(sampled_heads.to(device))
        motif_vectors = F.normalize(motif_latents.float().flatten(start_dim=1), dim=1, eps=1e-8)
        similarities = motif_vectors @ intro_vector
        for start, similarity_tensor in zip(starts, similarities):
            similarity = similarity_tensor.item()
            score = similarity + prompt_weight * float(entry.get("match_score", 0.0))
            ranked_results.append((score, entry, start, similarity))

    ranked_results.sort(key=lambda item: item[0], reverse=True)
    quality_check_count = max(1, int(getattr(args, "window_energy_check_top", 12)))
    valid_motifs = []
    for score, entry, start, similarity in ranked_results[:quality_check_count]:
        motif_codes = entry["codes"][..., start:start + requested_steps].clone()
        motif_audio = tokenizer_model.decode_codes(
            _batched_codes(motif_codes, tokenizer_model).to(device)
        ).detach().cpu()
        quiet_window = continued.structured.find_quiet_audio_window(
            motif_audio,
            tokenizer_config.sample_rate,
            args,
        )
        has_energy = continued.structured.clip_has_sufficient_energy(
            motif_audio,
            tokenizer_config.sample_rate,
            args,
        )
        if quiet_window is None and has_energy:
            valid_motifs.append((score, entry, start, motif_codes, similarity))
            continue
        quiet_ac_rms = None if quiet_window is None else quiet_window.get("ac_rms")
        print(
            "Rejected quiet latent-transition motif "
            f"{entry['file']} start_step={start} "
            f"quiet_ac_rms={quiet_ac_rms if quiet_ac_rms is not None else 'n/a'}"
        )

    if not valid_motifs:
        return None

    _, entry, start, motif_codes, similarity = choose_ranked_motif(valid_motifs, args)
    return entry, start, motif_codes, similarity


@torch.no_grad()
def build_evolving_latent_target(
    intro_codes: torch.Tensor,
    motif_codes: torch.Tensor,
    tokenizer_model,
    device: torch.device,
) -> torch.Tensor:
    transition_steps = min(intro_codes.shape[-1], motif_codes.shape[-1])
    intro_tail = intro_codes[..., -transition_steps:]
    motif_codes = motif_codes[..., :transition_steps]
    intro_latent = tokenizer_model.lookup_codes(_batched_codes(intro_tail, tokenizer_model).to(device))
    motif_latent = tokenizer_model.lookup_codes(_batched_codes(motif_codes, tokenizer_model).to(device))

    progress = torch.linspace(0.0, 1.0, transition_steps, device=device, dtype=intro_latent.dtype)
    motif_weight = smootherstep(progress).view(1, 1, -1)
    intro_boundary = intro_latent[..., -1:].expand_as(motif_latent)
    translated_motif = intro_boundary + motif_latent - motif_latent[..., :1]
    target = translated_motif * (1.0 - motif_weight) + motif_latent * motif_weight
    return target.detach().cpu()


@torch.no_grad()
def build_valid_token_overlap(
    intro_codes: torch.Tensor,
    motif_codes: torch.Tensor,
    tokenizer_model,
    device: torch.device,
    neighborhood_steps: int = 0,
    neighborhood_candidates: int = 1,
    continuity_weight: float = 0.0,
    offset_weight: float = 0.0,
    progression_weight: float = 0.5,
    return_diagnostics: bool = False,
):
    intro_batched = _batched_codes(intro_codes, tokenizer_model).to(device)
    motif_batched = _batched_codes(motif_codes, tokenizer_model).to(device)
    if len(tokenizer_model.quantizers) == 1:
        if intro_batched.dim() == 2:
            intro_batched = intro_batched.unsqueeze(1)
        if motif_batched.dim() == 2:
            motif_batched = motif_batched.unsqueeze(1)
    transition_steps = min(intro_batched.shape[-1], motif_batched.shape[-1])
    intro_tail = intro_batched[..., -transition_steps:]
    motif_head = motif_batched[..., :transition_steps]
    intro_latent = tokenizer_model.lookup_codes(intro_tail)
    motif_latent = tokenizer_model.lookup_codes(motif_head)

    progress = torch.linspace(0.0, 1.0, transition_steps, device=device, dtype=intro_latent.dtype)
    regular_weight = smootherstep(progress).view(1, 1, -1)
    averaged_target = intro_latent * (1.0 - regular_weight) + motif_latent * regular_weight

    radius = max(0, int(neighborhood_steps))
    requested_candidates = max(1, int(neighborhood_candidates))
    if radius <= 0 or requested_candidates == 1:
        offsets = torch.zeros(1, device=device, dtype=torch.long)
    else:
        requested_candidates = min(requested_candidates, 2 * radius + 1)
        if requested_candidates % 2 == 0:
            requested_candidates = min(2 * radius + 1, requested_candidates + 1)
        offsets = torch.linspace(
            -radius,
            radius,
            requested_candidates,
            device=device,
        ).round().long().unique(sorted=True)

    aligned_positions = torch.arange(transition_steps, device=device).unsqueeze(1)
    candidate_positions = (aligned_positions + offsets.unsqueeze(0)).clamp(0, transition_steps - 1)
    candidate_count = candidate_positions.shape[1]

    def gather_code_candidates(codes: torch.Tensor) -> torch.Tensor:
        gather_indices = candidate_positions.view(1, 1, transition_steps, candidate_count).expand(
            codes.shape[0],
            codes.shape[1],
            -1,
            -1,
        )
        expanded = codes.unsqueeze(-1).expand(-1, -1, -1, candidate_count)
        return torch.gather(expanded, 2, gather_indices)

    intro_candidates = gather_code_candidates(intro_tail)
    motif_candidates = gather_code_candidates(motif_head)
    candidate_codes = torch.cat([intro_candidates, motif_candidates], dim=-1)
    flat_candidates = candidate_codes.permute(0, 1, 2, 3).reshape(
        candidate_codes.shape[0],
        candidate_codes.shape[1],
        transition_steps * candidate_codes.shape[-1],
    )
    candidate_latents = tokenizer_model.lookup_codes(flat_candidates).reshape(
        intro_latent.shape[0],
        intro_latent.shape[1],
        transition_steps,
        candidate_codes.shape[-1],
    )

    target_scores = (
        candidate_latents - averaged_target.unsqueeze(-1)
    ).pow(2).mean(dim=1)
    normalized_offsets = offsets.float().abs() / max(1, radius)
    position_penalty = normalized_offsets.pow(2).repeat(2).view(1, 1, -1)
    base_scores = target_scores + max(0.0, float(offset_weight)) * position_penalty

    total_candidates = candidate_codes.shape[-1]
    source_ids = torch.cat(
        [
            torch.zeros(candidate_count, device=device, dtype=torch.long),
            torch.ones(candidate_count, device=device, dtype=torch.long),
        ]
    )
    zero_offset_index = int(torch.argmin(offsets.abs()).item())
    start_state = zero_offset_index
    end_state = candidate_count + zero_offset_index
    infinity = torch.tensor(float("inf"), device=device, dtype=base_scores.dtype)
    path_scores = torch.full(
        (intro_tail.shape[0], total_candidates),
        infinity,
        device=device,
        dtype=base_scores.dtype,
    )
    path_scores[:, start_state] = base_scores[:, 0, start_state]
    backpointers = []

    for step_idx in range(1, transition_steps):
        previous_latents = candidate_latents[..., step_idx - 1, :]
        current_latents = candidate_latents[..., step_idx, :]
        latent_jump = (
            previous_latents.unsqueeze(-1) - current_latents.unsqueeze(-2)
        ).pow(2).mean(dim=1)

        previous_positions = candidate_positions[step_idx - 1].repeat(2)
        current_positions = candidate_positions[step_idx].repeat(2)
        movement = current_positions.unsqueeze(0) - previous_positions.unsqueeze(1)
        same_source = source_ids.unsqueeze(0) == source_ids.unsqueeze(1)
        progression_penalty = torch.where(
            same_source,
            (movement.float() - 1.0).pow(2),
            torch.zeros_like(movement, dtype=base_scores.dtype),
        )
        reverse_source_switch = (source_ids.unsqueeze(1) == 1) & (source_ids.unsqueeze(0) == 0)

        transition_scores = (
            max(0.0, float(continuity_weight)) * latent_jump
            + max(0.0, float(progression_weight)) * progression_penalty.unsqueeze(0)
        )
        transition_scores = transition_scores.masked_fill(reverse_source_switch.unsqueeze(0), infinity)
        scores_from_previous = path_scores.unsqueeze(-1) + transition_scores
        best_previous_scores, best_previous_states = scores_from_previous.min(dim=1)
        path_scores = best_previous_scores + base_scores[:, step_idx, :]
        backpointers.append(best_previous_states)

    selected_states = torch.empty(
        (intro_tail.shape[0], transition_steps),
        device=device,
        dtype=torch.long,
    )
    selected_states[:, -1] = end_state if transition_steps > 1 else start_state
    for step_idx in range(transition_steps - 1, 0, -1):
        selected_states[:, step_idx - 1] = torch.gather(
            backpointers[step_idx - 1],
            1,
            selected_states[:, step_idx].unsqueeze(1),
        ).squeeze(1)

    code_indices = selected_states.view(intro_tail.shape[0], 1, transition_steps, 1).expand(
        -1,
        intro_tail.shape[1],
        -1,
        -1,
    )
    overlap_codes = torch.gather(candidate_codes, 3, code_indices).squeeze(-1)
    selected_sources = selected_states >= candidate_count
    selected_positions = torch.gather(
        candidate_positions.unsqueeze(0).expand(intro_tail.shape[0], -1, -1),
        2,
        (selected_states % candidate_count).unsqueeze(-1),
    ).squeeze(-1)
    motif_positions = torch.nonzero(selected_sources[0], as_tuple=False)
    first_motif_step = int(motif_positions[0].item()) if motif_positions.numel() > 0 else transition_steps
    if not return_diagnostics:
        return overlap_codes.detach().cpu(), first_motif_step

    selected_latents = tokenizer_model.lookup_codes(overlap_codes)
    latent_jumps = (selected_latents[..., 1:] - selected_latents[..., :-1]).pow(2).mean(dim=1)
    source_switches = (selected_sources[:, 1:] != selected_sources[:, :-1]).sum().item()
    max_offset_steps = (selected_positions - aligned_positions[:, 0].unsqueeze(0)).abs().max().item()
    diagnostics = {
        "source_switches": int(source_switches),
        "max_offset_steps": int(max_offset_steps),
        "mean_latent_jump": float(latent_jumps.mean().item()) if latent_jumps.numel() else 0.0,
        "max_latent_jump": float(latent_jumps.max().item()) if latent_jumps.numel() else 0.0,
        "candidates_per_source": int(candidate_count),
        "nonunit_source_steps": int(
            (
                (selected_positions[:, 1:] - selected_positions[:, :-1] != 1)
                & (selected_sources[:, 1:] == selected_sources[:, :-1])
            ).sum().item()
        ),
    }
    return overlap_codes.detach().cpu(), first_motif_step, diagnostics


@torch.no_grad()
def prepare_latent_guided_clip(
    effective_clip_args,
    section_name: str,
    previous_accepted_section: Optional[str],
    accepted_code_clips: List[torch.Tensor],
    section_entries: List[Dict],
    tokenizer_model,
    tokenizer_config,
    prior_config,
    device: torch.device,
):
    del prior_config
    if previous_accepted_section != "intro" or section_name != "body" or not accepted_code_clips:
        return effective_clip_args
    if effective_clip_args.latent_transition_seconds <= 0:
        return effective_clip_args

    effective_clip_args.min_dropout_ac_rms = max(
        0.0,
        float(effective_clip_args.latent_transition_min_ac_rms),
    )
    intro_codes = accepted_code_clips[-1]
    result = retrieve_body_motif(
        intro_codes,
        section_entries,
        tokenizer_model,
        tokenizer_config,
        effective_clip_args,
        device,
    )
    if result is None:
        print("No regular source was long enough for latent-guided intro-to-body token generation")
        return effective_clip_args

    motif_entry, motif_start, motif_codes, similarity = result
    steps_per_second = tokenizer_config.sample_rate / tokenizer_config.total_stride
    valid_overlap_codes, first_motif_step, overlap_diagnostics = build_valid_token_overlap(
        intro_codes,
        motif_codes,
        tokenizer_model,
        device,
        neighborhood_steps=max(
            0,
            int(round(effective_clip_args.latent_overlap_neighborhood_seconds * steps_per_second)),
        ),
        neighborhood_candidates=effective_clip_args.latent_overlap_neighborhood_candidates,
        continuity_weight=effective_clip_args.latent_overlap_continuity_weight,
        offset_weight=effective_clip_args.latent_overlap_offset_weight,
        progression_weight=effective_clip_args.latent_overlap_progression_weight,
        return_diagnostics=True,
    )
    effective_clip_args.latent_guidance_valid_overlap_codes = valid_overlap_codes
    effective_clip_args.latent_transition_first_motif_step = first_motif_step
    effective_seconds = (
        valid_overlap_codes.shape[-1]
        * tokenizer_config.total_stride
        / tokenizer_config.sample_rate
    )
    intro_overlap_seconds = (
        first_motif_step * tokenizer_config.total_stride / tokenizer_config.sample_rate
    )
    print(
        "Prepared valid-pair intro-to-body overlap: "
        f"motif={motif_entry['file']} source_start_step={motif_start} "
        f"intro_match_cosine={similarity:.4f} duration={effective_seconds:.2f}s "
        f"intro_valid_pairs={intro_overlap_seconds:.2f}s "
        f"regular_valid_pairs={effective_seconds - intro_overlap_seconds:.2f}s "
        f"candidates_per_source={overlap_diagnostics['candidates_per_source']} "
        f"max_offset_steps={overlap_diagnostics['max_offset_steps']} "
        f"source_switches={overlap_diagnostics['source_switches']} "
        f"nonunit_source_steps={overlap_diagnostics['nonunit_source_steps']} "
        f"mean_latent_jump={overlap_diagnostics['mean_latent_jump']:.5f} "
        f"max_latent_jump={overlap_diagnostics['max_latent_jump']:.5f}"
    )
    return effective_clip_args


@torch.no_grad()
def decode_contiguous_latent_song(
    accepted_codes,
    accepted_sections,
    tokenizer_model,
    tokenizer_config,
    args,
):
    device = next(tokenizer_model.parameters()).device
    decoded_clips = []
    clip_metadata = getattr(args, "accepted_clip_metadata", [{} for _ in accepted_codes])
    for clip_idx, codes in enumerate(accepted_codes):
        segment_ranges = clip_metadata[clip_idx].get("segment_ranges")
        if segment_ranges:
            stitched = continued.structured.decode_and_stitch_segments(
                codes.to(device),
                segment_ranges,
                tokenizer_model,
                tokenizer_config,
                args.source_overlap,
                args.theme_crossfade_ms,
            )
            expected_samples = max(1, int(round(codes.shape[-1] * tokenizer_config.total_stride)))
            if stitched.shape[-1] != expected_samples:
                stitched = F.interpolate(
                    stitched.unsqueeze(0),
                    size=expected_samples,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)
            decoded_clips.append(stitched.cpu())
        else:
            latents = tokenizer_model.lookup_codes(_batched_codes(codes, tokenizer_model).to(device))
            decoded = tokenizer_model.decoder(tokenizer_model.post_quant(latents))
            decoded_samples = max(1, int(round(latents.shape[-1] * tokenizer_config.total_stride)))
            decoded_clips.append(match_audio_length(decoded, decoded_samples).squeeze(0).cpu())

    output = decoded_clips[0]
    for clip_idx, waveform in enumerate(decoded_clips[1:], start=1):
        previous_section = accepted_sections[clip_idx - 1]
        next_section = accepted_sections[clip_idx]
        previous_generated_section = clip_metadata[clip_idx - 1].get(
            "generated_section",
            previous_section,
        )
        next_generated_section = clip_metadata[clip_idx].get(
            "generated_section",
            next_section,
        )
        section_changed = previous_generated_section == "intro" and next_generated_section == "body"
        if section_changed:
            fade_ms = args.section_crossfade_ms
            first_motif_step = clip_metadata[clip_idx].get("latent_transition_first_motif_step")
            if first_motif_step is not None:
                switch_sample = int(round(first_motif_step * tokenizer_config.total_stride))
                requested_fade_samples = int(
                    round(tokenizer_config.sample_rate * max(0, fade_ms) / 1000.0)
                )
                transition_end = min(
                    waveform.shape[-1],
                    switch_sample + requested_fade_samples // 2,
                )
                transition_start = max(0, transition_end - requested_fade_samples)
                waveform = waveform[..., transition_start:]
                print(
                    "Aligned intro-to-regular crossfade around token handoff: "
                    f"switch={1000.0 * switch_sample / tokenizer_config.sample_rate:.1f}ms "
                    f"body_overlap_start={1000.0 * transition_start / tokenizer_config.sample_rate:.1f}ms"
                )
        else:
            fade_ms = args.clip_crossfade_ms
        output = continued.structured.crossfade_theme_waveforms(
            [output, waveform],
            tokenizer_config.sample_rate,
            max(0, int(fade_ms)),
        )
        boundary_name = "intro-to-regular section" if section_changed else "clip/sequence"
        print(
            f"Applied {boundary_name} crossfade: requested={max(0, int(fade_ms))}ms "
            f"sections={previous_generated_section}->{next_generated_section} "
            f"at accepted clip boundary {clip_idx}/{clip_idx + 1}"
        )
    return output


if __name__ == "__main__":
    continued.main(
        configure_parser=configure_parser,
        decode_song=decode_contiguous_latent_song,
        prepare_clip_args=prepare_latent_guided_clip,
    )
