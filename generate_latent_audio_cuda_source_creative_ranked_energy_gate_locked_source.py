import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

import torch

import generate_latent_audio_cuda_source_creative_ranked as base
import generate_latent_audio_cuda_source_creative_ranked_energy_gate as energy_gate


def parse_args():
    return energy_gate.parse_args()


get_device = energy_gate.get_device
make_output_name = energy_gate.make_output_name
prompt_tokens = energy_gate.prompt_tokens
score_prompt_match = energy_gate.score_prompt_match
encode_source_codes = energy_gate.encode_source_codes
build_source_entries = energy_gate.build_source_entries
find_source_window_candidates = energy_gate.find_source_window_candidates
choose_source_window_creatively = energy_gate.choose_source_window_creatively
apply_repetition_penalty = energy_gate.apply_repetition_penalty
filter_logits = energy_gate.filter_logits
sample_rank_relaxed_next_code = energy_gate.sample_rank_relaxed_next_code
generate_rank_relaxed_window = energy_gate.generate_rank_relaxed_window
inject_creative_spans = energy_gate.inject_creative_spans
fuse_source_and_proposal_window = energy_gate.fuse_source_and_proposal_window
measure_window_energy = energy_gate.measure_window_energy
measure_audio_chunk_energies = energy_gate.measure_audio_chunk_energies
clip_has_sufficient_energy = energy_gate.clip_has_sufficient_energy
code_step_count = base.code_step_count
empty_code_sequence = base.empty_code_sequence
extract_code_tail = base.extract_code_tail
ensure_batched_codes = base.ensure_batched_codes
concat_code_sequences = base.concat_code_sequences
slice_code_steps = base.slice_code_steps
select_source_window_with_energy_gate = energy_gate.select_source_window_with_energy_gate


def filter_entries_to_locked_source(candidate_entries: List[Dict], locked_source_path: Optional[str]) -> List[Dict]:
    if not locked_source_path:
        return candidate_entries
    filtered = [entry for entry in candidate_entries if entry["path"] == locked_source_path]
    return filtered or candidate_entries


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
) -> Tuple[torch.Tensor, Optional[Dict]]:
    total_steps = config.latent_steps
    window_size = max(32, min(args.source_window, total_steps))
    overlap_size = max(0, min(args.source_overlap, window_size // 2))
    generated = empty_code_sequence(getattr(prior_model, "num_quantizers", 1))
    rng = energy_gate.random.Random(args.seed)
    first_selected_entry: Optional[Dict] = None

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
            max_candidates=args.window_energy_check_top,
        )
        chosen = select_source_window_with_energy_gate(
            candidates,
            args,
            tokenizer_model,
            device,
            rng,
        )

        if chosen is not None:
            if first_selected_entry is None:
                first_selected_entry = chosen["entry"]
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

    return generated.unsqueeze(0).to(device), first_selected_entry


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

    locked_source_path: Optional[str] = None
    locked_source_file: Optional[str] = None
    clips = []
    for clip_idx in range(clip_count):
        clip_candidate_entries = filter_entries_to_locked_source(candidate_entries, locked_source_path)
        if locked_source_file is not None:
            print(f"Generating clip {clip_idx + 1}/{clip_count} using locked source {locked_source_file} on {device}...")
        else:
            print(f"Generating source-creative-ranked latent clip {clip_idx + 1}/{clip_count} on {device}...")

        accepted_waveform = None
        accepted_source_entry = None
        retry_count = max(1, int(args.clip_retry_count))
        candidate_waveform = None
        candidate_source_entry = None

        for retry_idx in range(retry_count):
            clip_args = argparse.Namespace(**vars(args))
            clip_args.seed = args.seed + clip_idx * 1009 + retry_idx * 7919
            codes, chosen_source_entry = generate_source_creative_ranked_codes(
                clip_args,
                prior_model,
                tokenizer_model,
                text_tokens,
                text_mask,
                prior_config,
                clip_candidate_entries,
                device,
            )
            codes = codes.to(device=device, dtype=torch.long)
            waveform = tokenizer_model.decode_codes(codes, target_length=tokenizer_config.clip_samples)
            candidate_waveform = waveform.squeeze(0).cpu()
            candidate_source_entry = chosen_source_entry

            if clip_has_sufficient_energy(candidate_waveform, tokenizer_config.sample_rate, args):
                accepted_waveform = candidate_waveform
                accepted_source_entry = chosen_source_entry
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
            accepted_source_entry = candidate_source_entry
            print(f"Using last attempt for clip {clip_idx + 1} after {retry_count} failed energy checks")

        if locked_source_path is None and accepted_source_entry is not None:
            locked_source_path = accepted_source_entry["path"]
            locked_source_file = accepted_source_entry["file"]
            print(f"Locked subsequent clips to source {locked_source_file}")

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
    print(f"Saved source-creative-ranked latent audio to {output_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()