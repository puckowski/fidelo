import argparse
import math
import os
import random
from typing import List

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from latent_audio_token_pipeline import (
    AudioTextDataset,
    LatentAudioConfig,
    SimpleWordTokenizer,
    TextConditionedLatentPrior,
    latent_bos_token,
    load_audio_tokenizer_bundle,
    load_dataset_items,
    safe_audio_collate,
    save_latent_prior_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditioned latent prior on CUDA.")
    parser.add_argument("--tokenizer-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--audio-dir", default="dataset/audio")
    parser.add_argument("--out-dir", default="latent_audio_prior_out")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for latent prior training. Re-run with --allow-cpu to override.")


def build_code_inputs_targets(indices: torch.Tensor, config: LatentAudioConfig):
    bos = torch.full(
        (indices.shape[0], 1),
        fill_value=latent_bos_token(config),
        dtype=torch.long,
        device=indices.device,
    )
    input_codes = torch.cat([bos, indices[:, :-1]], dim=1)
    target_codes = indices.long()
    return input_codes, target_codes


@torch.no_grad()
def evaluate(audio_tokenizer, prior, loader, device, config):
    prior.eval()
    losses: List[float] = []
    for batch in loader:
        if batch is None:
            continue
        waveform = batch["waveform"].to(device, non_blocking=True)
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        codes = audio_tokenizer.encode_codes(waveform)
        input_codes, target_codes = build_code_inputs_targets(codes, config)
        logits, _ = prior(input_codes, text_tokens, text_mask)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_codes.reshape(-1))
        losses.append(float(loss.item()))
    prior.train()
    return sum(losses) / max(1, len(losses))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.allow_cpu)
    print(f"Latent prior training device: {device}")

    audio_tokenizer, config = load_audio_tokenizer_bundle(args.tokenizer_dir, device)
    audio_tokenizer.eval()
    for param in audio_tokenizer.parameters():
        param.requires_grad = False

    items = load_dataset_items(args.metadata_csv, args.audio_dir)
    if not items:
        raise RuntimeError("No valid audio files found for latent prior training.")
    print(f"Loaded {len(items)} prompt/audio pairs")

    text_tokenizer = SimpleWordTokenizer.build(
        [item["text"] for item in items],
        min_freq=args.min_word_freq,
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Text vocab size: {text_tokenizer.vocab_size}")

    config.metadata_csv = args.metadata_csv
    config.audio_dir = args.audio_dir
    dataset = AudioTextDataset(items, config, text_tokenizer=text_tokenizer, random_crop=args.random_crop)

    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 10 else 1
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=safe_audio_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=safe_audio_collate,
        pin_memory=(device.type == "cuda"),
    )

    prior = TextConditionedLatentPrior(text_tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = math.inf
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        prior.train()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"prior epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            if batch is None:
                continue
            waveform = batch["waveform"].to(device, non_blocking=True)
            text_tokens = batch["text_tokens"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                codes = audio_tokenizer.encode_codes(waveform)
            input_codes, target_codes = build_code_inputs_targets(codes, config)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = prior(input_codes, text_tokens, text_mask)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_codes.reshape(-1))
                loss = loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            should_step = ((steps + 1) % max(1, args.grad_accum_steps) == 0)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * max(1, args.grad_accum_steps)
            steps += 1
            pbar.set_postfix(loss=f"{running_loss / max(1, steps):.4f}")

        if steps % max(1, args.grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_loss = evaluate(audio_tokenizer, prior, val_loader, device, config)
        train_loss = running_loss / max(1, steps)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_latent_prior_bundle(args.out_dir, prior, text_tokenizer, config)
        torch.save(prior.state_dict(), os.path.join(args.out_dir, f"latent_prior_epoch_{epoch + 1:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(prior.state_dict(), os.path.join(args.out_dir, "best_latent_prior.pt"))
            print(f"Saved best latent prior to {os.path.join(args.out_dir, 'best_latent_prior.pt')}")

    print(f"Latent prior artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
