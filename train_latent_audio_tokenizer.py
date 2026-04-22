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
    VQAudioAutoencoder,
    load_dataset_items,
    safe_audio_collate,
    save_audio_tokenizer_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learned audio tokenizer on CUDA.")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--audio-dir", default="dataset/audio")
    parser.add_argument("--out-dir", default="latent_audio_tokenizer_out")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--code-dim", type=int, default=384)
    parser.add_argument("--commitment-cost", type=float, default=0.1)
    parser.add_argument("--encoder-channels", type=int, nargs="*", default=[128, 256, 512])
    parser.add_argument("--encoder-strides", type=int, nargs="*", default=[4, 4, 2])
    parser.add_argument("--residual-layers-per-stage", type=int, default=2)
    parser.add_argument("--bottleneck-layers", type=int, default=4)
    parser.add_argument("--quantizer-pre-layers", type=int, default=2)
    parser.add_argument("--quantizer-post-layers", type=int, default=2)
    parser.add_argument("--max-text-tokens", type=int, default=40)
    parser.add_argument("--text-embed-dim", type=int, default=256)
    parser.add_argument("--prior-hidden-size", type=int, default=768)
    parser.add_argument("--prior-num-layers", type=int, default=3)
    parser.add_argument("--prior-dropout", type=float, default=0.15)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--stft-weight", type=float, default=0.35)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def multi_resolution_stft_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    resolutions = [
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
    ]
    total = reconstructed.new_tensor(0.0)
    rec = reconstructed.squeeze(1)
    tgt = target.squeeze(1)

    for n_fft, hop_length, win_length in resolutions:
        window = torch.hann_window(win_length, device=reconstructed.device)
        rec_spec = torch.stft(
            rec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        tgt_spec = torch.stft(
            tgt,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        rec_mag = rec_spec.abs().clamp_min(1e-5)
        tgt_mag = tgt_spec.abs().clamp_min(1e-5)
        mag_loss = torch.mean(torch.abs(rec_mag - tgt_mag))
        log_mag_loss = torch.mean(torch.abs(torch.log(rec_mag) - torch.log(tgt_mag)))
        total = total + mag_loss + log_mag_loss

    return total / len(resolutions)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required for this tokenizer training script. Re-run with --allow-cpu to override.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    recon_losses: List[float] = []
    vq_losses: List[float] = []
    stft_losses: List[float] = []
    for batch in loader:
        if batch is None:
            continue
        waveform = batch["waveform"].to(device, non_blocking=True)
        recon, vq_loss, _ = model(waveform)
        recon_loss = torch.mean(torch.abs(recon - waveform))
        stft_loss = multi_resolution_stft_loss(recon, waveform)
        recon_losses.append(float(recon_loss.item()))
        vq_losses.append(float(vq_loss.item()))
        stft_losses.append(float(stft_loss.item()))
    model.train()
    return (
        sum(recon_losses) / max(1, len(recon_losses)),
        sum(vq_losses) / max(1, len(vq_losses)),
        sum(stft_losses) / max(1, len(stft_losses)),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.allow_cpu)
    print(f"Tokenizer training device: {device}")

    config = LatentAudioConfig(
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        commitment_cost=args.commitment_cost,
        residual_layers_per_stage=args.residual_layers_per_stage,
        bottleneck_layers=args.bottleneck_layers,
        quantizer_pre_layers=args.quantizer_pre_layers,
        quantizer_post_layers=args.quantizer_post_layers,
        max_text_tokens=args.max_text_tokens,
        text_embed_dim=args.text_embed_dim,
        prior_hidden_size=args.prior_hidden_size,
        prior_num_layers=args.prior_num_layers,
        prior_dropout=args.prior_dropout,
        encoder_channels=list(args.encoder_channels),
        encoder_strides=list(args.encoder_strides),
        metadata_csv=args.metadata_csv,
        audio_dir=args.audio_dir,
    )

    items = load_dataset_items(args.metadata_csv, args.audio_dir)
    if not items:
        raise RuntimeError("No valid audio files found for tokenizer training.")
    print(f"Loaded {len(items)} audio files")

    dataset = AudioTextDataset(items, config, text_tokenizer=None, random_crop=args.random_crop)
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

    model = VQAudioAutoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = math.inf
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_recon = 0.0
        running_vq = 0.0
        running_stft = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"tokenizer epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            if batch is None:
                continue
            waveform = batch["waveform"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                recon, vq_loss, _ = model(waveform)
                recon_loss = torch.mean(torch.abs(recon - waveform))
                stft_loss = multi_resolution_stft_loss(recon, waveform)
                loss = (
                    (args.recon_weight * recon_loss)
                    + (args.vq_weight * vq_loss)
                    + (args.stft_weight * stft_loss)
                )
                loss = loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            should_step = ((steps + 1) % max(1, args.grad_accum_steps) == 0)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_recon += float(recon_loss.item())
            running_vq += float(vq_loss.item())
            running_stft += float(stft_loss.item())
            steps += 1
            pbar.set_postfix(
                recon=f"{running_recon / max(1, steps):.4f}",
                stft=f"{running_stft / max(1, steps):.4f}",
                vq=f"{running_vq / max(1, steps):.4f}",
            )

        if steps % max(1, args.grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_recon, val_vq, val_stft = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1}: train_recon={running_recon / max(1, steps):.4f} "
            f"train_stft={running_stft / max(1, steps):.4f} train_vq={running_vq / max(1, steps):.4f} "
            f"val_recon={val_recon:.4f} val_stft={val_stft:.4f} val_vq={val_vq:.4f}"
        )

        save_audio_tokenizer_bundle(args.out_dir, model, config)
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"tokenizer_epoch_{epoch + 1:03d}.pt"))

        score = (args.recon_weight * val_recon) + (args.vq_weight * val_vq) + (args.stft_weight * val_stft)
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_audio_tokenizer.pt"))
            print(f"Saved best tokenizer to {os.path.join(args.out_dir, 'best_audio_tokenizer.pt')}")

    print(f"Tokenizer artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
