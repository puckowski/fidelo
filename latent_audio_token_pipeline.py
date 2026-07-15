import csv
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, default_collate


WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class LatentAudioConfig:
    sample_rate: int = 44100
    clip_seconds: float = 5.0
    encoder_channels: Sequence[int] = field(default_factory=lambda: [128, 256, 512])
    encoder_strides: Sequence[int] = field(default_factory=lambda: [4, 4, 2])
    code_dim: int = 384
    codebook_size: int = 4096
    num_quantizers: int = 1
    commitment_cost: float = 0.1
    residual_layers_per_stage: int = 3
    bottleneck_layers: int = 4
    quantizer_pre_layers: int = 2
    quantizer_post_layers: int = 3
    max_text_tokens: int = 40
    text_embed_dim: int = 256
    prior_hidden_size: int = 768
    prior_num_layers: int = 3
    prior_dropout: float = 0.15
    metadata_csv: str = "dataset/metadata.csv"
    audio_dir: str = "dataset/audio"

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def total_stride(self) -> int:
        stride = 1
        for value in self.encoder_strides:
            stride *= int(value)
        return stride

    @property
    def latent_steps(self) -> int:
        return math.ceil(self.clip_samples / self.total_stride)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["encoder_channels"] = list(self.encoder_channels)
        payload["encoder_strides"] = list(self.encoder_strides)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict) -> "LatentAudioConfig":
        payload = dict(payload)
        payload["encoder_channels"] = list(payload.get("encoder_channels", [128, 256, 512]))
        payload["encoder_strides"] = list(payload.get("encoder_strides", [4, 4, 2]))
        payload["num_quantizers"] = max(1, int(payload.get("num_quantizers", 1)))
        valid_keys = {item.name for item in fields(cls)}
        filtered_payload = {key: value for key, value in payload.items() if key in valid_keys}
        return cls(**filtered_payload)


def make_norm(num_channels: int) -> nn.Module:
    for groups in (32, 16, 8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            make_norm(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            make_norm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class SimpleWordTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {index: token for token, index in stoi.items()}
        self.pad_id = stoi[self.PAD]
        self.unk_id = stoi[self.UNK]

    @classmethod
    def build(cls, texts: List[str], min_freq: int = 1, max_vocab_size: int = 20000) -> "SimpleWordTokenizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in WORD_RE.findall(text.lower()):
                counts[token] = counts.get(token, 0) + 1

        vocab = [cls.PAD, cls.UNK]
        for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if freq < min_freq:
                continue
            vocab.append(token)
            if len(vocab) >= max_vocab_size:
                break
        return cls({token: idx for idx, token in enumerate(vocab)})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.stoi.get(token, self.unk_id) for token in WORD_RE.findall(text.lower())][:max_length]
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids != self.pad_id).long()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleWordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["stoi"])


def load_dataset_items(metadata_csv: str, audio_dir: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file", "")
            text = row.get("text", "")
            path = os.path.join(audio_dir, file_name)
            if os.path.isfile(path):
                items.append({"file": file_name, "text": text, "path": path})
    return items


def _resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_length = max(1, int(round(waveform.shape[-1] * float(target_rate) / float(source_rate))))
    return F.interpolate(
        waveform.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def save_audio_waveform(path: str, waveform: torch.Tensor, sample_rate: int):
    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    try:
        import soundfile as sf

        sf.write(path, waveform.transpose(0, 1).numpy(), sample_rate)
        return
    except Exception:
        pass

    try:
        import torchaudio

        torchaudio.save(path, waveform, sample_rate)
        return
    except Exception as exc:
        raise RuntimeError(f"Could not save audio with soundfile or torchaudio: {exc}") from exc


def load_audio_mono(path: str, sample_rate: int) -> torch.Tensor:
    try:
        import soundfile as sf

        waveform_np, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np).transpose(0, 1)
    except Exception:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load audio with soundfile or torchaudio: {exc}") from exc

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = _resample_waveform(waveform, sr, sample_rate)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")
    return waveform.clamp(-1.0, 1.0)


def crop_or_pad(waveform: torch.Tensor, clip_samples: int, random_crop: bool = True) -> torch.Tensor:
    total = waveform.shape[-1]
    if total > clip_samples:
        start = random.randint(0, total - clip_samples) if random_crop else max(0, (total - clip_samples) // 2)
        return waveform[:, start:start + clip_samples]
    if total < clip_samples:
        return F.pad(waveform, (0, clip_samples - total))
    return waveform


def match_audio_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = waveform.shape[-1]
    if current_length > target_length:
        return waveform[..., :target_length]
    if current_length < target_length:
        return F.pad(waveform, (0, target_length - current_length))
    return waveform


class AudioTextDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, str]],
        config: LatentAudioConfig,
        text_tokenizer: Optional[SimpleWordTokenizer] = None,
        random_crop: bool = True,
    ):
        self.items = items
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.random_crop = random_crop
        self._warned_paths = set()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        attempts = min(8, max(1, len(self.items)))
        current_index = index
        for _ in range(attempts):
            item = self.items[current_index]
            try:
                waveform = load_audio_mono(item["path"], self.config.sample_rate)
                waveform = crop_or_pad(waveform, self.config.clip_samples, random_crop=self.random_crop)
                output = {
                    "waveform": waveform.float(),
                    "path": item["path"],
                    "text": item["text"],
                }
                if self.text_tokenizer is not None:
                    text_tokens = self.text_tokenizer.encode(item["text"], self.config.max_text_tokens)
                    output["text_tokens"] = text_tokens.long()
                    output["text_mask"] = self.text_tokenizer.attention_mask(text_tokens).long()
                return output
            except Exception as exc:
                if item["path"] not in self._warned_paths:
                    self._warned_paths.add(item["path"])
                    print(f"Skipping invalid audio file: {item['path']} ({exc})")
                current_index = random.randrange(len(self.items))
        return None


def safe_audio_collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    collated = default_collate([{k: v for k, v in item.items() if k not in {"path", "text"}} for item in valid])
    collated["path"] = [item["path"] for item in valid]
    collated["text"] = [item["text"] for item in valid]
    return collated


class ConvEncoder(nn.Module):
    def __init__(self, config: LatentAudioConfig):
        super().__init__()
        channels = [1, *config.encoder_channels]
        layers = []
        for idx, stride in enumerate(config.encoder_strides):
            in_ch = channels[idx]
            out_ch = channels[idx + 1]
            kernel_size = stride * 2
            padding = stride // 2
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                    make_norm(out_ch),
                    nn.GELU(),
                ]
            )
            for block_idx in range(config.residual_layers_per_stage):
                layers.append(ResidualConvBlock(out_ch, dilation=(2 ** block_idx)))
        layers.append(nn.Conv1d(channels[-1], config.code_dim, kernel_size=3, padding=1))
        for block_idx in range(config.bottleneck_layers):
            layers.append(ResidualConvBlock(config.code_dim, dilation=(2 ** block_idx)))
        self.net = nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.net(waveform)


class ConvDecoder(nn.Module):
    def __init__(self, config: LatentAudioConfig):
        super().__init__()
        reversed_channels = [config.code_dim, *reversed(config.encoder_channels)]
        layers = []
        for block_idx in range(config.bottleneck_layers):
            layers.append(ResidualConvBlock(config.code_dim, dilation=(2 ** block_idx)))
        reversed_strides = list(reversed(config.encoder_strides))
        for idx, stride in enumerate(reversed_strides):
            in_ch = reversed_channels[idx]
            out_ch = reversed_channels[idx + 1]
            kernel_size = stride * 2
            padding = stride // 2
            output_padding = max(0, stride % 2)
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    ),
                    make_norm(out_ch),
                    nn.GELU(),
                ]
            )
            for block_idx in range(config.residual_layers_per_stage):
                layers.append(ResidualConvBlock(out_ch, dilation=(2 ** block_idx)))
        layers.append(nn.Conv1d(reversed_channels[-1], 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(latents))


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # latents: [B, C, T]
        with torch.amp.autocast(latents.device.type, enabled=False):
            latents_float = latents.float()
            flat = latents_float.permute(0, 2, 1).reshape(-1, self.code_dim)
            codebook = self.codebook.weight.float()
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ codebook.t()
                + codebook.pow(2).sum(dim=1)
            )
            indices = torch.argmin(distances, dim=1)
            quantized_float = F.embedding(indices, codebook).view(
                latents.shape[0], latents.shape[2], self.code_dim
            ).permute(0, 2, 1)

            codebook_loss = F.mse_loss(quantized_float, latents_float.detach())
            commitment_loss = F.mse_loss(quantized_float.detach(), latents_float)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        indices = indices.view(latents.shape[0], latents.shape[2])
        quantized = quantized_float.to(latents.dtype)
        return quantized, loss, indices

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, loss, indices = self.quantize(latents)
        quantized = latents + (quantized - latents).detach()
        return quantized, loss, indices

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.codebook(indices)
        return quantized.permute(0, 2, 1)


class VQAudioAutoencoder(nn.Module):
    def __init__(self, config: LatentAudioConfig):
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(config)
        pre_quant_layers = []
        for block_idx in range(config.quantizer_pre_layers):
            pre_quant_layers.append(ResidualConvBlock(config.code_dim, dilation=(2 ** block_idx)))
        self.pre_quant = nn.Sequential(*pre_quant_layers) if pre_quant_layers else nn.Identity()
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(config.codebook_size, config.code_dim, config.commitment_cost)
                for _ in range(max(1, config.num_quantizers))
            ]
        )
        post_quant_layers = []
        for block_idx in range(config.quantizer_post_layers):
            post_quant_layers.append(ResidualConvBlock(config.code_dim, dilation=(2 ** block_idx)))
        self.post_quant = nn.Sequential(*post_quant_layers) if post_quant_layers else nn.Identity()
        self.decoder = ConvDecoder(config)

    @property
    def quantizer(self) -> VectorQuantizer:
        return self.quantizers[0]

    def quantize_latents(self, latents: torch.Tensor, return_all_codes: bool = False):
        residual = latents
        quantized_total = torch.zeros_like(latents)
        total_vq_loss = latents.new_tensor(0.0)
        indices_per_quantizer: List[torch.Tensor] = []

        for quantizer in self.quantizers:
            quantized_raw, vq_loss, indices = quantizer.quantize(residual)
            quantized_total = quantized_total + quantized_raw
            total_vq_loss = total_vq_loss + vq_loss
            indices_per_quantizer.append(indices)
            residual = residual - quantized_raw

        quantized = latents + (quantized_total - latents).detach()
        indices = torch.stack(indices_per_quantizer, dim=1) if return_all_codes else indices_per_quantizer[0]
        total_vq_loss = total_vq_loss / max(1, len(self.quantizers))
        return quantized, total_vq_loss, indices

    def lookup_codes(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            return self.quantizer.lookup(indices)
        if indices.dim() != 3:
            raise ValueError(f"Expected code indices with shape [B, T] or [B, Q, T], got {tuple(indices.shape)}")
        if indices.shape[1] > len(self.quantizers):
            raise ValueError(
                f"Received {indices.shape[1]} code streams, but tokenizer only has {len(self.quantizers)} quantizers"
            )

        quantized = self.quantizers[0].lookup(indices[:, 0, :])
        for quantizer_idx in range(1, indices.shape[1]):
            quantized = quantized + self.quantizers[quantizer_idx].lookup(indices[:, quantizer_idx, :])
        return quantized

    def forward(self, waveform: torch.Tensor):
        latents = self.pre_quant(self.encoder(waveform))
        quantized, vq_loss, indices = self.quantize_latents(latents)
        recon = self.decoder(self.post_quant(quantized))
        recon = match_audio_length(recon, waveform.shape[-1])
        return recon, vq_loss, indices

    @torch.no_grad()
    def encode_codes(self, waveform: torch.Tensor, return_all_codes: bool = False) -> torch.Tensor:
        latents = self.pre_quant(self.encoder(waveform))
        _, _, indices = self.quantize_latents(latents, return_all_codes=return_all_codes)
        return indices

    @torch.no_grad()
    def decode_codes(self, indices: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        quantized = self.post_quant(self.lookup_codes(indices))
        waveform = self.decoder(quantized)
        if target_length is not None:
            waveform = match_audio_length(waveform, target_length)
        return waveform.clamp(-1.0, 1.0)


def latent_bos_token(config: LatentAudioConfig) -> int:
    return config.codebook_size


class TextConditionedLatentPrior(nn.Module):
    def __init__(self, text_vocab_size: int, config: LatentAudioConfig):
        super().__init__()
        self.config = config
        self.num_quantizers = max(1, int(config.num_quantizers))
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(config.codebook_size + 1, config.code_dim) for _ in range(self.num_quantizers)]
        )
        self.text_embedding = nn.Embedding(text_vocab_size, config.text_embed_dim)
        self.text_proj = nn.Linear(config.text_embed_dim, config.text_embed_dim)
        self.rnn = nn.GRU(
            input_size=config.code_dim + config.text_embed_dim,
            hidden_size=config.prior_hidden_size,
            num_layers=config.prior_num_layers,
            dropout=config.prior_dropout if config.prior_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_heads = nn.ModuleList(
            [nn.Linear(config.prior_hidden_size, config.codebook_size) for _ in range(self.num_quantizers)]
        )

    @property
    def code_embedding(self) -> nn.Embedding:
        return self.code_embeddings[0]

    @property
    def output_head(self) -> nn.Linear:
        return self.output_heads[0]

    def _normalize_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 2:
            if self.num_quantizers != 1:
                raise ValueError(
                    f"Expected multi-stream codes with shape [B, Q, T] for {self.num_quantizers} quantizers, "
                    f"got {tuple(codes.shape)}"
                )
            return codes.unsqueeze(1)
        if codes.dim() != 3:
            raise ValueError(f"Expected codes with shape [B, T] or [B, Q, T], got {tuple(codes.shape)}")
        if codes.shape[1] != self.num_quantizers:
            raise ValueError(
                f"Expected {self.num_quantizers} quantizer streams, got {codes.shape[1]}"
            )
        return codes

    def _denormalize_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.num_quantizers == 1:
            return codes[:, 0, :]
        return codes

    def _embed_code_sequence(self, codes: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_codes(codes)
        stream_embeddings = []
        for quantizer_idx, embedding in enumerate(self.code_embeddings):
            stream_embeddings.append(embedding(normalized[:, quantizer_idx, :]))
        return torch.stack(stream_embeddings, dim=0).sum(dim=0) / math.sqrt(len(stream_embeddings))

    def _logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([head(hidden_states) for head in self.output_heads], dim=1)
        return logits

    def bos_codes(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, self.num_quantizers),
            fill_value=latent_bos_token(self.config),
            dtype=torch.long,
            device=device,
        )

    def forward_step(self, current_codes: torch.Tensor, text_cond: torch.Tensor, hidden=None):
        current_codes = current_codes.to(dtype=torch.long)
        if current_codes.dim() == 1:
            current_codes = current_codes.unsqueeze(1)
        if current_codes.dim() == 2:
            current_codes = current_codes.unsqueeze(-1)
        code_emb = self._embed_code_sequence(current_codes)
        cond = text_cond.unsqueeze(1)
        x = torch.cat([code_emb, cond], dim=-1)
        out, hidden = self.rnn(x, hidden)
        logits = self._logits_from_hidden(out[:, -1:, :]).squeeze(2)
        return logits, hidden

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        history: List[torch.Tensor],
        repetition_penalty: float,
        repetition_window: int,
    ) -> torch.Tensor:
        if repetition_penalty <= 1.0 or not history:
            return logits
        recent = torch.stack(history[-max(1, repetition_window):], dim=-1)
        adjusted_logits = logits.clone()
        for batch_idx in range(recent.shape[0]):
            for quantizer_idx in range(recent.shape[1]):
                unique_tokens = torch.unique(recent[batch_idx, quantizer_idx])
                token_logits = adjusted_logits[batch_idx, quantizer_idx, unique_tokens]
                adjusted = torch.where(
                    token_logits >= 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty,
                )
                adjusted_logits[batch_idx, quantizer_idx, unique_tokens] = adjusted
        return adjusted_logits

    def _sample_next_codes(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits / max(temperature, 1e-5)
        next_codes = []
        for batch_idx in range(scaled_logits.shape[0]):
            stream_codes = []
            for quantizer_idx in range(scaled_logits.shape[1]):
                stream_logits = scaled_logits[batch_idx, quantizer_idx]
                if top_k is not None and 0 < top_k < stream_logits.shape[-1]:
                    values, indices = torch.topk(stream_logits, k=top_k, dim=-1)
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_probs = torch.softmax(values, dim=-1)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        keep_mask = cumulative <= top_p
                        keep_mask[..., 0] = True
                        values = values.masked_fill(~keep_mask, float("-inf"))
                    probs = torch.softmax(values, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    stream_code = indices.gather(-1, sampled).squeeze(-1)
                else:
                    filtered_logits = stream_logits
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        keep_mask = cumulative <= top_p
                        keep_mask[..., 0] = True
                        sorted_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
                        filtered_logits = torch.full_like(filtered_logits, float("-inf"))
                        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                    probs = torch.softmax(filtered_logits, dim=-1)
                    stream_code = torch.multinomial(probs, num_samples=1).squeeze(-1)
                stream_codes.append(stream_code)
            next_codes.append(torch.stack(stream_codes, dim=0))
        return torch.stack(next_codes, dim=0)

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_embedding(text_tokens)
        mask = text_mask.unsqueeze(-1).float()
        pooled = (text_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.tanh(self.text_proj(pooled))

    def forward(self, input_codes: torch.Tensor, text_tokens: torch.Tensor, text_mask: torch.Tensor, hidden=None):
        normalized_codes = self._normalize_codes(input_codes)
        code_emb = self._embed_code_sequence(normalized_codes)
        text_cond = self.encode_text(text_tokens, text_mask)
        cond = text_cond.unsqueeze(1).expand(-1, normalized_codes.shape[-1], -1)
        x = torch.cat([code_emb, cond], dim=-1)
        out, hidden = self.rnn(x, hidden)
        logits = self._logits_from_hidden(out)
        if self.num_quantizers == 1:
            logits = logits[:, 0, :, :]
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
        top_k: int = 64,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 128,
        prefix_codes: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        self.eval()
        device = device or next(self.parameters()).device
        text_tokens = text_tokens.to(device)
        text_mask = text_mask.to(device)
        hidden = None
        text_cond = self.encode_text(text_tokens, text_mask)
        current = self.bos_codes(text_tokens.shape[0], device)
        outputs: List[torch.Tensor] = []
        history: List[torch.Tensor] = []

        if prefix_codes is not None and prefix_codes.numel() > 0:
            prefix_codes = self._normalize_codes(prefix_codes.to(device=device, dtype=torch.long))
            for step_idx in range(prefix_codes.shape[-1]):
                _, hidden = self.forward_step(current, text_cond, hidden)
                current = prefix_codes[:, :, step_idx]
                history.append(current)

        for _ in range(num_steps):
            logits, hidden = self.forward_step(current, text_cond, hidden)
            logits = self._apply_repetition_penalty(logits, history, repetition_penalty, repetition_window)
            next_code = self._sample_next_codes(logits, temperature, top_k, top_p)
            outputs.append(next_code)
            history.append(next_code)
            current = next_code
        stacked = torch.stack(outputs, dim=-1)
        return self._denormalize_codes(stacked)


def stitch_waveforms(chunks: List[torch.Tensor], sample_rate: int, fade_ms: int = 40) -> torch.Tensor:
    if not chunks:
        raise ValueError("No chunks provided for stitching")
    fade_samples = int(sample_rate * fade_ms / 1000)
    out = chunks[0].clone()
    for nxt in chunks[1:]:
        effective = min(fade_samples, out.shape[-1] // 2, nxt.shape[-1] // 2)
        if effective <= 0:
            out = torch.cat([out, nxt], dim=-1)
            continue
        fade_shape = [1] * out.dim()
        fade_shape[-1] = effective
        fade_out = torch.linspace(1.0, 0.0, effective, device=out.device).view(*fade_shape)
        fade_in = torch.linspace(0.0, 1.0, effective, device=out.device).view(*fade_shape)
        mixed = out[..., -effective:] * fade_out + nxt[..., :effective] * fade_in
        out = torch.cat([out[..., :-effective], mixed, nxt[..., effective:]], dim=-1)
    return out


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _remap_tokenizer_state_key(key: str) -> str:
    if key.startswith("quantizer."):
        return key.replace("quantizer.", "quantizers.0.", 1)
    if key.startswith("residual_quantizer."):
        return key.replace("residual_quantizer.", "quantizers.1.", 1)
    return key


def _remap_prior_state_key(key: str) -> str:
    if key.startswith("code_embedding."):
        return key.replace("code_embedding.", "code_embeddings.0.", 1)
    if key.startswith("output_head."):
        return key.replace("output_head.", "output_heads.0.", 1)
    return key


def _is_optional_quantizer_key(key: str, num_quantizers: int) -> bool:
    if not key.startswith("quantizers."):
        return False
    parts = key.split(".")
    if len(parts) < 3:
        return False
    try:
        quantizer_idx = int(parts[1])
    except ValueError:
        return False
    return quantizer_idx >= num_quantizers or quantizer_idx > 0


def _is_optional_prior_key(key: str, num_quantizers: int) -> bool:
    if key.startswith("code_embeddings.") or key.startswith("output_heads."):
        parts = key.split(".")
        if len(parts) < 3:
            return False
        try:
            quantizer_idx = int(parts[1])
        except ValueError:
            return False
        return quantizer_idx > 0 or quantizer_idx >= num_quantizers
    return False


def load_tokenizer_state_dict(model: VQAudioAutoencoder, state: Dict[str, torch.Tensor]):
    ignored_prefixes = (
        "residual_correction.",
        "post_quant_refine.",
        "residual_proj.",
        "residual_mix",
    )
    filtered_state: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()

    for key, value in state.items():
        if key.startswith(ignored_prefixes):
            continue
        remapped_key = _remap_tokenizer_state_key(key)
        if remapped_key in model_state:
            filtered_state[remapped_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    disallowed_missing = [
        key for key in missing_keys if not _is_optional_quantizer_key(key, num_quantizers=1)
    ]
    disallowed_unexpected = [
        key for key in unexpected_keys if not key.startswith(ignored_prefixes)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Tokenizer checkpoint is incompatible. "
            f"Missing keys: {disallowed_missing}; unexpected keys: {disallowed_unexpected}"
        )
    return missing_keys, unexpected_keys


def load_latent_prior_state_dict(model: TextConditionedLatentPrior, state: Dict[str, torch.Tensor]):
    filtered_state: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()

    for key, value in state.items():
        remapped_key = _remap_prior_state_key(key)
        if remapped_key in model_state:
            filtered_state[remapped_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    disallowed_missing = [
        key for key in missing_keys if not _is_optional_prior_key(key, num_quantizers=model.num_quantizers)
    ]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            "Latent prior checkpoint is incompatible. "
            f"Missing keys: {disallowed_missing}; unexpected keys: {unexpected_keys}"
        )
    return missing_keys, unexpected_keys


def save_audio_tokenizer_bundle(out_dir: str, model: VQAudioAutoencoder, config: LatentAudioConfig):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "audio_tokenizer.pt"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_audio_tokenizer_bundle(model_dir: str, device: torch.device) -> Tuple[VQAudioAutoencoder, LatentAudioConfig]:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = LatentAudioConfig.from_dict(json.load(f))
    model = VQAudioAutoencoder(config)
    best_path = os.path.join(model_dir, "best_audio_tokenizer.pt")
    fallback_path = os.path.join(model_dir, "audio_tokenizer.pt")
    state = safe_torch_load(best_path if os.path.isfile(best_path) else fallback_path, device)
    load_tokenizer_state_dict(model, state)
    model.to(device).eval()
    return model, config


def save_latent_prior_bundle(out_dir: str, model: TextConditionedLatentPrior, text_tokenizer: SimpleWordTokenizer, config: LatentAudioConfig):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "latent_prior.pt"))
    text_tokenizer.save(os.path.join(out_dir, "text_tokenizer.json"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_latent_prior_bundle(model_dir: str, device: torch.device) -> Tuple[TextConditionedLatentPrior, SimpleWordTokenizer, LatentAudioConfig]:
    text_tokenizer = SimpleWordTokenizer.load(os.path.join(model_dir, "text_tokenizer.json"))
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = LatentAudioConfig.from_dict(json.load(f))
    model = TextConditionedLatentPrior(text_tokenizer.vocab_size, config)
    best_path = os.path.join(model_dir, "best_latent_prior.pt")
    fallback_path = os.path.join(model_dir, "latent_prior.pt")
    state = safe_torch_load(best_path if os.path.isfile(best_path) else fallback_path, device)
    load_latent_prior_state_dict(model, state)
    model.to(device).eval()
    return model, text_tokenizer, config
