import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
DEFAULT_AUDIO_DIR = os.path.join("dataset", "audio")
TARGET_SR = 24000
MAX_COMPARE_SECONDS = 20.0
HASH_BITS = 64
TOP_K_RESULTS = 5
PROJECTION_SEED = 12345


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuzzy-hash dataset audio files and print the nearest matches to an input audio file."
    )
    parser.add_argument("input_audio", help="Input filename or path to compare against the dataset audio files.")
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="Directory containing dataset audio files.")
    parser.add_argument("--top-k", type=int, default=TOP_K_RESULTS, help="Number of nearest matches to print.")
    return parser.parse_args()


def list_audio_files(audio_dir: str) -> List[Path]:
    base = Path(audio_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    files = []
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def resolve_audio_path(input_audio: str, audio_dir: str) -> Path:
    direct = Path(input_audio)
    if direct.is_file():
        return direct
    candidate = Path(audio_dir) / input_audio
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Input audio not found: {input_audio}")


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


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
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
    waveform = waveform.float().clamp(-1.0, 1.0)
    if sr != target_sr:
        waveform = _resample_waveform(waveform, sr, target_sr)
    max_samples = int(target_sr * MAX_COMPARE_SECONDS)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]
    return waveform


def extract_feature(path: str) -> torch.Tensor:
    waveform = load_audio_mono(path, TARGET_SR)
    audio = waveform.squeeze(0)
    if audio.numel() < 1024:
        audio = F.pad(audio, (0, max(0, 1024 - audio.numel())))

    window = torch.hann_window(1024)
    spec = torch.stft(
        audio,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-5)
    log_mag = torch.log(mag)

    band_mean = log_mag.mean(dim=1)
    band_std = log_mag.std(dim=1)
    frame_energy = log_mag.mean(dim=0)
    rms = torch.sqrt(torch.mean(audio.pow(2))).unsqueeze(0)
    zcr = ((audio[1:] * audio[:-1]) < 0).float().mean().unsqueeze(0)
    dyn = (audio.abs().quantile(0.95) - audio.abs().quantile(0.50)).unsqueeze(0)

    feature = torch.cat(
        [
            band_mean,
            band_std,
            frame_energy.mean().unsqueeze(0),
            frame_energy.std().unsqueeze(0),
            rms,
            zcr,
            dyn,
        ]
    )
    return F.normalize(feature, dim=0)


def build_projection_matrix(feature_dim: int, bits: int = HASH_BITS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(PROJECTION_SEED)
    projection = torch.randn(bits, feature_dim, generator=generator)
    return F.normalize(projection, dim=1)


def simhash(feature: torch.Tensor, projection: torch.Tensor) -> int:
    signs = torch.mv(projection, feature) >= 0
    value = 0
    for bit in signs.tolist():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


def format_hash(value: int, bits: int = HASH_BITS) -> str:
    hex_width = bits // 4
    return f"0x{value:0{hex_width}x}"


def main():
    args = parse_args()
    dataset_files = list_audio_files(args.audio_dir)
    if not dataset_files:
        raise RuntimeError("No dataset audio files found.")

    input_path = resolve_audio_path(args.input_audio, args.audio_dir)
    input_feature = extract_feature(str(input_path))
    projection = build_projection_matrix(input_feature.shape[0], HASH_BITS)
    input_hash = simhash(input_feature, projection)

    ranked: List[Tuple[int, float, Path, int]] = []
    for audio_path in dataset_files:
        try:
            if audio_path.resolve() == input_path.resolve():
                continue
            feature = extract_feature(str(audio_path))
            hashed = simhash(feature, projection)
            distance = hamming_distance(input_hash, hashed)
            similarity = cosine_similarity(input_feature, feature)
            ranked.append((distance, similarity, audio_path, hashed))
        except Exception as exc:
            print(f"Skipping unreadable candidate {audio_path.name} ({exc})")

    ranked.sort(key=lambda item: (item[0], -item[1], item[2].name.lower()))

    print(f"Input audio: {input_path}")
    print(f"Input fuzzy hash: {format_hash(input_hash)}")
    print(f"Dataset files checked: {len(dataset_files) - 1}")
    print()
    print(f"Top {min(args.top_k, len(ranked))} nearest matches:")
    for index, (distance, similarity, path, hashed) in enumerate(ranked[:args.top_k], start=1):
        print(
            f"{index:>2}. {path.name} | hamming={distance:>2} | cosine={similarity:.4f} | hash={format_hash(hashed)}"
        )


if __name__ == "__main__":
    main()
