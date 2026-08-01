import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


WORD_RE = re.compile(r"[a-z0-9']+")
DEFAULT_METADATA = os.path.join("dataset", "metadata.csv")
DEFAULT_AUDIO_DIR = os.path.join("dataset", "audio")
TARGET_SR = 24000
TOP_K_METADATA = 64
TOP_K_RESULTS = 15
MAX_COMPARE_SECONDS = 20.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find nearest dataset audio matches. With 2 arguments: filter by query text then compare to a reference file. With 1 argument: compare the full dataset to that reference file."
    )
    parser.add_argument("arg1", help="Query text, or a reference audio filename/path if only one argument is provided.")
    parser.add_argument("arg2", nargs="?", default="", help="Optional reference audio filename/path.")
    return parser.parse_args()


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def load_metadata(metadata_csv: str, audio_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file", "")
            text = row.get("text", "")
            if not file_name:
                continue
            path = os.path.join(audio_dir, file_name)
            if os.path.isfile(path):
                rows.append({"file": file_name, "text": text, "path": path})
    return rows


def score_text_match(query: str, text: str) -> float:
    query_l = query.lower().strip()
    text_l = (text or "").lower()
    query_tokens = tokenize(query_l)
    text_tokens = set(tokenize(text_l))

    score = 0.0
    if query_l and query_l in text_l:
        score += 5.0

    overlap = sum(1 for token in query_tokens if token in text_tokens)
    score += overlap
    if query_tokens:
        score += overlap / len(query_tokens)

    return score


def resolve_audio_path(reference_audio: str, audio_dir: str) -> str:
    if os.path.isfile(reference_audio):
        return reference_audio
    candidate = os.path.join(audio_dir, reference_audio)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Reference audio not found: {reference_audio}")


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
    max_samples = int(TARGET_SR * MAX_COMPARE_SECONDS)
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

    feature = torch.cat([
        band_mean,
        band_std,
        frame_energy.mean().unsqueeze(0),
        frame_energy.std().unsqueeze(0),
        rms,
        zcr,
        dyn,
    ])
    return F.normalize(feature, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


def main():
    args = parse_args()
    metadata_rows = load_metadata(DEFAULT_METADATA, DEFAULT_AUDIO_DIR)
    if not metadata_rows:
        raise RuntimeError("No readable dataset entries found.")

    query = args.arg1 if args.arg2 else ""
    reference_audio = args.arg2 if args.arg2 else args.arg1

    reference_path = resolve_audio_path(reference_audio, DEFAULT_AUDIO_DIR)
    reference_feature = extract_feature(reference_path)

    scored_rows: List[Tuple[float, Dict[str, str]]] = []
    for row in metadata_rows:
        score = score_text_match(query, row["text"]) if query else 0.0
        scored_rows.append((score, row))
    scored_rows.sort(key=lambda item: item[0], reverse=True)

    if query:
        candidate_rows = [row for score, row in scored_rows[:TOP_K_METADATA] if score > 0]
        if not candidate_rows:
            candidate_rows = [row for _, row in scored_rows[:TOP_K_METADATA]]
    else:
        candidate_rows = [row for _, row in scored_rows[:TOP_K_METADATA]]

    ranked: List[Tuple[float, float, Dict[str, str]]] = []
    for row in candidate_rows:
        try:
            feature = extract_feature(row["path"])
            audio_score = cosine_similarity(reference_feature, feature)
            text_score = score_text_match(query, row["text"]) if query else 0.0
            ranked.append((audio_score, text_score, row))
        except Exception as exc:
            print(f"Skipping unreadable candidate {row['file']} ({exc})")

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    if query:
        print(f"Query: {query}")
    else:
        print("Query: <none; comparing against full dataset>")
    print(f"Reference audio: {reference_path}")
    print(f"Metadata candidates checked: {len(candidate_rows)}")
    print()
    print("Nearest matches:")
    for idx, (audio_score, text_score, row) in enumerate(ranked[:TOP_K_RESULTS], start=1):
        print(
            f"{idx:>2}. {row['file']} | audio_sim={audio_score:.4f} | text_score={text_score:.2f} | {row['text']}"
        )


if __name__ == "__main__":
    main()
