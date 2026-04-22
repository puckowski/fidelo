import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "dataset"
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")

OUT_DIR = "prepared"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SR = 24000
CHUNK_SEC = 0.25   # sub-second
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

N_CLUSTERS = 512   # number of discrete chunk IDs
MAX_CHUNKS_PER_FILE = None  # set to an int to limit dataset size


# -----------------------------
# Feature extractor
# -----------------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def split_chunks(wav: torch.Tensor, chunk_samples: int = CHUNK_SAMPLES) -> List[torch.Tensor]:
    total = wav.shape[1]
    chunks = []
    for start in range(0, total - chunk_samples + 1, chunk_samples):
        chunks.append(wav[:, start:start + chunk_samples].clone())
    return chunks


def chunk_feature(chunk: torch.Tensor) -> np.ndarray:
    """
    Returns a fixed-size feature vector for clustering.
    """
    with torch.no_grad():
        mel = mel_transform(chunk)               # [1, n_mels, frames]
        mel_db = amplitude_to_db(mel)            # [1, n_mels, frames]
        mel_db = mel_db.squeeze(0)               # [n_mels, frames]

        # Summary stats over time
        mean = mel_db.mean(dim=1)                # [n_mels]
        std = mel_db.std(dim=1)                  # [n_mels]

        # Additional simple temporal stats
        frame_energy = mel_db.mean(dim=0)
        energy_mean = frame_energy.mean().unsqueeze(0)
        energy_std = frame_energy.std().unsqueeze(0)
        zcr = ((chunk[:, 1:] * chunk[:, :-1]) < 0).float().mean().unsqueeze(0)

        feat = torch.cat([mean, std, energy_mean, energy_std, zcr], dim=0)
        return feat.cpu().numpy().astype(np.float32)


@dataclass
class ChunkRecord:
    chunk_index_global: int
    file: str
    text: str
    chunk_index_in_file: int
    start_sample: int
    end_sample: int
    cluster_id: int = -1
    wav_path: str = ""


def main():
    df = pd.read_csv(METADATA_CSV)

    all_features = []
    records: List[ChunkRecord] = []
    skipped_files: List[Dict[str, str]] = []

    chunk_wav_dir = os.path.join(OUT_DIR, "chunk_wavs")
    os.makedirs(chunk_wav_dir, exist_ok=True)

    global_idx = 0

    print("Extracting chunk features...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        file_name = row.file
        text = row.text
        path = os.path.join(AUDIO_DIR, file_name)

        if not os.path.isfile(path):
            skipped_files.append({"file": file_name, "reason": "missing file"})
            tqdm.write(f"Skipping missing file: {path}")
            continue

        try:
            wav = load_audio_mono(path)
            chunks = split_chunks(wav)
        except Exception as exc:
            skipped_files.append({"file": file_name, "reason": str(exc)})
            tqdm.write(f"Skipping unreadable file: {path} ({exc})")
            continue

        if not chunks:
            skipped_files.append({"file": file_name, "reason": "audio too short for one chunk"})
            tqdm.write(f"Skipping short file: {path}")
            continue

        if MAX_CHUNKS_PER_FILE is not None:
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        for i, chunk in enumerate(chunks):
            feat = chunk_feature(chunk)
            all_features.append(feat)

            out_wav = os.path.join(chunk_wav_dir, f"{global_idx:08d}.wav")
            torchaudio.save(out_wav, chunk, TARGET_SR)

            rec = ChunkRecord(
                chunk_index_global=global_idx,
                file=file_name,
                text=text,
                chunk_index_in_file=i,
                start_sample=i * CHUNK_SAMPLES,
                end_sample=(i + 1) * CHUNK_SAMPLES,
                wav_path=out_wav,
            )
            records.append(rec)
            global_idx += 1

    if not all_features:
        raise RuntimeError("No valid audio chunks were extracted. Check dataset/audio and metadata.csv.")

    X = np.stack(all_features, axis=0)
    print("Feature matrix shape:", X.shape)

    print(f"Clustering into {N_CLUSTERS} chunk IDs...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=4096,
        random_state=42,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(X)

    for rec, cid in zip(records, cluster_ids):
        rec.cluster_id = int(cid)

    # Save features, centroids, records
    np.save(os.path.join(OUT_DIR, "chunk_features.npy"), X)
    np.save(os.path.join(OUT_DIR, "cluster_centers.npy"), kmeans.cluster_centers_.astype(np.float32))

    with open(os.path.join(OUT_DIR, "chunk_records.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # Also save per-file token sequences
    per_file: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.file not in per_file:
            per_file[rec.file] = {
                "file": rec.file,
                "text": rec.text,
                "chunk_ids": [],
                "chunk_record_ids": [],
            }
        per_file[rec.file]["chunk_ids"].append(rec.cluster_id)
        per_file[rec.file]["chunk_record_ids"].append(rec.chunk_index_global)

    with open(os.path.join(OUT_DIR, "train_sequences.jsonl"), "w", encoding="utf-8") as f:
        for _, item in per_file.items():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if skipped_files:
        skipped_path = os.path.join(OUT_DIR, "skipped_files.jsonl")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for item in skipped_files:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done.")
    print("Wrote:")
    print(" -", os.path.join(OUT_DIR, "chunk_records.jsonl"))
    print(" -", os.path.join(OUT_DIR, "train_sequences.jsonl"))
    print(" -", os.path.join(OUT_DIR, "chunk_features.npy"))
    print(" -", os.path.join(OUT_DIR, "cluster_centers.npy"))
    if skipped_files:
        print(" -", os.path.join(OUT_DIR, "skipped_files.jsonl"))
        print(f"Skipped {len(skipped_files)} bad files.")


if __name__ == "__main__":
    main()