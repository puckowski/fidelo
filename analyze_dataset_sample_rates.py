import argparse
import collections
import csv
import os
from typing import Dict, List, Tuple


def get_audio_info(path: str):
    try:
        import soundfile as sf

        info = sf.info(path)
        return {
            "sample_rate": int(info.samplerate),
            "frames": int(info.frames),
            "channels": int(info.channels),
            "format": getattr(info, "format", "unknown"),
        }
    except Exception:
        pass

    try:
        import torchaudio

        info = torchaudio.info(path)
        return {
            "sample_rate": int(info.sample_rate),
            "frames": int(info.num_frames),
            "channels": int(info.num_channels),
            "format": getattr(info, "encoding", "unknown"),
        }
    except Exception as exc:
        raise RuntimeError(f"Could not read audio metadata with soundfile or torchaudio: {exc}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sample-rate distribution in the dataset audio files.")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--audio-dir", default="dataset/audio")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on number of files to inspect. 0 means all files.")
    parser.add_argument("--show-missing", action="store_true", help="Print missing files.")
    return parser.parse_args()


def load_paths(metadata_csv: str, audio_dir: str, max_files: int = 0) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file", "")
            path = os.path.join(audio_dir, file_name)
            paths.append((file_name, path))
            if max_files > 0 and len(paths) >= max_files:
                break
    return paths


def main():
    args = parse_args()
    entries = load_paths(args.metadata_csv, args.audio_dir, args.max_files)
    if not entries:
        raise RuntimeError("No dataset entries found.")

    sample_rate_counts: Dict[int, int] = collections.Counter()
    extension_counts: Dict[str, int] = collections.Counter()
    unreadable: List[Tuple[str, str]] = []
    missing: List[str] = []

    total_checked = 0
    for file_name, path in entries:
        if not os.path.isfile(path):
            missing.append(file_name)
            continue

        extension_counts[os.path.splitext(file_name)[1].lower()] += 1
        try:
            info = get_audio_info(path)
            sample_rate_counts[int(info["sample_rate"])] += 1
            total_checked += 1
        except Exception as exc:
            unreadable.append((file_name, str(exc)))

    print("Dataset sample-rate analysis")
    print("============================")
    print(f"Entries in metadata checked: {len(entries)}")
    print(f"Readable audio files: {total_checked}")
    print(f"Missing files: {len(missing)}")
    print(f"Unreadable files: {len(unreadable)}")
    print()

    print("Sample-rate distribution:")
    if sample_rate_counts:
        for sample_rate, count in sorted(sample_rate_counts.items(), key=lambda item: (-item[1], item[0])):
            pct = 100.0 * count / max(1, total_checked)
            print(f"  {sample_rate:>6} Hz : {count:>6} files ({pct:5.2f}%)")
    else:
        print("  No readable files found.")
    print()

    print("File extension distribution:")
    for ext, count in sorted(extension_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {ext or '<no extension>'} : {count}")
    print()

    if sample_rate_counts:
        most_common_rate, most_common_count = max(sample_rate_counts.items(), key=lambda item: item[1])
        print(f"Most common sample rate: {most_common_rate} Hz ({most_common_count} files)")
        print("Recommendation:")
        print("  - Use the most common native sample rate when possible for best reconstruction fidelity.")
        print("  - If compute is limited, choose a lower target rate deliberately as a tradeoff, not by default.")
        print("  - For music, if many files are >= 24000 Hz, prefer 24000 Hz over 16000 Hz when hardware allows.")
        print()

    if args.show_missing and missing:
        print("Missing files:")
        for file_name in missing[:100]:
            print(f"  {file_name}")
        if len(missing) > 100:
            print(f"  ... and {len(missing) - 100} more")
        print()

    if unreadable:
        print("Unreadable files (first 50):")
        for file_name, error in unreadable[:50]:
            print(f"  {file_name}: {error}")
        if len(unreadable) > 50:
            print(f"  ... and {len(unreadable) - 50} more")


if __name__ == "__main__":
    main()
