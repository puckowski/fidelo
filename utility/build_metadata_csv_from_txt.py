import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build dataset/metadata.csv from audio files and matching .txt description files."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing .txt files and the audio folder.")
    parser.add_argument("--audio-dir", default="", help="Optional audio directory override. Defaults to <dataset-dir>/audio.")
    parser.add_argument("--output", default="", help="Optional metadata.csv output path. Defaults to <dataset-dir>/metadata.csv.")
    parser.add_argument(
        "--skip-missing-text",
        action="store_true",
        help="Skip audio files without matching .txt files instead of failing.",
    )
    return parser.parse_args()


def normalize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = stem.casefold()
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip()


def discover_text_files(dataset_dir: Path) -> Dict[str, Path]:
    text_files: Dict[str, Path] = {}
    duplicates = []

    for path in sorted(dataset_dir.glob("*.txt")):
        key = normalize_stem(path.name)
        if key in text_files:
            duplicates.append((text_files[key], path))
            continue
        text_files[key] = path

    if duplicates:
        duplicate_list = ", ".join(f"{first.name} / {second.name}" for first, second in duplicates)
        raise RuntimeError(f"Found duplicate normalized text names: {duplicate_list}")

    return text_files


def discover_audio_files(audio_dir: Path) -> Iterable[Path]:
    for path in sorted(audio_dir.iterdir()):
        if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS:
            yield path


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_rows(audio_files: Iterable[Path], text_files: Dict[str, Path], skip_missing_text: bool) -> Tuple[list[dict], list[str]]:
    rows = []
    missing = []

    for audio_path in audio_files:
        key = normalize_stem(audio_path.name)
        text_path = text_files.get(key)
        if text_path is None:
            missing.append(audio_path.name)
            if skip_missing_text:
                continue
            continue

        text = load_text(text_path)
        rows.append({"file": audio_path.name, "text": text})

    return rows, missing


def write_metadata_csv(output_path: Path, rows: Iterable[dict]):
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)



def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    audio_dir = Path(args.audio_dir) if args.audio_dir else dataset_dir / "audio"
    output_path = Path(args.output) if args.output else dataset_dir / "metadata.csv"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    text_files = discover_text_files(dataset_dir)
    audio_files = list(discover_audio_files(audio_dir))
    if not audio_files:
        raise RuntimeError(f"No audio files found in: {audio_dir}")

    rows, missing = build_rows(audio_files, text_files, args.skip_missing_text)
    if missing and not args.skip_missing_text:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise RuntimeError(
            "Missing matching .txt files for audio files: "
            f"{preview}{extra}. Re-run with --skip-missing-text to ignore them."
        )

    write_metadata_csv(output_path, rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    if missing:
        print(f"Skipped {len(missing)} audio files without matching text files")


if __name__ == "__main__":
    main()
