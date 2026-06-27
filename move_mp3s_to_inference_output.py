import argparse
import shutil
from pathlib import Path


MEDIA_EXTENSIONS = {".mp3", ".wav"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move MP3 and WAV files into an inference_output folder."
    )
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Directory to scan for MP3 and WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="inference_output",
        help="Destination folder for moved MP3 files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without changing files.",
    )
    return parser.parse_args()


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if args.recursive else "*"
    media_files = [
        path
        for path in source_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    media_files = [path for path in media_files if output_dir not in path.parents and path.parent != output_dir]

    if not media_files:
        print("No MP3 or WAV files found to move.")
        return

    moved = 0
    for media_path in sorted(media_files):
        destination = unique_destination(output_dir / media_path.name)
        if args.dry_run:
            print(f"Would move {media_path} -> {destination}")
            moved += 1
            continue

        shutil.move(str(media_path), str(destination))
        print(f"Moved {media_path} -> {destination}")
        moved += 1

    print(f"Processed {moved} media file(s).")


if __name__ == "__main__":
    main()
