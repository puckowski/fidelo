import argparse
import csv
import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove metadata rows whose audio files cannot be decoded."
    )
    parser.add_argument(
        "--metadata-csv",
        default="dataset/metadata.csv",
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--audio-dir",
        default="dataset/audio",
        help="Directory containing the audio files referenced by metadata.csv.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to overwriting the input metadata file.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional text file listing removed rows and reasons.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use ffmpeg validation when available to catch partially corrupted files that still decode in Python.",
    )
    return parser.parse_args()


STRICT_ERROR_MARKERS = [
    "error",
    "invalid data",
    "header missing",
    "dequantization failed",
    "cannot read next header",
    "illegal audio-mpeg-header",
    "giving up resync",
    "hit end of (available) data during resync",
    "part2_3_length",
    "invalid/unsupported frame",
]


def _try_soundfile_read(audio_path: Path):
    try:
        sf = importlib.import_module("soundfile")
    except Exception:
        return False, "soundfile unavailable"

    try:
        sf.read(str(audio_path), dtype="float32", always_2d=True)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _try_torchaudio_read(audio_path: Path):
    try:
        torchaudio = importlib.import_module("torchaudio")
    except Exception:
        return False, "torchaudio unavailable"

    try:
        torchaudio.load(str(audio_path))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _try_ffmpeg_validate(audio_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg unavailable"

    command = [
        ffmpeg,
        "-v",
        "warning",
        "-nostdin",
        "-i",
        str(audio_path),
        "-f",
        "null",
        os.devnull,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return None, str(exc)

    stderr = (result.stderr or "").strip()
    stderr_lower = stderr.casefold()
    if result.returncode != 0:
        return False, stderr or f"ffmpeg exited with code {result.returncode}"

    for marker in STRICT_ERROR_MARKERS:
        if marker in stderr_lower:
            return False, stderr

    return True, stderr


def validate_audio_file(audio_path: Path, strict: bool = False):
    if not audio_path.is_file():
        return False, "file not found"

    suffix = audio_path.suffix.casefold()
    if suffix and suffix not in AUDIO_EXTENSIONS:
        return False, f"unsupported extension: {audio_path.suffix}"

    ok, reason = _try_soundfile_read(audio_path)
    if ok:
        return True, ""

    ok2, reason2 = _try_torchaudio_read(audio_path)
    if ok2:
        if strict:
            strict_ok, strict_reason = _try_ffmpeg_validate(audio_path)
            if strict_ok is False:
                return False, f"strict ffmpeg validation failed: {strict_reason}"
        return True, ""

    strict_reason = ""
    if strict:
        strict_ok, strict_reason = _try_ffmpeg_validate(audio_path)
        if strict_ok is False:
            return False, f"soundfile: {reason}; torchaudio: {reason2}; strict: {strict_reason}"

    return False, f"soundfile: {reason}; torchaudio: {reason2}{'; strict: ' + strict_reason if strict_reason else ''}"


def load_rows(metadata_csv: Path):
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "file" not in fieldnames:
            raise RuntimeError("Metadata CSV must contain at least a 'file' column")
        return fieldnames, list(reader)


def write_rows(output_csv: Path, fieldnames: Iterable[str], rows: list[dict]):
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report_path: Path, removed: list[tuple[str, str]]):
    with report_path.open("w", encoding="utf-8") as f:
        for file_name, reason in removed:
            f.write(f"{file_name}\t{reason}\n")


def main():
    args = parse_args()
    metadata_csv = Path(args.metadata_csv)
    audio_dir = Path(args.audio_dir)
    output_csv = Path(args.output) if args.output else metadata_csv

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    fieldnames, rows = load_rows(metadata_csv)
    kept_rows = []
    removed_rows = []

    for row in rows:
        file_name = (row.get("file") or "").strip()
        audio_path = audio_dir / file_name
        is_valid, reason = validate_audio_file(audio_path, strict=args.strict)
        if is_valid:
            kept_rows.append(row)
        else:
            removed_rows.append((file_name, reason))

    write_rows(output_csv, fieldnames, kept_rows)

    if args.report:
        write_report(Path(args.report), removed_rows)

    print(f"Checked {len(rows)} metadata rows")
    print(f"Kept {len(kept_rows)} rows")
    print(f"Removed {len(removed_rows)} invalid rows")
    if removed_rows:
        preview = ", ".join(name for name, _ in removed_rows[:10])
        extra = "" if len(removed_rows) <= 10 else f" ... and {len(removed_rows) - 10} more"
        print(f"Removed files: {preview}{extra}")
    elif args.strict:
        print("No rows were removed in strict mode. The remaining warnings may be recoverable decoder warnings rather than hard failures.")
    if args.report:
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
