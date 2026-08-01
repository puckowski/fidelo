import argparse
import csv
import re
from pathlib import Path


WORD_RE = re.compile(r"\S+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Limit metadata.csv text values to a maximum number of words."
    )
    parser.add_argument(
        "--input",
        default="dataset/metadata.csv",
        help="Input metadata CSV path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=16,
        help="Maximum number of words allowed in the text column.",
    )
    return parser.parse_args()


def trim_to_word_limit(text: str, max_words: int) -> str:
    words = WORD_RE.findall(text.strip())
    return " ".join(words[:max_words])


def process_metadata(input_path: Path, output_path: Path, max_words: int):
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "file" not in fieldnames or "text" not in fieldnames:
            raise RuntimeError("CSV must contain 'file' and 'text' columns")

        rows = []
        changed = 0
        for row in reader:
            original = row.get("text", "")
            trimmed = trim_to_word_limit(original, max_words)
            if trimmed != original.strip():
                changed += 1
            row["text"] = trimmed
            rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Trimmed {changed} rows to at most {max_words} words")


def main():
    args = parse_args()
    if args.max_words <= 0:
        raise ValueError("--max-words must be greater than 0")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    process_metadata(input_path, output_path, args.max_words)


if __name__ == "__main__":
    main()
