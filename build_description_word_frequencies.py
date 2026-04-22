import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set


WORD_RE = re.compile(r"[a-z0-9']+")
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with", "your",
    "genre", "genres", "style", "styles", "tag", "tags", "recorded", "live", "unreleased",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build word-frequency data from dataset audio descriptions for HTML/JavaScript word cloud visualization."
    )
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-json", default="wordcloud_word_frequencies.json")
    parser.add_argument("--output-csv", default="wordcloud_word_frequencies.csv")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument(
        "--extra-stopwords",
        nargs="*",
        default=[],
        help="Additional words to exclude from the counts.",
    )
    return parser.parse_args()


def tokenize(text: str) -> Iterable[str]:
    return WORD_RE.findall(text.lower())


def load_rows(metadata_csv: str, text_column: str) -> List[str]:
    rows: List[str] = []
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row.get(text_column, "") or "")
    return rows


def build_counter(texts: List[str], stopwords: Set[str], min_word_length: int) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        for token in tokenize(text):
            if len(token) < min_word_length:
                continue
            if token in stopwords:
                continue
            counter[token] += 1
    return counter


def write_json(path: str, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def write_csv(path: str, items: List[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "value"])
        writer.writeheader()
        writer.writerows(items)


def main():
    args = parse_args()
    texts = load_rows(args.metadata_csv, args.text_column)
    stopwords = set(DEFAULT_STOPWORDS)
    stopwords.update(word.lower() for word in args.extra_stopwords)

    counter = build_counter(texts, stopwords, args.min_word_length)
    most_common = [
        {"text": word, "value": count}
        for word, count in counter.most_common(args.top_k)
        if count >= args.min_count
    ]

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    write_json(str(output_json), most_common)
    write_csv(str(output_csv), most_common)

    print(f"Processed {len(texts)} rows from {args.metadata_csv}")
    print(f"Unique kept words: {len(counter)}")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote CSV: {output_csv}")
    print("Top 20 words:")
    for item in most_common[:20]:
        print(f"  {item['text']}: {item['value']}")


if __name__ == "__main__":
    main()
