import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

import statistics


NUMERIC_COLUMNS = [
    "exit_code",
    "theme_gate_rejects",
    "clip_gate_rejects",
    "total_gate_instances",
    "static_score",
    "high_ratio",
    "flatness",
    "zcr",
    "centroid_norm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show per-report modes for every report.csv under batch_reports."
    )
    parser.add_argument(
        "--root",
        default="batch_reports",
        help="Root folder to scan recursively for report.csv files.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Decimal places for float mode grouping.",
    )
    return parser.parse_args()


def discover_reports(root: str) -> List[Path]:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted([p for p in base.glob("**/report.csv") if p.is_file()])


def parse_numeric(value: str, round_digits: int):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None

    try:
        if any(char in text for char in [".", "e", "E"]):
            return round(float(text), round_digits)
        return int(text)
    except ValueError:
        return None


def mode_of(values: List):
    if not values:
        return None, 0
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value, count


def summarize_report(path: Path, round_digits: int) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summary: Dict[str, str] = {
        "folder": str(path.parent.resolve()),
        "rows": str(len(rows)),
    }

    for col in NUMERIC_COLUMNS:
        values = [parse_numeric(row.get(col, ""), round_digits) for row in rows]
        values = [v for v in values if v is not None]
        mode_value, mode_count = mode_of(values)
        if mode_value is None:
            summary[f"{col}_mode"] = "n/a"
            summary[f"{col}_median"] = "n/a"
        else:
            summary[f"{col}_mode"] = f"{mode_value} (count={mode_count})"
            median_value = statistics.median(values)
            summary[f"{col}_median"] = f"{median_value}"

    return summary


def main():
    args = parse_args()
    reports = discover_reports(args.root)
    if not reports:
        raise RuntimeError(f"No report.csv files found under {args.root}")

    for report in reports:
        summary = summarize_report(report, args.round)
        print(f"report: {report}")
        print(f"folder: {summary['folder']}")
        print(f"rows: {summary['rows']}")
        for col in NUMERIC_COLUMNS:
            print(f"  {col}_mode: {summary[f'{col}_mode']}")
            print(f"  {col}_median: {summary[f'{col}_median']}")
        print()


if __name__ == "__main__":
    main()
