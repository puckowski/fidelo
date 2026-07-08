import argparse
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stash repository changes while preserving selected output directories."
    )
    parser.add_argument("--repo-root", required=True, help="Repository root directory.")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Relative path prefix to preserve from stashing. Can be passed multiple times.",
    )
    parser.add_argument(
        "--message",
        default="Auto stash from tokenizer reconstruction loop",
        help="Stash message.",
    )
    return parser.parse_args()


def normalize_prefix(prefix: str) -> str:
    normalized = prefix.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return normalized + "/"


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def should_exclude(path: str, excluded_prefixes: List[str]) -> bool:
    normalized = normalize_path(path)
    return any(normalized == prefix[:-1] or normalized.startswith(prefix) for prefix in excluded_prefixes)


def collect_changed_paths(repo_root: str, excluded_prefixes: List[str]) -> List[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    entries = result.stdout.decode("utf-8", errors="replace").split("\0")
    changed_paths: List[str] = []
    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        idx += 1
        if not entry:
            continue
        status = entry[:2]
        path = entry[3:]
        candidate_paths = [path]
        if status.startswith("R") or status.startswith("C") or status[1:2] in {"R", "C"}:
            if idx < len(entries):
                candidate_paths.append(entries[idx])
                idx += 1
        for candidate in candidate_paths:
            if candidate and not should_exclude(candidate, excluded_prefixes):
                changed_paths.append(candidate)
    deduped: List[str] = []
    seen = set()
    for path in changed_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def main() -> int:
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    excluded_prefixes = [normalize_prefix(prefix) for prefix in args.exclude_prefix]

    try:
        changed_paths = collect_changed_paths(repo_root, excluded_prefixes)
    except FileNotFoundError:
        print("git executable not found in PATH.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr.decode("utf-8", errors="replace"))
        return exc.returncode or 1

    if not changed_paths:
        print("No stashable repository changes found.")
        return 0

    command = [
        "git",
        "stash",
        "push",
        "--include-untracked",
        "--message",
        args.message,
        "--",
        *changed_paths,
    ]

    try:
        completed = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        return exc.returncode or 1

    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())