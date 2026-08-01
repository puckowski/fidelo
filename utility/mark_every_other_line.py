import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Change trailing ,0 to ,1 on odd-numbered lines of a text file."
    )
    parser.add_argument("path", nargs="?", default="tmp", help="File to update in place (default: tmp).")
    return parser.parse_args()


def update_file(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as source:
        lines = source.readlines()

    changed = 0
    for index in range(0, len(lines), 2):
        line = lines[index]
        newline = ""
        if line.endswith("\r\n"):
            line, newline = line[:-2], "\r\n"
        elif line.endswith(("\r", "\n")):
            line, newline = line[:-1], line[-1]

        if line.endswith(",0"):
            lines[index] = f"{line[:-2]},1{newline}"
            changed += 1

    with path.open("w", encoding="utf-8", newline="") as destination:
        destination.writelines(lines)

    return changed, len(lines)


def main():
    args = parse_args()
    path = Path(args.path)
    changed, line_count = update_file(path)
    print(f"Updated {changed} of {line_count} lines in {path}")


if __name__ == "__main__":
    main()