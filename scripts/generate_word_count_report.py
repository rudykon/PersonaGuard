#!/usr/bin/env python3
"""Generate or verify the formal CHI manuscript word-count report."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
COUNTER = ROOT / "paper" / "count_words.pl"
MAIN = ROOT / "paper" / "main.tex"
BODY = ROOT / "paper" / "body.tex"
OUTPUT = ROOT / "paper" / "word_count_report.txt"
RECOMMENDED_MIN = 5000
RECOMMENDED_MAX = 8000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def count() -> tuple[str, int, int, int]:
    completed = subprocess.run(
        ["perl", str(COUNTER), str(MAIN), str(BODY)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    values: dict[str, int] = {}
    for key in ("abstract_words", "body_words", "total_words"):
        match = re.search(rf"^{key}=(\d+)$", completed.stdout, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Counter output lacks {key}")
        values[key] = int(match.group(1))
    return (
        completed.stdout.splitlines()[0],
        values["abstract_words"],
        values["body_words"],
        values["total_words"],
    )


def render(scope: str, abstract: int, body: int, total: int) -> str:
    if total < RECOMMENDED_MIN:
        decision = f"below the recorded {RECOMMENDED_MIN:,}--{RECOMMENDED_MAX:,}-word range"
    elif total <= RECOMMENDED_MAX:
        decision = f"within the recorded {RECOMMENDED_MIN:,}--{RECOMMENDED_MAX:,}-word range"
    else:
        decision = (
            f"{total - RECOMMENDED_MAX:,} words above the recorded "
            f"{RECOMMENDED_MAX:,}-word upper recommendation"
        )
    return f"""CHI 2027 manuscript word-count report

Source files:
  paper/main.tex
  paper/body.tex
  generated table rows expanded through LaTeX input

Counting scope:
  {scope}
  Alphabetic tokens are counted; hyphenated and apostrophe forms remain one token.

Result:
  abstract_words = {abstract}
  body_words     = {body}
  total_words    = {total}

Decision:
  The manuscript is {decision}.
  The project record uses the CHI 2027 recommended 5,000--8,000-word range;
  it does not use the superseded internal 9,500-word target.

Reproduce:
  perl paper/count_words.pl paper/main.tex paper/body.tex
"""


def main() -> None:
    args = parse_args()
    expected = render(*count())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != expected:
            raise RuntimeError("Formal word-count report is stale")
    else:
        OUTPUT.write_text(expected, encoding="utf-8")
    print(expected.split("Decision:\n", 1)[1].splitlines()[0].strip())


if __name__ == "__main__":
    main()
