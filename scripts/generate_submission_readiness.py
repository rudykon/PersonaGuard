#!/usr/bin/env python3
"""Validate author-only submission fields and generate the ethics sentence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    "en": ROOT / "paper" / "generated" / "secondary_analysis_ethics_statement.tex",
}

DISCLOSURE_COMPLETE = "DISCLOSURE_COMPLETE"
REQUIRED_DISCLOSURE_FIELDS = {
    "source_approval_reference",
    "data_access_basis",
    "analysis_scope",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True,
        help="path to an author-supplied submission-readiness JSON record",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless the secondary-analysis disclosure is complete",
    )
    return parser.parse_args(argv)


def validate(document: dict[str, object], *, strict: bool = False) -> dict[str, object]:
    if document.get("schema_version") != "submission-readiness-v2":
        raise ValueError("Unexpected submission-readiness schema")
    record = document.get("secondary_analysis_ethics")
    if not isinstance(record, dict):
        raise ValueError("secondary_analysis_ethics must be an object")
    status = str(record.get("status", ""))
    if status != DISCLOSURE_COMPLETE:
        raise ValueError(f"Unknown secondary-analysis ethics status: {status!r}")
    for field in ("manuscript_sentence", "manuscript_sentence_zh"):
        sentence = str(record.get(field, "")).strip()
        if not sentence or "TODO" in sentence.upper() or "PLACEHOLDER" in sentence.upper():
            raise ValueError(f"{field} must be complete and contain no placeholder")
    missing = [
        field for field in sorted(REQUIRED_DISCLOSURE_FIELDS)
        if not str(record.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(
            "Secondary-analysis disclosure lacks fields: " + ", ".join(missing)
        )
    if record.get("additional_institutional_determination_claimed") is not False:
        raise ValueError(
            "The record must explicitly state that no additional institutional "
            "determination is claimed"
        )
    return record


def render(record: dict[str, object], *, language: str = "en") -> str:
    field = "manuscript_sentence" if language == "en" else "manuscript_sentence_zh"
    sentence = str(record[field]).strip()
    if language == "zh":
        sentence = sentence.replace("gated research release", "受限研究发布集")
        comment = "% 由作者提供的披露记录生成；请通过独立生成工具更新。\n"
    else:
        comment = "% Generated from an author-supplied disclosure record; update with the standalone generator.\n"
    return comment + sentence + "\n"


def main() -> None:
    args = parse_args()
    document = json.loads(args.source.read_text(encoding="utf-8"))
    record = validate(document, strict=args.strict)
    for language, output in OUTPUTS.items():
        expected = render(record, language=language)
        if args.check:
            if output.read_text(encoding="utf-8") != expected:
                raise RuntimeError(
                    f"Generated secondary-analysis ethics statement is stale: {language}"
                )
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(expected, encoding="utf-8")
    status = record["status"]
    print(f"Secondary-analysis ethics readiness: {status}")


if __name__ == "__main__":
    main()
