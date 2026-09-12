#!/usr/bin/env python3
"""Run tagged-PDF preflight and optional formal veraPDF PDF/UA-1 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "paper_support" / "pdfua_validation_report.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--verapdf", type=Path)
    parser.add_argument("--require-verapdf", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def run_capture(parts: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(parts),
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def parse_pdfinfo(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_pdffonts(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        embedded, subset, unicode_map = parts[-5:-2]
        rows.append(
            {
                "name": parts[0],
                "embedded": embedded,
                "subset": subset,
                "unicode_map": unicode_map,
            }
        )
    return rows


def vera_validation(executable: Path | None, pdf: Path) -> dict[str, object]:
    if executable is None:
        return {"available": False, "profile": "PDF/UA-1", "compliant": None}
    version = run_capture([str(executable), "--version"])
    completed = run_capture(
        [
            str(executable),
            "-f",
            "ua1",
            "--format",
            "json",
            "--maxfailures",
            "100",
            str(pdf),
        ]
    )
    try:
        payload = json.loads(completed.stdout)
        result = payload["report"]["jobs"][0]["validationResult"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "veraPDF did not return a parseable PDF/UA-1 report. "
            + completed.stderr.strip()
        ) from exc
    details = result.get("details", {})
    summaries = details.get("ruleSummaries", [])
    return {
        "available": True,
        "executable": str(executable),
        "version": version.stdout.splitlines()[0] if version.stdout else "unknown",
        "profile": result.get("profileName", "PDF/UA-1 validation profile"),
        "compliant": bool(result.get("compliant")),
        "statement": result.get("statement"),
        "passed_rules": details.get("passedRules"),
        "failed_rules": details.get("failedRules"),
        "passed_checks": details.get("passedChecks"),
        "failed_checks": details.get("failedChecks"),
        "failed_rule_summaries": [
            {
                "clause": item.get("clause"),
                "test_number": item.get("testNumber"),
                "description": item.get("description"),
                "failed_checks": item.get("failedChecks"),
                "tags": item.get("tags", []),
            }
            for item in summaries
            if item.get("ruleStatus") == "FAILED"
        ],
        "return_code": completed.returncode,
        "stderr_warnings": completed.stderr.splitlines(),
    }


def validate_pdf(pdf: Path, vera: Path | None) -> dict[str, object]:
    resolved = pdf if pdf.is_absolute() else ROOT / pdf
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    info_run = run_capture(["pdfinfo", str(resolved)])
    meta_run = run_capture(["pdfinfo", "-meta", str(resolved)])
    fonts_run = run_capture(["pdffonts", str(resolved)])
    structure_run = run_capture(["pdfinfo", "-struct-text", str(resolved)])
    if info_run.returncode or meta_run.returncode or fonts_run.returncode:
        raise RuntimeError(f"Poppler preflight could not parse {resolved}")
    info = parse_pdfinfo(info_run.stdout)
    fonts = parse_pdffonts(fonts_run.stdout)
    checks = {
        "tagged": info.get("Tagged", "").lower() == "yes",
        "no_suspects": info.get("Suspects", "").lower() == "no",
        "not_encrypted": info.get("Encrypted", "").lower() == "no",
        "metadata_stream": info.get("Metadata Stream", "").lower() == "yes",
        "pdfua_part_1_metadata": "<pdfuaid:part>1</pdfuaid:part>" in meta_run.stdout,
        "document_language_metadata": "<dc:language>" in meta_run.stdout,
        "title_metadata": "<dc:title>" in meta_run.stdout,
        "all_fonts_embedded": bool(fonts)
        and all(row["embedded"] == "yes" for row in fonts),
        "all_fonts_have_unicode_maps": bool(fonts)
        and all(row["unicode_map"] == "yes" for row in fonts),
    }
    formal = vera_validation(vera, resolved)
    machine_compliant = all(checks.values()) and formal.get("compliant") is True
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "pages": int(info.get("Pages", "0")),
        "pdf_version": info.get("PDF version"),
        "preflight_checks": checks,
        "font_count": len(fonts),
        "poppler_structure_return_code": structure_run.returncode,
        "poppler_structure_warnings": structure_run.stderr.splitlines(),
        "formal_machine_validation": formal,
        "machine_compliant": machine_compliant,
    }


def main() -> None:
    args = parse_args()
    vera = args.verapdf
    if vera is None:
        detected = shutil.which("verapdf") or shutil.which("veraPDF")
        vera = Path(detected) if detected else None
    if vera is not None and not vera.is_file():
        raise FileNotFoundError(vera)
    if args.require_verapdf and vera is None:
        raise RuntimeError("veraPDF is required but was not found")
    reports = [validate_pdf(pdf, vera) for pdf in args.pdfs]
    payload = {
        "schema_version": "chi-pdfua-validation-v1",
        "validator_scope": (
            "Machine-verifiable PDF/UA-1 checks only; reading order, alternative "
            "text quality, table semantics, link purpose, and visual contrast "
            "still require human accessibility review."
        ),
        "documents": reports,
        "all_machine_compliant": all(
            bool(report["machine_compliant"]) for report in reports
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    for report in reports:
        print(
            f"{report['path']}: machine_compliant={report['machine_compliant']}"
        )
    if not payload["all_machine_compliant"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
