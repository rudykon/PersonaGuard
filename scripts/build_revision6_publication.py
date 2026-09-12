#!/usr/bin/env python3
"""Build or verify every Revision-6 publication-facing consumer.

The formal analysis artifacts remain immutable upstream records.  This command
refreshes their aggregate evidence registry, rebuilds the sole publication
source, generates manuscript/README consumers and figures, and runs the
consistency tests.  ``--check`` is read-only and fails on stale generated text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
FIGURE_STEMS = (
    "Evidence-to-Action_Audit_Protocol",
    "Protocol_Evaluation_refined_source",
    "protocol_overview",
    "evaluation_design",
    "protocol_evaluation",
    "estimator_reference_actionability",
    "dense_trajectory_examples",
    "Evidence_Graph",
    "Validation_Boundaries_portraits",
    "dense_algorithm_sensitivity",
    "decision_boundary_sensitivity",
)
# Editable sources/exports live outside the manuscript; paper gets cited PDFs only.
FIGURE_SUFFIXES = (".svg", "_embed.pdf")
RASTER_FIGURE_SUFFIXES = (".png", ".tiff")
CUSTOM_SVG_STEMS = frozenset((
    "Evidence-to-Action_Audit_Protocol", "Protocol_Evaluation_refined_source",
    "Evidence_Graph", "Validation_Boundaries_portraits",
))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed generated text and tests without writing",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip the targeted standard-library unittest suite",
    )
    parser.add_argument(
        "--require-raster", action="store_true",
        help="also require optional PNG/TIFF exports (not LaTeX dependencies)",
    )
    return parser.parse_args(argv)


def run(*parts: str) -> None:
    command = [PYTHON, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_tests() -> None:
    run(
        "-m",
        "unittest",
        "-v",
        "tests.test_revision6_analyses",
        "tests.test_revision6_publication",
        "tests.test_evidence_traceability",
        "tests.test_protocol_replay",
        "tests.test_submission_readiness",
        "tests.test_revision6_synthetic_smoke",
        "tests.test_pdfua_validator",
        "tests.test_paper_asset_layout",
    )


def cited_figure_names() -> set[str]:
    """Read current TeX references, including both languages and supplements."""
    names: set[str] = set()
    pattern = re.compile(r"\\includegraphics\s*(?:\[[\s\S]*?\])?\s*\{([^}]+)\}")
    for path in (ROOT / "paper").rglob("*.tex"):
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
        for name in pattern.findall(text):
            if Path(name).name != name or not name.endswith("_embed.pdf"):
                raise RuntimeError(f"Review nonstandard figure reference: {name}")
            names.add(name)
    registered = {f"{stem}_embed.pdf" for stem in FIGURE_STEMS}
    if names != registered:
        raise RuntimeError(
            f"Figure registration differs from TeX references: "
            f"unregistered={sorted(names - registered)}, unused={sorted(registered - names)}"
        )
    return names


def publish_figure_exports() -> None:
    """Copy only currently cited PDFs from support exports into paper/figures."""
    destination = ROOT / "paper" / "figures"
    names = cited_figure_names()
    sources = {name: ROOT / "paper_support" / "figures" / name for name in names}
    for source in sources.values():
        if not source.is_file() or not source.stat().st_size:
            raise RuntimeError(f"Missing or empty figure export: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sorted(sources.items()):
        target = destination / name
        if not target.is_file() or source.read_bytes() != target.read_bytes():
            shutil.copy2(source, target)


def verify_figure_exports(*, require_raster: bool = False) -> None:
    names = cited_figure_names()
    missing = []
    support = ROOT / "paper_support" / "figures"
    for stem in FIGURE_STEMS:
        suffixes = FIGURE_SUFFIXES
        if require_raster and stem not in CUSTOM_SVG_STEMS:
            suffixes += RASTER_FIGURE_SUFFIXES
        paths = [support / f"{stem}{suffix}" for suffix in suffixes]
        paths.append(ROOT / "paper" / "figures" / f"{stem}_embed.pdf")
        for path in paths:
            if not path.is_file() or not path.stat().st_size:
                missing.append(path.relative_to(ROOT).as_posix())
    if missing:
        raise RuntimeError("Missing or empty publication figure exports: " + ", ".join(missing))
    for name in names:
        if (support / name).read_bytes() != (ROOT / "paper" / "figures" / name).read_bytes():
            raise RuntimeError(f"Published figure differs from support export: {name}")
    extras = {
        path.name for path in (ROOT / "paper" / "figures").iterdir()
        if path.name not in names
    }
    if extras:
        raise RuntimeError(f"Unreferenced items in paper/figures: {sorted(extras)}")


def main() -> None:
    args = parse_args()
    if args.check:
        run("scripts/generate_submission_readiness.py", "--check", "--strict")
        run("scripts/generate_word_count_report.py", "--check")
        run("scripts/generate_evidence_traceability.py", "--check")
        run("scripts/run_protocol_replay.py", "--check")
        run("scripts/build_revision6_source.py", "--check")
        run("scripts/generate_revision6_publication.py", "--check")
        run("paper_support/figures/export_current_protocol_svgs.py", "--check")
        run("paper_support/figures/export_current_supplementary_svgs.py", "--check")
        verify_figure_exports(require_raster=args.require_raster)
    else:
        run("scripts/generate_submission_readiness.py", "--strict")
        run("scripts/refresh_evidence_traceability.py")
        run("scripts/generate_evidence_traceability.py")
        run("scripts/generate_evidence_traceability.py", "--check")
        run("scripts/run_protocol_replay.py")
        run("scripts/build_revision6_source.py")
        run("scripts/generate_revision6_publication.py")
        run("paper_support/figures/make_revision6_protocol_figure.py")
        run("paper_support/figures/make_dense_trajectory_examples.py")
        run("paper_support/figures/make_revision6_validity_figure.py")
        run("paper_support/figures/make_revision5_figures.py")
        run("paper_support/figures/make_supplementary_figures.py")
        run("paper_support/figures/make_revision6_decision_figures.py")
        run("paper_support/figures/export_current_protocol_svgs.py")
        publish_figure_exports()
        run("scripts/generate_word_count_report.py")
        run("scripts/build_revision6_source.py", "--check")
        run("scripts/generate_revision6_publication.py", "--check")
        verify_figure_exports(require_raster=args.require_raster)
    if not args.skip_tests:
        run_tests()
    print("Revision-6 publication build is current.")


if __name__ == "__main__":
    main()
