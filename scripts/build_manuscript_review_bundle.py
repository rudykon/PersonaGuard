#!/usr/bin/env python3
"""Package both complete manuscript sources and PDFs without research data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/submission/chi2027_manuscript_review_bundle.zip"
DIRECT = (
    "paper/main.tex", "paper/body.tex", "paper/main.pdf",
    "paper/main_zh.tex", "paper/body_zh.tex", "paper/main_zh.pdf",
    "paper/supplementary_information.tex", "paper/supplementary_information.pdf",
    "paper/supplementary_information_zh.tex", "paper/supplementary_information_zh.pdf",
    "paper/accessibility_patches.tex", "paper/accessibility_patches_zh.tex",
    "paper/references.bib", "paper_support/revision6_source.json",
    "paper_support/protocol_rules.json", "paper_support/protocol_replay_cases.json",
    "paper_support/external_reuse_cases.json", "paper/word_count_report.txt",
    "paper_support/pdfua_validation_report.json",
    "paper_support/reference_verification_2026-09-11.json",
)
README = """# Complete bilingual manuscript review bundle

This is a reading and LaTeX-rebuild bundle, not a newly executed study or
an external submission. Both main.tex entry points include their body files.
The Chinese PDFs are reading versions, not additional research reports.

From the extracted paper directory, using TeX Live 2026 with acmart,
latexmk, LuaLaTeX, XeLaTeX and Noto Serif/Sans/Mono CJK SC fonts:

    latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
    latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
    latexmk -lualatex -interaction=nonstopmode -halt-on-error supplementary_information.tex
    latexmk -xelatex -interaction=nonstopmode -halt-on-error supplementary_information_zh.tex

All generated LaTeX fragments and vector figure dependencies are included.
revision6_source.json remains the publication-facing numerical source;
do not edit numerical values in generated fragments. The separate anonymous
supplement archive contains the executable research reproduction materials.
The only trial-level release is the three authorized anonymous Figure 5
examples and their plotted coordinates in the numerical source.
This bundle excludes full restricted data, original participant identifiers, media,
repository metadata, credentials, and private handoff notes.

MANIFEST.json lists each included file and its SHA-256 digest. It identifies
the packaged local version; it does not imply a Git commit or remote update.
"""


def payloads() -> dict[str, bytes]:
    paths = [ROOT / name for name in DIRECT]
    paths.extend(sorted((ROOT / "paper/generated").glob("*.tex")))
    paths.extend(sorted((ROOT / "paper/figures").glob("*_embed.pdf")))
    result = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    result["README.md"] = README.encode("utf-8")
    manifest = {name: hashlib.sha256(data).hexdigest()
                for name, data in sorted(result.items())}
    result["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return result


def build_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads().items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build_bytes()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != data:
            raise SystemExit("Manuscript review bundle is missing or stale")
        print("Manuscript review bundle is byte-current")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
        print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
