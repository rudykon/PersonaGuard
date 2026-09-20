#!/usr/bin/env python3
"""Generate the current validity figure from the sole Revision-6 source."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

from make_revision3_figures import figure_validity_identifiability


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "revision6_source.json"
FIGURE_DIR = ROOT / "artifacts" / "figures"
FIGURE_STEM = "validity_identifiability"
FIGURE_WIDTH_MM = 182.1
OUTPUT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600

# Mirror the settings used by the imported renderer so this canonical wrapper
# carries a complete, statically inspectable publication contract.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("schema_version") != "revision6-publication-source-v1":
        raise RuntimeError("Stale Revision-6 publication source")
    if source.get("revision") != 6:
        raise RuntimeError("Revision 6 is not the active publication source")
    figure_validity_identifiability(source)
    missing = [
        suffix
        for suffix in OUTPUT_SUFFIXES
        if not (FIGURE_DIR / f"{FIGURE_STEM}{suffix}").is_file()
    ]
    if missing:
        raise RuntimeError(f"Validity figure export contract failed: {missing}")
    print("Generated canonical Revision-6 validity figure and source data")


if __name__ == "__main__":
    main()
