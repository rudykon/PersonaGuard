"""Export author-supplied Supplementary Figures S1/S2 without editing the SVGs.

Reuse the existing TeX-Libertine outlining and librsvg/Cairo renderer. Its
3:2 viewport adds horizontal letterboxing around these 4:3 SVGs; remove only
that viewport padding when placing the vector page on the original artboard.
No artwork is rasterized, recolored, relaid out, or replaced with system fonts.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import fitz

import export_current_protocol_svgs as native

FIGURES = Path(__file__).resolve().parents[2] / "paper" / "figures"
SOURCES = ("Evidence_Graph", "Validation_Boundaries_portraits")
RENDER_WIDTH, RENDER_HEIGHT = 1536.0, 1024.0


def dimensions(source: Path) -> tuple[float, float]:
    root = ET.parse(source).getroot()
    bounds = tuple(float(value) for value in root.attrib["viewBox"].split())
    if len(bounds) != 4 or bounds[:2] != (0.0, 0.0) or min(bounds[2:]) <= 0:
        raise ValueError("Expected a positive, origin-based SVG artboard")
    if root.get("preserveAspectRatio", "xMidYMid meet") not in {"xMidYMid", "xMidYMid meet"}:
        raise ValueError("Unexpected SVG aspect-ratio policy")
    for element in root.iter():
        if element.tag in {native.tag("image"), native.tag("script"), native.tag("foreignObject")}:
            raise ValueError("Expected self-contained vector artwork")
        if any(key.endswith("href") and not value.startswith("#")
               for key, value in element.attrib.items()):
            raise ValueError("External SVG resource is not permitted")
    return bounds[2], bounds[3]


def signature(source: Path, fonts: dict[bool, Path]) -> str:
    return hashlib.sha256(native.signature(source, fonts).encode("ascii")
                          + Path(__file__).read_bytes()).hexdigest()


def verify(output: Path, expected: str, size: tuple[float, float]) -> None:
    if not output.is_file():
        raise RuntimeError(f"Missing author SVG export: {output.name}")
    with fitz.open(output) as document:
        if len(document) != 1 or document.metadata.get("keywords") != expected:
            raise RuntimeError(f"Stale author SVG export: {output.name}")
        page = document[0]
        if abs(page.rect.width - size[0]) > .01 or abs(page.rect.height - size[1]) > .01:
            raise RuntimeError(f"SVG artboard size changed: {output.name}")
        if page.get_fonts(full=True) or page.get_images(full=True):
            raise RuntimeError(f"Expected font-free vector export: {output.name}")


def export_one(stem: str, *, check: bool = False, preview_dir: Path | None = None) -> Path:
    if stem not in SOURCES:
        raise ValueError(f"Unregistered supplementary SVG: {stem}")
    source = FIGURES / f"{stem}.svg"
    output = FIGURES / f"{stem}_embed.pdf"
    fonts = {bold: native.tex_font(bold) for bold in (False, True)}
    size = dimensions(source)
    expected = signature(source, fonts)
    if not check:
        with TemporaryDirectory(prefix="chi2027-si-svg-") as directory:
            outlined = Path(directory) / "outlined.svg"
            outlined.write_bytes(native.outline(source, fonts))
            rendered = Path(directory) / "rendered.pdf"
            native.render_svg(outlined, rendered)
            scale = min(RENDER_WIDTH / size[0], RENDER_HEIGHT / size[1])
            width, height = size[0] * scale, size[1] * scale
            left, top = (RENDER_WIDTH - width) / 2, (RENDER_HEIGHT - height) / 2
            viewport = fitz.Rect(left, top, left + width, top + height)
            with fitz.open(rendered) as source_pdf, fitz.open() as result:
                page = result.new_page(width=size[0], height=size[1])
                page.show_pdf_page(page.rect, source_pdf, 0, clip=viewport)
                result.set_metadata({
                    "title": stem,
                    "subject": "Vector export of unchanged author-supplied supplementary SVG",
                    "keywords": expected,
                    "producer": "librsvg/Cairo; TeX Libertine outlines; original SVG artboard",
                })
                result.save(output, garbage=4, deflate=True, no_new_id=True)
    verify(output, expected, size)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        with fitz.open(output) as document:
            # Approximately 300 dpi at the manuscript's 182.1 mm figure width.
            zoom = (182.1 / 25.4 * 300) / size[0]
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.set_dpi(300, 300)
            pixmap.save(preview_dir / f"{stem}.png")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.check and args.preview_dir:
        parser.error("--check is read-only; omit --preview-dir")
    for stem in SOURCES:
        output = export_one(stem, check=args.check, preview_dir=args.preview_dir)
        print(f"{'Verified' if args.check else 'Exported'} {output.name}")


if __name__ == "__main__":
    main()
