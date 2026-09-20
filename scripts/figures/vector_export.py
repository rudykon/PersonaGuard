"""Shared publication-vector export helpers.

Matplotlib's editable TrueType PDFs are retained as the figure masters. When
LuaLaTeX imports those PDFs into a tagged PDF/UA document, its PDF importer can
introduce references to the ``.notdef`` glyph. The submission-facing copy is
therefore produced from a temporary path-only SVG. PyMuPDF preserves the
geometry as vector paths, while the editable PDF and text-bearing SVG remain
available beside it.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import matplotlib as mpl
from matplotlib.figure import Figure


def save_pdfua_embed(
    fig: Figure,
    output: Path,
    savefig_kwargs: dict[str, object],
) -> None:
    """Write a font-free vector PDF for inclusion in a tagged manuscript."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="chi2027-vector-") as temporary:
        path_svg = Path(temporary) / "figure-paths.svg"
        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(path_svg, format="svg", **savefig_kwargs)
        with fitz.open(path_svg) as svg_document:
            pdf_bytes = svg_document.convert_to_pdf()
        with fitz.open("pdf", pdf_bytes) as pdf_document:
            pdf_document.save(output, garbage=4, deflate=True)

    with fitz.open(output) as audit_document:
        raster_images = sum(
            len(page.get_images(full=True)) for page in audit_document
        )
        font_references = sum(
            len(page.get_fonts(full=True)) for page in audit_document
        )
    if raster_images or font_references:
        raise RuntimeError(
            f"PDF/UA embed must remain font-free vector artwork: {output} "
            f"(images={raster_images}, fonts={font_references})"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Empty PDF/UA vector export: {output}")
