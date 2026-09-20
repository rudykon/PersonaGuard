"""Export the current author-supplied Figure 1/2 SVGs as font-free vector PDFs.

SVG originals remain editable and unchanged. Embedded SVG glyphs are used
where supplied; Figure 1's external Libertine font comes from TeX Live.
"""
from __future__ import annotations

import argparse
import hashlib
import ctypes as C
from ctypes.util import find_library
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import fitz
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

FIGURES = Path(__file__).resolve().parents[2] / "paper" / "figures"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
SOURCES = (
    "Evidence-to-Action_Audit_Protocol",
    "Protocol_Evaluation_refined_source",
)


def tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def styles(element: ET.Element) -> dict[str, str]:
    return dict(part.strip().split(":", 1) for part in
                element.get("style", "").split(";") if ":" in part)


def tex_font(bold: bool) -> Path:
    name = "LinLibertine_RB.otf" if bold else "LinLibertine_R.otf"
    result = subprocess.check_output(["kpsewhich", name], text=True).strip()
    if not result or not Path(result).is_file():
        raise RuntimeError(f"Required original font unavailable: {name}")
    return Path(result)


def signature(source: Path, fonts: dict[bool, Path]) -> str:
    parts = [source.read_bytes(), Path(__file__).read_bytes()]
    parts.extend(path.read_bytes() for _, path in sorted(fonts.items()))
    return hashlib.sha256(b"\0".join(parts)).hexdigest()


def path_data(path: MplPath) -> str:
    commands = {MplPath.MOVETO: "M", MplPath.LINETO: "L",
                MplPath.CURVE3: "Q", MplPath.CURVE4: "C"}
    result = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            result.append("Z")
        elif code in commands:
            result.append(commands[code] + " ".join(f"{v:.8g}" for v in vertices))
        else:
            raise RuntimeError(f"Unexpected glyph path command: {code}")
    return " ".join(result)


def outline(source: Path, fonts: dict[bool, Path]) -> bytes:
    root = ET.parse(source).getroot()
    embedded = {}
    for font in root.iter(tag("font")):
        face = font.find(tag("font-face"))
        if face is None:
            raise RuntimeError("Embedded font has no font-face")
        embedded[face.get("font-family")] = (
            float(face.get("units-per-em", "1000")),
            float(font.get("horiz-adv-x", "1000")),
            {g.get("unicode"): g for g in font.findall(tag("glyph"))},
        )
    parents = {child: parent for parent in root.iter() for child in parent}
    for text in list(root.iter(tag("text"))):
        if list(text):
            raise RuntimeError("Nested SVG text requires explicit layout handling")
        content = text.text or ""
        style = styles(text)
        family = style["font-family"].strip("'\"")
        size = float(style["font-size"].removesuffix("px"))
        group = ET.Element(tag("g"), {k: v for k, v in text.attrib.items()
                                     if k not in {"style", "x", "y"}})
        group.set("style", ";".join(f"{k}:{v}" for k, v in style.items()
                                   if not k.startswith("font-")))
        if text.get("x") or text.get("y"):
            raise RuntimeError("Unexpected text positioning outside source transform")
        if family in embedded:
            em, default_advance, glyphs = embedded[family]
            run = ET.SubElement(group, tag("g"), {"transform": f"scale({size/em}, {-size/em})"})
            x = 0.0
            for char in content:
                if char not in glyphs:
                    raise RuntimeError(f"Missing embedded glyph {char!r} in {family}")
                glyph = glyphs[char]
                if glyph.get("d"):
                    ET.SubElement(run, tag("path"), {"d": glyph.get("d"), "transform": f"translate({x},0)"})
                x += float(glyph.get("horiz-adv-x", str(default_advance)))
        elif family == "Linux Libertine O":
            font = fonts[style.get("font-weight") == "bold"]
            cmap = FT2Font(str(font)).get_charmap()
            missing = {char for char in content if ord(char) not in cmap}
            if missing:
                raise RuntimeError(f"Missing original-font glyphs: {missing}")
            path = TextPath((0, 0), content, size=size, prop=FontProperties(fname=str(font)))
            ET.SubElement(group, tag("path"), {"d": path_data(path), "transform": "scale(1,-1)"})
        else:
            raise RuntimeError(f"Unexpected font family: {family}")
        parent = parents[text]
        position = list(parent).index(text)
        parent.remove(text)
        parent.insert(position, group)
    for font in list(root.iter(tag("font"))):
        parents[font].remove(font)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def verify(pdf: Path, expected_signature: str) -> None:
    if not pdf.is_file():
        raise RuntimeError(f"Missing SVG export: {pdf.name}")
    with fitz.open(pdf) as document:
        if len(document) != 1 or document.metadata.get("keywords") != expected_signature:
            raise RuntimeError(f"Stale SVG export: {pdf.name}")
        page = document[0]
        if page.get_fonts(full=True) or page.get_images(full=True):
            raise RuntimeError(f"Expected font-free vector artwork: {pdf.name}")
        if abs(page.rect.width / page.rect.height - 1.5) > 1e-6:
            raise RuntimeError(f"SVG aspect ratio changed: {pdf.name}")


def render_svg(source: Path, output: Path) -> None:
    """Use installed librsvg/Cairo through Python, retaining vector gradients."""
    def library(name: str):
        found = find_library(name)
        if not found:
            raise RuntimeError(f"Required SVG rendering library missing: {name}")
        return C.CDLL(found)

    rsvg, cairo, gobject, glib = map(library, ("rsvg-2", "cairo", "gobject-2.0", "glib-2.0"))
    class Rectangle(C.Structure):
        _fields_ = [(name, C.c_double) for name in ("x", "y", "width", "height")]
    class Error(C.Structure):
        _fields_ = [("domain", C.c_uint), ("code", C.c_int), ("message", C.c_char_p)]
    error = C.POINTER(Error)()
    def bind(lib, name, arguments, result):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = arguments, result
        return fn
    new = bind(rsvg, "rsvg_handle_new_from_file", [C.c_char_p, C.POINTER(C.POINTER(Error))], C.c_void_p)
    render = bind(rsvg, "rsvg_handle_render_document", [C.c_void_p, C.c_void_p, C.POINTER(Rectangle), C.POINTER(C.POINTER(Error))], C.c_int)
    pdf_surface = bind(cairo, "cairo_pdf_surface_create", [C.c_char_p, C.c_double, C.c_double], C.c_void_p)
    create = bind(cairo, "cairo_create", [C.c_void_p], C.c_void_p)
    status = bind(cairo, "cairo_status", [C.c_void_p], C.c_int)
    surface_status = bind(cairo, "cairo_surface_status", [C.c_void_p], C.c_int)
    finish = bind(cairo, "cairo_surface_finish", [C.c_void_p], None)
    destroy = bind(cairo, "cairo_destroy", [C.c_void_p], None)
    destroy_surface = bind(cairo, "cairo_surface_destroy", [C.c_void_p], None)
    unref = bind(gobject, "g_object_unref", [C.c_void_p], None)
    free_error = bind(glib, "g_error_free", [C.POINTER(Error)], None)
    handle = new(str(source).encode(), C.byref(error))
    surface = context = None
    try:
        if not handle:
            raise RuntimeError(error.contents.message.decode() if error else "Cannot load SVG")
        surface = pdf_surface(str(output).encode(), 1536, 1024)
        context = create(surface)
        if not render(handle, context, C.byref(Rectangle(0, 0, 1536, 1024)), C.byref(error)):
            raise RuntimeError(error.contents.message.decode() if error else "Cannot render SVG")
        finish(surface)
        if status(context) or surface_status(surface):
            raise RuntimeError("Cairo reported a PDF export error")
    finally:
        if context:
            destroy(context)
        if surface:
            destroy_surface(surface)
        if handle:
            unref(handle)
        if error:
            free_error(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fonts = {bold: tex_font(bold) for bold in (False, True)}
    for stem in SOURCES:
        source = FIGURES / f"{stem}.svg"
        output = FIGURES / f"{stem}_embed.pdf"
        expected = signature(source, fonts)
        if not args.check:
            with TemporaryDirectory(prefix="chi2027-svg-export-") as temporary:
                path = Path(temporary) / "outlined.svg"
                path.write_bytes(outline(source, fonts))
                rendered = Path(temporary) / "rendered.pdf"
                render_svg(path, rendered)
                with fitz.open(rendered) as pdf:
                    pdf.set_metadata({"title": stem, "subject": "Vector export of the current source SVG",
                                      "keywords": expected, "producer": "librsvg/Cairo via Python; original SVG glyphs / TeX Libertine"})
                    pdf.save(output, garbage=4, deflate=True, no_new_id=True)
        verify(output, expected)
        print(f"{'Verified' if args.check else 'Exported'} {output.relative_to(FIGURES.parent)}")


if __name__ == "__main__":
    main()
