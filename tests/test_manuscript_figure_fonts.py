"""Guard the five requested figure fonts without redrawing scientific figures."""

import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import MathTextParser


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "scripts" / "figures"
SPEC = importlib.util.spec_from_file_location(
    "manuscript_fonts_under_test", FIGURES / "manuscript_fonts.py"
)
FONTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FONTS)

TARGETS = {
    "make_revision6_protocol_figure.py": "figure_protocol_evaluation",
    "make_revision5_figures.py": "figure_estimator_reference_actionability",
    "make_dense_trajectory_examples.py": "make_figure",
    "make_supplementary_figures.py": "figure_algorithms",
    "make_revision6_decision_figures.py": "main",
}
VARIANTS = {
    "regular": ("normal", "normal"),
    "bold": ("bold", "normal"),
    "italic": ("normal", "italic"),
    "bold_italic": ("bold", "italic"),
}
MATH_SLOTS = ("rm", "it", "bf", "bfit", "sf", "tt", "cal")


def font_family_names(value):
    """Accept either form supported by Matplotlib's family configuration."""
    return [value] if isinstance(value, str) else list(value)


def uses_font_decorator(decorator):
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (
        isinstance(decorator, ast.Name)
        and decorator.id == "with_manuscript_fonts"
    ) or (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "with_manuscript_fonts"
    )


class ManuscriptFigureFontsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = FONTS.font_paths()
        cls.settings = FONTS.font_settings()

    def test_four_real_opentype_faces_match_manuscript_family(self):
        self.assertEqual(FONTS.FONT_FAMILY, "Linux Libertine O")
        self.assertEqual(set(self.paths), set(VARIANTS))
        self.assertEqual(len({path.resolve() for path in self.paths.values()}), 4)
        for variant, path in self.paths.items():
            with self.subTest(variant=variant):
                self.assertIsInstance(path, Path)
                self.assertTrue(path.is_file(), path)
                self.assertEqual(path.suffix.lower(), ".otf")
                face = FT2Font(str(path))
                self.assertEqual(face.family_name, FONTS.FONT_FAMILY)

    def test_registered_regular_bold_italic_and_bold_italic_resolve_exactly(self):
        for variant, (weight, style) in VARIANTS.items():
            with self.subTest(variant=variant):
                properties = font_manager.FontProperties(
                    family=FONTS.FONT_FAMILY, weight=weight, style=style
                )
                resolved = Path(font_manager.findfont(
                    properties, fallback_to_default=False
                )).resolve()
                self.assertEqual(resolved, self.paths[variant].resolve())

    def test_threshold_and_axis_symbols_exist_in_every_font_face(self):
        for variant, path in self.paths.items():
            face = FT2Font(str(path))
            characters = face.get_charmap()
            for symbol in "δ≥≤−":
                with self.subTest(variant=variant, symbol=symbol):
                    self.assertIn(ord(symbol), characters)

    def test_text_and_every_math_slot_use_libertine_without_fallback(self):
        self.assertIn(
            FONTS.FONT_FAMILY,
            font_family_names(self.settings["font.family"]),
        )
        self.assertEqual(self.settings["mathtext.fontset"], "custom")
        self.assertIsNone(self.settings["mathtext.fallback"])
        for slot in MATH_SLOTS:
            with self.subTest(slot=slot):
                pattern = self.settings[f"mathtext.{slot}"]
                properties = font_manager.FontProperties(pattern)
                self.assertIn(FONTS.FONT_FAMILY, properties.get_family())
                resolved = font_manager.findfont(
                    properties, fallback_to_default=False
                )
                self.assertEqual(FT2Font(resolved).family_name, FONTS.FONT_FAMILY)

    def test_figure_four_math_label_has_no_dejavu_glyphs(self):
        with mpl.rc_context(FONTS.font_settings()):
            parsed = MathTextParser("path").parse(
                r"$\lambda_s=0$", prop=font_manager.FontProperties(size=9)
            )
        self.assertTrue(parsed.glyphs)
        for glyph in parsed.glyphs:
            # Matplotlib versions may append fields to the glyph tuple.
            self.assertEqual(glyph[0].family_name, FONTS.FONT_FAMILY)

    def test_decorator_applies_fonts_and_preserves_metadata_and_return_value(self):
        @FONTS.with_manuscript_fonts
        def figure_probe(value):
            """A rendering-free probe of the figure-local font context."""
            self.assertIn(
                FONTS.FONT_FAMILY, font_family_names(mpl.rcParams["font.family"])
            )
            self.assertEqual(mpl.rcParams["mathtext.fontset"], "custom")
            self.assertIsNone(mpl.rcParams["mathtext.fallback"])
            return value

        marker = object()
        with mpl.rc_context({
            "font.family": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "mathtext.fallback": "stix",
        }):
            before = {key: copy.deepcopy(mpl.rcParams[key]) for key in self.settings}
            self.assertIs(figure_probe(marker), marker)
            after = {key: mpl.rcParams[key] for key in self.settings}
            self.assertEqual(after, before)
        self.assertEqual(figure_probe.__name__, "figure_probe")
        self.assertEqual(figure_probe.__doc__, figure_probe.__wrapped__.__doc__)

    def test_decorator_restores_caller_fonts_after_failure(self):
        @FONTS.with_manuscript_fonts
        def failing_probe():
            self.assertIn(
                FONTS.FONT_FAMILY, font_family_names(mpl.rcParams["font.family"])
            )
            raise RuntimeError("font-context-test")

        with mpl.rc_context({
            "font.family": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "mathtext.fallback": "stix",
        }):
            before = {key: copy.deepcopy(mpl.rcParams[key]) for key in self.settings}
            with self.assertRaisesRegex(RuntimeError, "font-context-test"):
                failing_probe()
            after = {key: mpl.rcParams[key] for key in self.settings}
            self.assertEqual(after, before)

    def test_only_the_five_requested_rendering_entries_are_decorated(self):
        for filename, expected_function in TARGETS.items():
            with self.subTest(script=filename):
                tree = ast.parse((FIGURES / filename).read_text(encoding="utf-8"))
                decorated = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and any(uses_font_decorator(d) for d in node.decorator_list)
                }
                self.assertEqual(decorated, {expected_function})


if __name__ == "__main__":
    unittest.main()
