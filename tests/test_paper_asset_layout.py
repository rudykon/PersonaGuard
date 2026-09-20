"""Keep manuscript inputs separate from editable/reproducibility materials."""

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from scripts import build_revision6_publication as builder


ROOT = Path(__file__).resolve().parents[1]


class PaperAssetLayoutTests(unittest.TestCase):
    def test_json_inputs_are_outside_manuscript_directory(self):
        self.assertEqual(list((ROOT / "paper").rglob("*.json")), [])
        required = (
            "configs/protocol/adjacent_frameworks.json",
            "configs/protocol/protocol_rules.json",
            "configs/protocol/protocol_replay_cases.json",
            "configs/protocol/external_reuse_cases.json",
            "results/evidence_traceability.json",
            "results/revision6_source.json",
            "results/statistical_analysis_registry.json",
            "scripts/figures/dense_trajectory_examples.json",
        )
        for relative in required:
            path = ROOT / relative
            self.assertTrue(path.is_file(), path)
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)
        self.assertFalse((ROOT / "paper_support").exists())

    def test_plotting_code_is_in_scripts_directory(self):
        for name in (
            "make_revision6_protocol_figure.py", "make_revision5_figures.py",
            "make_dense_trajectory_examples.py", "make_supplementary_figures.py",
            "make_revision6_decision_figures.py", "make_revision6_validity_figure.py",
            "make_revision3_figures.py", "make_revision4_figures.py",
            "export_current_protocol_svgs.py", "export_current_supplementary_svgs.py",
            "manuscript_fonts.py", "vector_export.py",
        ):
            path = ROOT / "scripts" / "figures" / name
            self.assertTrue(path.is_file(), path)

    def test_figure_data_lives_with_generating_code(self):
        self.assertEqual(len(list((ROOT / "scripts" / "figures").glob("source_data_*.csv"))), 14)

    def test_figure_directory_matches_actual_tex_references(self):
        expected = builder.cited_figure_names() | {f"{stem}.svg" for stem in builder.CUSTOM_SVG_STEMS}
        paths = list((ROOT / "paper" / "figures").iterdir())
        self.assertEqual({path.name for path in paths}, expected)
        self.assertTrue(all(path.is_file() and path.suffix in {".pdf", ".svg"} for path in paths))

    def test_editable_svg_sources_remain_available(self):
        for stem in builder.CUSTOM_SVG_STEMS:
            source = ROOT / "paper" / "figures" / f"{stem}.svg"
            self.assertTrue(source.is_file(), source)
            self.assertTrue(ET.parse(source).getroot().tag.endswith("svg"), source)


if __name__ == "__main__":
    unittest.main()
