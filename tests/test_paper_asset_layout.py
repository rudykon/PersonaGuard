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
            "adjacent_frameworks", "dense_trajectory_examples",
            "evidence_traceability", "external_reuse_cases",
            "pdfua_validation_report", "protocol_replay_cases",
            "protocol_rules", "reference_verification_2026-09-11",
            "revision6_source", "statistical_analysis_registry",
            "submission_readiness",
        )
        for stem in required:
            path = ROOT / "paper_support" / f"{stem}.json"
            self.assertTrue(path.is_file(), path)
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_figure_directory_matches_actual_tex_references(self):
        expected = builder.cited_figure_names()
        paths = list((ROOT / "paper" / "figures").iterdir())
        self.assertEqual({path.name for path in paths}, expected)
        self.assertTrue(all(path.is_file() and path.suffix == ".pdf" for path in paths))

    def test_editable_svg_sources_remain_available(self):
        for name in builder.cited_figure_names():
            source = ROOT / "paper_support" / "figures" / name.replace("_embed.pdf", ".svg")
            self.assertTrue(source.is_file(), source)
            self.assertTrue(ET.parse(source).getroot().tag.endswith("svg"), source)


if __name__ == "__main__":
    unittest.main()
