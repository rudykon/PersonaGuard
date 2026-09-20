from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Revision6PublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_module(
            "build_revision6_source_for_test",
            ROOT / "scripts" / "build_revision6_source.py",
        )
        cls.generator = load_module(
            "generate_revision6_publication_for_test",
            ROOT / "scripts" / "generate_revision6_publication.py",
        )
        cls.source_path = ROOT / "results" / "revision6_source.json"
        cls.source = json.loads(cls.source_path.read_text(encoding="utf-8"))

    def test_revision6_is_unique_canonical_source(self) -> None:
        self.assertEqual(
            self.source["schema_version"], "revision6-publication-source-v1"
        )
        self.assertEqual(self.source["revision"], 6)
        self.assertEqual(
            self.source["policy"]["single_numeric_source"],
            "results/revision6_source.json",
        )
        self.assertEqual(self.source, self.builder.build_source())
        self.assertIn("protocol_rules", self.source["summaries"])
        self.assertIn("external_reuse_cases", self.source["summaries"])
        self.assertIn("adjacent_frameworks", self.source["summaries"])
        self.assertIn("algorithm_experiments", self.source["summaries"])
        algorithms = self.source["summaries"]["algorithm_experiments"]
        self.assertEqual(
            algorithms["schema_version"],
            "merps-algorithm-experiment-summary-v2",
        )
        self.assertTrue(
            algorithms["oof_artifact_consistency"]["sample_identity_consistent"]
        )

    def test_publication_source_contains_only_portable_paths(self) -> None:
        serialized = json.dumps(self.source, ensure_ascii=False)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn("/home/", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("file" + "://", serialized)
        self.assertIsNone(re.search(r'"[A-Za-z]:[\\/]', serialized))
        self.builder.assert_no_private_paths(self.source)
        self.assertIn("path_policy", self.source["policy"])

    def test_generated_consumers_are_byte_current(self) -> None:
        for path, expected in self.generator.expected_files(self.source).items():
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), expected, path)

    def test_publication_figure_scripts_do_not_read_revision_directories(self) -> None:
        paths = [
            ROOT / "scripts" / "figures" / "make_revision6_protocol_figure.py",
            ROOT / "scripts" / "figures" / "make_revision5_figures.py",
            ROOT / "scripts" / "figures" / "make_revision6_validity_figure.py",
            ROOT / "scripts" / "figures" / "make_revision6_decision_figures.py",
        ]
        forbidden = [
            "artifacts/revision2",
            "artifacts/revision3",
            "artifacts/revision4",
            "artifacts/revision5",
            "artifacts/revision6",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("revision6_source.json", text, path)
            for token in forbidden:
                self.assertNotIn(token, text, path)

    def test_manuscript_tables_use_generated_rows(self) -> None:
        main_inputs = [
            "revision6_availability_rows",
            "protocol_contract_rows",
            "protocol_replay_rows",
            "algorithm_route_rows",
        ]
        text = (ROOT / "paper" / "body.tex").read_text(encoding="utf-8")
        for stem in main_inputs:
            self.assertIn(rf"\input{{generated/{stem}}}", text)
        self.assertNotIn("revision6_signal_audit_rows", text)
        self.assertNotIn("revision6_calibration_rows", text)
        self.assertNotIn("decision_boundary_sensitivity.png", text)
        supplement = (
            ROOT / "paper" / "supplementary_information.tex"
        ).read_text(encoding="utf-8")
        self.assertIn(
            r"\input{generated/revision6_signal_audit_rows}", supplement
        )
        self.assertIn(
            r"\input{generated/revision6_sensing_rows}", supplement
        )
        self.assertIn(
            r"\input{generated/evidence_traceability_rows}", supplement
        )
        for stem in (
            "adjacent_framework_rows",
            "algorithm_content_rows",
            "algorithm_physiology_rows",
            "algorithm_sparse_rows",
            "protocol_rule_rows",
            "external_reuse_rows",
            "statistical_analysis_registry_rows",
            "revision6_calibration_rows",
        ):
            self.assertIn(rf"\input{{generated/{stem}}}", supplement)
        self.assertIn("decision_boundary_sensitivity_embed.pdf", supplement)

    def test_generated_table_fragments_close_booktabs_internally(self) -> None:
        stems = [
            "revision6_signal_audit_rows",
            "revision6_availability_rows",
            "revision6_sensing_rows",
            "revision6_calibration_rows",
            "protocol_replay_rows",
            "external_reuse_rows",
            "protocol_rule_rows",
            "protocol_contract_rows",
            "adjacent_framework_rows",
            "statistical_analysis_registry_rows",
            "algorithm_route_rows",
            "algorithm_content_rows",
            "algorithm_physiology_rows",
            "algorithm_sparse_rows",
        ]
        for stem in stems:
            text = (ROOT / "paper" / "generated" / f"{stem}.tex").read_text(
                encoding="utf-8"
            )
            self.assertEqual(text.splitlines()[-1], r"\bottomrule", stem)

    def test_main_and_si_line_art_use_vector_assets(self) -> None:
        main_stems = [
            "protocol_overview",
            "evaluation_design",
            "protocol_evaluation",
            "estimator_reference_actionability",
            "dense_trajectory_examples",
        ]
        current_svg_stems = [
            "Evidence-to-Action_Audit_Protocol",
            "Protocol_Evaluation_refined_source",
        ]
        for manuscript in ("body.tex",):
            text = (ROOT / "paper" / manuscript).read_text(encoding="utf-8")
            stems = current_svg_stems + main_stems[2:]
            for stem in stems:
                self.assertIn(f"{stem}_embed.pdf", text)
                self.assertNotIn(f"{stem}.png", text)
            self.assertNotIn("protocol_transfer_ablation", text)

        supplementary_svg_stems = [
            "Evidence_Graph",
            "Validation_Boundaries_portraits",
        ]
        supplement_stems = [
            *supplementary_svg_stems,
            "dense_algorithm_sensitivity",
            "decision_boundary_sensitivity",
        ]
        for manuscript in (
            "supplementary_information.tex",
        ):
            text = (ROOT / "paper" / manuscript).read_text(encoding="utf-8")
            for stem in supplement_stems:
                self.assertIn(f"{stem}_embed.pdf", text)
                self.assertNotIn(f"{stem}.png", text)

        for stem in [*main_stems, *supplement_stems[2:]]:
            for suffix in ("_embed.pdf",):
                path = ROOT / "paper" / "figures" / f"{stem}{suffix}"
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

        for stem in [*current_svg_stems, *supplementary_svg_stems]:
            for suffix in (".svg", "_embed.pdf"):
                path = ROOT / "paper" / "figures" / f"{stem}{suffix}"
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_generated_protocol_tables_use_short_human_rule_labels(self) -> None:
        protocol_lines = (
            ROOT / "paper" / "generated" / "protocol_rule_rows.tex"
        ).read_text(encoding="utf-8").splitlines()[:-1]
        external_lines = (
            ROOT / "paper" / "generated" / "external_reuse_rows.tex"
        ).read_text(encoding="utf-8").splitlines()[:-1]

        self.assertEqual(
            {line.split(" & ", 1)[0] for line in protocol_lines},
            {"R1", "R2", "R3", "R4", "R5"},
        )
        for line in [*protocol_lines, *external_lines]:
            self.assertRegex(line, r"^R[1-5] &|^[^&]+ & [^&]+ & R[1-5] &")

    def test_statistical_registry_is_complete_and_generated(self) -> None:
        registry = self.source["statistical_analysis_registry"]
        self.assertEqual(
            registry["schema_version"], "statistical-analysis-registry-v1"
        )
        self.assertEqual(len(registry["rows"]), 11)
        self.assertTrue(
            {
                "content_conditioned_trajectory",
                "causal_physiology_residual",
                "post_trial_sam_sparse_recovery",
            }.issubset({row["id"] for row in registry["rows"]})
        )
        required = {
            "endpoint",
            "population_unit",
            "estimator_or_test",
            "uncertainty",
            "multiplicity",
            "resampling_or_folds",
            "seed",
            "status",
        }
        for row in registry["rows"]:
            self.assertTrue(required.issubset(row), row["id"])
            self.assertTrue(all(str(row[field]).strip() for field in required), row["id"])
        generated = json.loads(
            (ROOT / "results" / "statistical_analysis_registry.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(generated, registry)

    def test_shared_reference_outputs_do_not_use_generalizability_symbols(self) -> None:
        publication_files = [
            ROOT / "paper" / "body.tex",
            ROOT / "paper" / "supplementary_information.tex",
            ROOT / "results" / "evidence_traceability.json",
        ]
        forbidden = (r"R^{\mathrm{SR}}", "R_G^SR", "R_Phi^SR")
        for path in publication_files:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, path)

    def test_readme_markers_are_unique(self) -> None:
        for path in (ROOT / "README.zh-CN.md", ROOT / "paper" / "README.md"):
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count(self.generator.START), 1, path)
            self.assertEqual(text.count(self.generator.END), 1, path)

    def test_decision_curve_contract(self) -> None:
        rows = self.source["tables"]["decision_threshold_curves"]
        analyses = {row["analysis"] for row in rows}
        self.assertEqual(
            analyses,
            {
                "sensing_increment_vs_video_mean",
                "signed_calibration_vs_video_only",
            },
        )
        for row in rows:
            proportion = float(row["proportion"])
            participants = int(row["participants"])
            count = int(row["count"])
            self.assertAlmostEqual(proportion, count / participants)
            self.assertEqual(participants, 24)
            self.assertNotIn("wilson_ci95_low", row)
            self.assertNotIn("wilson_ci95_high", row)

        figure_source = (
            ROOT / "scripts" / "figures" / "make_revision6_decision_figures.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("fill_between(", figure_source)

    def test_repeated_grouped_cv_rebuilds_reference_and_target(self) -> None:
        result = self.source["summaries"]["revision6"][
            "primary_antialias_reservation_sensing"
        ]["repeated_grouped_cv_sensitivity"]
        self.assertTrue(result["reference_and_target_rebuilt_per_partition"])
        self.assertGreaterEqual(int(result["repeats"]), 5)


if __name__ == "__main__":
    unittest.main()
