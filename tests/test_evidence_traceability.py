from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate_evidence_traceability.py"
SPEC = importlib.util.spec_from_file_location("generate_evidence_traceability", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load evidence traceability generator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvidenceTraceabilityTest(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(
            (PROJECT_ROOT / "results" / "evidence_traceability.json").read_text(
                encoding="utf-8"
            )
        )

    def test_committed_graph_is_valid(self):
        MODULE.validate_graph(self.graph)

    def test_machine_scope_is_explicitly_structural(self):
        checked = " ".join(self.graph["machine_check_scope"]["checked"]).lower()
        unchecked = " ".join(
            self.graph["machine_check_scope"]["not_checked"]
        ).lower()
        self.assertIn("sha-256", checked)
        self.assertIn("substantive", unchecked)
        self.assertIn("human analyst agreement", unchecked)

    def test_all_four_edge_types_are_exercised(self):
        observed = {edge["type"] for edge in self.graph["edges"]}
        self.assertEqual(observed, MODULE.ALLOWED_EDGE_TYPES)

    def test_partial_node_cannot_license_user_benefit(self):
        invalid = json.loads(json.dumps(self.graph))
        invalid["nodes"][0]["permits"].append("user_benefit")
        with self.assertRaises(ValueError):
            MODULE.validate_graph(invalid)
    def test_graph_shape_and_empirical_library_node(self):
        self.assertEqual(len(self.graph["nodes"]), 11)
        self.assertEqual(len(self.graph["edges"]), 14)
        ids = {node["id"] for node in self.graph["nodes"]}
        self.assertIn("empirical_library_resampling_stability", ids)
        self.assertIn("reference_representativeness_equity", ids)
        self.assertNotIn("within_session_generalizability", ids)

    def test_unassessed_reference_equity_blocks_retention(self):
        self.assertIn(
            {
                "source": "reference_representativeness_equity",
                "target": "retention_and_transfer",
                "type": "blocks_deployment",
            },
            self.graph["edges"],
        )

    def test_every_node_has_lifecycle_and_registered_evidence(self):
        registered = {item["id"] for item in self.graph["artifact_registry"]}
        for node in self.graph["nodes"]:
            self.assertTrue(MODULE.REQUIRED_NODE_LIFECYCLE_FIELDS.issubset(node))
            self.assertTrue(node["status_rationale"])
            self.assertTrue(node["decision_owner_role"])
            self.assertTrue(node["review_trigger"])
            self.assertTrue(set(node["evidence_artifact_ids"]).issubset(registered))

    def test_artifact_hash_mismatch_is_rejected(self):
        invalid = json.loads(json.dumps(self.graph))
        invalid["artifact_registry"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            MODULE.validate_graph(invalid)

    def test_unknown_artifact_reference_is_rejected(self):
        invalid = json.loads(json.dumps(self.graph))
        invalid["nodes"][0]["evidence_artifact_ids"].append("missing_artifact")
        with self.assertRaises(ValueError):
            MODULE.validate_graph(invalid)

    def test_generated_rows_match_schema(self):
        self.assertEqual(
            MODULE.table_rows(self.graph, "en"),
            (PROJECT_ROOT / "paper" / "generated" / "evidence_traceability_rows.tex").read_text(
                encoding="utf-8"
            ),
        )

    def test_generated_rows_close_booktabs_before_input_file_hooks(self):
        for language in ("en", "zh"):
            generated = MODULE.table_rows(self.graph, language)
            self.assertTrue(
                generated.endswith(r"\bottomrule" + "\n")
            )


if __name__ == "__main__":
    unittest.main()
