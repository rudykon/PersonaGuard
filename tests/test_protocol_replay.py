from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_protocol_replay.py"
SPEC = importlib.util.spec_from_file_location("run_protocol_replay_for_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load protocol replay runner")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProtocolReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(
            (ROOT / "artifacts" / "revision6" / "protocol_replay.json").read_text(
                encoding="utf-8"
            )
        )

    def test_committed_result_is_deterministic_and_current(self):
        self.assertEqual(self.result, MODULE.build_result())
        validation = self.result["validation"]
        self.assertTrue(validation["order_invariant"])
        self.assertEqual(validation["order_permutations"], 100)
        self.assertEqual(
            validation["invalid_mutations_rejected"],
            validation["invalid_mutations"],
        )
        self.assertEqual(
            validation["route_locality_probes_passed"],
            validation["route_locality_probes"],
        )
        self.assertEqual(validation["permission_boundary_probes"], 30)
        self.assertEqual(
            validation["permission_boundary_probes_passed"],
            validation["permission_boundary_probes"],
        )
        self.assertEqual(validation["precedence_probes"], 4)
        self.assertEqual(
            validation["precedence_probes_passed"],
            validation["precedence_probes"],
        )
        self.assertEqual(validation["combined_distinct_actions"], 5)

    def test_replay_produces_distinct_non_tautological_actions(self):
        results = self.result["case_results"]
        self.assertEqual(len(results), 4)
        self.assertEqual(len({row["audited_action"] for row in results}), 4)
        self.assertTrue(all(row["decision_changed"] for row in results))
        actions = {row["audited_action"] for row in results}
        self.assertIn("BOUND_TO_EVIDENCE_SCOPE", actions)
        self.assertIn("RETAIN_EVALUATED_COMPARATOR", actions)
        self.assertIn("RUN_PREREGISTERED_USER_STUDY", actions)
        self.assertIn("WITHHOLD_RETENTION_AND_TRANSFER", actions)

    def test_external_cases_use_same_schema_and_include_positive_routes(self):
        transfer = self.result["external_transfer"]
        self.assertEqual(transfer["source_count"], 6)
        self.assertEqual(transfer["case_count"], 7)
        self.assertTrue(transfer["represented_without_schema_extension"])
        self.assertEqual(transfer["fallback_actions"], 0)
        self.assertEqual(transfer["positive_bounded_actions"], 5)
        actions = {
            row["audited_action"] for row in self.result["external_case_results"]
        }
        self.assertIn("PROCEED_WITHIN_EVALUATED_BOUNDARY", actions)
        self.assertIn("RETAIN_EVALUATED_COMPARATOR", actions)
        self.assertIn("RUN_PREREGISTERED_USER_STUDY", actions)

    def test_route_inputs_do_not_encode_expected_actions_or_rule_ids(self):
        for filename in ("protocol_replay_cases.json", "external_reuse_cases.json"):
            specification = json.loads(
                (ROOT / "configs" / "protocol" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(specification["schema_version"], "route-audit-cases-v2")
            for case in specification["cases"]:
                self.assertNotIn("expected_action", case)
                self.assertNotIn("audited_action", case)
                self.assertNotIn("rule_id", case)

    def test_each_normative_rule_has_sources_and_material_ablation(self):
        inventory = {row["id"]: row for row in self.result["rule_inventory"]}
        ablations = {row["rule_id"]: row for row in self.result["rule_ablation"]}
        self.assertEqual(len(inventory), 5)
        self.assertEqual(set(inventory), set(ablations))
        for rule_id, rule in inventory.items():
            self.assertTrue(rule["derivation_sources"], rule_id)
            self.assertGreater(ablations[rule_id]["cases_changed"], 0, rule_id)

    def test_positive_permission_requires_capability_and_control_or_reversibility(self):
        protocol = json.loads(
            (ROOT / "configs" / "protocol" / "protocol_rules.json").read_text(encoding="utf-8")
        )
        external = json.loads(
            (ROOT / "configs" / "protocol" / "external_reuse_cases.json").read_text(
                encoding="utf-8"
            )
        )
        positive = next(
            case
            for case in external["cases"]
            if MODULE.evaluate_case(case, protocol)["audited_action"]
            == "PROCEED_WITHIN_EVALUATED_BOUNDARY"
        )
        unavailable = copy.deepcopy(positive)
        unavailable["route_capability"] = "NOT_AVAILABLE"
        self.assertNotEqual(
            MODULE.evaluate_case(unavailable, protocol)["audited_action"],
            "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        )
        uncontrolled = copy.deepcopy(positive)
        uncontrolled["evidence"]["user_control"] = "NOT_TESTED"
        uncontrolled["evidence"]["safety_reversibility"] = "NOT_TESTED"
        self.assertNotEqual(
            MODULE.evaluate_case(uncontrolled, protocol)["audited_action"],
            "PROCEED_WITHIN_EVALUATED_BOUNDARY",
        )

    def test_every_declared_evidence_field_is_used_by_a_rule(self):
        protocol = json.loads(
            (ROOT / "configs" / "protocol" / "protocol_rules.json").read_text(encoding="utf-8")
        )
        used = set()
        for rule in protocol["rules"]:
            used.update(MODULE.predicate_fields(rule["predicate"]))
        declared = {
            f"evidence.{field}" for field in protocol["evidence_fields"]
        }
        self.assertEqual(declared - used, set())

    def test_machine_scope_disclaims_substantive_validity(self):
        unchecked = " ".join(self.result["not_machine_checked"]).lower()
        self.assertIn("substantive", unchecked)
        self.assertIn("human analyst agreement", unchecked)
        self.assertIn("fairness", unchecked)
        boundary = self.result["interpretation_boundary"].lower()
        self.assertIn("not an independent analyst study", boundary)

    def test_supplied_case_set_entrypoint_reuses_frozen_resolver(self):
        resolved = MODULE.resolve_case_set(
            ROOT / "configs" / "protocol" / "protocol_replay_cases.json",
            ROOT / "configs" / "protocol" / "protocol_rules.json",
            external=False,
        )
        self.assertEqual(
            resolved["case_results"],
            self.result["worked_case_results"],
        )
        self.assertEqual(
            resolved["schema_version"],
            "protocol-case-set-resolution-v1",
        )
        self.assertEqual(len(resolved["protocol_sha256"]), 64)
        self.assertEqual(len(resolved["case_set_sha256"]), 64)
        unchecked = " ".join(resolved["not_machine_checked"]).lower()
        self.assertIn("substantive validity", unchecked)
        self.assertIn("human analyst agreement", unchecked)


if __name__ == "__main__":
    unittest.main()
