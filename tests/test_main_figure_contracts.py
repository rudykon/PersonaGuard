"""Guard the source-to-figure mappings without rerunning scientific analyses."""
from collections import Counter
import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def rows(stem):
    with (ROOT / "scripts/figures" / f"source_data_{stem}.csv").open() as handle:
        return list(csv.DictReader(handle))


class MainFigureContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((ROOT / "results/revision6_source.json").read_text())
        cls.protocol = cls.source["summaries"]["protocol_replay"]

    def test_action_counts_preserve_both_record_sets(self):
        data = rows("protocol_evaluation")
        for group, key in (("worked", "worked_case_results"),
                           ("external", "external_case_results")):
            expected = Counter(r["audited_action"] for r in self.protocol[key])
            actual = {r["action"]: int(r["value"]) for r in data
                      if r["kind"] == "action_count" and r["case_set"] == group}
            self.assertEqual(sum(actual.values()), len(self.protocol[key]))
            for action, count in actual.items():
                self.assertEqual(count, expected[action])

    def test_structural_responses_use_frozen_counts(self):
        data = [r for r in rows("protocol_evaluation") if r["panel"] == "C"]
        self.assertEqual(len(data), 5)
        validation = self.protocol["validation"]
        pairs = [
            ("order_permutations", "order_permutations"),
            ("invalid_mutations_rejected", "invalid_mutations"),
            ("route_locality_probes_passed", "route_locality_probes"),
            ("permission_boundary_probes_passed", "permission_boundary_probes"),
            ("precedence_probes_passed", "precedence_probes"),
        ]
        self.assertTrue(validation["order_invariant"])
        self.assertEqual([(int(r["value"]), int(r["denominator"])) for r in data],
                         [(validation[a], validation[b]) for a, b in pairs])

    def test_dense_analyses_do_not_acquire_formal_route_links(self):
        data = rows("evaluation_design")
        formal = [r for r in data if r["panel"] == "C"]
        self.assertEqual(len(formal), 4)
        self.assertEqual({r["unit"] for r in formal},
                         {"Trace interpretation", "Optional profile sensing",
                          "Signed calibration", "Retention / transfer"})
        dense = [r for r in data if "additional dense" in r["evaluation_lane"]]
        self.assertEqual(len(dense), 1)
        self.assertIn("additional resolved audit routes", dense[0]["does_not_establish"])
        self.assertIn("known video", dense[0]["probe"])

    def test_measurement_figure_uses_separate_unjittered_axes(self):
        script = (ROOT / "scripts/figures/make_revision5_figures.py").read_text()
        function = script.split("def figure_estimator_reference_actionability(", 1)[1]
        function = function.split("\ndef main()", 1)[0]
        self.assertNotIn("twinx", function)
        self.assertNotIn("display_jitter", function)
        data = rows("estimator_reference_actionability")
        self.assertEqual(sum(r["panel"] == "A" for r in data), 5)
        self.assertEqual(sum(r["panel"] == "B" for r in data), 4)
        self.assertEqual(sum(r["panel"] == "C" for r in data), 6)
        self.assertEqual(sum(r["panel"] == "D" for r in data), 4)


if __name__ == "__main__":
    unittest.main()
