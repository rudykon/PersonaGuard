from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_revision6_analyses.py"
SPEC = importlib.util.spec_from_file_location("run_revision6_analyses", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load run_revision6_analyses.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Revision6AnalysisTest(unittest.TestCase):
    def test_calibration_videos_are_the_exact_evaluation_complement(self):
        rows = [{"video": str(video)} for video in range(2, 16)]
        self.assertEqual(MODULE.calibration_videos(rows, 1), (1,))

    def test_residual_offset_uses_requested_matched_video_set(self):
        lookup = {}
        for subject in (1, 2):
            for video in range(1, 16):
                lookup[(0, subject, video)] = (
                    float(subject * video),
                    float(video),
                )
        first = MODULE.residual_offset(
            lookup, fold=0, subject=2, calibration=(1, 2), shrinkage=0.0
        )
        second = MODULE.residual_offset(
            lookup, fold=0, subject=2, calibration=(14, 15), shrinkage=0.0
        )
        self.assertAlmostEqual(first, 1.5)
        self.assertAlmostEqual(second, 14.5)
        self.assertNotEqual(first, second)

    def test_blocked_permutation_detects_large_category_pattern(self):
        matrix = np.tile(np.asarray([0.4, 0.2, 0.0, -0.2, -0.4]), (24, 1))
        result = MODULE.blocked_category_permutation(
            matrix, repeats=999, seed=11
        )
        self.assertLess(result["participant_blocked_permutation_p"], 0.01)

    def test_blocked_permutation_is_one_for_identical_categories(self):
        matrix = np.tile(np.asarray([0.1] * 5), (24, 1))
        result = MODULE.blocked_category_permutation(
            matrix, repeats=99, seed=7
        )
        self.assertAlmostEqual(result["participant_blocked_permutation_p"], 1.0)

    def test_holm_adjustment_is_bounded_and_monotone_in_sorted_order(self):
        raw = [0.03, 0.001, 0.2, 0.04]
        adjusted = MODULE.holm_adjust(raw)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        order = np.argsort(raw)
        ordered = [adjusted[int(index)] for index in order]
        self.assertTrue(all(a <= b for a, b in zip(ordered, ordered[1:])))
        self.assertTrue(all(a >= b for a, b in zip(adjusted, raw)))

    def test_combined_eeg_name_and_dimension_are_explicit(self):
        metadata = MODULE.MODEL_METADATA["eeg_only"]
        self.assertEqual(metadata["label"], "Combined EEG")
        self.assertEqual(metadata["dimension"], 980)
        self.assertEqual(MODULE.MODEL_METADATA["fnirs_only"]["dimension"], 1260)

    def test_model_table_covers_every_evaluated_method(self):
        rows = MODULE.sensing_model_rows()
        self.assertEqual([row["method"] for row in rows], list(MODULE.METHODS))
        self.assertNotIn("subject", {key for row in rows for key in row})


    def test_threshold_decision_curve_reports_benefit_and_degradation(self):
        rows = MODULE.threshold_decision_rows(
            [-0.10, 0.00, 0.20],
            analysis="test",
            route="test",
            unit="units",
            thresholds=(0.05,),
        )
        counts = {row["direction"]: row["count"] for row in rows}
        self.assertEqual(counts["meets_gain_threshold"], 1)
        self.assertEqual(counts["exceeds_degradation_threshold"], 1)
        self.assertTrue(
            all(0.0 <= row["wilson_ci95_low"] <= row["wilson_ci95_high"] <= 1.0 for row in rows)
        )

    def test_repeated_participant_fold_plans_are_grouped_and_complete(self):
        primary = MODULE.participant_fold_plan(0, seed=17)
        shuffled = MODULE.participant_fold_plan(1, seed=17)
        for plan in (primary, shuffled):
            held_out = []
            for training, validation in plan:
                self.assertEqual(np.intersect1d(training, validation).size, 0)
                held_out.extend(int(value) for value in validation)
            self.assertEqual(sorted(held_out), list(range(1, 25)))
        self.assertNotEqual(
            MODULE.fold_plan_digest(primary),
            MODULE.fold_plan_digest(shuffled),
        )
        self.assertEqual(
            MODULE.fold_plan_digest(shuffled),
            MODULE.fold_plan_digest(MODULE.participant_fold_plan(1, seed=17)),
        )


if __name__ == "__main__":
    unittest.main()
