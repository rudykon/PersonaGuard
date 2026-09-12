from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_revision5_analyses.py"
SPEC = importlib.util.spec_from_file_location("run_revision5_analyses", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load run_revision5_analyses.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Revision5AnalysisTest(unittest.TestCase):
    def test_absolute_deviation_matches_manual_definition(self):
        mapping = np.asarray([0, 0, 1, 4, 4], dtype=np.int16)
        lag = np.arange(len(mapping), dtype=np.float64) - mapping
        q = float(np.mean(np.abs(lag)))
        b = float(np.mean(lag))
        self.assertAlmostEqual(q, 0.6)
        self.assertAlmostEqual(b, 0.2)
        self.assertGreaterEqual(q, abs(b))

    def test_signal_summary_uses_reservation_aware_revision4_methods(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            revision4_dir = Path(temp_dir)
            payload = {
                "nested_matched_target_sensing": {"methods": {"fnirs_only": {"participant_macro_mae_seconds": 0.848}}},
                "reservation_aware_sensing": {"nested_matched_target_sensing": {"methods": {"fnirs_only": {"participant_macro_mae_seconds": 0.897}}}},
            }
            (revision4_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            methods = MODULE.load_reservation_aware_sensing_methods(revision4_dir)
            self.assertAlmostEqual(methods["fnirs_only"]["participant_macro_mae_seconds"], 0.897)

    def test_cbramod_manifest_checkpoint_is_portable(self):
        manifest = {"checkpoint": str(PROJECT_ROOT / "checkpoints" / "model.pth"), "sha256": "abc"}
        sanitized = MODULE.sanitize_cbramod_manifest(manifest)
        self.assertEqual(sanitized["checkpoint"], "checkpoints/model.pth")
        self.assertNotIn(str(PROJECT_ROOT), str(sanitized))
        self.assertEqual(sanitized["sha256"], "abc")

    def test_cluster_reference_excludes_every_focal_identity_copy(self):
        selected = np.asarray([3, 7, 3, 9, 7, 2], dtype=np.int16)
        for focal in range(len(selected)):
            sources = MODULE.cluster_reference_sources(selected, focal)
            self.assertFalse(np.any(sources == selected[focal]))
            expected = selected[selected != selected[focal]]
            np.testing.assert_array_equal(sources, expected)

    def test_derangement_has_no_fixed_points(self):
        rng = np.random.default_rng(17)
        for size in (2, 4, 5, 24):
            permutation = MODULE.random_derangement(size, rng)
            self.assertEqual(sorted(permutation.tolist()), list(range(size)))
            self.assertTrue(np.all(permutation != np.arange(size)))

    def test_percentile_interval_is_ordered(self):
        values = np.arange(100, dtype=np.float64)
        low, high = MODULE.percentile_interval(values)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, values.min())
        self.assertLessEqual(high, values.max())


if __name__ == "__main__":
    unittest.main()
