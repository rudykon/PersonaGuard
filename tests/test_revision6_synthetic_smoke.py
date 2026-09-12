from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_revision6_synthetic_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_revision6_synthetic_smoke", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load Revision-6 synthetic smoke test")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Revision6SyntheticSmokeTest(unittest.TestCase):
    def test_publication_structure_runs_without_gated_data(self):
        result = MODULE.run_smoke()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["uses_gated_data"])
        self.assertTrue(result["fold_digests_distinct"])
        self.assertTrue(result["machine_scope_structural_only"])
        self.assertEqual(result["threshold_rows"], 6)


if __name__ == "__main__":
    unittest.main()
