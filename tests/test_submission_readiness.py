from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_submission_readiness.py"
SPEC = importlib.util.spec_from_file_location("generate_submission_readiness", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load submission-readiness generator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SubmissionReadinessTest(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(
            (ROOT / "paper_support" / "submission_readiness.json").read_text(encoding="utf-8")
        )

    def test_complete_disclosure_passes_strict_gate(self):
        record = MODULE.validate(self.document, strict=True)
        self.assertEqual(record["status"], MODULE.DISCLOSURE_COMPLETE)
        self.assertFalse(record["additional_institutional_determination_claimed"])

    def test_generated_sentence_matches_record(self):
        record = MODULE.validate(self.document)
        self.assertEqual(
            MODULE.render(record, language="en"),
            (ROOT / "paper" / "generated" / "secondary_analysis_ethics_statement.tex").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(
            MODULE.render(record, language="zh"),
            (ROOT / "paper" / "generated" / "secondary_analysis_ethics_statement_zh.tex").read_text(
                encoding="utf-8"
            ),
        )

    def test_complete_disclosure_requires_source_provenance(self):
        incomplete = json.loads(json.dumps(self.document))
        record = incomplete["secondary_analysis_ethics"]
        record["source_approval_reference"] = ""
        with self.assertRaises(ValueError):
            MODULE.validate(incomplete, strict=True)


if __name__ == "__main__":
    unittest.main()
