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
        self.document = {
            "schema_version": "submission-readiness-v2",
            "secondary_analysis_ethics": {
                "status": MODULE.DISCLOSURE_COMPLETE,
                "manuscript_sentence": "Example secondary-analysis disclosure.",
                "manuscript_sentence_zh": "示例分析使用 gated research release。",
                "source_approval_reference": "Example source approval",
                "data_access_basis": "Example research access terms",
                "analysis_scope": "Example secondary analysis",
                "additional_institutional_determination_claimed": False,
            },
        }

    def test_complete_disclosure_passes_strict_gate(self):
        record = MODULE.validate(self.document, strict=True)
        self.assertEqual(record["status"], MODULE.DISCLOSURE_COMPLETE)
        self.assertFalse(record["additional_institutional_determination_claimed"])

    def test_generated_sentence_matches_record(self):
        record = MODULE.validate(self.document)
        self.assertEqual(
            MODULE.render(record, language="en").splitlines()[1:],
            ["Example secondary-analysis disclosure."],
        )
        self.assertEqual(
            MODULE.render(record, language="zh").splitlines()[1:],
            ["示例分析使用 受限研究发布集。"],
        )

    def test_cli_accepts_explicit_author_record(self):
        args = MODULE.parse_args(["--source", "author-record.json", "--check", "--strict"])
        self.assertEqual(args.source, Path("author-record.json"))
        self.assertTrue(args.check)
        self.assertTrue(args.strict)

    def test_complete_disclosure_requires_source_provenance(self):
        incomplete = json.loads(json.dumps(self.document))
        record = incomplete["secondary_analysis_ethics"]
        record["source_approval_reference"] = ""
        with self.assertRaises(ValueError):
            MODULE.validate(incomplete, strict=True)


if __name__ == "__main__":
    unittest.main()
