from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_submission_pdf.py"
SPEC = importlib.util.spec_from_file_location("validate_submission_pdf", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load PDF/UA validator")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PdfUaValidatorTest(unittest.TestCase):
    def test_pdfinfo_parser(self):
        parsed = MODULE.parse_pdfinfo(
            "Tagged: yes\nSuspects: no\nMetadata Stream: yes\nPages: 21\n"
        )
        self.assertEqual(parsed["Tagged"], "yes")
        self.assertEqual(parsed["Pages"], "21")

    def test_pdffonts_parser_uses_right_hand_status_columns(self):
        rows = MODULE.parse_pdffonts(
            "name type encoding emb sub uni object ID\n"
            "---- ---- -------- --- --- --- ---------\n"
            "ABC+Font Type 1 Custom yes yes yes 10 0\n"
            "XYZ+Font CID TrueType Identity-H yes yes yes 11 0\n"
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["embedded"] == "yes" for row in rows))
        self.assertTrue(all(row["unicode_map"] == "yes" for row in rows))

    def test_capture_tolerates_non_utf8_diagnostic_bytes(self):
        completed = MODULE.run_capture(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'\\xfe')",
            ]
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "\ufffd")


if __name__ == "__main__":
    unittest.main()
