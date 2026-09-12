from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManuscriptStructureTest(unittest.TestCase):
    def test_seven_sections_and_two_evaluation_layers(self):
        body = (ROOT / "paper/body.tex").read_text()
        self.assertEqual(re.findall(r"\\section\{([^}]+)\}", body), [
            "Introduction", "Background and Related Work",
            "Evidence-to-Action Audit Protocol", "Evaluation Methods",
            "Results", "Discussion", "Conclusion",
        ])
        self.assertIn(r"\subsection{Protocol Evaluation}", body)
        self.assertIn(r"\subsection{Worked-Case Evidence Generation}", body)
        self.assertIn("permission-boundary changes, and rule ablations", body)
        self.assertIn("not counted as additional resolved audit routes", body)
        self.assertIn("Candidate MAE", body)
        self.assertIn("not this reported candidate estimate", body)

    def test_copyediting_boundaries_remain_distinct(self):
        body = (ROOT / "paper/body.tex").read_text()
        rules = body.split(r"\subsection{Decision Rules and Precedence}", 1)[1]
        rules = rules.split(r"\subsection{Applying the Protocol:", 1)[0]
        self.assertEqual(rules.count("normative"), 1)
        self.assertEqual(rules.count("optimal"), 1)
        self.assertIn("Rule ablation tests whether each declared constraint", rules)
        self.assertIn("\n\nEstimator and reference choices materially changed", body)
        self.assertIn(
            "The present evaluations establish neither independent analyst reuse "
            "nor direct interaction benefit.", body
        )
        self.assertNotIn("the present reorganization of evidence", body)

    def test_abstract_and_keywords(self):
        main = (ROOT / "paper/main.tex").read_text()
        abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                             main, re.S).group(1)
        self.assertEqual(len(abstract.split()), 139)
        self.assertIn("profile interpretation remained interface-contingent", abstract)
        self.assertIn("HCI methodology", main)

    def test_review_bundle_includes_complete_inputs_without_data(self):
        path = ROOT / "scripts/build_manuscript_review_bundle.py"
        spec = importlib.util.spec_from_file_location("review_bundle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        files = module.payloads()
        for suffix in ("", "_zh"):
            self.assertIn(f"paper/body{suffix}.tex", files)
            self.assertIn(f"paper/main{suffix}.pdf", files)
        for name in files:
            self.assertNotIn("data", Path(name).parts)
            self.assertNotIn(".git", Path(name).parts)
            self.assertNotIn("HANDOFF", name)
            self.assertNotIn(Path(name).suffix, {".npy", ".npz", ".mp4", ".mat"})
        self.assertEqual(module.build_bytes(), module.build_bytes())


if __name__ == "__main__":
    unittest.main()
