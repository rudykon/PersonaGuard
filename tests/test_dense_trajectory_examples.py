"""Publication tests for the narrowly authorized Figure 5 curve release."""
import copy
import csv
import json
from pathlib import Path
import unittest

from scripts import prepare_dense_trajectory_examples as preparation
from scripts import build_anonymous_supplement as archive_builder

ROOT = Path(__file__).resolve().parents[1]


class DenseTrajectoryExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = json.loads((ROOT / "scripts/figures/dense_trajectory_examples.json").read_text())
        cls.source = json.loads((ROOT / "results/revision6_source.json").read_text())

    def test_three_examples_have_closed_anonymous_schema(self):
        preparation.validate_release(self.release)
        self.assertEqual([e["rank"] for e in self.release["examples"]], [91, 181, 270])
        self.assertEqual([len(e["time_seconds"]) for e in self.release["examples"]],
                         [122, 111, 122])
        self.assertEqual(self.release["selection"]["candidate_trials"], 360)
        self.assertEqual(self.release["selection"]["excluded_trials"], 0)
        self.assertEqual(self.source["summaries"]["dense_trajectory_examples"], self.release)

    def test_extra_identifier_or_curve_is_rejected(self):
        for key in ("participant_id", "video_id", "sample_ids"):
            changed = copy.deepcopy(self.release)
            changed["examples"][0][key] = "not-authorized"
            with self.assertRaises(ValueError):
                preparation.validate_release(changed)
        changed = copy.deepcopy(self.release)
        changed["examples"][0]["curves"]["extra_prediction"] = []
        with self.assertRaises(ValueError):
            preparation.validate_release(changed)

    def test_release_exactly_reproduces_frozen_archives_when_available(self):
        if not all((ROOT / p).exists() for p in preparation.ARCHIVES.values()):
            self.skipTest("Restricted OOF archives are not redistributed")
        expected, _ = preparation.prepare()
        self.assertEqual(self.release, expected)
        # Matches the already reported estimates; no new model fit is performed.
        observed = self.release["verification"]["all_trial_macro_mae"]
        for name, rounded in (("metadata", 31.333), ("content", 30.493),
                              ("residual", 30.499), ("sam", 26.289)):
            self.assertEqual(round(observed[name], 3), rounded)

    def test_csv_contains_every_authorized_coordinate(self):
        path = ROOT / "scripts/figures/source_data_dense_trajectory_examples.csv"
        with path.open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames,
                             ["example", "percentile", "time_seconds", "axis", *preparation.CURVES])
            rows = list(reader)
        self.assertEqual(len(rows), 710)
        lookup = {(r["example"], int(r["time_seconds"]), r["axis"]): r for r in rows}
        self.assertEqual(len(lookup), len(rows))
        for example in self.release["examples"]:
            for i, time in enumerate(example["time_seconds"]):
                for dimension, axis in enumerate(("valence", "arousal")):
                    row = lookup[example["label"], time, axis]
                    for curve in preparation.CURVES:
                        self.assertEqual(float(row[curve]), example["curves"][curve][i][dimension])

    def test_package_excludes_private_mapping_and_complete_archives(self):
        paths = [p.relative_to(ROOT).as_posix() for p in archive_builder.collect_files()]
        self.assertIn("scripts/figures/dense_trajectory_examples.json", paths)
        self.assertIn("scripts/figures/source_data_dense_trajectory_examples.csv", paths)
        self.assertFalse(any("private_selection" in p or p.endswith(".npz") for p in paths))

    def test_five_main_figures_and_canonical_renderer(self):
        for name in ("body.tex", "body_zh.tex"):
            body = (ROOT / "paper" / name).read_text()
            self.assertEqual(body.count(r"\begin{figure*}"), 5)
            self.assertIn("dense_trajectory_examples_embed.pdf", body)
        renderer = (ROOT / "scripts/figures/make_dense_trajectory_examples.py").read_text()
        self.assertIn("revision6_source.json", renderer)
        self.assertNotIn("np.load(", renderer)
        self.assertIn('SOURCE = ROOT / "results/revision6_source.json"', renderer)
        self.assertTrue((ROOT / "paper/figures/dense_trajectory_examples_embed.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
