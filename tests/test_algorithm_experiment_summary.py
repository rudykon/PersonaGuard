from __future__ import annotations

import math
import unittest

from scripts.summarize_algorithm_experiments import build_summary, markdown


class AlgorithmExperimentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = build_summary()

    def test_frozen_deployment_choices_match_metrics(self) -> None:
        self.assertEqual(
            self.summary["schema_version"],
            "merps-algorithm-experiment-summary-v2",
        )
        zero = self.summary[
            "zero_interaction_held_out_participant_held_out_released_video"
        ]
        candidates = zero["candidates"]
        observed = min(candidates, key=lambda row: row["trial_macro_mae"])
        self.assertEqual(observed["name"], zero["selected_model"])
        self.assertEqual(zero["selected_model"], "clip_siglip")
        self.assertFalse(zero["router"]["adopted"])
        self.assertAlmostEqual(
            zero["metadata_prior_trial_macro_mae"]
            - zero["selected_trial_macro_mae"],
            zero["selected_gain_vs_metadata"],
        )
        self.assertIn("fixed 15-video support", self.summary["evidence_scope"][
            "video_inference_boundary"
        ])

    def test_stimulus_and_offset_evidence_is_complete(self) -> None:
        integrity = self.summary["stimulus_integrity"]
        self.assertEqual(integrity["files_expected"], 15)
        self.assertEqual(integrity["files_present"], 15)
        self.assertEqual(integrity["identities_verified"], 15)
        offsets = integrity["media_minus_annotation_seconds"]["per_video"]
        self.assertEqual(len(offsets), 15)
        self.assertTrue(all(math.isfinite(value) for value in offsets))
        self.assertAlmostEqual(min(offsets), 1.7)
        self.assertAlmostEqual(max(offsets), 2.833333, places=5)

    def test_physiology_and_sparse_conditions_are_not_conflated(self) -> None:
        physiology = self.summary["causal_physiology_residual"]
        sparse = self.summary["one_post_trial_sam_sparse_recovery"]
        self.assertFalse(physiology["full_modalities"]["adopted"])
        self.assertFalse(physiology["temporal_token_modalities"]["adopted"])
        self.assertTrue(sparse["adopted_for_post_trial_only"])
        self.assertTrue(sparse["not_zero_interaction"])
        self.assertGreater(sparse["gain_vs_canonical_prior"], 0.0)
        self.assertAlmostEqual(
            physiology["full_modalities"]["primary_trial_macro_mae"],
            30.498715394073063,
        )
        self.assertAlmostEqual(
            physiology["temporal_token_modalities"]["primary_trial_macro_mae"],
            30.49315748612086,
        )

    def test_oof_artifacts_share_identity_targets_and_finite_values(self) -> None:
        audit = self.summary["oof_artifact_consistency"]
        self.assertEqual(audit["artifacts_checked"], 12)
        self.assertEqual(audit["samples"], 36864)
        self.assertTrue(audit["sample_identity_consistent"])
        self.assertTrue(audit["targets_identical"])
        self.assertTrue(audit["all_numeric_arrays_finite"])
        self.assertGreaterEqual(audit["numeric_arrays_checked"], 270)

    def test_all_input_hashes_are_sha256(self) -> None:
        identities = list(self.summary["input_identities"].values())
        identities.extend(
            self.summary["oof_artifact_consistency"]["artifact_identities"].values()
        )
        identities.append(self.summary["generator_identity"])
        for identity in identities:
            digest = identity["sha256"]
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_markdown_contains_frozen_numbers(self) -> None:
        rendered = markdown(self.summary)
        self.assertIn("30.493", rendered)
        self.assertIn("26.289", rendered)
        self.assertIn("15/15", rendered)


if __name__ == "__main__":
    unittest.main()
