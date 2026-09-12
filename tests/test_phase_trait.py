from __future__ import annotations

import unittest

import numpy as np

from merps.innovation.phase_trait import (
    corrected_item_total_correlations,
    estimate_variance_components,
    reliability_for_videos,
    videos_required,
)


class PhaseTraitTests(unittest.TestCase):
    def test_variance_components_recover_balanced_additive_structure(self):
        participant = np.asarray([-1.5, -0.5, 0.5, 1.5])
        video = np.asarray([-2.0, 0.0, 2.0])
        matrix = 5.0 + participant[:, None] + video[None, :]
        components = estimate_variance_components(matrix)
        self.assertGreater(components.participant, 0.0)
        self.assertGreater(components.video, 0.0)
        self.assertAlmostEqual(components.residual, 0.0, places=10)
        self.assertAlmostEqual(
            reliability_for_videos(components, 1)["relative_g"], 1.0
        )

    def test_reliability_increases_with_video_count(self):
        rng = np.random.default_rng(7)
        participant = rng.normal(0.0, 1.0, size=60)
        video = rng.normal(0.0, 0.4, size=12)
        matrix = (
            participant[:, None]
            + video[None, :]
            + rng.normal(0.0, 1.0, size=(60, 12))
        )
        components = estimate_variance_components(matrix)
        one = reliability_for_videos(components, 1)["relative_g"]
        four = reliability_for_videos(components, 4)["relative_g"]
        self.assertGreater(four, one)
        self.assertIsNotNone(videos_required(components, 0.7))

    def test_item_total_flags_video_that_tracks_participant_trait(self):
        participant = np.arange(10, dtype=np.float64)
        matrix = np.stack(
            [
                participant,
                participant + np.asarray([0, 1] * 5),
                np.asarray([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.float64),
            ],
            axis=1,
        )
        correlations = corrected_item_total_correlations(matrix)
        self.assertGreater(correlations[0], correlations[2])


if __name__ == "__main__":
    unittest.main()
